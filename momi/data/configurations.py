import itertools as it
import autograd.numpy as np
from scipy.special import comb
from .compressed_counts import _config2hashable
from ..util import memoize_instance


def build_config_list(sampled_pops, counts, sampled_n=None, ascertainment_pop=None):
    """
    if sampled_n is not None, counts is the derived allele counts

    if sampled_n is None, counts has an extra trailing axis:
      counts[...,0] is ancestral allele count,
      counts[...,1] is derived allele count
    """
    if sampled_n is not None:
        sampled_n = np.array(sampled_n, dtype=int)
        counts1 = np.array(counts, dtype=int, ndmin=2)
        counts0 = sampled_n - counts1
        counts = np.array([counts0, counts1], dtype=int)
        counts = np.transpose(counts, axes=[1, 2, 0])
    counts = np.array(counts, ndmin=3, dtype=int)
    assert counts.shape[1:] == (len(sampled_pops), 2)
    counts.setflags(write=False)
    return ConfigList(sampled_pops, counts, sampled_n, ascertainment_pop)


def build_full_config_list(sampled_pops, sampled_n, ascertainment_pop=None):
    sampled_n = np.array(sampled_n)
    if ascertainment_pop is None:
        ascertainment_pop = [True] * len(sampled_pops)
    ascertainment_pop = np.array(ascertainment_pop)

    ranges = [list(range(n + 1)) for n in sampled_n]
    config_list = []
    for x in it.product(*ranges):
        x = np.array(x, dtype=int)
        if not (np.all(x[ascertainment_pop] == 0) or np.all(
                x[ascertainment_pop] == sampled_n[ascertainment_pop])):
            config_list.append(x)
    return build_config_list(
        sampled_pops, np.array(config_list, dtype=int), sampled_n,
        ascertainment_pop=ascertainment_pop)


class ConfigList(object):
    """
    Stores a list of configs. Important methods/attributes:

    ConfigList.sampled_pops: the population labels
    ConfigList[i] : the i-th config in the list
    ConfigList.sampled_n : the number of alleles sampled per population.
                        used to construct the likelihood vectors for
                        junction tree algorithm.
    """
    def __init__(self, sampled_pops, conf_arr, sampled_n=None,
                 ascertainment_pop=None):
        """Use build_config_list() instead of calling this constructor directly"""
        # If sampled_n=None, ConfigList.sampled_n will be the max number of
        # observed individuals/alleles per population.
        self.sampled_pops = tuple(sampled_pops)
        self.value = conf_arr

        if ascertainment_pop is None:
            ascertainment_pop = [True] * len(sampled_pops)
        self.ascertainment_pop = np.array(ascertainment_pop)
        self.ascertainment_pop.setflags(write=False)
        if all(not a for a in self.ascertainment_pop):
            raise ValueError(
                "At least one of the populations must be used for "
                "ascertainment of polymorphic sites")

        max_n = np.max(np.sum(self.value, axis=2), axis=0)

        if sampled_n is None:
            sampled_n = max_n
        sampled_n = np.array(sampled_n)
        if np.any(sampled_n < max_n):
            raise ValueError("config greater than sampled_n")
        self.sampled_n = sampled_n
        if not np.sum(sampled_n[self.ascertainment_pop]) >= 2:
            raise ValueError("The total sample size of the ascertainment "
                             "populations must be >= 2")

        config_sampled_n = np.sum(self.value, axis=2)
        self.has_missing_data = np.any(config_sampled_n != self.sampled_n)

        if np.any(np.sum(self.value[:, self.ascertainment_pop, :], axis=1)
                  == 0):
            raise ValueError("Monomorphic sites not allowed. In addition, all"
                             " sites must be polymorphic when restricted to"
                             " the ascertainment populations")

    @property
    def config_array(self):
        return self.value

    def __getitem__(self, args):
        return self.value.__getitem__(args)

    def __len__(self): return len(self.value)

    def __eq__(self, other):
        conf_arr = self.value
        try:
            return np.all(conf_arr == other.value)
        except AttributeError:
            return False

    @memoize_instance
    def as_tuple(self):
        ret = []
        for c in self:
            c = tuple(map(tuple, c))
            ret.append(c)
        return tuple(ret)

    def count_subsets(self, derived_weights_dict, total_counts_dict):
        """
        For each config, count number of subsets with given sample
        size, weighted according to the number of derived alleles.

        Parameters
        total_counts_dict: dict mapping pop to n_pop
        derived_weights_dict: dict mapping pop to list of floats
           with length n_pop+1, giving the weight for each
           derived allele count

        Returns
        numpy.ndarray of weighted counts for each config
        """
        assert (set(derived_weights_dict.keys())
                <= set(total_counts_dict.keys()))

        ret = np.ones(len(self))
        for p, n in total_counts_dict.items():
            i = self.sampled_pops.index(p)
            if p in derived_weights_dict:
                curr = np.zeros(len(self))
                assert len(derived_weights_dict[p]) == n+1
                for d, w in enumerate(derived_weights_dict[p]):
                    if w == 0:
                        continue
                    a = n - d
                    curr += w * (comb(self.value[:, i, 0], a)
                                 * comb(self.value[:, i, 1], d))
                ret *= curr
            else:
                ret *= comb(self.value[:, i, :].sum(axis=1), n)
        return ret

    def subsample_probs(self, subconfig):
        """
        Returns the probability of subsampling subconfig
        from each config.
        """
        subconfig = np.array(subconfig)
        total_counts_dict = {p: n for p, n in zip(self.sampled_pops,
                                                  subconfig.sum(axis=1))
                             if n > 0}
        derived_counts_dict = {p: [0]*(n+1)
                               for p, n in total_counts_dict.items()}
        for p, d in zip(self.sampled_pops, subconfig[:, 1]):
            if p in derived_counts_dict:
                derived_counts_dict[p][d] = 1

        num = self.count_subsets(derived_counts_dict, total_counts_dict)
        denom = self.count_subsets({}, total_counts_dict)

        # avoid 0/0
        assert np.all(num[denom == 0] == 0)
        denom[denom == 0] = 1
        return num / denom

    # TODO: remove this method (and self.sampled_n attribute)
    def _copy(self, sampled_n=None):
        """
        Notes
        -----
        Note that momi.expected_sfs, momi.composite_log_likelihood require
        Demography.sampled_n == ConfigList.sampled_n.
        If this is not the case, you can use _copy() to create a copy with the
        correct sampled_n.
        Note this has no affect on the actual allele counts, as missing data
        is allowed. sampled_n is just used to construct (and store) certain
        vectors for the SFS algorithm.
        """
        if sampled_n is None:
            sampled_n = self.sampled_n
        return ConfigList(self.sampled_pops, self.value, sampled_n=sampled_n,
                           ascertainment_pop=self.ascertainment_pop)

    def _vecs_and_idxs(self, folded):
        augmented_configs = self._augmented_configs(folded)
        augmented_idxs = self._augmented_idxs(folded)

        # construct the vecs
        vecs = [np.zeros((len(augmented_configs), n + 1))
                for n in self.sampled_n]

        for i in range(len(vecs)):
            n = self.sampled_n[i]
            derived = np.einsum(
                "i,j->ji", np.ones(len(augmented_configs)), np.arange(n + 1))
            curr = (comb(derived, augmented_configs[:, i, 1])
                    * comb(n - derived, augmented_configs[:, i, 0])
                    / comb(n, np.sum(augmented_configs[:, i, :], axis=1)))
            assert not np.any(np.isnan(curr))
            vecs[i] = np.transpose(curr)

        # copy augmented_idxs to make it safe
        return vecs, dict(augmented_idxs)

    # def _config_str_iter(self):
    #     for c in self.value:
    #         yield _config2hashable(c)

    def _augmented_configs(self, folded):
        return self._build_augmented_configs_idxs(folded)[0]

    def _augmented_idxs(self, folded):
        return self._build_augmented_configs_idxs(folded)[1]

    @memoize_instance
    def _build_augmented_configs_idxs(self, folded):
        augmented_configs = []
        augmented_config_2_idx = {}  # maps config -> row in vecs

        def augmented_idx(config):
            hashed = _config2hashable(config)
            try:
                return augmented_config_2_idx[hashed]
            except KeyError:
                idx = len(augmented_configs)
                assert idx == len(augmented_config_2_idx)

                augmented_config_2_idx[hashed] = idx
                augmented_configs.append(config)
                return idx

        # initial row is a "zero" config
        null_idx = augmented_idx(
            np.array([(-1, 1)] * len(self.sampled_pops), dtype=int))

        # next row is normalization constant
        denom_idx = augmented_idx(np.zeros((len(self.sampled_pops), 2)))

        # get row indices for each config
        idx_2_row = np.array(list(map(augmented_idx, self)), dtype=int)

        # remove monomorphic configs
        # (if there is missing data or error matrices,
        # expected_sfs_tensor_prod will return nonzero SFS
        # for monomorphic configs)
        monomorphic = np.any(np.sum(self.value, axis=1) == 0, axis=1)
        idx_2_row[monomorphic] = null_idx

        # get row indices for each denominator
        sample_sizes_array = np.sum(self.value, axis=2)
        if np.any(sample_sizes_array > self.sampled_n):
            raise Exception("There is a config that is larger than the"
                            " specified sample size!")

        sample_sizes = [tuple(s) for s in sample_sizes_array]
        # corrections for monomorphic sites (all ancestral & all derived)
        ssize_2_corrections = [{}, {}]
        for s in set(sample_sizes):
            # add rows for monomorphic correction terms
            for mono_allele in (0, 1):
                mono_config = tuple(ss if asc else 0 for ss,
                                    asc in zip(s, self.ascertainment_pop))
                mono_config = np.array(
                    [mono_config, [0] * len(mono_config)], dtype=int, ndmin=2)
                if mono_allele == 1:
                    mono_config = mono_config[::-1, :]
                mono_config = np.transpose(mono_config)
                ssize_2_corrections[mono_allele][
                    s] = augmented_idx(mono_config)
        corrections_2_denom = [np.array([corr_row[s] for s in sample_sizes],
                                        dtype=int)
                               for corr_row in ssize_2_corrections]

        # get row indices for folded configs
        if folded:
            rev_confs = self.value[:, :, ::-1]
            is_symm = np.all(self.value == rev_confs, axis=(1, 2))
            folded_2_row = []
            for rc, symm in zip(rev_confs, is_symm):
                if symm:
                    # map to 0 if symmetric
                    folded_2_row += [null_idx]
                else:
                    folded_2_row += [augmented_idx(rc)]
            folded_2_row = np.array(folded_2_row, dtype=int)
            # dont use monomorphic configs
            folded_2_row[monomorphic] = null_idx

        idxs = {'denom_idx': denom_idx, 'idx_2_row': idx_2_row}
        assert len(corrections_2_denom) == 2
        idxs.update({('corrections_2_denom', 0): corrections_2_denom[0],
                     ('corrections_2_denom', 1): corrections_2_denom[1]})
        try:
            idxs['folded_2_row'] = folded_2_row
        except UnboundLocalError:
            pass

        return np.array(augmented_configs, dtype=int), idxs


class _ConfigList_Subset(ConfigList):
    # Efficient access to subset of configs
    def __init__(self, configs, sub_idxs):
        self.sub_idxs = sub_idxs
        self.full_configs = configs
        for a in ("sampled_n", "sampled_pops",
                  "has_missing_data", "ascertainment_pop"):
            setattr(self, a, getattr(self.full_configs, a))

    @property
    def value(self):
        return self.full_configs.value[self.sub_idxs, :, :]

    def __getitem__(self, args):
        if isinstance(args, tuple):
            arg0 = self.sub_idxs[args[0]]
            rest = tuple([slice(None)] + list(args[1:]))
            return self.full_configs[arg0][rest]
        else:
            return self.full_configs[self.sub_idxs[args]]

    def __iter__(self):
        for i in self.value:
            yield i

    def __len__(self):
        return len(self.sub_idxs)

    # def _vecs_and_idxs(self, folded):
    #     vecs,_ = self.full_configs._vecs_and_idxs(folded)
    #     old_idxs, idxs = self._build_idxs(folded)

    #     vecs = [v[old_idxs,:] for v in vecs]
    #     ## copy idxs to make it safe
    #     return vecs, dict(idxs)

    def _augmented_configs(self, folded):
        return self.full_configs._augmented_configs(
            folded)[self._build_old_new_idxs(folded)[0], :, :]

    def _augmented_idxs(self, folded):
        return self._build_old_new_idxs(folded)[1]

    @memoize_instance
    def _build_old_new_idxs(self, folded):
        idxs = self.full_configs._augmented_idxs(folded)

        denom_idx_key = 'denom_idx'
        denom_idx = idxs[denom_idx_key]
        idxs = {k: v[self.sub_idxs]
                for k, v in list(idxs.items()) if k != denom_idx_key}

        old_idxs = np.array(
            list(set(sum(map(list, idxs.values()), [denom_idx]))))
        old_2_new_idxs = {old_id: new_id for new_id,
                          old_id in enumerate(old_idxs)}

        idxs = {k: np.array([old_2_new_idxs[old_id]
                             for old_id in v], dtype=int)
                for k, v in list(idxs.items())}
        idxs[denom_idx_key] = old_2_new_idxs[denom_idx]
        return old_idxs, idxs
