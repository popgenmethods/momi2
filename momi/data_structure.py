
from cached_property import cached_property
from .util import memoize_instance
import autograd.numpy as np
import scipy
import scipy.misc
import scipy.sparse
from scipy.misc import comb
from .math_functions import _apply_error_matrices
from collections import Counter
import warnings
import itertools as it
import json


def config_array(sampled_pops, counts, sampled_n=None, ascertainment_pop=None):
    """
    if sampled_n is not None, counts is the derived allele counts
    if sampled_n is None, counts has an extra trailing axis, counts[...,0] is ancestral allele count, counts[...,1] is derived allele count
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
    return ConfigArray(sampled_pops, counts, sampled_n, ascertainment_pop)


def full_config_array(sampled_pops, sampled_n, ascertainment_pop=None):
    sampled_n = np.array(sampled_n)
    if ascertainment_pop is None:
        ascertainment_pop = [True] * len(sampled_pops)
    ascertainment_pop = np.array(ascertainment_pop)

    ranges = [list(range(n + 1)) for n in sampled_n]
    config_list = []
    for x in it.product(*ranges):
        x = np.array(x, dtype=int)
        if not (np.all(x[ascertainment_pop] == 0) or np.all(x[ascertainment_pop] == sampled_n[ascertainment_pop])):
            config_list.append(x)
    return config_array(sampled_pops, np.array(config_list, dtype=int), sampled_n,
                        ascertainment_pop=ascertainment_pop)


class ConfigArray(object):
    """
    Stores a list of configs. Important methods/attributes:

    ConfigArray.sampled_pops: the population labels
    ConfigArray[i] : the i-th config in the list
    ConfigArray.sampled_n : the number of alleles sampled per population.
                        used to construct the likelihood vectors for
                        junction tree algorithm.
    """

    def __init__(self, sampled_pops, conf_arr, sampled_n=None, ascertainment_pop=None):
        """Use config_array() instead of calling this constructor directly"""
        # If sampled_n=None, ConfigArray.sampled_n will be the max number of
        # observed individuals/alleles per population.
        self.sampled_pops = tuple(sampled_pops)
        self.value = conf_arr

        if ascertainment_pop is None:
            ascertainment_pop = [True] * len(sampled_pops)
        self.ascertainment_pop = np.array(ascertainment_pop)
        self.ascertainment_pop.setflags(write=False)
        if all(not a for a in self.ascertainment_pop):
            raise ValueError(
                "At least one of the populations must be used for ascertainment of polymorphic sites")

        max_n = np.max(np.sum(self.value, axis=2), axis=0)

        if sampled_n is None:
            sampled_n = max_n
        sampled_n = np.array(sampled_n)
        if np.any(sampled_n < max_n):
            raise ValueError("config greater than sampled_n")
        self.sampled_n = sampled_n
        if not np.sum(sampled_n[self.ascertainment_pop]) >= 2:
            raise ValueError(
                "The total sample size of the ascertainment populations must be >= 2")

        config_sampled_n = np.sum(self.value, axis=2)
        self.has_missing_data = np.any(config_sampled_n != self.sampled_n)

        #self.has_monomorphic = _has_monomorphic(self.value)
        if np.any(np.sum(self.value[:, self.ascertainment_pop, :], axis=1) == 0):
            raise ValueError(
                "Monomorphic sites not allowed. In addition, all sites must be polymorphic when restricted to the ascertainment populations")

    def __getitem__(self, *args): return self.value.__getitem__(*args)

    def __len__(self): return len(self.value)

    def __eq__(self, other):
        conf_arr = self.value
        try:
            return np.all(conf_arr == other.value)
        except AttributeError:
            return False

    # TODO: remove this method (and self.sampled_n attribute)
    def _copy(self, sampled_n=None):
        """
        Notes
        -----
        Note that momi.expected_sfs, momi.composite_log_likelihood require
        Demography.sampled_n == ConfigArray.sampled_n.
        If this is not the case, you can use _copy() to create a copy with the correct
        sampled_n.
        Note this has no affect on the actual allele counts, as missing data is allowed.
        sampled_n is just used to construct (and store) certain vectors for the SFS algorithm.
        """
        if sampled_n is None:
            sampled_n = self.sampled_n
        return ConfigArray(self.sampled_pops, self.value, sampled_n=sampled_n, ascertainment_pop=self.ascertainment_pop)

    def _vecs_and_idxs(self, folded):
        #augmented_configs, augmented_idxs = self._build_augmented_configs_idxs(folded)
        augmented_configs = self._augmented_configs(folded)
        augmented_idxs = self._augmented_idxs(folded)

        # construct the vecs
        vecs = [np.zeros((len(augmented_configs), n + 1))
                for n in self.sampled_n]

        for i in range(len(vecs)):
            n = self.sampled_n[i]
            derived = np.einsum(
                "i,j->ji", np.ones(len(augmented_configs)), np.arange(n + 1))
            curr = comb(derived, augmented_configs[:, i, 1]) * comb(n - derived, augmented_configs[
                :, i, 0]) / comb(n, np.sum(augmented_configs[:, i, :], axis=1))
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
            raise Exception(
                "There is a config that is larger than the specified sample size!")

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
        corrections_2_denom = [np.array([corr_row[s] for s in sample_sizes], dtype=int)
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


def site_freq_spectrum(sampled_pops, loci):
    """
    Parameters
    ----------
    sampled_pops : list of the population labels
    loci : list of dicts, or list of lists

           if loci[i] is a dict, then loci[i][config]
           is the count of the config at locus i

           if loci[i] is a list, then loci[i][j]
           is the config of the j-th SNP at locus i

    See also
    --------
    site_freq_spectrum.load(): load in sfs from file created by Sfs.dump()
    """
    index2loc = []
    index2count = []
    loci_counters = []

    def chained_sequences():
        for loc, locus_iter in enumerate(loci):
            loci_counters.append(Counter())
            try:
                locus_iter = locus_iter.items()
            except AttributeError:
                locus_iter = zip(locus_iter, iter(lambda: 1, 0))
            for config, count in locus_iter:
                index2loc.append(loc)
                index2count.append(count)
                yield config

    compressed_counts = CompressedAlleleCounts.from_iter(
        chained_sequences(), len(sampled_pops))
    config_array = compressed_counts.config_array
    index2uniq = compressed_counts.index2uniq

    assert len(index2loc) == len(index2count) and len(
        index2count) == len(index2uniq)
    for loc, count, uniq in zip(index2loc, index2count, index2uniq):
        loci_counters[loc][uniq] += count

    configs = ConfigArray(sampled_pops, config_array, None, None)
    return Sfs(loci_counters, configs)


def load_sfs(f):
    """
    Read in Sfs that has been written to file by Sfs.dump()
    """
    info = json.load(f)

    loci = []
    for locus, locus_rows in it.groupby(info["(locus,config_id,count)"], lambda x: x[0]):
        loci.append({config_id: count
                     for _, config_id, count in locus_rows})

    configs = ConfigArray(info["sampled_pops"],
                          np.array(info["configs"]))

    return Sfs(loci, configs)
site_freq_spectrum.load = load_sfs


class Sfs(object):
    """
    Represents an observed SFS across several loci.

    Important methods/attributes:
    """

    def __init__(self, loci, configs):
        self.configs = configs

        self.loc_idxs, self.loc_counts = [], []
        for loc in loci:
            if len(loc) == 0:
                self.loc_idxs.append(np.array([], dtype=int))
                self.loc_counts.append(np.array([], dtype=float))
            else:
                try:
                    items = loc.items()
                except:
                    loc = np.array(loc)
                    if len(loc.shape) == 2:
                        assert loc.shape[0] == 2
                        idxs, cnts = loc[0,:], loc[1,:]
                    else:
                        idxs, cnts = np.unique(loc, return_counts=True)
                else:
                    idxs, cnts = zip(*loc.items())
                self.loc_idxs.append(np.array(idxs, dtype=int))
                self.loc_counts.append(np.array(cnts, dtype=float))

        if len(self.loc_idxs) > 1:
            self._total_freqs = np.array(np.squeeze(np.asarray(self.freqs_matrix.sum(axis=1))),
                                        ndmin=1)
        else:
            # avoid costly building of frequency matrix, when there are many Sfs's of a single locus
            # (e.g. in many stochastic minibatches)
            idxs, = self.loc_idxs
            cnts, = self.loc_counts
            self._total_freqs = np.zeros(len(self.configs))
            self._total_freqs[idxs] = cnts

    def dump(self, f):
        """
        Write the Sfs in a compressed JSON format,
        that can be read in by site_freq_spectrum.load()
        """
        print("{", file=f)
        print('\t"sampled_pops": {},'.format(
            json.dumps(list(self.sampled_pops))), file=f)
        print('\t"configs": [', file=f)
        n_configs = len(self.configs.value)
        for i, c in enumerate(self.configs.value.tolist()):
            if i != n_configs - 1:
                print("\t\t{},".format(c), file=f)
            else:
                # no trailing comma
                print("\t\t{}".format(c), file=f)
        print("\t],", file=f)
        print('\t"(locus,config_id,count)": [', file=f)
        for loc in range(self.n_loci):
            n_loc_rows = len(self.loc_idxs[loc])
            assert n_loc_rows == len(self.loc_counts[loc])
            for i in range(n_loc_rows):
                config_id = int(self.loc_idxs[loc][i])
                count = float(self.loc_counts[loc][i])
                if loc == self.n_loci - 1 and i == n_loc_rows - 1:
                    # no trailing comma
                    print("\t\t{}".format(json.dumps(
                        (loc, config_id, count))), file=f)
                else:
                    print("\t\t{},".format(json.dumps(
                        (loc, config_id, count))), file=f)
        print("\t]", file=f)
        print("}", file=f)

    def combine_loci(self):
        """
        Returns a copy of this SFS, but with all loci combined into a single locus
        """
        return _sub_sfs(self.configs, self._total_freqs)

    @property
    def freqs_matrix(self):
        """
        Returns the frequencies as a sparse matrix;
        freqs_matrix[i, j] is the frequency of Sfs.configs[i] at locus j
        """
        try:
            self._freqs_matrix
        except:
            self._freqs_matrix = _freq_matrix_from_counters(
                self.loc_idxs, self.loc_counts, len(self.configs))
        return self._freqs_matrix

    @cached_property
    def avg_pairwise_hets(self):
        """
        Returns the number of SNPs where a single individual is heterozygote,
        averaged over all individuals within each population

        Returns numpy.ndarray pairwise_hets, where
        pairwise_hets[i,j] is the average number of hets in population j at locus i
        """
        n_nonmissing = np.sum(self.configs.value, axis=2)
        # for denominator, assume 1 allele is drawn from whole sample, and 1
        # allele is drawn only from nomissing alleles
        denoms = np.maximum(n_nonmissing * (self.sampled_n - 1), 1.0)
        p_het = 2 * self.configs.value[:, :, 0] * \
            self.configs.value[:, :, 1] / denoms

        return self.freqs_matrix.T.dot(p_het)

    @cached_property
    def p_missing(self):
        if not self.configs.has_missing_data:
            return 0.0
        return 1.0 - np.einsum("ijk,i->j", self.configs.value, self._total_freqs / float(self.n_snps())) / self.sampled_n
        #n_missing = self.sampled_n - np.sum(self.configs.value, axis=2)
        #ret = [Counter(n_missing[:,i]) for i in range(n_missing.shape[1])]
        #n_snps = len(self.configs)
        #ret = [[cnts[i]/float(n_snps) for i in range(n+1)] for cnts,n in zip(ret, self.sampled_n)]
        # return ret

    @property
    def sampled_n(self):
        return self.configs.sampled_n

    @property
    def ascertainment_pop(self):
        return self.configs.ascertainment_pop

    @property
    def sampled_pops(self):
        return self.configs.sampled_pops

    @property
    def n_loci(self):
        return len(self.loc_idxs)

    @property
    def n_nonzero_entries(self):
        return len(self.configs)

    @memoize_instance
    def n_snps(self, vector=False, locus=None):
        if vector:
            assert locus is None
            return np.array([self.n_snps(locus=loc) for loc in range(self.n_loci)])
        if locus is None:
            return np.sum(self._total_freqs)
        else:
            return sum(self.loc_counts[locus])

    def __eq__(self, other):
        loci_dicts = self._get_dict(vector=True)
        try:
            return loci_dicts == other._get_dict(vector=True)
        except AttributeError:
            return False

    def _get_dict(self, vector=False, locus=None):
        if vector:
            assert locus is None
            return [self._get_dict(locus=loc) for loc in range(self.n_loci)]
        elif locus is None:
            return dict(zip(map(_config2hashable, self.configs), self._total_freqs))
        #idxs, counts = self._idxs_counts(locus)
        return dict(zip((_config2hashable(self.configs[i]) for i in self.loc_idxs[locus]), self.loc_counts[locus]))

    def to_dict(self, vector=False):
        if not vector:
            return {_hashed2config(k): v for k, v in self._get_dict().items()}
        else:
            return [{_hashed2config(k): v for k, v in d.items()}
                    for d in self._get_dict(vector=True)]

    @cached_property
    def _entropy(self):
        counts = self._total_freqs
        n_snps = float(self.n_snps())
        p = counts / n_snps
        # return np.sum(p * np.log(p))
        ret = np.sum(p * np.log(p))

        # correct for missing data
        sampled_n = np.sum(self.configs.value, axis=2)
        sampled_n_counts = Counter()
        assert len(counts) == len(sampled_n)
        for c, n in zip(counts, sampled_n):
            n = tuple(n)
            sampled_n_counts[n] += c
        sampled_n_counts = np.array(
            list(sampled_n_counts.values()), dtype=float)

        ret = ret + np.sum(sampled_n_counts / n_snps *
                           np.log(n_snps / sampled_n_counts))
        assert not np.isnan(ret)
        return ret

    def _get_muts_poisson_entropy(self, use_pairwise_diffs):
        if use_pairwise_diffs:
            return self._pairwise_muts_poisson_entropy
        else:
            return self._total_muts_poisson_entropy

    @cached_property
    def _total_muts_poisson_entropy(self):
        lambd = self.n_snps(vector=True)
        lambd = lambd[lambd > 0]
        return np.sum(-lambd + lambd * np.log(lambd) - scipy.special.gammaln(lambd + 1))

    @cached_property
    def _pairwise_muts_poisson_entropy(self):
        lambd = self.avg_pairwise_hets
        ret = -lambd + lambd * np.log(lambd) - scipy.special.gammaln(lambd + 1)
        ret[lambd <= 1e-16] = 0
        ret = ret * self.sampled_n / float(np.sum(self.sampled_n[self.ascertainment_pop]))
        return np.sum(ret[:,self.ascertainment_pop])

    def fold(self):
        """
        Returns a copy of the SFS, but with folded entries.
        """
        loci = []
        # for l in self.loci:
        for li, lc in zip(self.loc_idxs, self.loc_counts):
            loci += [Counter()]
            # for k,v in list(l.items()):
            for k, v in zip(li, lc):
                k = np.array(self.configs[k])
                if tuple(k[:, 0]) < tuple(k[:, 1]):
                    k = k[:, ::-1]
                k = _config2hashable(k)
                loci[-1][k] = loci[-1][k] + v

        def convert_loc(loc):
            ret = lambda: None

            def ret_items():
                for k, v in loc.items():
                    k = _hashed2config(k)
                    yield k, v
            ret.items = ret_items
            return ret

        loci = [convert_loc(loc) for loc in loci]
        ret = site_freq_spectrum(self.sampled_pops, loci)
        if np.any(ret.sampled_n != self.sampled_n):
            ret = ret._copy(sampled_n=self.sampled_n)
        return ret

    def _copy(self, sampled_n=None):
        """
        See also: ConfigArray._copy()
        """
        if sampled_n is None:
            sampled_n = self.sampled_n
        return Sfs([dict(zip(li, lc)) for li, lc in zip(self.loc_idxs, self.loc_counts)],
                   ConfigArray(self.sampled_pops, self.configs.value, sampled_n=sampled_n, ascertainment_pop=self.ascertainment_pop))

    def _integrate_sfs(self, weights, vector=False, locus=None):
        if vector:
            assert locus is None
            return np.array([self._integrate_sfs(weights, locus=loc) for loc in range(self.n_loci)])
        if locus is None:
            idxs, counts = slice(None), self._total_freqs
        else:
            idxs, counts = self.loc_idxs[locus], self.loc_counts[locus]
        #idxs, counts = self._idxs_counts(locus)
        return np.sum(weights[idxs] * counts)

    def subsample_inds(self, n):
        """
        Return the induced SFS on all subsets of n
        individuals.

        See also: momi.SegSites.subsample_inds()
        """
        subconfigs, weights = _get_subsample_counts(self.configs, n)
        freqs = np.array(self.freqs_matrix.T.dot(weights.T).T)
        assert freqs.shape == (weights.shape[0], self.n_loci)
        return site_freq_spectrum(self.sampled_pops,
                                  [{c: f for c, f in zip(subconfigs, loc_freqs) if f != 0}
                                   for loc_freqs in freqs.T])

    def _get_pairwise_missing_probs(self):
        return np.dot(self._total_freqs,
                      self.configs._get_pairwise_missing_probs())


def _freq_matrix_from_counters(idxs_by_loc, cnts_by_loc, n_configs):
    data = []
    indices = []
    indptr = []
    n_loc = 0
    for idxs, cnts in zip(idxs_by_loc, cnts_by_loc):
        indptr.append(len(data))
        data.extend(cnts)
        indices.extend(idxs)
        n_loc += 1
    indptr.append(len(data))
    return scipy.sparse.csc_matrix((data, indices, indptr), shape=(n_configs, n_loc)).tocsr()


def _get_subsample_counts(configs, n):
    config_sampled_n = np.sum(configs.value, axis=-1)

    def get_cnt(super_n, sub_n):
        assert super_n.shape[1:] == sub_n.shape
        denom = np.prod(
            comb(config_sampled_n, np.sum(sub_n, axis=-1)), axis=-1)
        cnt = np.prod(comb(super_n, sub_n), axis=(1, 2)) / denom
        cnt[denom == 0] = 0
        return cnt

    ret = {}
    for pop_comb in it.combinations_with_replacement(configs.sampled_pops, n):
        subsample_n = Counter(pop_comb)
        subsample_n = np.array([subsample_n[pop]
                                for pop in configs.sampled_pops], dtype=int)
        if np.any(subsample_n > configs.sampled_n):
            continue

        for sfs_entry in it.product(*(range(sub_n + 1)
                                             for sub_n in subsample_n)):
            sfs_entry = np.array(sfs_entry, dtype=int)
            if np.all(sfs_entry == 0) or np.all(sfs_entry == subsample_n):
                # monomorphic
                continue

            sfs_entry = np.transpose([subsample_n - sfs_entry, sfs_entry])
            sfs_entry = tuple(map(tuple, sfs_entry))
            cnt_vec = get_cnt(configs.value, np.array(sfs_entry, dtype=int))
            if not np.all(cnt_vec == 0):
                ret[sfs_entry] = cnt_vec

    subconfigs, weights = zip(*ret.items())
    return list(subconfigs), np.array(weights)


def _randomly_drop_alleles(seg_sites, p_missing, ascertainment_pop=None):
    p_missing = p_missing * np.ones(len(seg_sites.sampled_n))
    if ascertainment_pop is None:
        ascertainment_pop = np.array([True] * len(seg_sites.sampled_n))

    p_sampled = 1.0 - np.transpose([p_missing, p_missing])
    ret = []
    for locus in seg_sites:
        ret.append([])
        for config in locus:
            newconfig = np.random.binomial(config, p_sampled)
            if np.any(newconfig[ascertainment_pop, :].sum(axis=0) == 0):
                # monomorphic
                continue
            ret[-1].append(newconfig)
    return seg_site_configs(seg_sites.sampled_pops, ret, ascertainment_pop=ascertainment_pop)


def seg_site_configs(sampled_pops, config_sequences, ascertainment_pop=None):
    """
    Parameters
    ----------
    sampled_pops : sequence of the population labels
    config_sequences : sequence of sequences of configs
                       config_sequences[i][j] is the configuration at the jth SNP of the ith locus
    """
    idx_list = []  # idx_list[i][j] is the index in configs of the jth SNP at locus i

    # index2loc[i] is the locus of the ith total SNP (after concatenating all
    # the loci)
    index2loc = []

    def chained_sequences():
        for loc, locus_configs in enumerate(config_sequences):
            idx_list.append([])  # add the locus, even if it has no configs!!
            for config in locus_configs:
                index2loc.append(loc)
                yield config

    #config_array, config2uniq, index2uniq = _build_data(chained_sequences(),
    #                                                    len(sampled_pops))
    compressed_counts = CompressedAlleleCounts.from_iter(
        chained_sequences(), len(sampled_pops))
    config_array = compressed_counts.config_array
    index2uniq = compressed_counts.index2uniq

    assert len(index2loc) == len(index2uniq)
    for loc, uniq_idx in zip(index2loc, index2uniq):
        idx_list[loc].append(uniq_idx)

    configs = ConfigArray(sampled_pops, config_array, None, ascertainment_pop)
    return SegSites(configs, idx_list)


class SegSitesLocus(object):

    def __init__(self, configs, idxs):
        self.configs = configs
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, site):
        if self.configs is None:
            raise NotImplementedError(
                "Iterating through the configs at each position is not supported, because each position is representing a mixture of configs, not a single config")
        return self.configs[self.idxs[site]]

    def _get_likelihoods(self, idx_likelihoods):
        return idx_likelihoods[self.idxs]


class SegSites(object):
    def __init__(self, configs, idx_list, config_mixture_by_idx=None):
        """
        This constructor should not be called directly, instead use the function
        momi.seg_site_configs().

        You can also use the method momi.SegSites.subsample_inds() to produce
        SegSites objects corresponding to subsamples of individuals.
        """
        self.configs = configs
        self.idx_list = [list(idxs) for idxs in idx_list]

        # if config_mixture_by_idx is not None, then each idx corresponds to a mixture of configs
        # in particular, this is used for constructing datasets over all
        # subsamples of individuals
        self.config_mixture_by_idx = config_mixture_by_idx
        if config_mixture_by_idx is None:
            self.sfs = Sfs(self.idx_list, self.configs)
            self.loci = [SegSitesLocus(self.configs, idxs)
                         for idxs in idx_list]
        else:
            self.loci = [SegSitesLocus(None, idxs) for idxs in self.idx_list]
            loc_idxs, loc_counts = zip(*[np.unique(loc, return_counts=True) for loc in self.idx_list])
            idx_freqs_matrix = _freq_matrix_from_counters(
                loc_idxs, loc_counts, len(config_mixture_by_idx))
            config_freqs_matrix = np.array(
                idx_freqs_matrix.T.dot(config_mixture_by_idx).T)
            arange = np.arange(len(configs))
            config_counts_list = [(arange[col != 0], col[col != 0])
                                  for col in config_freqs_matrix.T]
            self.sfs = Sfs(config_counts_list, self.configs)

    def __getitem__(self, loc):
        return self.loci[loc]

    def __len__(self):
        return len(self.loci)

    @property
    def ascertainment_pop(self): return self.sfs.ascertainment_pop

    @property
    def sampled_pops(self): return self.sfs.sampled_pops

    @property
    def sampled_n(self): return self.sfs.sampled_n

    @property
    def n_loci(self): return self.sfs.n_loci

    def n_snps(self, locus=None):
        ret = self.sfs.n_snps(locus=locus)
        assert int(ret) == ret
        return int(ret)

    def __eq__(self, other):
        configs, idx_list, ascertainment_pop = self.configs, self.idx_list, self.ascertainment_pop
        try:
            return configs == other.configs and idx_list == other.idx_list and np.all(ascertainment_pop == other.ascertainment_pop)
        except AttributeError:
            return False

    def subsample_inds(self, n):
        """
        Returns a new SegSites object, corresponding to a mixture of all SegSites objects
        that would be obtained by drawing subsamples of n individuals.

        In particular, the log-likelihood of a corresponding SNP in the new SegSites object,
        is a mixture of the log-likelihood of all subsets of n individuals that could be drawn
        from the original samples. The mixture weight is equal to the probability of drawing a
        particular subsample, conditional on the populations we are subsampling from.

        See also Sfs.subsample_inds(), which is produces an equivalent subsampled object for the Sfs.

        Confidence intervals computed by ConfidenceRegion and likelihoods computed by SfsLikelihoodSurface
        should all work properly. However, they are only implemented for the multivariate case, and not the
        Poisson case (i.e., don't specify the mutation rate for such likelihoods).
        """
        subconfigs, weights = _get_subsample_counts(self.configs, n)
        if not np.all(self.configs.ascertainment_pop):
            raise NotImplementedError(
                "Generating subsamples of individuals not implemented for data with ascertainment populations")
        subconfigs = config_array(self.sampled_pops, subconfigs)
        return SegSites(subconfigs, self.idx_list, config_mixture_by_idx=weights.T)

    # used for confidence intervals
    def _get_likelihood_sequences(self, config_likelihoods):
        if self.config_mixture_by_idx is not None:
            idx_likelihoods = np.dot(
                self.config_mixture_by_idx, config_likelihoods)
        else:
            idx_likelihoods = config_likelihoods
        for loc in self:
            yield loc._get_likelihoods(idx_likelihoods)

    ## reorganize the sites into n_chunks equally sized loci
    #def _make_equal_len_chunks(self, n_chunks):
    #    all_idxs = list(it.chain.from_iterable(self.idx_list))
    #    chunk_len = len(all_idxs) / float(n_chunks)
    #    count = it.count()
    #    def new_idx_chunks():
    #        for _, sub_idxs in it.groupby(all_idxs, lambda x: int(np.floor(next(count)/ chunk_len))):
    #            yield sub_idxs
    #    return SegSites(self.configs, new_idx_chunks(), self.config_mixture_by_idx)


# to hash configs, represent it as a str
# (this seems to be more memory efficient than representing it as a tuple)
def _config2hashable(config):
    return "\t".join("%d,%d" % (a, d) for a, d in config)

# the inverse function of _config2hashable


def _hashed2config(config_str):
    return tuple((int(a), int(d))
                 for a, d in (x.split(",")
                              for x in config_str.strip().split()))


class _CompressedList(object):
    def __init__(self):
        self.uniq_values = []
        self.value2uniq = {}
        self.index2uniq = []

    def __len__(self):
        return len(self.index2uniq)

    def __getitem__(self, index):
        return self.uniq_values[self.index2uniq[index]]

    def append(self, value):
        try:
            uniq_idx = self.value2uniq[value]
        except KeyError:
            uniq_idx = len(self.uniq_values)
            self.value2uniq[value] = uniq_idx
            self.uniq_values.append(value)
        self.index2uniq.append(uniq_idx)


class _CompressedHashedCounts(object):
    def __init__(self, npops):
        self.compressed_list = _CompressedList()
        self.npops = npops

    def append(self, config):
        self.compressed_list.append(_config2hashable(config))

    def index2uniq(self):
        return self.compressed_list.index2uniq

    def config_array(self):
        ret = np.zeros((len(self.compressed_list.uniq_values), self.npops, 2),
                       dtype=int)
        for i, config_str in enumerate(self.compressed_list.uniq_values):
            ret[i, :, :] = _hashed2config(config_str)
        return ret

    def compressed_allele_counts(self):
        return CompressedAlleleCounts(self.config_array(),
                                      self.index2uniq())

class CompressedAlleleCounts(object):
    @classmethod
    def from_iter(cls, config_iter, npops, sort=True):
        compressed_hashes = _CompressedHashedCounts(npops)
        for config in config_iter:
            compressed_hashes.append(config)
        return cls(compressed_hashes.config_array(),
                   compressed_hashes.index2uniq(),
                   sort=sort)

    def __init__(self, config_array, index2uniq,
                 sort=True):
        self.config_array = config_array
        self.index2uniq = np.array(index2uniq, dtype=int)
        if sort:
            self.sort_configs()

    def __getitem__(self, i):
        return self.config_array[self.index2uniq[i], :, :]

    def __len__(self):
        return len(self.index2uniq)

    def filter(self, idxs):
        to_keep = self.index2uniq[idxs]
        uniq_to_keep, uniq_to_keep_inverse = np.unique(
            to_keep, return_inverse=True)
        return CompressedAlleleCounts(self.config_array[uniq_to_keep, :, :],
                                      uniq_to_keep_inverse)

    def sort_configs(self):
        # sort configs so that "(very) similar" configs are next to each other
        # and will end up in the same batch,
        # thus avoiding redundant computation
        # "similar" == configs have same num missing alleles
        # "very similar" == configs are folded copies of each other
        a = self.config_array[:, :, 0]  # ancestral counts
        d = self.config_array[:, :, 1]  # derived counts
        n = a + d  # totals

        n = list(map(tuple, n))
        a = list(map(tuple, a))
        d = list(map(tuple, d))

        folded = list(map(min, list(zip(a, d))))

        keys = list(zip(n, folded))
        sorted_idxs = sorted(range(len(n)), key=lambda i: keys[i])
        sorted_idxs = np.array(sorted_idxs, dtype=int)

        unsorted_idxs = [None] * len(sorted_idxs)
        for i, j in enumerate(sorted_idxs):
            unsorted_idxs[j] = i
        unsorted_idxs = np.array(unsorted_idxs, dtype=int)

        self.config_array = self.config_array[sorted_idxs, :, :]
        self.index2uniq = unsorted_idxs[self.index2uniq]

    def count_configs(self):
        return np.bincount(self.index2uniq)

    @cached_property
    def n_samples(self):
        return np.max(np.sum(self.config_array, axis=2), axis=0)


def write_seg_sites(sequences_file, seg_sites):
    sampled_pops = seg_sites.sampled_pops

    sequences_file.write("\t".join(map(str, sampled_pops)) + "\n")

    if not np.all(seg_sites.ascertainment_pop):
        sequences_file.write("# Population used for ascertainment?\n")
        sequences_file.write(
            "\t".join(map(str, seg_sites.ascertainment_pop)) + "\n")

    for locus_configs in seg_sites:
        sequences_file.write("\n//\n\n")
        for config in locus_configs:
            #sequences_file.write("\t".join([",".join(map(str,x)) for x in config]) + "\n")
            sequences_file.write(_config2hashable(config) + "\n")


def read_seg_sites(sequences_file):
    #ret = []
    stripped_lines = (line.strip() for line in sequences_file)
    lines = (line for line in stripped_lines if line != "" and line[0] != "#")

    def get_loc(line):
        if line.startswith("//"):
            get_loc.curr += 1
        return get_loc.curr
    get_loc.curr = -1

    loci = it.groupby(lines, get_loc)

    _, header = next(loci)
    sampled_pops = tuple(next(header).split())

    def str2bool(s):
        if any(a.startswith(s.lower()) for a in ("true", "yes")):
            return True
        elif any(a.startswith(s.lower()) for a in ("false", "no")):
            return False
        raise ValueError("Can't convert %s to boolean" % s)

    try:
        ascertainment_pop = list(map(str2bool, next(header).split()))
    except (ValueError, StopIteration):
        ascertainment_pop = None

    def get_configs(locus):
        assert next(locus).startswith("//")
        for line in locus:
            # yield tuple(tuple(map(int,x.split(","))) for x in line.split())
            yield _hashed2config(line)

    return seg_site_configs(sampled_pops, (get_configs(loc) for i, loc in loci), ascertainment_pop=ascertainment_pop)


def _sub_sfs(configs, counts, subidxs=None):
    assert len(counts.shape) == 1
    if subidxs is None:
        assert len(counts) == len(configs.value)
        # make array copies to prevent views keeping references to larger
        # objects
        subidxs = np.array(np.arange(len(counts))[counts != 0])
        counts = np.array(counts[counts != 0])
    assert len(subidxs) == len(counts)
    sub_configs = _ConfigArray_Subset(configs, subidxs)
    return Sfs([{i: c for i, c in enumerate(counts)}], sub_configs)


class _ConfigArray_Subset(ConfigArray):
    # Efficient access to subset of configs

    def __init__(self, configs, sub_idxs):
        self.sub_idxs = sub_idxs
        self.full_configs = configs
        for a in ("sampled_n", "sampled_pops", "has_missing_data", "ascertainment_pop"):
            setattr(self, a, getattr(self.full_configs, a))
        #self.has_monomorphic = _has_monomorphic(self.value)

    @property
    def value(self):
        return self.full_configs.value[self.sub_idxs, :, :]

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
        return self.full_configs._augmented_configs(folded)[self._build_old_new_idxs(folded)[0], :, :]

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
