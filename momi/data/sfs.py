import itertools as it
import collections as co
import json
import autograd.numpy as np
import numpy as raw_np
from cached_property import cached_property
import scipy
import scipy.sparse
import scipy.misc
import gzip
import os
import logging
from .compressed_counts import _hashed2config, _config2hashable
from .compressed_counts import CompressedAlleleCounts
from .config_array import ConfigArray
from .config_array import _ConfigArray_Subset
from ..util import memoize_instance


def site_freq_spectrum(sampled_pops, loci, length=None):
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
            loci_counters.append(co.Counter())
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
    return Sfs(loci_counters, configs, folded=False, length=length)


def load_sfs(f):
    """
    Read in Sfs that has been written to file by Sfs.dump()
    """
    if isinstance(f, str):
        fname = os.path.expanduser(f)
        if fname.endswith(".gz"):
            with gzip.open(fname, "rt") as f:
                return load_sfs(f)
        else:
            with open(fname) as f:
                return load_sfs(f)

    logging.getLogger(__name__).info("Reading json...")
    info = json.load(f)
    logging.getLogger(__name__).info("Finished reading json...")

    logging.getLogger(__name__).info("Constructing configs...")
    configs = info.pop("configs")
    logging.getLogger(__name__).info(f"{len(configs)} unique configs detected")
    # don't use autograd here for performance
    configs = raw_np.array(configs, dtype=int)
    logging.getLogger(__name__).info("Converted to numpy array...")
    configs = ConfigArray(info.pop("sampled_pops"), configs)
    logging.getLogger(__name__).info("Finished constructing configs")

    logging.getLogger(__name__).info("Constructing SFS...")
    loci = []
    for locus, locus_rows in it.groupby(
            info.pop("(locus,config_id,count)"), lambda x: x[0]):
        loci.append({config_id: count
                     for _, config_id, count in locus_rows})

    ret = Sfs(loci, configs, **info)
    logging.getLogger(__name__).info("Finished constructing SFS")

    return ret


site_freq_spectrum.load = load_sfs


class Sfs(object):
    @classmethod
    def from_matrix(cls, mat, configs, *args, **kwargs):
        # convert to csc
        mat = scipy.sparse.csc_matrix(mat)
        indptr = mat.indptr
        loci = []
        for i in range(mat.shape[1]):
            loci.append(np.array([
                mat.indices[indptr[i]:indptr[i+1]],
                mat.data[indptr[i]:indptr[i+1]]]))

        return cls(loci, configs, *args, **kwargs)

    def __init__(self, loci, configs, folded, length):
        self.folded = folded
        self.length = length

        self.configs = configs

        self.loc_idxs, self.loc_counts = [], []
        for loc in loci:
            if len(loc) == 0:
                self.loc_idxs.append(np.array([], dtype=int))
                self.loc_counts.append(np.array([], dtype=float))
            else:
                try:
                    loc.items()
                except:
                    loc = np.array(loc)
                    if len(loc.shape) == 2:
                        assert loc.shape[0] == 2
                        idxs, cnts = loc[0, :], loc[1, :]
                    else:
                        idxs, cnts = np.unique(loc, return_counts=True)
                else:
                    idxs, cnts = zip(*loc.items())
                self.loc_idxs.append(np.array(idxs, dtype=int))
                self.loc_counts.append(np.array(cnts, dtype=float))

        if len(self.loc_idxs) > 1:
            self._total_freqs = np.array(np.squeeze(np.asarray(
                self.freqs_matrix.sum(axis=1))), ndmin=1)
        else:
            # avoid costly building of frequency matrix, when there are many
            # Sfs's of a single locus (e.g. in many stochastic minibatches)
            idxs, = self.loc_idxs
            cnts, = self.loc_counts
            self._total_freqs = np.zeros(len(self.configs))
            self._total_freqs[idxs] = cnts

        assert not np.any(self._total_freqs == 0)

    def dump(self, f):
        """
        Write the Sfs in a compressed JSON format,
        that can be read in by site_freq_spectrum.load()
        """
        if isinstance(f, str):
            fname = os.path.expanduser(f)
            if fname.endswith(".gz"):
                with gzip.open(fname, "wt") as f:
                    self.dump(f)
            else:
                with open(fname, 'w') as f:
                    self.dump(f)
            return

        print("{", file=f)
        print('\t"sampled_pops": {},'.format(
            json.dumps(list(self.sampled_pops))), file=f)
        print('\t"folded": {},'.format(
            json.dumps(self.folded)), file=f)
        print('\t"length": {},'.format(
            json.dumps(self.length)), file=f)
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

    @property
    def populations(self):
        return self.sampled_pops

    @memoize_instance
    def combine_loci(self):
        """
        Returns a copy of this SFS, but with all loci
        combined into a single locus
        """
        return self.from_matrix(self.freqs_matrix.sum(axis=1),
                                self.configs, self.folded,
                                self.length)

    @property
    def freqs_matrix(self):
        """
        Returns the frequencies as a sparse matrix;
        freqs_matrix[i, j] is the frequency of Sfs.configs[i] at locus j
        """
        return self.csr_freqs_matrix

    @cached_property
    def csr_freqs_matrix(self):
        return _csr_freq_matrix_from_counters(
            self.loc_idxs, self.loc_counts, len(self.configs))

    @cached_property
    def avg_pairwise_hets(self):
        """
        Returns the number of SNPs where a single individual is heterozygote,
        averaged over all individuals within each population

        Returns numpy.ndarray pairwise_hets, where pairwise_hets[i,j] is the
        average number of hets in population j at locus i
        """
        n_nonmissing = np.sum(self.configs.value, axis=2)
        # for denominator, assume 1 allele is drawn from whole sample, and 1
        # allele is drawn only from nomissing alleles
        denoms = np.maximum(n_nonmissing * (self.sampled_n - 1), 1.0)
        p_het = 2 * self.configs.value[:, :, 0] * \
            self.configs.value[:, :, 1] / denoms

        return self.freqs_matrix.T.dot(p_het)

    def resample(self):
        loci = np.random.choice(
            np.arange(self.n_loci), size=self.n_loci, replace=True)
        mat = self.freqs_matrix[:, loci]
        to_keep = np.asarray(mat.sum(axis=1) > 0).squeeze()
        to_keep = np.arange(len(self.configs))[to_keep]
        mat = mat[to_keep, :]
        configs = _ConfigArray_Subset(self.configs, to_keep)

        # NOTE not exactly right but close enough
        length = self.length * mat.sum() / self.n_snps()

        return self.from_matrix(mat, configs, self.folded, length)

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
            return np.array([self.n_snps(locus=loc)
                             for loc in range(self.n_loci)])
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
            return dict(zip(map(_config2hashable, self.configs),
                            self._total_freqs))
        return dict(zip(
            (_config2hashable(self.configs[i]) for i in self.loc_idxs[locus]),
            self.loc_counts[locus]))

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
        sampled_n_counts = co.Counter()
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
        return np.sum(-lambd + lambd * np.log(lambd)
                      - scipy.special.gammaln(lambd + 1))

    @cached_property
    def _pairwise_muts_poisson_entropy(self):
        lambd = self.avg_pairwise_hets
        ret = -lambd + lambd * np.log(lambd) - scipy.special.gammaln(lambd + 1)
        ret[lambd <= 1e-16] = 0
        ret = ret * self.sampled_n / float(np.sum(
            self.sampled_n[self.ascertainment_pop]))
        return np.sum(ret[:, self.ascertainment_pop])

    def fold(self):
        """
        Returns a copy of the SFS, but with folded entries.
        """
        def get_folded(config):
            if tuple(config[:, 0]) < tuple(config[:, 1]):
                return config[:, ::-1]
            else:
                return config

        compressed_folded = CompressedAlleleCounts.from_iter(
            map(get_folded, self.configs),
            len(self.sampled_pops), sort=False)

        return self.from_matrix(
            compressed_folded.index2uniq_mat @ self.freqs_matrix,
            ConfigArray(self.sampled_pops, compressed_folded.config_array,
                        sampled_n=self.sampled_n,
                        ascertainment_pop=self.ascertainment_pop),
            folded=True, length=self.length)

    def _copy(self, sampled_n=None):
        """
        See also: ConfigArray._copy()
        """
        if sampled_n is None:
            sampled_n = self.sampled_n
        return self.from_matrix(
            self.csr_freqs_matrix,
            ConfigArray(self.sampled_pops, self.configs.value,
                        sampled_n=sampled_n,
                        ascertainment_pop=self.ascertainment_pop),
            self.folded, self.length)

    def _integrate_sfs(self, weights, vector=False, locus=None):
        if vector:
            assert locus is None
            return np.array([self._integrate_sfs(weights, locus=loc)
                             for loc in range(self.n_loci)])
        if locus is None:
            idxs, counts = slice(None), self._total_freqs
        else:
            idxs, counts = self.loc_idxs[locus], self.loc_counts[locus]
        return np.sum(weights[idxs] * counts)

    def subset_populations(self, populations, non_ascertained_pops=None):
        if non_ascertained_pops is None:
            non_ascertained_pops = []
            for pop, asc in zip(self.sampled_pops, self.ascertainment_pop):
                if not asc:
                    non_ascertained_pops.append(pop)
        return self._subset_populations(tuple(populations),
                                        tuple(non_ascertained_pops))

    @memoize_instance
    def _subset_populations(self, populations, non_ascertained_pops):
        # old indexes of the new population ordering
        old_sampled_pops = list(self.sampled_pops)
        old_pop_idx = np.array([
            old_sampled_pops.index(p) for p in populations], dtype=int)

        # old ascertainment
        ascertained = dict(zip(self.sampled_pops,
                               self.ascertainment_pop))
        # restrict to new populations
        ascertained = {p: ascertained[p] for p in populations}
        # previously non-ascertained pops should remain so
        for pop, is_asc in ascertained.items():
            if not is_asc:
                assert pop in non_ascertained_pops
        # update ascertainment
        for pop in non_ascertained_pops:
            ascertained[pop] = False
        # convert from dict to numpy array
        ascertained = np.array([ascertained[p] for p in populations])

        # keep only polymorphic configs
        asc_only = self.configs[:, old_pop_idx[ascertained], :]
        asc_is_poly = (asc_only.sum(axis=1) != 0).all(axis=1)
        asc_is_poly = np.arange(len(asc_is_poly))[asc_is_poly]
        sub_sfs = self._subset_configs(asc_is_poly)

        # get the new configs
        new_configs = CompressedAlleleCounts.from_iter(
            sub_sfs.configs[:, old_pop_idx, :],
            len(populations), sort=False)

        return self.from_matrix(
            new_configs.index2uniq_mat @ sub_sfs.freqs_matrix,
            ConfigArray(populations, new_configs.config_array,
                        ascertainment_pop=ascertained),
            self.folded, self.length)

    def _subset_configs(self, idxs):
        return self.from_matrix(
            self.csr_freqs_matrix[idxs, :],
            _ConfigArray_Subset(self.configs, idxs),
            self.folded, self.length)

    @property
    def sfs(self):
        return self

    @cached_property
    def _p_missing(self):
        """
        Estimates the probability that a random allele
        from each population is missing.

        To estimate missingness, first remove random allele;
        if the resulting config is monomorphic, then ignore.
        If polymorphic, then count whether the removed allele
        is missing or not.

        This avoids bias from fact that we don't observe
        some polymorphic configs that appear monomorphic
        after removing the missing alleles.
        """
        counts = self._total_freqs
        sampled_n = self.sampled_n
        n_pops = len(self.sampled_pops)

        config_arr = self.configs.value
        # augment config_arr to contain the missing counts
        n_miss = sampled_n - np.sum(config_arr, axis=2)
        config_arr = np.concatenate((config_arr, np.reshape(
            n_miss, list(n_miss.shape)+[1])), axis=2)

        ret = []
        for i in range(n_pops):
            n_valid = []
            for allele in (0, 1, -1):
                # configs with removed allele
                removed = np.array(config_arr)
                removed[:, i, allele] -= 1
                # is the resulting config polymorphic?
                valid_removed = (removed[:, i, allele] >= 0) & np.all(
                    np.sum(removed[:, :, :2], axis=1) > 0, axis=1)

                # sum up the valid configs
                n_valid.append(np.sum(
                    (counts * config_arr[:, i, allele])[valid_removed]))
            # fraction of valid configs with missing additional allele
            ret.append(n_valid[-1] / float(sum(n_valid)))
        return np.array(ret)


def _csr_freq_matrix_from_counters(idxs_by_loc, cnts_by_loc,
                                   n_configs):
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
    return scipy.sparse.csc_matrix((data, indices, indptr),
                                   shape=(n_configs, n_loc)).tocsr()


def _get_subsample_counts(configs, n):
    subconfigs, weights = [], []
    for pop_comb in it.combinations_with_replacement(configs.sampled_pops, n):
        subsample_n = co.Counter(pop_comb)
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
            cnt_vec = configs.subsample_probs(sfs_entry)
            if not np.all(cnt_vec == 0):
                subconfigs.append(sfs_entry)
                weights.append(cnt_vec)

    return np.array(subconfigs), np.array(weights)


class SimpleNamespace(object):
    pass
