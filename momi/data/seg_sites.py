import itertools as it
import autograd.numpy as np
from .compressed_counts import _hashed2config, _config2hashable
from .compressed_counts import CompressedAlleleCounts
from .config_array import config_array
from .config_array import ConfigArray
from .sfs import Sfs, _freq_matrix_from_counters, _get_subsample_counts


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
