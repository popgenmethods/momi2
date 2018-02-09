from functools import partial
from collections import OrderedDict
import networkx as nx
import scipy
import scipy.misc
from scipy.misc import comb
import scipy.sparse
import autograd.numpy as np
import autograd
from autograd import primitive
import msprime
from .compute_sfs import expected_total_branch_len
from .data.compressed_counts import CompressedAlleleCounts, _CompressedHashedCounts
from .data.seg_sites import seg_site_configs
from .data.snps import SnpAlleleCounts
from .util import memoize_instance, memoize
from .math_functions import sum_antidiagonals, convolve_axes, binom_coeffs, roll_axes, hypergeom_quasi_inverse, par_einsum, convolve_sum_axes
from .events import get_event_from_old, DemographyError, _set_sizes, _build_demo_graph

import os
import itertools

from cached_property import cached_property

import logging
logger = logging.getLogger(__name__)



def demographic_history(events, archaic_times_dict=None, default_N=1.0):
    return DemographicHistory(events, archaic_times_dict, default_N)


def make_demography(events, sampled_pops, sampled_n,
                    sampled_t=None, default_N=1.0, time_scale='ms'):
    """
      Parameters
      ----------
      events : list of tuples
           The demographic history as a list of events.
           Events are represented as tuples. There are 4 kinds of events:
                ('-en',t,i,N) : size change
                     At time t, scaled size of pop. i is set to N,
                     and its growth rate is set to 0.
                ('-eg',t,i,g) : exponential growth
                     At time t, exponential growth rate of pop. i is
                     set to g.
                     So for s >= t, the pop size of i will be
                          N(s) = N(t) exp( (t-s) * g)
                ('-ej',t,i,j) : join event
                     At time t, all lineages in pop. i move into pop. j.
                     Additionally, pop. i is removed, and not allowed to
                     be affected by further events.
                ('-ep',t,i,j,p_ij) : pulse event
                     At time t, each lineage in pop. i moves into pop. j
                     independently with probability p_ij.
                     (Forward-in-time, migration is from j to i, with
                      fraction p_ij of the alleles in i replaced)
           Time is measured backwards from the present (so t==0 is the present, t>0 is the past)
           Events are processed in order, backwards in time from the present.
           If two events occur at the same time, they will be processed according to their
           order in the list.
      sampled_pops : list of population labels
            labels can be any hashable type (e.g. str, int, tuple)
      sampled_n : list of ints
            the number of alleles sampled from each pop
            should satisfy len(sampled_n) == len(sampled_pops)
      sampled_t : None, or list of floats
            the time each pop was sampled.
            if None, all populations are sampled at the present (t=0)
            if not None, should have len(sampled_t) == len(sampled_pops)
      default_N : float
            the scaled size N of all populations, unless changed by -en or -eg
      time_scale : str or float
            if time_scale=='ms', coalescence rate is 2/N per unit time
            if time_scale=='standard', coalescence rate is 1/2N
            if float, coalescence rate is 2/(N*time_scale)
    """
    #logger.warn("momi.make_demography() is depracated, use momi.demographic_history() instead")
    return _make_multipop_moran(events, sampled_pops, sampled_n, sampled_t, default_N, time_scale)


def _make_multipop_moran(events, sampled_pops, sampled_n, sampled_t=None, default_N=1.0, time_scale='ms'):
    if sampled_t is None:
        sampled_t = (0.0,) * len(sampled_n)

    if time_scale == 'ms':
        time_scale = 1.0
    elif time_scale == 'standard':
        time_scale = 4.0
    elif isinstance(time_scale, str):
        raise DemographyError("time_scale must be float, 'ms', or 'standard'")

    default_N = default_N * time_scale
    old_events, events = events, []
    for e in old_events:
        if e[0] == '-en':
            flag, t, i, N = e
            e = flag, t, i, N * time_scale
        events += [e]

    # create sampling events
    sampling_events = [('-eSample', t, i, n)
                       for i, n, t in zip(sampled_pops, sampled_n, sampled_t)]
    events = sampling_events + list(events)

    # sort events by time
    events = sorted(events, key=lambda x: x[1])

    events = [get_event_from_old(e) for e in events]
    sample_sizes = OrderedDict(zip(sampled_pops, sampled_n))

    _G = _build_demo_graph(events, sample_sizes, {}, default_N)
    return Demography(_G)


class DemographicHistory(object):
    def __init__(self, events, archaic_times_dict, default_N):
        self.events = events
        self.archaic_times_dict = archaic_times_dict
        self.default_N = default_N

    def _get_multipop_moran(self, sampled_pops, sampled_n):
        if sampled_pops is None or sampled_n is None:
            raise ValueError(
                "Need to provide sampled_n/sampled_pops parameters")
        return self._get_multipop_moran_helper(tuple(sampled_pops), tuple(sampled_n))

    def simulate_trees(self, sampled_pops, sampled_n,
                       **kwargs):
        return self._get_multipop_moran(
            sampled_pops, sampled_n
        ).simulate_trees(**kwargs)

    def simulate_data(self, sampled_pops, sampled_n,
                      **kwargs):
        return self._get_multipop_moran(
            sampled_pops, sampled_n
        ).simulate_data(**kwargs)

    def simulate_vcf(self, sampled_pops, sampled_n, *args, **kwargs):
        return self._get_multipop_moran(
            sampled_pops, sampled_n
        ).simulate_vcf(*args, **kwargs)

    @memoize_instance
    def _get_multipop_moran_helper(self, sampled_pops, sampled_n):
        return _make_multipop_moran(self.events, sampled_pops, sampled_n,
                                    self.get_sampled_t(sampled_pops),
                                    self.default_N, time_scale="ms")

    def get_sampled_t(self, sampled_pops):
        if self.archaic_times_dict:
            return [self.archaic_times_dict[pop]
                    if pop in self.archaic_times_dict else 0.0
                    for pop in sampled_pops]
        else:
            return None

    def rescaled(self, factor=None):
        if factor is None:
            factor = 1.0 / self.default_N
        rescaled_events = rescale_events(self.events, factor)
        default_N = self.default_N * factor
        if self.archaic_times_dict:
            archaic_times_dict = {p: t * factor for p,
                                  t in self.archaic_times_dict.items()}
        else:
            archaic_times_dict = None
        return DemographicHistory(rescaled_events, archaic_times_dict, default_N)


class differentiable_method(object):
    """
    a descriptor for cacheing all the differentiable objects in the demography
    this is used to reorganize some of the computations during automatic differentiation,
    which can be very resource intensive

    based on memoize_instance in util.py, which is itself based on http://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        obj = args[0]
        cache = obj._diff_cache

        key = (self.func, args[1:], frozenset(list(kw.items())))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res

class Demography(object):
    """
    The demographic history relating a sample of individuals.
    """
    def __init__(self, G, cache=None):
        """
        For internal use only.
        Use make_demography() to create a Demography.
        """
        self._G = G
        self._event_tree = _build_event_tree(self._G)

        if cache is not None:
            self._diff_cache = cache
        else:
            self._diff_cache = {}

    def _get_differentiable_part(self):
        # used with self._get_graph_structure() and autograd.checkpoint
        # returns a dict of the memoized values so we can
        # compute their derivatives easily
        expected_total_branch_len(self)
        return self._diff_cache

    def _get_graph_structure(self):
        # returns just the graph structure, i.e. the "non-differentiable" part of the Demography
        # use this with _get_differentiable_part()
        # to re-organize certain computations during automatic differentiation
        ret = nx.DiGraph()
        ret.add_nodes_from(self._G.nodes(data=False))
        ret.add_edges_from(self._G.edges(data=False))

        for v, d in self._G.nodes(data=True):
            if 'lineages' in d:
                ret.node[v]['lineages'] = d['lineages']

        ret.graph['events_as_edges'] = tuple(self._G.graph['events_as_edges'])
        ret.graph['sampled_pops'] = self.sampled_pops

        return ret

    def copy(self, sampled_n=None):
        """
        Notes
        -----
        Note that momi.expected_sfs, momi.composite_log_likelihood require
        Demography.sampled_n == ConfigArray.sampled_n.
        If this is not the case, you can use copy() to create a copy with the correct
        sampled_n.
        """
        if sampled_n is None:
            sampled_n = self.sampled_n
        return _make_multipop_moran(self.events, self.sampled_pops, sampled_n, self.sampled_t, self.default_N)

    #@property
    #def events(self):
    #    """
    #    The list of events (tuples) making up the demographic history
    #    """
    #    return self._G.graph['event_cmds']

    @property
    def sampled_pops(self):
        """
        The list of population labels
        """
        return self._G.graph['sampled_pops']

    @property
    def sampled_n(self):
        """
        The list of number of samples per population
        """
        return np.array(tuple(self._G.node[(l, 0)]['lineages'] for l in self.sampled_pops), dtype=int)

    @cached_property
    def _demo_hist(self):
        return DemographicHistory(
            self.events,
            dict(zip(self.sampled_pops, self.sampled_t)),
            self.default_N)

    def _get_multipop_moran(self, sampled_pops, sampled_n):
        if sampled_pops is None:
            sampled_pops = self.sampled_pops
        if sampled_n is None:
            sampled_n = self.sampled_n

        ## converting sampled_pops to array breaks if pops are labeled as ints
        #sampled_pops = np.array(sampled_pops)
        sampled_n = np.array(sampled_n)

        if (any([p1 != p2 for p1, p2 in zip(sampled_pops,
                                            self.sampled_pops)])
            or np.any(self.sampled_n != sampled_n)):
            return self._demo_hist._get_multipop_moran(
                sampled_pops, sampled_n)

        return self

    def rescaled(self, factor=None):
        """
        Returns the equivalent Demography, but with time rescaled by factor

        Parameters
        ----------
        factor : float or None
             The factor to rescale time by.
             If None, rescale by 1/default_N (so that rescaled_demo is in ms units, and rescaled_demo.default_N == 1.0)

        Returns
        -------
        rescaled_demo : Demography
             The same demography, but with population sizes N*factor, times t*factor,
             and growth rates g/factor.
        """
        if factor is None:
            factor = 1.0 / self.default_N
        rescaled_events = rescale_events(self.events, factor)
        default_N = self.default_N * factor
        try:
            sampled_t = self.sampled_t * factor
        except:
            sampled_t = None
        return _make_multipop_moran(rescaled_events,
                                    self.sampled_pops, self.sampled_n,
                                    sampled_t=sampled_t, default_N=default_N)

    @memoize_instance
    def _n_at_node(self, node):
        return np.sum(self._G.node[(pop, idx)]['lineages']
                      for pop, idx in nx.dfs_preorder_nodes(self._G, node)
                      if idx == 0)

    @property
    def _root(self):
        ret, = self._parent_pops(self._event_root)
        return ret

    @property
    def _event_root(self):
        return self._event_tree.root

    def _event_type(self, event):
        if len(event) == 1:
            return 'leaf'
        elif len(event) == 3:
            return 'pulse'
        elif len(self._event_tree[event]) == 2:
            return 'merge_clusters'
        else:
            return 'merge_subpops'

    def _sub_pops(self, event):
        '''
        The group of subpopulations corresponding to this event in the junction tree.
        '''
        return self._event_tree.node[event]['subpops']

    def _parent_pops(self, event):
        '''The populations arising due to this event, backwards in time.'''
        return self._event_tree.node[event]['parent_pops']

    def _child_pops(self, event):
        '''
        Returns dict of
        {child_pop : child_event},
        which gives populations arising from this event forward in time,
        and the corresponding child events in the junction tree.
        '''
        return self._event_tree.node[event]['child_pops']

    def _pulse_nodes(self, event):
        parent_pops = self._parent_pops(event)
        child_pops_events = self._child_pops(event)
        assert len(child_pops_events) == 2
        child_pops, child_events = list(zip(*list(child_pops_events.items())))

        child_in = dict(self._G.in_degree(child_pops))
        recipient, = [k for k, v in list(child_in.items()) if v == 2]
        non_recipient, = [k for k, v in list(child_in.items()) if v == 1]

        parent_out = dict(self._G.out_degree(parent_pops))
        donor, = [k for k, v in list(parent_out.items()) if v == 2]
        non_donor, = [k for k, v in list(parent_out.items()) if v == 1]

        return recipient, non_recipient, donor, non_donor

    """
    ALL differentiable methods used by compute_sfs
    should be decorated by @differentiable_method !!!

    This is so that we can extract the memoized values with
    _get_differentiable_part(), which can be used in conjunction with
    autograd.checkpoint() to reduce memory usage by computing
    in batches of SNPs

    TODO refactor so that we don't need to decorate, it has high
    congnitive load!!!
    """

    @property
    @differentiable_method
    def sampled_t(self):
        """
        An array of times at which each population was sampled
        """
        return np.array(tuple(self._G.node[(l, 0)]['sizes'][0]['t'] for l in self.sampled_pops))

    @property
    @differentiable_method
    def default_N(self):
        """
        The scaled size N of all populations, unless changed by -en or -eg
        """
        return self._G.graph['default_N']

    @differentiable_method
    def _truncated_sfs(self, node):
        return self._G.node[node]['model'].sfs(self._n_at_node(node))

    @differentiable_method
    def _scaled_time(self, node):
        return self._G.node[node]['model'].scaled_time

    def _pulse_prob(self, event):
        return self._pulse_prob_helper(event), self._pulse_prob_idxs(event)

    def _pulse_prob_idxs(self, event):
        recipient, non_recipient, donor, non_donor = self._pulse_nodes(event)
        admixture_idxs = self._admixture_prob_idxs(recipient)
        return admixture_idxs + [non_recipient]

    @differentiable_method
    def _pulse_prob_helper(self, event):
        # returns 4-tensor
        # running time is O(n^5), because of pseudo-inverse
        # if pulse from ghost population, only costs O(n^4)
        recipient, non_recipient, donor, non_donor = self._pulse_nodes(event)

        admixture_prob, admixture_idxs = self._admixture_prob(recipient)

        pulse_idxs = admixture_idxs + [non_recipient]
        assert pulse_idxs == self._pulse_prob_idxs(event)

        pulse_prob = par_einsum(admixture_prob, admixture_idxs,
                                binom_coeffs(self._n_at_node(non_recipient)), [
                                    non_recipient],
                                pulse_idxs)
        pulse_prob = par_einsum(pulse_prob, pulse_idxs,
                                binom_coeffs(self._n_at_node(recipient)), [
                                    donor],
                                pulse_idxs)
        pulse_prob = roll_axes(pulse_prob, pulse_idxs, non_recipient, donor)

        donor_idx = pulse_idxs.index(donor)
        pulse_prob = par_einsum(pulse_prob, pulse_idxs,
                                1.0 /
                                binom_coeffs(pulse_prob.shape[
                                             donor_idx] - 1), [donor],
                                pulse_idxs)

        # reduce the number of lineages in donor to only the number necessary
        N, n = pulse_prob.shape[donor_idx] - 1, self._n_at_node(donor)
        assert N >= n
        if N > n:
            assert -1 not in pulse_idxs
            tmp_idxs = [-1 if x == donor else x for x in pulse_idxs]
            pulse_prob = par_einsum(pulse_prob, tmp_idxs,
                                    hypergeom_quasi_inverse(N, n),
                                    [-1, donor], pulse_idxs)
        assert pulse_prob.shape[donor_idx] == n + 1

        return pulse_prob

    def _admixture_prob(self, admixture_node):
        return self._admixture_prob_helper(admixture_node), self._admixture_prob_idxs(admixture_node)

    def _admixture_prob_idxs(self, admixture_node):
        edge1, edge2 = sorted(self._G.in_edges(
            [admixture_node], data=True), key=lambda x: str(x[:2]))
        parent1, parent2 = [e[0] for e in (edge1, edge2)]
        return [admixture_node, parent1, parent2]

    @differentiable_method
    def _admixture_prob_helper(self, admixture_node):
        '''
        Array with dim [n_admixture_node+1, n_parent1_node+1, n_parent2_node+1],
        giving probability of derived counts in child, given derived counts in parents
        '''
        n_node = self._n_at_node(admixture_node)

        # admixture node must have two parents
        edge1, edge2 = sorted(self._G.in_edges(
            [admixture_node], data=True), key=lambda x: str(x[:2]))
        parent1, parent2 = [e[0] for e in (edge1, edge2)]
        prob1, prob2 = [e[2]['prob'] for e in (edge1, edge2)]
        assert prob1 + prob2 == 1.0

        #n_from_1 = np.arange(n_node + 1)
        #n_from_2 = n_node - n_from_1
        #binom_coeffs = (prob1**n_from_1) * (prob2**n_from_2) * \
        #    scipy.misc.comb(n_node, n_from_1)
        #ret = par_einsum(_der_in_admixture_node(n_node), list(range(4)),
        #                 binom_coeffs, [0],
        #                 [1, 2, 3])
        ret = np.transpose(admixture_operator(n_node, prob1))
        assert ret.shape == tuple([n_node + 1] * 3)

        assert [admixture_node, parent1,
                parent2] == self._admixture_prob_idxs(admixture_node)
        return ret

    def simulate_data(self, **kwargs):
        treeseq = self.simulate_trees(**kwargs)
        try:
            treeseq.variants
        except:
            pass
        else:
            treeseq = [treeseq]

        mat = np.zeros((len(self.sampled_n), sum(self.sampled_n)), dtype=int)
        j = 0
        for i, n in enumerate(self.sampled_n):
            for _ in range(n):
                mat[i, j] = 1
                j += 1
        mat = scipy.sparse.csr_matrix(mat)

        def get_config(genos):
            derived_counts = mat.dot(genos)
            return np.array([
                self.sampled_n - derived_counts,
                derived_counts
            ]).T

        chrom = []
        pos = []
        compressed_counts = _CompressedHashedCounts(len(self.sampled_pops))

        for c, locus in enumerate(treeseq):
            for v in locus.variants():
                compressed_counts.append(get_config(v.genotypes))
                chrom.append(c)
                pos.append(v.position)

        return SnpAlleleCounts(chrom, pos, compressed_counts.compressed_allele_counts(),
                               self.sampled_pops)

    def simulate_vcf(self, outfile, mutation_rate, recombination_rate,
                     length, chrom_names=[1], ploidy=1, random_seed=None):
        if np.any(self.sampled_n % ploidy != 0):
            raise ValueError("Sampled alleles per population must be integer multiple of ploidy")

        treeseq = self.simulate_trees(
            mutation_rate=mutation_rate, recombination_rate=recombination_rate,
            length=length, num_replicates=len(chrom_names), random_seed=random_seed)

        outfile.write("##fileformat=VCFv4.2\n")
        outfile.write('##source="VCF simulated by momi2 using msprime backend"' + "\n")
        for chrom in chrom_names:
            outfile.write("##contig=<ID={0},length={1}>\n".format(chrom, length))

        n_samples = int(np.sum(self.sampled_n) / ploidy)
        fields = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        for pop, n in zip(self.sampled_pops, self.sampled_n):
            for i in range(int(n / ploidy)):
                fields.append("{}_{}".format(pop, i))
        outfile.write("\t".join(fields) + "\n")
        for chrom, locus in zip(chrom_names, treeseq):
            for v in locus.variants():
                gt = np.reshape(v.genotypes, (n_samples, ploidy))
                row = [str(chrom), str(int(np.floor(v.position))), ".", "A", "T", ".", ".", ".", "GT"] + [
                    "|".join(map(str, sample)) for sample in gt]
                outfile.write("\t".join(row) + "\n")




    def simulate_trees(self, **kwargs):
        sampled_t = self.sampled_t
        if sampled_t is None:
            sampled_t = 0.0
        sampled_t = np.array(sampled_t) * np.ones(len(self.sampled_pops))

        pops = {p: i for i, p in enumerate(self.sampled_pops)}
        sampled_n = self.sampled_n

        #events = sorted(self.events, key=lambda x: x[1])
        #demographic_events = []
        #for event in events:
        #    flag = event[0]
        #    if flag == '-ep':
        #        _, t, i, j, pij = event
        #        for k in (i, j):
        #            if k not in pops:
        #                pops[k] = len(pops)
        #        demographic_events.append(msprime.MassMigration(
        #            t, pops[i], pops[j], proportion=pij))
        #        continue
        #    elif flag == '-ej':
        #        _, t, i, j = event
        #        for k in (i, j):
        #            if k not in pops:
        #                pops[k] = len(pops)
        #        demographic_events.append(
        #            msprime.MassMigration(t, pops[i], pops[j]))
        #    elif flag == '-eg':
        #        _, t, i, alpha = event
        #        if i not in pops:
        #            pops[i] = len(pops)
        #        demographic_events.append(msprime.PopulationParametersChange(
        #            t, growth_rate=alpha, population_id=pops[i]))
        #    elif flag == '-en':
        #        _, t, i, N = event
        #        if i not in pops:
        #            pops[i] = len(pops)
        #        demographic_events.append(msprime.PopulationParametersChange(
        #            t, initial_size=N / 4, growth_rate=0, population_id=pops[i]))
        #    else:
        #        assert False

        demographic_events = []
        for e in self._G.graph["events"]:
            e = e.get_msprime_event(self._G.graph["params"], pops)
            if e is not None:
                demographic_events.append(e)

        return msprime.simulate(population_configurations=[msprime.PopulationConfiguration()
                                                           for _ in range(len(pops))],
                                Ne=self.default_N / 4,
                                demographic_events=demographic_events,
                                samples=[msprime.Sample(population=pops[p], time=t)
                                         for p, t, n in zip(self.sampled_pops, self.sampled_t, self.sampled_n)
                                         for _ in range(n)],
                                **kwargs)


def rescale_events(events, factor):
    rescaled_events = []
    for event in events:
        if event[0] == '-ej':
            flag, t, i, j = event
            event = (flag, t * factor, i, j)
        elif event[0] == '-en':
            flag, t, i, N = event
            event = (flag, t * factor, i, N * factor)
        elif event[0] == '-eg':
            flag, t, i, alpha = event
            event = (flag, t * factor, i, alpha / (1.0 * factor))
        elif event[0] == '-ep':
            flag, t, i, j, p = event
            event = (flag, t * factor, i, j, p)
        else:
            assert False
        rescaled_events += [event]
    return rescaled_events


def admixture_operator(n_node, p):
    # axis0=n_from_parent, axis1=der_from_parent, axis2=der_in_parent
    der_in_parent = np.tile(np.arange(n_node + 1), (n_node + 1, n_node + 1, 1))
    n_from_parent = np.transpose(der_in_parent, [2, 0, 1])
    der_from_parent = np.transpose(der_in_parent, [0, 2, 1])

    anc_in_parent = n_node - der_in_parent
    anc_from_parent = n_from_parent - der_from_parent

    x = comb(der_in_parent, der_from_parent) * comb(
        anc_in_parent, anc_from_parent) / comb(n_node, n_from_parent)
    # rearrange so axis0=1, axis1=der_in_parent, axis2=der_from_parent, axis3=n_from_parent
    x = np.transpose(x)
    x = np.reshape(x, [1] + list(x.shape))

    n = np.arange(n_node+1)
    B = comb(n_node, n)

    # the two arrays to convolve_sum_axes
    x1 = (x * B * ((1-p)**n) * (p**(n[::-1])))
    x2 = x[:, :, :, ::-1]

    # reduce array size; approximate low probability events with 0
    mu = n_node * (1-p)
    sigma = np.sqrt(n_node * p * (1-p))
    n_sd = 4
    lower = np.max((0, np.floor(mu - n_sd*sigma)))
    upper = np.min((n_node, np.ceil(mu + n_sd*sigma))) + 1
    lower, upper = int(lower), int(upper)

    ##x1 = x1[:,:,:upper,lower:upper]
    ##x2 = x2[:,:,:(n_node-lower+1),lower:upper]

    ret = convolve_sum_axes(x1, x2)
    # axis0=der_in_parent1, axis1=der_in_parent2, axis2=der_in_child
    ret = np.reshape(ret, ret.shape[1:])
    if ret.shape[2] < (n_node+1):
        ret = np.pad(ret, [(0,0),(0,0),(0,n_node+1-ret.shape[2])], "constant")
    return ret[:, :, :(n_node+1)]


#@memoize
#def _der_in_admixture_node(n_node):
#    '''
#    returns 4d-array, [n_from_parent1, der_in_child, der_in_parent1, der_in_parent2].
#    Used by Demography._admixture_prob_helper
#    '''
#    # axis0=n_from_parent, axis1=der_from_parent, axis2=der_in_parent
#    der_in_parent = np.tile(np.arange(n_node + 1), (n_node + 1, n_node + 1, 1))
#    n_from_parent = np.transpose(der_in_parent, [2, 0, 1])
#    der_from_parent = np.transpose(der_in_parent, [0, 2, 1])
#
#    anc_in_parent = n_node - der_in_parent
#    anc_from_parent = n_from_parent - der_from_parent
#
#    x = scipy.misc.comb(der_in_parent, der_from_parent) * scipy.misc.comb(
#        anc_in_parent, anc_from_parent) / scipy.misc.comb(n_node, n_from_parent)
#
#    ret, labels = convolve_axes(
#        x, x[::-1, ...], [[c for c in 'ijk'], [c for c in 'ilm']], ['j', 'l'], 'n')
#    return np.einsum('%s->inkm' % ''.join(labels), ret[..., :(n_node + 1)])


def _build_event_tree(G):
    # def node_time(v):
    #     return G.node[v]['sizes'][0]['t']

    eventEdgeList = []
    currEvents = {k: (k,) for k, v in list(dict(G.out_degree()).items()) if v == 0}
    eventDict = {e: {'subpops': (v,), 'parent_pops': (
        v,), 'child_pops': {}} for v, e in list(currEvents.items())}
    for e in G.graph['events_as_edges']:
        # get the population edges forming the event
        parent_pops, child_pops = list(map(set, list(zip(*e))))
        child_events = set([currEvents[c] for c in child_pops])

        sub_pops = set(itertools.chain(
            *[eventDict[c]['subpops'] for c in child_events]))
        sub_pops.difference_update(child_pops)
        sub_pops.update(parent_pops)

        # try:
        #     times = [t for t in map(node_time, parent_pops)]
        #     assert np.allclose(times, times[0])
        # except TypeError:
        #     ## autograd sometimes raise TypeError for this assertion
        #     pass

        eventDict[e] = {'parent_pops': tuple(parent_pops), 'subpops': tuple(
            sub_pops), 'child_pops': {c: currEvents[c] for c in child_pops}}
        currEvents.update({p: e for p in sub_pops})
        for p in child_pops:
            del currEvents[p]
        eventEdgeList += [(e, c) for c in child_events]
    ret = nx.DiGraph(eventEdgeList)
    for e in eventDict:
        ret.add_node(e, **(eventDict[e]))

    assert len(currEvents) == 1
    root, = [v for k, v in list(currEvents.items())]
    ret.root = root

    return ret

# methods for constructing demography from string



def get_treeseq_configs(treeseq, sampled_n):
    mat = np.zeros((len(sampled_n), sum(sampled_n)), dtype=int)
    j = 0
    for i, n in enumerate(sampled_n):
        for _ in range(n):
            mat[i, j] = 1
            j += 1
    mat = scipy.sparse.csr_matrix(mat)

    def get_config(genos):
        derived_counts = mat.dot(genos)
        return np.array([
            sampled_n - derived_counts,
            derived_counts
        ]).T

    for v in treeseq.variants():
        yield get_config(v.genotypes)
