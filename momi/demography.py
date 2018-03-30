from functools import partial
import networkx as nx
import scipy
import scipy.special
from scipy.special import comb
import scipy.sparse
import autograd.numpy as np
import msprime
from .compute_sfs import expected_total_branch_len
from .data.compressed_counts import _CompressedHashedCounts, _CompressedList
from .data.snps import SnpAlleleCounts
from .util import memoize_instance
from .math_functions import (
    binom_coeffs, roll_axes, hypergeom_quasi_inverse,
    par_einsum, convolve_sum_axes)

import pysam
import os
import itertools

import logging
logger = logging.getLogger(__name__)


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

    #def copy(self, sampled_n=None):
    #    """
    #    Notes
    #    -----
    #    Note that momi.expected_sfs, momi.composite_log_likelihood require
    #    Demography.sampled_n == ConfigList.sampled_n.
    #    If this is not the case, you can use copy() to create a copy with the correct
    #    sampled_n.
    #    """
    #    if sampled_n is None:
    #        sampled_n = self.sampled_n
    #    return _make_multipop_moran(self.events, self.sampled_pops, sampled_n, self.sampled_t, self.default_N)

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
        #    scipy.special.comb(n_node, n_from_1)
        #ret = par_einsum(_der_in_admixture_node(n_node), list(range(4)),
        #                 binom_coeffs, [0],
        #                 [1, 2, 3])
        ret = np.transpose(admixture_operator(n_node, prob1))
        assert ret.shape == tuple([n_node + 1] * 3)

        assert [admixture_node, parent1,
                parent2] == self._admixture_prob_idxs(admixture_node)
        return ret

    def simulate_data(self, length, num_replicates=1, **kwargs):
        treeseq = self.simulate_trees(length=length, num_replicates=num_replicates,
                                      **kwargs)
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

        #chrom = []
        chrom = _CompressedList()
        pos = []
        compressed_counts = _CompressedHashedCounts(len(self.sampled_pops))

        for c, locus in enumerate(treeseq):
            for v in locus.variants():
                compressed_counts.append(get_config(v.genotypes))
                chrom.append(c)
                pos.append(v.position)

        return SnpAlleleCounts(
            chrom, pos, compressed_counts.compressed_allele_counts(),
            self.sampled_pops, use_folded_sfs=False,
            non_ascertained_pops=[], length=length*num_replicates,
            n_read_snps=len(compressed_counts), n_excluded_snps=0)

    def simulate_vcf(self, out_prefix, mutation_rate,
                     recombination_rate, length,
                     chrom_name=1, ploidy=1, random_seed=None,
                     force=False, print_aa=True):
        out_prefix = os.path.expanduser(out_prefix)
        vcf_name = out_prefix + ".vcf"
        bed_name = out_prefix + ".bed"
        for fname in (vcf_name, bed_name):
            if not force and os.path.isfile(fname):
                raise FileExistsError(
                    "{} exists and force=False".format(fname))

        if np.any(self.sampled_n % ploidy != 0):
            raise ValueError("Sampled alleles per population must be"
                             " integer multiple of ploidy")

        with open(bed_name, "w") as bed_f:
            print(chrom_name, 0, length, sep="\t", file=bed_f)

        with open(vcf_name, "w") as vcf_f:
            treeseq = self.simulate_trees(
                mutation_rate=mutation_rate,
                recombination_rate=recombination_rate,
                length=length, num_replicates=1,
                random_seed=random_seed)

            print("##fileformat=VCFv4.2", file=vcf_f)
            print('##source="VCF simulated by momi2 using'
                  ' msprime backend"', file=vcf_f)
            print("##contig=<ID={chrom_name},length={length}>".format(
                chrom_name=chrom_name, length=length), file=vcf_f)
            print('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
                  file=vcf_f)
            print('##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">',
                  file=vcf_f)

            n_samples = int(np.sum(self.sampled_n) / ploidy)
            fields = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL",
                      "FILTER", "INFO", "FORMAT"]
            for pop, n in zip(self.sampled_pops, self.sampled_n):
                for i in range(int(n / ploidy)):
                    fields.append("{}_{}".format(pop, i))
            print(*fields, sep="\t", file=vcf_f)

            loc = next(treeseq)
            if print_aa:
                info_str = "AA=A"
            else:
                info_str = "."

            for v in loc.variants():
                gt = np.reshape(v.genotypes, (n_samples, ploidy))
                print(chrom_name, int(np.floor(v.position)),
                      ".", "A", "T", ".", ".", info_str, "GT",
                      *["|".join(map(str, sample)) for sample in gt],
                      sep="\t", file=vcf_f)

        pysam.tabix_index(vcf_name, preset="vcf", force=force)

    def simulate_trees(self, **kwargs):
        sampled_t = self.sampled_t
        if sampled_t is None:
            sampled_t = 0.0
        sampled_t = np.array(sampled_t) * np.ones(len(self.sampled_pops))

        pops = {p: i for i, p in enumerate(self.sampled_pops)}

        demographic_events = []
        for e in self._G.graph["events"]:
            e = e.get_msprime_event(self._G.graph["params"], pops)
            if e is not None:
                demographic_events.append(e)

        return msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration()
                for _ in range(len(pops))],
            Ne=self.default_N / 4,
            demographic_events=demographic_events,
            samples=[
                msprime.Sample(population=pops[p], time=t)
                for p, t, n in zip(
                        self.sampled_pops, self.sampled_t,
                        self.sampled_n)
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
#    x = scipy.special.comb(der_in_parent, der_from_parent) * scipy.special.comb(
#        anc_in_parent, anc_from_parent) / scipy.special.comb(n_node, n_from_parent)
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
