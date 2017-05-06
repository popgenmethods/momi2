import networkx as nx
from .util import memoize_instance, memoize
from .math_functions import sum_antidiagonals, convolve_axes, binom_coeffs, roll_axes, hypergeom_quasi_inverse, par_einsum, convolve_sum_axes
import scipy
import scipy.misc
from scipy.misc import comb
import scipy.sparse
import autograd.numpy as np
import autograd
from autograd import primitive
import msprime

from .size_history import ConstantHistory, ExponentialHistory, PiecewiseHistory
from .compute_sfs import expected_total_branch_len
from .data_structure import seg_site_configs
from functools import partial

import os
import itertools

import logging
logger = logging.getLogger(__name__)


def getval(x):
    try:
        return x.value
    except AttributeError:
        return x


def demographic_history(events, archaic_times_dict=None, default_N=1.0):
    return DemographicHistory(events, archaic_times_dict, default_N)


def make_demography(events, sampled_pops, sampled_n, sampled_t=None, default_N=1.0, time_scale='ms'):
    """
    Depracated. Use momi.demographic_history() instead.

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

    logger.debug("Making demography, {0}".format({'events': [tuple(getval(x) for x in ev) for ev in events],
                                                  'sampled_t': [getval(x) for x in sampled_t],
                                                  'sampled_pops': list(sampled_pops),
                                                  'sampled_n': list(sampled_n),
                                                  'default_N': getval(default_N),
                                                  'time_scale': getval(time_scale)}))

    if time_scale == 'ms':
        time_scale = 1.0
    elif time_scale == 'standard':
        time_scale = 4.0
    elif isinstance(time_scale, str):
        raise DemographyError("time_scale must be float, 'ms', or 'standard'")

    old_default_N = default_N
    default_N = default_N * time_scale
    old_events, events = events, []
    for e in old_events:
        if e[0] == '-en':
            flag, t, i, N = e
            e = flag, t, i, N * time_scale
        events += [e]

    # process all events
    _G = nx.DiGraph()
    _G.graph['event_cmds'] = tuple(events)
    _G.graph['default_N'] = default_N
    _G.graph['events_as_edges'] = []
    # the nodes currently at the root of the graph, as we build it up from the
    # leafs
    _G.graph['roots'] = {}

    # create sampling events
    sampling_events = [('-eSample', t, i, n)
                       for i, n, t in zip(sampled_pops, sampled_n, sampled_t)]
    events = sampling_events + list(events)

    # sort events by time
    events = sorted(events, key=lambda x: x[1])

    event_funs = {
        "-" + f.__name__[1:]: f for f in [_ep, _eg, _en, _ej, _es, _eSample]}
    for event in events:
        flag, args = event[0], event[1:]
        event_funs[flag](_G, *args)

    assert _G.node
    _G.graph['roots'] = [r for _, r in list(
        _G.graph['roots'].items()) if r is not None]

    if len(_G.graph['roots']) != 1:
        raise DemographyError("Must have a single root population")

    node, = _G.graph['roots']
    _set_sizes(_G.node[node], float('inf'))

    _G.graph['sampled_pops'] = tuple(sampled_pops)
    return Demography(_G)


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


class Demography(object):
    """
    The demographic history relating a sample of individuals.
    """

    def __init__(self, G, diff_cache_keys=[], diff_cache_vals=[]):
        """
        For internal use only.
        Use make_demography() to create a Demography.
        """
        self._G = G
        self._event_tree = _build_event_tree(self._G)

        # a hack that allows us reorganize some computations during auto differentiation
        # methods decorated by @differentiable_method will first look if result
        # is in diff_cache before computing it
        assert len(diff_cache_keys) == len(diff_cache_vals)
        self._diff_cache = dict(list(zip(diff_cache_keys, diff_cache_vals)))

    def _get_differentiable_part(self):
        # use this with _get_graph_structure()
        # to re-organize certain computations during automatic differentiation
        expected_total_branch_len(self)
        assert self._diff_cache

        keys, vals = list(zip(*list(self._diff_cache.items())))
        # convert vals to autograd.TupleNode (as opposed to a tuple of
        # autograd.Node)
        vals = autograd.container_types.make_tuple(*vals)

        return keys, vals

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

    @property
    def events(self):
        """
        The list of events (tuples) making up the demographic history
        """
        return self._G.graph['event_cmds']

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

    def _get_multipop_moran(self, sampled_pops, sampled_n):
        if sampled_pops is not None and sampled_n is not None:
            if tuple(sampled_pops) != tuple(self.sampled_pops) or tuple(sampled_n) != tuple(self.sampled_n):
                raise ValueError("sampled_pops/sampled_n attributes already present in Demography object, and disagrees with the provided values. To avoid this error, replace the depracated momi.make_demography() with momi.demographic_history().")
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

        child_in = self._G.in_degree(child_pops)
        recipient, = [k for k, v in list(child_in.items()) if v == 2]
        non_recipient, = [k for k, v in list(child_in.items()) if v == 1]

        parent_out = self._G.out_degree(parent_pops)
        donor, = [k for k, v in list(parent_out.items()) if v == 2]
        non_donor, = [k for k, v in list(parent_out.items()) if v == 1]

        return recipient, non_recipient, donor, non_donor

    """
    ALL methods returning floats BELOW HERE
    They should be decorated by @differentiable_method to ensure cacheing of differentiable objects!!!!
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

        return seg_site_configs(
            self.sampled_pops,
            (get_treeseq_configs(locus, self.sampled_n)
             for locus in treeseq)
        )

    def simulate_trees(self, **kwargs):
        sampled_t = self.sampled_t
        if sampled_t is None:
            sampled_t = 0.0
        sampled_t = np.array(sampled_t) * np.ones(len(self.sampled_pops))

        pops = {p: i for i, p in enumerate(self.sampled_pops)}
        sampled_n = self.sampled_n

        events = sorted(self.events, key=lambda x: x[1])
        demographic_events = []
        for event in events:
            flag = event[0]
            if flag == '-ep':
                _, t, i, j, pij = event
                for k in (i, j):
                    if k not in pops:
                        pops[k] = len(pops)
                demographic_events.append(msprime.MassMigration(
                    t, pops[i], pops[j], proportion=pij))
                continue
            elif flag == '-ej':
                _, t, i, j = event
                for k in (i, j):
                    if k not in pops:
                        pops[k] = len(pops)
                demographic_events.append(
                    msprime.MassMigration(t, pops[i], pops[j]))
            elif flag == '-eg':
                _, t, i, alpha = event
                if i not in pops:
                    pops[i] = len(pops)
                demographic_events.append(msprime.PopulationParametersChange(t,
                                                                             growth_rate=alpha,
                                                                             population_id=pops[i]))
            elif flag == '-en':
                _, t, i, N = event
                if i not in pops:
                    pops[i] = len(pops)
                demographic_events.append(msprime.PopulationParametersChange(t,
                                                                             initial_size=N / 4,
                                                                             growth_rate=0,
                                                                             population_id=pops[i]))
            else:
                assert False
        return msprime.simulate(population_configurations=[msprime.PopulationConfiguration()
                                                           for _ in range(len(pops))],
                                Ne=self.default_N / 4,
                                demographic_events=demographic_events,
                                samples=[msprime.Sample(population=pops[p], time=t)
                                         for p, t, n in zip(self.sampled_pops, self.sampled_t, self.sampled_n)
                                         for _ in range(n)],
                                **kwargs)

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

    x1 = x1[:,:,:upper,lower:upper]
    x2 = x2[:,:,:(n_node-lower+1),lower:upper]

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
    currEvents = {k: (k,) for k, v in list(G.out_degree().items()) if v == 0}
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


def _es(G, t, i, p):
    raise DemographyError(
        "Flag -es not implemented, use -ep instead. See help(demography).")


def _ej(G, t, i, j):
    if i not in G.graph['roots']:
        G.graph['roots'][i] = None
        # don't need to do anything else
        return
    _check_ej_ep_pops(G, '-ej', t, i, j)

    i0, j0 = (G.graph['roots'][k] for k in (i, j))
    j1 = (j, j0[1] + 1)
    assert j1 not in G.nodes()

    for k in i0, j0:
        # sets the TruncatedSizeHistory, and N_top and growth_rate for all
        # epochs
        _set_sizes(G.node[k], t)
    _ej_helper(G, t, i0, j0, j1)

    G.graph['roots'][j] = j1
    G.graph['roots'][i] = None


def _ej_helper(G, t, i0, j0, j1):
    prev = G.node[j0]['sizes'][-1]
    G.add_node(j1, sizes=[{'t': t, 'N': prev['N_top'],
                           'growth_rate':prev['growth_rate']}])

    new_edges = ((j1, i0), (j1, j0))
    G.graph['events_as_edges'].append(new_edges)
    G.add_edges_from(new_edges)


def _en(G, t, i, N):
    _check_en_eg_pops(G, '-en', t, i, N)
    G.node[G.graph['roots'][i]]['sizes'].append(
        {'t': t, 'N': N, 'growth_rate': None})


def _eg(G, t, i, growth_rate):
    _check_en_eg_pops(G, '-eg', t, i, growth_rate)
    G.node[G.graph['roots'][i]]['sizes'].append(
        {'t': t, 'growth_rate': growth_rate})


def _ep(G, t, i, j, pij):
    if pij < 0. or pij > 1.:
        raise DemographyError("Invalid event {0}: pulse probability must be between 0,1".format(
            tuple(getval(v) for v in ('-ep', t, i, j, pij))))

    if i not in G.graph['roots']:
        # don't need to do anything
        return

    _check_ej_ep_pops(G, '-ep', t, i, j, pij)

    children = {k: G.graph['roots'][k] for k in (i, j)}
    for v in list(children.values()):
        _set_sizes(G.node[v], t)

    parents = {k: (v[0], v[1] + 1) for k, v in list(children.items())}
    assert all([par not in G.node for par in list(parents.values())])

    prev_sizes = {k: G.node[c]['sizes'][-1] for k, c in list(children.items())}
    for k, s in list(prev_sizes.items()):
        G.add_node(parents[k], sizes=[
                   {'t': t, 'N': s['N_top'], 'growth_rate':s['growth_rate']}])

    G.add_edge(parents[i], children[i], prob=1. - pij)
    G.add_edge(parents[j], children[i], prob=pij)
    G.add_edge(parents[j], children[j])

    new_event = tuple((parents[u], children[v])
                      for u, v in ((i, i), (j, i), (j, j))
                      )
    G.graph['events_as_edges'] += [new_event]

    for k, v in list(parents.items()):
        G.graph['roots'][k] = v


def _eSample(G, t, i, n):
    G.add_node((i, 0),
               lineages=n)

    if i in G.graph['roots']:
        if G.graph['roots'][i] is None:
            raise DemographyError(
                "Invalid events: pop {0} removed by -ej before sample time".format(i))

        #G.node[(i,0)]['model'] = _TrivialHistory()
        G.node[(i, 0)]['sizes'] = [
            {'t': t, 'N': G.graph['default_N'], 'growth_rate':None}]
        _set_sizes(G.node[(i, 0)], t)

        prev = G.graph['roots'][i]
        _set_sizes(G.node[prev], t)

        assert prev[0] == i and prev[1] != 0
        newpop = (i, prev[1] + 1)
        _ej_helper(G, t, (i, 0), prev, newpop)
    else:
        newpop = (i, 0)
        G.node[newpop]['sizes'] = [
            {'t': t, 'N': G.graph['default_N'], 'growth_rate':None}]
    G.graph['roots'][i] = newpop


def _check_en_eg_pops(G, *event):
    flag, t, i = event[:3]
    if i in G.graph['roots'] and G.graph['roots'][i] is None:
        raise DemographyError(
            "Invalid event {0}: pop {1} was already removed by previous -ej".format(tuple(getval(e) for e in event), i))

    if i not in G.graph['roots']:
        G.graph['roots'][i] = (i, 1)
        G.add_node(G.graph['roots'][i],
                   sizes=[{'t': t, 'N': G.graph['default_N'], 'growth_rate':None}],
                   )


def _check_ej_ep_pops(G, *event):
    flag, t, i, j = event[:4]
    for k in (i, j):
        if k in G.graph['roots'] and G.graph['roots'][k] is None:
            raise DemographyError(
                "Invalid event {0}: pop {1} was already removed by previous -ej".format(tuple(getval(e) for e in event), k))

    if j not in G.graph['roots']:
        G.graph['roots'][j] = (j, 1)
        G.add_node(G.graph['roots'][j],
                   sizes=[{'t': t, 'N': G.graph['default_N'], 'growth_rate':None}],
                   )


def _set_sizes(node_data, end_time):
    assert 'model' not in node_data

    # add 'model_func' to node_data, add information to node_data['sizes']
    sizes = node_data['sizes']
    # add a dummy epoch with the end time
    sizes.append({'t': end_time})

    # do some processing
    N, growth_rate = sizes[0]['N'], sizes[0]['growth_rate']
    pieces = []
    for i in range(len(sizes) - 1):
        sizes[i]['tau'] = tau = (sizes[i + 1]['t'] - sizes[i]['t'])

        if 'N' not in sizes[i]:
            sizes[i]['N'] = N
        if 'growth_rate' not in sizes[i]:
            sizes[i]['growth_rate'] = growth_rate
        growth_rate = sizes[i]['growth_rate']
        N = sizes[i]['N']

        if growth_rate is not None and tau != float('inf'):
            pieces.append(ExponentialHistory(
                tau=tau, growth_rate=growth_rate, N_bottom=N))
            N = pieces[-1].N_top
        else:
            if growth_rate != 0. and growth_rate is not None and tau == float('inf'):
                raise DemographyError("Final epoch must have 0 growth rate")
            pieces.append(ConstantHistory(tau=tau, N=N))

        sizes[i]['N_top'] = N

        if not all([sizes[i][x] >= 0.0 for x in ('tau', 'N', 'N_top')]):
            raise DemographyError("Negative time or population size in {sizes}".format(
                sizes=[{k: getval(v) for k, v in s.items()} for s in sizes]))
    sizes.pop()  # remove the final dummy epoch

    assert len(pieces) > 0
    if len(pieces) == 0:
        node_data['model'] = pieces[0]
    else:
        node_data['model'] = PiecewiseHistory(pieces)


class DemographyError(Exception):
    pass
