import networkx as nx
from util import default_ms_path, memoize_instance, memoize, truncate0
from math_functions import einsum2, fft_einsum
import scipy, scipy.misc
import autograd.numpy as np

from sum_product import compute_sfs

import parse_ms
import os, itertools

def make_demography(ms_cmd, *args, **kwargs):
    return Demography(parse_ms._to_nx(ms_cmd, *args, **kwargs))

class Demography(nx.DiGraph):
    def __init__(self, to_copy, *args, **kwargs):
        '''
        to_copy: a networkx.DiGraph returned by parse_ms.to_nx(),
                 or another Demography object
        '''
        super(Demography, self).__init__(to_copy, *args, **kwargs)
        self.leaves = set([k for k, v in self.out_degree().items() if v == 0])
        self.event_tree = build_event_tree(self)

    @property
    def ms_cmd(self):
        '''The ms command line equivalent to this demography'''
        return self.graph['cmd']

    def simulate_sfs(self, num_sims, ms_path=default_ms_path(), theta=None, seed=None, additionalParams=""):
        '''
        Simulates num_sims independent SFS's from the demography, using ms or
        similar program (e.g. scrm, macs).

        Default value of ms_path is system variable $MS_PATH.

        If theta = None, uses total branch lengths for frequencies (ala fastsimcoal).

        returns list [{tuple(config) : count}] of length num_sims
        '''
        return parse_ms.simulate_sfs(self, num_sims, ms_path, theta, seed, additionalParams)

    @memoize_instance
    def n_lineages(self, node):
        return np.sum(self.node[l]['lineages'] for l in self.leaves_subtended_by(node))

    @memoize_instance
    def leaves_subtended_by(self, node):
        return self.leaves & set(nx.dfs_preorder_nodes(self, node))

    def truncated_sfs(self, node):
        '''The truncated SFS at node.'''
        return self.node[node]['model'].sfs(self.n_lineages(node))

    def apply_transition(self, node, array, axis):
        '''Apply Moran model transition at node to array along axis.'''
        return self.node[node]['model'].transition_prob(array, axis)
   
    @property
    def root(self):
        '''The root (ancestral) population.'''
        ret, = self.parent_pops(self.event_root)
        return ret

    @property
    def event_root(self):
        '''The root of the junction tree.'''
        return self.event_tree.root

    def event_type(self, event):
        if len(event) == 1:
            return 'leaf'
        elif len(self.parent_pops(event)) == 2:
            return 'admixture'
        elif len(self.event_tree[event]) == 2:
            return 'merge_clusters'
        else:
            return 'merge_subpops'

    def sub_pops(self, event):
        '''
        The group of subpopulations corresponding to this event in the junction tree.
        '''
        return self.event_tree.node[event]['subpops']

    def parent_pops(self, event):
        '''The populations arising due to this event, backwards in time.'''
        return self.event_tree.node[event]['parent_pops']

    def child_pops(self, event):
        '''
        Returns dict of 
        {child_pop : child_event},
        which gives populations arising from this event forward in time,
        and the corresponding child events in the junction tree.
        '''
        return self.event_tree.node[event]['child_pops']
   
    @memoize_instance
    def admixture_prob(self, admixture_node):
        '''
        Array with dim [n_admixture_node+1, n_parent1_node+1, n_parent2_node+1],
        giving probability of derived counts in child, given derived counts in parents
        '''
        n_node = self.n_lineages(admixture_node)

        # admixture node must have two parents
        edge1,edge2 = self.in_edges([admixture_node], data=True)
        nd = self.node[admixture_node]
        parent1,parent2 = edge1[0], edge2[0]
        prob1,prob2 = nd['splitprobs'][parent1], nd['splitprobs'][parent2]
        assert prob1 + prob2 == 1.0

        n_from_1 = np.arange(n_node+1)
        n_from_2 = n_node - n_from_1
        binom_coeffs = (prob1**n_from_1) * (prob2**n_from_2) * scipy.misc.comb(n_node, n_from_1)
        ret = einsum2(der_in_admixture_node(n_node), range(4),
                      binom_coeffs, [0],
                      [1,2,3])
        assert ret.shape == tuple([n_node+1] * 3)
        return ret, [admixture_node, parent1, parent2]

@memoize
def der_in_admixture_node(n_node):
    '''
    returns 4d-array, [n_from_parent1, der_in_child, der_in_parent1, der_in_parent2].
    Used by Demography.admixture_prob
    '''
    # axis0=n_from_parent, axis1=der_from_parent, axis2=der_in_parent
    der_in_parent = np.tile(np.arange(n_node+1), (n_node+1,n_node+1,1))
    n_from_parent = np.transpose(der_in_parent, [2,0,1])
    der_from_parent = np.transpose(der_in_parent, [0,2,1])
    
    anc_in_parent = n_node - der_in_parent
    anc_from_parent = n_from_parent - der_from_parent
    
    x = scipy.misc.comb(der_in_parent, der_from_parent) * scipy.misc.comb(anc_in_parent, anc_from_parent) / scipy.misc.comb(n_node, n_from_parent)

    ret = fft_einsum(x, [0, 1, 2],
                     x[::-1,...], [0, 1, 3],
                     [0,1,2,3], [1])[:,:(n_node+1),:,:]
    # deal with small negative numbers from fft
    ret = truncate0(ret, axis=1)
    return ret

def build_event_tree(demo):
    eventEdgeList = []
    currEvents = {l : (l,) for l in demo.leaves}
    eventDict = {e : {'subpops' : (l,), 'parent_pops' : (l,), 'child_pops' : {}} for l,e in currEvents.iteritems()}

    for e in demo.graph['events']:
        # get the population edges forming the event
        parent_pops, child_pops = map(set, zip(*e))
        child_events = set([currEvents[c] for c in child_pops])
        assert len(e) == 2 and len(parent_pops) + len(child_pops) == 3 and len(child_events) in (1,2)

        sub_pops = set(itertools.chain(*[eventDict[c]['subpops'] for c in child_events]))
        sub_pops.difference_update(child_pops)
        sub_pops.update(parent_pops)

        eventDict[e] = {'parent_pops' : tuple(parent_pops), 'subpops' : tuple(sub_pops), 'child_pops' : {c : currEvents[c] for c in child_pops}}
        currEvents.update({p : e for p in sub_pops})
        for p in child_pops:
            del currEvents[p]
        eventEdgeList += [(e, c) for c in child_events]
    ret = nx.DiGraph(eventEdgeList)
    for e in eventDict:
        ret.add_node(e, **(eventDict[e]))

    assert len(currEvents) == 1
    root, = [v for k,v in currEvents.iteritems()]
    ret.root = root

    return ret
