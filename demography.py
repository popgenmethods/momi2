import networkx as nx
from util import memoize_instance, memoize
from math_functions import einsum2, fft_einsum
import scipy, scipy.misc
import autograd.numpy as np

from sum_product import compute_sfs

import parse_ms
import random
import itertools

def make_demography(ms_cmd, *params):
    return Demography(parse_ms._to_nx(ms_cmd, *params))

class Demography(nx.DiGraph):
    ## TODO: remove this method
    @classmethod
    def from_newick(cls, newick, default_lineages=None, default_N=1.0):
        ms_cmd = parse_ms._from_newick(newick, default_lineages, default_N)
        return make_demography(ms_cmd)

    def __init__(self, to_copy, *args, **kwargs):
        '''
        Input:
        to_copy: either a Demography object, or networkx.DiGraph returned by parse_ms.to_nx()
        '''
        super(Demography, self).__init__(to_copy, *args, **kwargs)
        self.leaves = set([k for k, v in self.out_degree().items() if v == 0])
        self.event_tree = self.build_event_tree()

    def build_event_tree(self):
        eventEdgeList = []
        currEvents = {l : (l,) for l in self.leaves}
        eventDict = {e : {'subpops' : (l,), 'parent_pops' : (l,), 'child_pops' : {}} for l,e in currEvents.iteritems()}
        
        for e in self.graph['events']:
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
    
    @property
    def event_root(self):
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
        return self.event_tree.node[event]['subpops']

    def parent_pops(self, event):
        return self.event_tree.node[event]['parent_pops']

    # returns dict of {childPop : childEvent}
    def child_pops(self, event):
        return self.event_tree.node[event]['child_pops']

    @property
    def root(self):
        ret, = self.parent_pops(self.event_root)
        return ret
    
    def truncated_sfs(self, node):
        return self.node[node]['model'].sfs(self.n_lineages(node))

    def apply_transition(self, node, array, idx):
        return self.node[node]['model'].transition_prob(array, idx)

    @memoize_instance
    def n_lineages(self, node):
        if self.is_leaf(node):
            return self.node[node]['lineages']
        ret = 0
        for child_node in self[node]:
            ret = ret + self.n_lineages(child_node)
        return ret

    '''
    Simulates the SFS from the demography.
    If theta = None, uses total branch lengths for frequencies (ala fastsimcoal)

    returns (sumFreqs,sumSqFreqs,nonzeroFreqs)
    where
    sumFreqs = sum of frequencies across all datasets
    sumSqFreqs = sum of squared frequencies across all datasets
    nonzeroFreqs = # of datasets where frequency was > 0
    '''
    def simulate_sfs(self, num_sims, theta=None, seed=None, additionalParams=""):
        return parse_ms.simulate_sfs(self, num_sims, theta, seed, additionalParams)

    @memoize_instance
    def admixture_prob(self, admixture_node):
        '''
        Returns ndarray with dimensions [child_der, par1_der, par2_der]

        child_der: # derived alleles in admixture_node
        par1_der, par2_der: # derived alleles in parent1, parent2 of admixture_node

        returns probability of child_der given par1_der, par2_der
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

    ## TODO: is this method used?
    def is_leaf(self, node):
        return node in self.leaves

@memoize
def der_in_admixture_node(n_node):
    '''
    returns 4d-array, [n_from_parent1, der_in_child, der_in_parent1, der_in_parent2]
    '''
    # axis0=n_from_parent, axis1=der_from_parent, axis2=der_in_parent
    der_in_parent = np.tile(np.arange(n_node+1), (n_node+1,n_node+1,1))
    n_from_parent = np.transpose(der_in_parent, [2,0,1])
    der_from_parent = np.transpose(der_in_parent, [0,2,1])
    
    anc_in_parent = n_node - der_in_parent
    anc_from_parent = n_from_parent - der_from_parent
    
    x = scipy.misc.comb(der_in_parent, der_from_parent) * scipy.misc.comb(anc_in_parent, anc_from_parent) / scipy.misc.comb(n_node, n_from_parent)

    return fft_einsum(x, [0, 1, 2],
                      x[::-1,...], [0, 1, 3],
                      [0,1,2,3], [1])[:,:(n_node+1),:,:]
