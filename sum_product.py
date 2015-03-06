from __future__ import print_function, division
import scipy.misc, scipy.signal
import math
import numpy as np

from util import memoize_instance, memoize

class SumProduct(object):
    ''' 
    compute joint SFS entry via a sum-product like algorithm,
    using Moran model to do transitions up the tree,
    and using Polanski Kimmel to compute frequency of mutations at each node in tree
    '''
    def __init__(self, demography):
        self.G = demography
        # TODO: make eventTree contain the demography rather than vice versa
        self.eventTree = self.G.eventTree
        # assert self.n_derived_leafs(tree) > 0 and self.n_derived_leafs(tree) < self.n_leaf_lins(tree)
        
    def p(self, normalized = False):
        '''Return joint SFS entry for the demography'''
        assert len(self.G.node_data[self.G.root]['model']) == 1
        ret = self.joint_sfs(self.eventTree.root)
        if normalized:
            ret /= self.G.totalSfsSum
        return ret

    @memoize_instance
    def leaf_likelihood_bottom(self, leaf):
        n_node = self.G.node_data[leaf]['lineages']
        ret = np.zeros(n_node + 1)
        ret[self.G.node_data[leaf]['derived']] = 1.0
        return ret

    def combinatorial_factors(self, node):
        n_node = self.G.n_lineages_subtended_by[node]
        return scipy.misc.comb(n_node, np.arange(n_node + 1))

    @memoize_instance
    def truncated_sfs(self, node):
        n_node = self.G.n_lineages_subtended_by[node]
        sfs = np.array([self.G.node_data[node]['model'].freq(n_derived, n_node) for n_derived in range(n_node + 1)])
        sfs[sfs == float("inf")] = 0.0
        return sfs

    @memoize_instance
    def partial_likelihood_top(self, top, bottom):
        ''' Partial likelihood of data at top of node, i.e.
        i.e. = P(n_top) P(x | n_derived_top, n_ancestral_top)
        note n_top is fixed in Moran model, so P(n_top)=1
        '''       
        bottom_likelihood = self.partial_likelihood_bottom(bottom)       
        return self.G.node_data[bottom]['model'].transition_prob(bottom_likelihood)

    @memoize_instance
    def partial_likelihood_bottom(self, node):
        '''Partial likelihood of data under Moran model, given alleles at bottom of node
        i.e. = P(n_bottom) P(x | n_derived_bottom, n_ancestral_bottom)
        note n_bottom is fixed in Moran model, so P(n_bottom)=1
        '''
        if self.G.is_leaf(node):
            return self.leaf_likelihood_bottom(node)
        liks = [self.partial_likelihood_top(node, child) * self.combinatorial_factors(child) 
                for child in self.G[node]]
        return scipy.signal.fftconvolve(*liks) / self.combinatorial_factors(node)
       
    @memoize_instance
    def joint_sfs(self, event):
        '''The joint SFS entry for the configuration under this node'''
        # if no derived leafs, return 0
        if all(self.G.n_derived_subtended_by[subpop] == 0 for subpop in event['subpops']):
            return 0.0
        
        newpop = event['newpop']
        # term for mutation occurring at the newpop
        # do some fancy slicing to consider only configs where derived alleles are all in newpop
        idx = [0] * len(event['subpops'])
        idx[event['subpops'].idx(newpop)] = slice(self.G.n_lineages_subtended_by[newpop]+1)
        ret = (self.partial_likelihood_bottom(newpop)[idx] * self.truncated_sfs(newpop)).sum()
        
        #if self.G.is_leaf(node):
        if event['type'] == 'leaf':
            return ret
        # add on terms for mutation occurring below this node
        # if no derived leafs on right, add on term from the left
        elif event['type'] == 'merge_clusters':
            c1, c2 = self.eventTree[event]
            for child, other_child in ((c1, c2), (c2, c1)):
                #if self.G.n_derived_subtended_by[child] == 0:
                if all(self.G.n_derived_subtended_by[subpop] == 0 for subpop in child['subpops']):
                    ret += self.joint_sfs(other_child)
            return ret
        else:
            raise Exception("Event type %s not yet implemented" % event['type'])
