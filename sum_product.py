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
    def __init__(self, demography, fftconvolve=False):
        self.G = demography
        self.fftconvolve = fftconvolve
        # assert self.n_derived_leafs(tree) > 0 and self.n_derived_leafs(tree) < self.n_leaf_lins(tree)

    def p(self):
        '''Return joint SFS entry for the demography'''
        return self.joint_sfs(self.G.root)

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
        if self.fftconvolve:
            ret = scipy.signal.fftconvolve(*liks)
        else:
            ret = np.convolve(*liks)
        return ret / self.combinatorial_factors(node)

    @memoize_instance
    def joint_sfs(self, node):
        '''The joint SFS entry for the configuration under this node'''
        # if no derived leafs, return 0
        if self.G.n_derived_subtended_by[node] == 0:
            return 0.0

        # term for mutation occurring at this node
        ret = (self.partial_likelihood_bottom(node) * self.truncated_sfs(node)).sum()

        if self.G.is_leaf(node):
            return ret

        # add on terms for mutation occurring below this node
        # if no derived leafs on right, add on term from the left
        c1, c2 = self.G[node]
        for child, other_child in ((c1, c2), (c2, c1)):
            if self.G.n_derived_subtended_by[child] == 0:
                ret += self.joint_sfs(other_child)
        return ret
