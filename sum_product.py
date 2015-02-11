from __future__ import print_function, division
from combinatorics import logbinom, log_urn_prob
import scipy.misc, scipy.signal
import warnings
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from util import memoize_instance, memoize
from collections import namedtuple
from huachen_eqs import SFS_Chen
from sfs_polanski_kimmel import SFS_PolanskiKimmel
from moran_model import MoranTransition


class SumProduct(object):
    ''' 
    compute joint SFS entry via a sum-product like algorithm,
    using Moran model to do transitions up the tree,
    and using Polanski Kimmel to compute frequency of mutations at each node in tree
    '''
    def __init__(self, demography):
        self.G = demography
         
        # assert self.n_derived_leafs(tree) > 0 and self.n_derived_leafs(tree) < self.n_leaf_lins(tree)
        
    def p(self):
        '''Return joint SFS entry for the demography'''
        return self.joint_sfs(self.G.root)

    def leaf_likelihood_bottom(self, node):
        n_node = self.G.n_leaf_lineages[node]
        ret = np.zeros(n_node + 1)
        ret[self.G[node]['n_derived']] = 1.0
        return ret

    def combinatorial_factors(self, node):
        n_node = self.G.n_leaf_lineages[node]
        return scipy.misc.comb(n_node, np.arange(n_node + 1))

    def truncated_sfs(self, node):
        n_node = self.G.n_leaf_lineages[node]
        sfs = np.array([self.G[node]['eta'].freq(n_derived, n_node) for n_derived in range(n_node + 1)])
        sfs[sfs == float("inf")] = 0.0
        return sfs

    def convolve(self, left, right):
        return scipy.signal.fftconvolve(left, right)

    def partial_top_likelihood(self, node):
        ''' Partial likelihood of data at top of node, i.e.
        i.e. = P(n_top) P(x | n_derived_top, n_ancestral_top)
        note n_top is fixed in Moran model, so P(n_top)=1
        '''       
        bottom_likelihood = self.partial_likelihood_bottom(node)       
        return node.moran.computeAction(bottom_likelihood)

    @memoize_instance    
    def partial_likelihood_bottom(self, node):
        '''Partial likelihood of data under Moran model, given alleles at bottom of node
        i.e. = P(n_bottom) P(x | n_derived_bottom, n_ancestral_bottom)
        note n_bottom is fixed in Moran model, so P(n_bottom)=1
        '''
        if self.G.is_leaf(node):
            return self.leaf_likelihood_bottom(node)

        liks = [self.partial_top_likelihood(child) * self.combinatorial_factors(child) for child in self.G[node]]
        return self.convolve(*liks) / self.combinatorial_factors(node)
       
    def joint_sfs(self, node):
        '''The joint SFS entry for the configuration under this node'''
        # if no derived leafs, return 0
        n_leaves_subtended = sum([self.G.n_leaf_lineages[leaf] for leaf in self.G.leaves_subtended_by(node)])
        if self.G.n_derived_leaves[node] == 0:
            return 0.0
               
        # term for mutation occurring at this node
        ret = (self.partial_likelihood_bottom(node) * self.truncated_sfs(node)).sum()
        
        if self.G.is_leaf(node):
            return ret
        
        # add on terms for mutation occurring below this node
        # if no derived leafs on right, add on term from the left
        for child in self.G[node]:
            if self.G.n_leaf_derived[child] == 0:
                ret += self.joint_sfs(child)
        return ret
