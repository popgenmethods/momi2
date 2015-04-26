import autograd.numpy as np
from autograd.numpy import sum
import scipy.misc
import bidict as bd
from util import memoize_instance, memoize, my_trace, swapaxes, my_einsum, fft_einsum

def sum_antidiagonals(arr, labels, axis0, axis1, new_axis):
    assert axis0 != axis1
    idx0,idx1 = labels.index(axis0), labels.index(axis1)

    ret = swapaxes(swapaxes(arr, idx0, 0), idx1, 1)[::-1,...]
    ret = np.array([my_trace(ret,offset=k) 
                    for k in range(-ret.shape[0]+1,ret.shape[1])])    

    labels = list(labels)
    labels[idx0],labels[idx1] = labels[0],labels[1]
    labels = [new_axis] + labels[2:]
   
    return ret,labels


class SumProduct(object):
    ''' 
    compute joint SFS entry via a sum-product like algorithm,
    using Moran model to do transitions up the tree,
    and using Polanski Kimmel to compute frequency of mutations at each node in tree
    '''
    def __init__(self, demography):
        self.G = demography
        
    def p(self, normalized = False):
        '''Return joint SFS entry for the demography'''
        ret = self.joint_sfs(self.G.event_root)
        if normalized:
            ret = ret / self.G.totalSfsSum
        return ret

    @memoize_instance
    def leaf_likelihood_bottom(self, leaf):
        n_node = self.G.node_data[leaf]['lineages']
        ret = np.zeros(n_node + 1)
        ret[self.G.node_data[leaf]['derived']] = 1.0
        return ret

    def combinatorial_factors(self, node):
        n_node = self.G.n_lineages_at_node[node]
        return scipy.misc.comb(n_node, np.arange(n_node + 1))

    ## TODO: make this a property of the demography, instead of the sum_product
    @memoize_instance
    def truncated_sfs(self, node):
        n_node = self.G.n_lineages_at_node[node]

        sfs = [self.G.node_data[node]['model'].freq(n_derived, n_node) for n_derived in range(n_node + 1)]
        if self.G.node_data[node]['model'].tau == float('inf'):
            assert node == self.G.root
            sfs[-1] = 0.0
        #sfs[sfs == float("inf")] = 0.0
        return np.array(sfs)

    @memoize_instance
    def partial_likelihood_top(self, event, popList):
        ''' Partial likelihood of data at top of node, i.e.
        i.e. = P(n_top) P(x | n_derived_top, n_ancestral_top)
        note n_top is fixed in Moran model, so P(n_top)=1
        '''       
        ret = self.partial_likelihood_bottom(event)
        for pop in popList:
            ret = self.G.node_data[pop]['model'].transition_prob(ret,
                                                                 self.G.sub_pops(event).index(pop))
        return ret

    @memoize_instance
    def partial_likelihood_bottom(self, event):
        '''Partial likelihood of data under Moran model, given alleles at bottom of node
        i.e. = P(n_bottom) P(x | n_derived_bottom, n_ancestral_bottom)
        note n_bottom is fixed in Moran model, so P(n_bottom)=1
        '''
        if self.G.event_type(event) == 'leaf':
            leafpop, = self.G.parent_pops(event)
            return self.leaf_likelihood_bottom(leafpop)
        elif self.G.event_type(event) == 'admixture':
            childpop, = self.G.child_pops(event).keys()
            p1,p2 = self.G.parent_pops(event)

            childEvent, = self.G.eventTree[event]
            childTopLik = self.partial_likelihood_top(childEvent, frozenset([childpop]))
            
            admixture_prob, admixture_idxs = self.G.admixture_prob(childpop)
            return my_einsum(childTopLik, self.G.sub_pops(childEvent),
                             admixture_prob, admixture_idxs,
                             self.G.sub_pops(event))
        elif self.G.event_type(event) == 'merge_subpops':
            newpop, = self.G.parent_pops(event)
            childPops = self.G[newpop]
            childEvent, = self.G.eventTree[event]

            ret = self.partial_likelihood_top(childEvent, frozenset(childPops))
            
            c1,c2 = childPops
            below_subpops = self.G.sub_pops(childEvent)
            for c in c1,c2:
                ret = my_einsum(ret, below_subpops,
                                self.combinatorial_factors(c), [c],
                                below_subpops)
            ret,idxs = sum_antidiagonals(ret, below_subpops, c1, c2, newpop)
            return my_einsum(ret, idxs,
                             1.0/self.combinatorial_factors(newpop), [newpop],
                             self.G.sub_pops(event))
        elif self.G.event_type(event) == 'merge_clusters':
            newpop, = self.G.parent_pops(event)
            child_liks = []
            child_sub_pops = []
            for childPop, childEvent in self.G.child_pops(event).iteritems():
                ## TODO: remove frozenset here (and elsewhere) after removing memoization
                childTopLik = self.partial_likelihood_top(childEvent, frozenset([childPop]))
                sub_pops = [newpop if x == childPop else x for x in self.G.sub_pops(childEvent)]
                child_sub_pops.append(sub_pops)
                childTopLik = my_einsum(childTopLik, sub_pops,
                                        self.combinatorial_factors(childPop), [newpop],
                                        sub_pops)
                child_liks.append(childTopLik)
            sub_pops = self.G.sub_pops(event)
            ret = fft_einsum(child_liks[0], child_sub_pops[0],
                             child_liks[1], child_sub_pops[1],
                             sub_pops,
                             [newpop])
            ret = my_einsum(ret, sub_pops,
                            1.0/self.combinatorial_factors(newpop), [newpop],
                            sub_pops)
            return ret
        else:
            raise Exception("Event type %s not yet implemented" % self.G.event_type(event))
       
    @memoize_instance
    def joint_sfs(self, event):
        '''The joint SFS entry for the configuration under this node'''
        event_subpops = self.G.sub_pops(event)
        # if no derived leafs, return 0
        if all(self.G.n_derived_subtended_by[subpop] == 0 for subpop in event_subpops):
            return 0.0
        
        ret = 0.0
        bottom_likelihood = self.partial_likelihood_bottom(event)
        for newpop in self.G.parent_pops(event):
            # term for mutation occurring at the newpop
            newpop_idx = event_subpops.index(newpop)
            idx = [0] * bottom_likelihood.ndim
            idx[newpop_idx] = slice(bottom_likelihood.shape[newpop_idx])
            ret = ret + np.sum(bottom_likelihood[idx] * self.truncated_sfs(newpop))

        if self.G.event_type(event) == 'leaf':
            return ret
        # add on terms for mutation occurring below this node
        # if no derived leafs on right, add on term from the left
        elif self.G.event_type(event) == 'merge_clusters':
            c1, c2 = self.G.eventTree[event]
            for child, other_child in ((c1, c2), (c2, c1)):
                if all(self.G.n_derived_subtended_by[subpop] == 0 for subpop in self.G.sub_pops(child)):
                    ret += self.joint_sfs(other_child)
            return ret
        elif self.G.event_type(event) == 'merge_subpops' or self.G.event_type(event) == 'admixture':
            childEvent, = self.G.eventTree[event]
            ret += self.joint_sfs(childEvent)
            return ret
        else:
            raise Exception("Event type %s not yet implemented" % self.G.event_type(event))
