import autograd.numpy as np
from autograd.numpy import sum
import scipy.misc
import bidict as bd
from util import memoize_instance, memoize, my_trace, swapaxes, my_einsum, fft_einsum, sum_antidiagonals

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
        #ret = self.joint_sfs(self.G.event_root)
        _,ret = self.partial_likelihood(self.G.event_root)
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
        lik,sfs = self.partial_likelihood(event)
        for pop in popList:
            lik = self.G.node_data[pop]['model'].transition_prob(lik,
                                                                 self.G.sub_pops(event).index(pop))
        return lik,sfs

    @memoize_instance
    def partial_likelihood(self, event):
        lik_fun = eval("_%s_likelihood" % self.G.event_type(event))
        lik,sfs = lik_fun(self, event)
        
        # add on sfs entry at this event
        event_subpops = self.G.sub_pops(event)
        for newpop in self.G.parent_pops(event):
            # term for mutation occurring at the newpop
            newpop_idx = event_subpops.index(newpop)
            idx = [0] * lik.ndim
            idx[newpop_idx] = slice(lik.shape[newpop_idx])
            sfs = sfs + np.sum(lik[idx] * self.truncated_sfs(newpop))

        return lik,sfs


def _leaf_likelihood(sp, event):
    leaf, = sp.G.parent_pops(event)
    n_node = sp.G.node_data[leaf]['lineages']
    lik = np.zeros(n_node + 1)
    lik[sp.G.node_data[leaf]['derived']] = 1.0
    return lik,0.0
    #return ret

def _admixture_likelihood(sp, event):
    child_pop, = sp.G.child_pops(event).keys()
    p1,p2 = sp.G.parent_pops(event)

    child_event, = sp.G.eventTree[event]
    ## TODO: remove frozenset
    lik,sfs = sp.partial_likelihood_top(child_event, frozenset([child_pop]))

    admixture_prob, admixture_idxs = sp.G.admixture_prob(child_pop)
    lik = my_einsum(lik, sp.G.sub_pops(child_event),
                    admixture_prob, admixture_idxs,
                    sp.G.sub_pops(event))

    return lik,sfs

def _merge_subpops_likelihood(sp, event):
    newpop, = sp.G.parent_pops(event)
    child_pops = sp.G[newpop]
    child_event, = sp.G.eventTree[event]

    lik,sfs = sp.partial_likelihood_top(child_event, frozenset(child_pops))

    c1,c2 = child_pops
    below_subpops = sp.G.sub_pops(child_event)
    for c in c1,c2:
        lik = my_einsum(lik, below_subpops,
                        sp.combinatorial_factors(c), [c],
                        below_subpops)
    lik,idxs = sum_antidiagonals(lik, below_subpops, c1, c2, newpop)
    lik = my_einsum(lik, idxs,
                    1.0/sp.combinatorial_factors(newpop), [newpop],
                    sp.G.sub_pops(event))

    return lik,sfs

def _merge_clusters_likelihood(sp, event):
    newpop, = sp.G.parent_pops(event)
    child_liks = []
    child_sub_pops = []

    for child_pop, child_event in sp.G.child_pops(event).iteritems():
        ## TODO: remove frozenset here (and elsewhere) after removing memoization
        lik,sfs = sp.partial_likelihood_top(child_event, frozenset([child_pop]))

        sub_pops = [newpop if x == child_pop else x for x in sp.G.sub_pops(child_event)]
        child_sub_pops.append(sub_pops)
        lik = my_einsum(lik, sub_pops,
                        sp.combinatorial_factors(child_pop), [newpop],
                        sub_pops)

        child_liks.append((lik,sfs))

    child_liks,child_sfs = zip(*child_liks)
    sub_pops = sp.G.sub_pops(event)

    lik = fft_einsum(child_liks[0], child_sub_pops[0],
                     child_liks[1], child_sub_pops[1],
                     sub_pops,
                     [newpop])
    lik = my_einsum(lik, sub_pops,
                    1.0/sp.combinatorial_factors(newpop), [newpop],
                    sub_pops)
    
    sfs = 0.0
    for freq, other_lik in zip(child_sfs, child_liks[::-1]):
        sfs = sfs + freq * np.squeeze(other_lik[ [slice(1)] * other_lik.ndim])
    return lik, sfs
