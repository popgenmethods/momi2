import autograd.numpy as np
import scipy.misc
from util import memoize_instance, memoize, my_trace, swapaxes, my_einsum, fft_einsum, sum_antidiagonals

## TODO: move this into __init__ ?
def compute_sfs(demography, config_list, normalized = False, ret_branch_len=False):
    '''Return joint SFS entry for the demography'''
    data = np.array(config_list, ndmin=2)
    if data.ndim != 2 or data.shape[1] != len(demography.leaves):
        raise IOError("Invalid config_list. config_list should either be a list of k-tuples or a 2d array with k columns, where k is the number of leaf populations.")

    _,ret,branch_len = partial_likelihood(data, demography, demography.event_root)
    branch_len = branch_len - ret[1]
    # first two indices correspond to all ancestral and all derived
    ret = ret[2:]
    if normalized:
        ret = ret / branch_len
    ret = np.squeeze(ret)
    if ret_branch_len:
        return ret, branch_len
    else:
        return ret

## TODO: make this a property of the demography, instead of the sum_product
def truncated_sfs(G, node):
    n_node = G.n_lineages_at_node[node]

    sfs = [G.node_data[node]['model'].freq(n_derived, n_node) for n_derived in range(n_node + 1)]
    if G.node_data[node]['model'].tau == float('inf'):
        assert node == G.root
        sfs[-1] = 0.0
    #sfs[sfs == float("inf")] = 0.0
    return np.array(sfs)


def partial_likelihood_top(data, G, event, popList):
    ''' Partial likelihood of data at top of node, i.e.
    i.e. = P(n_top) P(x | n_derived_top, n_ancestral_top)
    note n_top is fixed in Moran model, so P(n_top)=1
    '''       
    lik,sfs,branch_len = partial_likelihood(data, G, event)
    for pop in popList:
        idx = (_lik_axes(G, event)).index(pop)
        lik = G.node_data[pop]['model'].transition_prob(lik,idx)
    return lik,sfs,branch_len

def partial_likelihood(data, G, event):
    lik_fun = eval("_%s_likelihood" % G.event_type(event))
    lik,sfs,branch_len = lik_fun(data, G, event)

    # add on sfs entry at this event
    axes = _lik_axes(G, event)
    for newpop in G.parent_pops(event):
        # term for mutation occurring at the newpop
        newpop_idx = axes.index(newpop)
        idx = [0] * lik.ndim
        idx[0], idx[newpop_idx] = slice(None), slice(None)

        sub_lik, trunc_sfs = lik[idx], truncated_sfs(G, newpop)
        sfs = sfs + my_einsum(sub_lik, ['',newpop],
                              trunc_sfs, [newpop],
                              [''])
        branch_len = branch_len + np.dot(1.0 - sub_lik[0,:] , trunc_sfs)
    return lik,sfs,branch_len


def combinatorial_factors(G, node):
    n_node = G.n_lineages_at_node[node]
    return scipy.misc.comb(n_node, np.arange(n_node + 1))

def _lik_axes(G, event):
    '''
    Returns axes labels of the partial likelihood tensor
    first axis corresponds to SFS entry
    subsequent axes correspond to subpopulations
    '''
    sub_pops = list(G.sub_pops(event))
    assert '' not in sub_pops
    return [''] + sub_pops

def _leaf_likelihood(data, G, event):
    leaf, = G.parent_pops(event)
    n_node = G.node_data[leaf]['lineages']

    n_der = data[:,sorted(G.leaves).index(leaf)]

    # the first two rows of lik correspond to all ancestral and all derived, respectively
    lik = np.zeros((len(n_der)+2, n_node + 1))
    lik[zip(*enumerate(n_der,start=2))] = 1.0
    lik[0,0] = 1.0
    lik[1,n_node] = 1.0

    return lik,0.,0.

def _admixture_likelihood(data, G, event):
    child_pop, = G.child_pops(event).keys()
    p1,p2 = G.parent_pops(event)

    child_event, = G.eventTree[event]
    lik,sfs,branch_len = partial_likelihood_top(data, G, child_event, [child_pop])

    admixture_prob, admixture_idxs = G.admixture_prob(child_pop)
    lik = my_einsum(lik, _lik_axes(G, child_event),
                    admixture_prob, admixture_idxs,
                    _lik_axes(G, event))

    return lik,sfs,branch_len

def _merge_subpops_likelihood(data, G, event):
    newpop, = G.parent_pops(event)
    child_pops = G[newpop]
    child_event, = G.eventTree[event]

    lik,sfs,branch_len = partial_likelihood_top(data, G, child_event, child_pops)

    c1,c2 = child_pops
    child_axes = _lik_axes(G, child_event)
    for c in c1,c2:
        lik = my_einsum(lik, child_axes,
                        combinatorial_factors(G, c), [c],
                        child_axes)
    lik,axes = sum_antidiagonals(lik, child_axes, c1, c2, newpop)

    assert set(axes) == set(_lik_axes(G,event))
    lik = my_einsum(lik, axes,
                    1.0/combinatorial_factors(G, newpop), [newpop],
                    _lik_axes(G, event))

    return lik,sfs,branch_len

def _merge_clusters_likelihood(data, G, event):
    newpop, = G.parent_pops(event)
    child_liks = []
    child_axes = []
    
    branch_len = 0.0
    for child_pop, child_event in G.child_pops(event).iteritems():
        lik,sfs,child_len = partial_likelihood_top(data, G, child_event, [child_pop])
        branch_len = branch_len + child_len

        axes = [newpop if x == child_pop else x for x in _lik_axes(G, child_event)]
        child_axes.append(axes)
        lik = my_einsum(lik, axes,
                        combinatorial_factors(G, child_pop), [newpop],
                        axes)

        child_liks.append((lik,sfs))

    child_liks,child_sfs = zip(*child_liks)
    axes = _lik_axes(G, event)

    lik = fft_einsum(child_liks[0], child_axes[0],
                     child_liks[1], child_axes[1],
                     axes,
                     [newpop])
    lik = my_einsum(lik, axes,
                    1.0/combinatorial_factors(G, newpop), [newpop],
                    axes)
    
    sfs = 0.0
    for freq, other_lik in zip(child_sfs, child_liks[::-1]):
        sfs = sfs + freq * np.squeeze(other_lik[[slice(None)] + [0] * (other_lik.ndim-1)])
    return lik, sfs, branch_len
