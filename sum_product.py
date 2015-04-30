import autograd.numpy as np
import scipy.misc
from util import memoize_instance, memoize, truncate0
from math_functions import einsum2, fft_einsum, sum_antidiagonals

def log_likelihood_prf(demo, theta, sfs_counts, EPSILON=0.0):
    '''
    Return log likelihood under Poisson random field model.

    demo: object returned by demography.make_demography
    theta: 2*mutation_rate
    sfs_counts: dictionary {config : counts}
    EPSILON: EPSILON/theta added onto SFS, to prevent taking log of 0
             default is 0. Try setting to a small positive number, 
             e.g. 1e-6, if optimizer is failing due to log(0).
    '''
    config_list,counts = zip(*sorted(sfs_counts.iteritems()))
    counts = np.array(counts)

    sfs_vals, branch_len = compute_sfs(demo, config_list)
    sfs_vals = sfs_vals + EPSILON / theta
    ret = -branch_len * theta / 2.0 + np.sum(np.log(sfs_vals * theta / 2.0) * counts - scipy.special.gammaln(counts+1))

    assert ret < 0.0
    return ret

def compute_sfs(demography, config_list):
    '''
    Return joint SFS entry for the demography.

    demography: object returned by demography.make_demography
    config_list: list of k-tuples of derived counts, 
                 where k=number of leaf pops in demography
    '''
    data = np.array(config_list, ndmin=2)
    if data.ndim != 2 or data.shape[1] != len(demography.leaves):
        raise IOError("Invalid config_list.")

    _,ret,branch_len = partial_likelihood(data, demography, demography.event_root)
    branch_len = branch_len - ret[1]

    # first two indices correspond to the monomorphic states
    ret = ret[2:]
    assert branch_len >= 0.0 and np.all(ret >= 0.0) and np.all(ret <= branch_len)
    return np.squeeze(ret), branch_len


def partial_likelihood_top(data, G, event, popList):
    ''' Partial likelihood of data at top of node, i.e.
    i.e. = P(n_top) P(x | n_derived_top, n_ancestral_top)
    note n_top is fixed in Moran model, so P(n_top)=1
    '''       
    lik,sfs,branch_len = partial_likelihood(data, G, event)
    for pop in popList:
        idx = (_lik_axes(G, event)).index(pop)
        lik = G.apply_transition(pop, lik, idx)

    _check_positive(lik,sfs,branch_len)

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

        sub_lik, trunc_sfs = lik[idx], G.truncated_sfs(newpop)
        sfs = sfs + einsum2(sub_lik, ['',newpop],
                            trunc_sfs, [newpop],
                            [''])
        branch_len = branch_len + np.dot(1.0 - sub_lik[0,:] , trunc_sfs)

    _check_positive(lik,sfs,branch_len)

    return lik,sfs,branch_len

def _check_positive(lik,sfs,branch_len):
    assert np.all(lik >= 0.0) and np.all(sfs >= 0.0) and np.all(branch_len >= 0.0)    

def combinatorial_factors(G, node):
    n_node = G.n_lineages(node)
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
    n_node = G.n_lineages(leaf)

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

    child_event, = G.event_tree[event]
    lik,sfs,branch_len = partial_likelihood_top(data, G, child_event, [child_pop])

    admixture_prob, admixture_idxs = G.admixture_prob(child_pop)
    lik = einsum2(lik, _lik_axes(G, child_event),
                  admixture_prob, admixture_idxs,
                  _lik_axes(G, event))

    return lik,sfs,branch_len

def _merge_subpops_likelihood(data, G, event):
    newpop, = G.parent_pops(event)
    child_pops = G[newpop]
    child_event, = G.event_tree[event]

    lik,sfs,branch_len = partial_likelihood_top(data, G, child_event, child_pops)

    c1,c2 = child_pops
    child_axes = _lik_axes(G, child_event)
    for c in c1,c2:
        lik = einsum2(lik, child_axes,
                      combinatorial_factors(G, c), [c],
                      child_axes)
    lik,axes = sum_antidiagonals(lik, child_axes, c1, c2, newpop)

    assert set(axes) == set(_lik_axes(G,event))
    lik = einsum2(lik, axes,
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
        lik = einsum2(lik, axes,
                      combinatorial_factors(G, child_pop), [newpop],
                      axes)

        child_liks.append((lik,sfs))

    child_liks,child_sfs = zip(*child_liks)
    axes = _lik_axes(G, event)

    lik = fft_einsum(child_liks[0], child_axes[0],
                     child_liks[1], child_axes[1],
                     axes,
                     [newpop])
    # deal with very small negative numbers from fft
    lik = truncate0(lik, axis=axes.index(newpop))

    lik = einsum2(lik, axes,
                  1.0/combinatorial_factors(G, newpop), [newpop],
                  axes)

    sfs = 0.0
    for freq, other_lik in zip(child_sfs, child_liks[::-1]):
        sfs = sfs + freq * np.squeeze(other_lik[[slice(None)] + [0] * (other_lik.ndim-1)])
    return lik, sfs, branch_len
