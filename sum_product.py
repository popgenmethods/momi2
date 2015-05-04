from __future__ import division
import autograd.numpy as np
import scipy
from util import memoize_instance, memoize, truncate0, make_constant
from math_functions import einsum2, fft_einsum, sum_antidiagonals, hypergeom_quasi_inverse
from autograd.core import primitive
from autograd import hessian

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

    n = np.sum([demography.n_lineages(l) for l in demography.leaves])
    data_rowsums = np.sum(data, axis=1)
    if np.any(data_rowsums == n) or np.any(data_rowsums == 0):
        raise IOError("Monomorphic sites in config_list.")

    _,ret,branch_len = _partial_likelihood(data, demography, demography.event_root)

    # first two indices of ret correspond to the monomorphic states
    branch_len = branch_len - ret[1]
    ret = ret[2:]

    assert branch_len >= 0.0 and np.all(ret >= 0.0) and np.all(ret <= branch_len)
    return np.squeeze(ret), branch_len

def _partial_likelihood(data, G, event):
    ''' 
    Partial likelihood of data at event,
    P(x | n_derived_node, n_ancestral_node)
    with all subpopulation nodes at their initial time.
    '''
    lik_fun = _event_lik_fun(G, event)
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

def _partial_likelihood_top(data, G, event, popList):
    ''' 
    Partial likelihood of data at top of nodes in popList,
    P(x | n_derived_top, n_ancestral_top)
    '''       
    lik,sfs,branch_len = _partial_likelihood(data, G, event)
    for pop in popList:
        idx = (_lik_axes(G, event)).index(pop)
        lik = G.apply_transition(pop, lik, idx)

    _check_positive(lik,sfs,branch_len)

    return lik,sfs,branch_len

def _check_positive(lik,sfs,branch_len):
    assert np.all(lik >= 0.0) and np.all(sfs >= 0.0) and np.all(branch_len >= 0.0)    

def combinatorial_factors(n):
    return scipy.misc.comb(n, np.arange(n + 1))

def _lik_axes(G, event):
    '''
    Returns axes labels of the partial likelihood tensor
    first axis corresponds to SFS entry
    subsequent axes correspond to subpopulations
    '''
    sub_pops = list(G.sub_pops(event))
    assert '' not in sub_pops
    return [''] + sub_pops

def _event_lik_fun(G, event):
    e_type = G.event_type(event)
    if e_type == 'leaf':
        return _leaf_likelihood
    elif e_type == 'admixture':
        return _admixture_likelihood
    elif e_type == 'merge_subpops':
        return _merge_subpops_likelihood
    elif e_type == 'merge_clusters':
        return _merge_clusters_likelihood
    else:
        raise Exception("Unrecognized event type.")

def _leaf_likelihood(data, G, event):
    leaf, = G.parent_pops(event)
    n_node = G.n_lineages(leaf)

    n_der = data[:,sorted(G.leaves).index(leaf)]

    # the matrix of likelihoods (rows=datapoints, columns=# derived alleles)
    lik = np.zeros((len(n_der)+2, n_node + 1))

    # add two additional datapoints for monomorphic ancestral and monomorphic derived, respectively
    lik[0,0] = 1.0
    lik[1,n_node] = 1.0

    # fill likelihoods with delta masses at the data points
    lik[zip(*enumerate(n_der,start=2))] = 1.0

    return lik,0.,0.

def _admixture_likelihood(data, G, event):
    child_pop, = G.child_pops(event).keys()
    p1,p2 = G.parent_pops(event)

    child_event, = G.event_tree[event]
    lik,sfs,branch_len = _partial_likelihood_top(data, G, child_event, [child_pop])

    admixture_prob, admixture_idxs = G.admixture_prob(child_pop)
    lik = einsum2(lik, _lik_axes(G, child_event),
                  admixture_prob, admixture_idxs,
                  _lik_axes(G, event))

    return lik,sfs,branch_len

def _merge_subpops_likelihood(data, G, event):
    newpop, = G.parent_pops(event)
    child_pops = G[newpop]
    child_event, = G.event_tree[event]

    lik,sfs,branch_len = _partial_likelihood_top(data, G, child_event, child_pops)

    c1,c2 = child_pops
    child_axes = _lik_axes(G, child_event)
    for c in c1,c2:
        lik = einsum2(lik, child_axes,
                      combinatorial_factors(G.n_lineages(c)), [c],
                      child_axes)
    lik,axes = sum_antidiagonals(lik, child_axes, c1, c2, newpop)

    event_axes = _lik_axes(G,event)
    assert set(axes) == set(event_axes)
    newidx = axes.index(newpop)
    lik = einsum2(lik, axes,
                  1.0/combinatorial_factors(lik.shape[newidx]-1), [newpop],
                  event_axes)

    # reduce the number of lineages in newpop to only the number necessary
    axes = event_axes
    newidx = axes.index(newpop)
    lik = einsum2(lik, event_axes[:newidx] + [c1] + axes[(newidx+1):],
                  hypergeom_quasi_inverse(lik.shape[newidx]-1, G.n_lineages(newpop)),
                  [c1,newpop], axes)
    truncate0(lik, axis=newidx)
    assert lik.shape[newidx] == G.n_lineages(newpop)+1

    return lik,sfs,branch_len

def _merge_clusters_likelihood(data, G, event):
    newpop, = G.parent_pops(event)
    child_liks = []
    child_axes = []
    
    branch_len = 0.0
    for child_pop, child_event in G.child_pops(event).iteritems():
        lik,sfs,child_len = _partial_likelihood_top(data, G, child_event, [child_pop])
        branch_len = branch_len + child_len

        axes = [newpop if x == child_pop else x for x in _lik_axes(G, child_event)]
        child_axes.append(axes)
        lik = einsum2(lik, axes,
                      combinatorial_factors(G.n_lineages(child_pop)), [newpop],
                      axes)

        child_liks.append((lik,sfs))

    child_liks,child_sfs = zip(*child_liks)
    axes = _lik_axes(G, event)

    lik = fft_einsum(child_liks[0], child_axes[0],
                     child_liks[1], child_axes[1],
                     axes,
                     [newpop])
    # deal with very small negative numbers from fft
    newidx = axes.index(newpop)
    lik = truncate0(lik, axis=newidx)

    lik = einsum2(lik, axes,
                  1.0/combinatorial_factors(G.n_lineages(newpop)), [newpop],
                  axes)

    sfs = 0.0
    for freq, other_lik in zip(child_sfs, child_liks[::-1]):
        sfs = sfs + freq * np.squeeze(other_lik[[slice(None)] + [0] * (other_lik.ndim-1)])
    return lik, sfs, branch_len
