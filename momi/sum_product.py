from __future__ import division
import warnings
import autograd.numpy as np
import scipy
from util import memoize_instance, memoize, truncate0, make_constant, set0
from math_functions import einsum2, sum_antidiagonals, hypergeom_quasi_inverse, convolve_axes
from autograd.core import primitive
from autograd import hessian

def compute_sfs(demography, config_list, error_matrices=None, min_freqs=1):
    '''
    Returns (sfs,normalizing_constant), where:
    sfs = array with sfs entry for each config given
    normalizing_constant = sum of all possible sfs entries
    i.e.,
    sfs = expected count of SNP
    sfs / normalizing_constant = conditional probability of SNP

    Inputs are
    demography: object returned by momi.make_demography()
    config_list: list of D-tuples of derived counts, 
                 where D=number of leaf pops in demography
    error_matrices: list whose length is number of leaf populations
                    i-th entry is a matrix, with
                    error_matrices[i][j,k] = P(observe j derived in pop i | k actual derived in pop i)
                    if None, assume no errors in measure/observation
    min_freqs: number or array-like whose length is the number of leaf populations
               only SFS entries where the minor allele attains a minimum frequency in at least one subpopulation
               are considered; all other SFS entries are assigned 0.
               This is also reflected in the normalization constant
    '''
    data = np.array(config_list, ndmin=2)
    if data.ndim != 2 or data.shape[1] != len(demography.leaves):
        raise IOError("Invalid config_list.")

    n_leaf_lins = np.array([demography.n_lineages(l) for l in demography.leaves])
    #n = np.sum()
    #data_rowsums = np.sum(data, axis=1)
#     if np.any(data_rowsums == n) or np.any(data_rowsums == 0):
#         raise IOError("Monomorphic sites in config_list.")

    min_freqs = np.array(min_freqs) * np.ones(len(demography.leaves), dtype='i')
    if np.any(min_freqs < 1) or np.any(min_freqs > n_leaf_lins):
        raise Exception("Minimum frequencies must be in (0,num_lins] for each leaf pop")
    max_freqs = n_leaf_lins - min_freqs

    leaf_liks = {}
    for col,leaf in enumerate(sorted(demography.leaves)):
        n_lins = demography.n_lineages(leaf)

        # an array of the likelihoods at the leaf population
        cur_lik = np.zeros( (data.shape[0] + 3, n_lins + 1) )
        cur_lik[0,:] = 1.0 # likelihood of all states
        cur_lik[1,:(min_freqs[col])] = 1.0 # likelihood of derived not attaining minimum frequency
        cur_lik[2,(max_freqs[col]+1):] = 1.0 # likelihood of ancestral not attaining minimum frequency
        cur_lik[zip(*enumerate(data[:,col], start=3))] = 1.0 # likelihoods for config_list

        if error_matrices is not None:
            err = error_matrices[col]
            if not np.allclose(np.sum(err, axis=0) , 1.0):
                raise Exception("Columns of error matrix should sum to 1")
            cur_lik = einsum2(cur_lik, ['entry','n_der_observed'],
                              err, ['n_der_observed','n_der_actual'],
                              ['entry','n_der_actual'])

        leaf_liks[leaf] = cur_lik

    sfs = raw_compute_sfs(leaf_liks, demography)

    # extract the normalizing constant
    normalizing_constant = sfs[0] - sfs[1] - sfs[2]
    sfs = sfs[3:]

    # set entries not attaining minimum frequency to 0
    attain_min_freq = np.logical_and(np.sum(data >= min_freqs, axis=1) > 0, # at least one entry above min_freqs
                                     np.sum(data <= max_freqs, axis=1) > 0 # at least one entry below max_freqs
                                     )
    if not np.all(attain_min_freq):
        warnings.warn("Entries that do not attain minimum minor allele frequency are set to 0.")
        sfs = set0(sfs, np.logical_not(attain_min_freq))

    assert normalizing_constant >= 0.0 and np.all(sfs >= 0.0) and np.all(sfs <= normalizing_constant)
    return np.squeeze(sfs), normalizing_constant

def raw_compute_sfs(leaf_liks, demography):
    '''
    Similar to compute_sfs (and in fact is called by it),
    but can be used to compute SFS under more complicated error/ascertainment models

    leaf_liks = dictionary whose keys are the leaf populations of demography
    leaf_liks[i] = matrix with dimensions (S , n_at_leaf_i)
    leaf_liks[i][s,d] = likelihood of d derived alleles in leaf population i for the s-th configuration

    The s-th configuration can essentially be viewed as a rank-1 tensor, given by the outer product
    leaf_liks[0][s,:] * leaf_liks[1][s,:] * ...
    '''
    _,sfs = _partial_likelihood(leaf_liks, demography, demography.event_root)
    return sfs

def _partial_likelihood(leaf_liks, G, event):
    ''' 
    Partial likelihood of data at event,
    P(x | n_derived_node, n_ancestral_node)
    with all subpopulation nodes at their initial time.
    '''
    lik_fun = _event_lik_fun(G, event)
    lik,sfs = lik_fun(leaf_liks, G, event)

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

    _check_positive(lik,sfs)

    return lik,sfs

def _partial_likelihood_top(leaf_liks, G, event, popList):
    ''' 
    Partial likelihood of data at top of nodes in popList,
    P(x | n_derived_top, n_ancestral_top)
    '''       
    lik,sfs = _partial_likelihood(leaf_liks, G, event)
    for pop in popList:
        idx = (_lik_axes(G, event)).index(pop)
        lik = G.apply_transition(pop, lik, idx)

    _check_positive(lik,sfs)

    return lik,sfs

def _check_positive(lik,sfs):
    assert np.all(lik >= 0.0) and np.all(sfs >= 0.0)

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

def _leaf_likelihood(leaf_liks, G, event):
    leaf, = G.parent_pops(event)
    return leaf_liks[leaf],0.

def _admixture_likelihood(leaf_liks, G, event):
    child_pop, = G.child_pops(event).keys()
    p1,p2 = G.parent_pops(event)

    child_event, = G.event_tree[event]
    lik,sfs = _partial_likelihood_top(leaf_liks, G, child_event, [child_pop])

    admixture_prob, admixture_idxs = G.admixture_prob(child_pop)
    lik = einsum2(lik, _lik_axes(G, child_event),
                  admixture_prob, admixture_idxs,
                  _lik_axes(G, event))

    return lik,sfs

def _merge_subpops_likelihood(leaf_liks, G, event):
    newpop, = G.parent_pops(event)
    child_pops = G[newpop]
    child_event, = G.event_tree[event]

    lik,sfs = _partial_likelihood_top(leaf_liks, G, child_event, child_pops)

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

    return lik,sfs

def _merge_clusters_likelihood(leaf_liks, G, event):
    newpop, = G.parent_pops(event)
    child_liks = []
    for child_pop, child_event in G.child_pops(event).iteritems():
        axes = _lik_axes(G, child_event)        
        lik,sfs = _partial_likelihood_top(leaf_liks, G, child_event, [child_pop])
        lik = einsum2(lik, axes,
                      combinatorial_factors(G.n_lineages(child_pop)), [child_pop],
                      axes)
        child_liks.append((child_pop,axes,lik,sfs))

    child_pops,child_axes,child_liks,child_sfs = zip(*child_liks)

    lik, old_axes = convolve_axes(child_liks[0], child_liks[1],
                                  child_axes, child_pops, newpop)
    
    axes = _lik_axes(G, event)    
    lik = einsum2(lik, old_axes,
                  1.0/combinatorial_factors(G.n_lineages(newpop)), [newpop],
                  axes)

    sfs = 0.0
    for freq, other_lik in zip(child_sfs, child_liks[::-1]):
        sfs = sfs + freq * np.squeeze(other_lik[[slice(None)] + [0] * (other_lik.ndim-1)])
    return lik, sfs
