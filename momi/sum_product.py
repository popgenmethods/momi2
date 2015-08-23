from __future__ import division
import warnings
import autograd.numpy as np
import scipy
from util import memoize_instance, memoize, truncate0, make_constant, set0
from math_functions import einsum2, sum_antidiagonals, hypergeom_quasi_inverse, convolve_axes
from autograd.core import primitive
from autograd import hessian

## TODO: clean up this function!
## make a separate function that returns the normalization constant, handles min_freqs, etc.
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

    vecs = []
    for col,leaf in enumerate(sorted(demography.leaves)):
        n_lins = demography.n_lineages(leaf)

        # an array of the likelihoods at the leaf population
        cur_lik = np.zeros( (data.shape[0] + 3, n_lins + 1) )
        cur_lik[0,:] = 1.0 # likelihood of all states
        cur_lik[1,:(min_freqs[col])] = 1.0 # likelihood of derived not attaining minimum frequency
        cur_lik[2,(max_freqs[col]+1):] = 1.0 # likelihood of ancestral not attaining minimum frequency
        cur_lik[zip(*enumerate(data[:,col], start=3))] = 1.0 # likelihoods for config_list

        vecs += [cur_lik]

    if error_matrices is not None:
        vecs = _apply_error_matrices(vecs, error_matrices)
        
    sfs = expected_sfs_tensor_prod(vecs, demography)

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
    
    assert normalizing_constant == expected_total_branch_len(demography, error_matrices, min_freqs)
    
    return np.squeeze(sfs), normalizing_constant


def expected_total_branch_len(demography, error_matrices=None, min_freqs=1):
    """
    The expected sum of SFS entries for all configs (as given by
    compute_sfs). Equivalently, the expected number of observed mutations
    when theta=1, or the expected total branch length of the sample
    genealogy (in ms-scaled units).

    Parameters
    ----------
    demography : Demography
         object constructed by make_demography

    Returns
    -------
    total : float-like
         the total expected number of SNPs/branch length

    Other Parameters
    ----------------
    error_matrices : optional, sequence of 2-dimensional numpy.ndarray
         length-D sequence, where D = number of demes in demography.
         error_matrices[i] describes the sampling error in deme i as:

         error_matrices[i][j,k] = P(observe j mutants in deme i | k mutants in deme i)

         If error_matrices is not None, then the returned value is adjusted
         to account for this sampling error, in particular the effect it
         has on the total number of observed mutations.
    min_freqs : optional, int or sequence of ints
         It is sometimes desirable to only consider SNPs with minor allele
         reaching a certain minimum frequency.
         min_freqs corrects the returned value to only consider configs with
         an allele reaching the minimum frequency within at least one deme.
         min_freqs should either be a positive integer giving this minimum
         frequency, or a sequence giving deme-specific minimum frequencies
         for each deme.

    See Also
    --------
    expected_sfs_tensor_prod : more general class of summary statistics
         (of which this function is a special case)
    compute_sfs : individual SFS entries
    """
    vecs = [np.ones(demography.n_lineages(l)+1) for l in sorted(demography.leaves)]

    if error_matrices is not None:
        vecs = _apply_error_matrices(vecs, error_matrices)

    total = expected_sfs_tensor_prod(vecs, demography)
    ## return in the simple case, without errors or minimum frequencies
    if np.all(min_freqs == 1) and error_matrices is None:
        return total
    
    ## for more complicated case, need to subtract off branch_len not to be considered
    ## either because of min_freqs, or because error_matrices has made some actually polymorphic sites appear to be monomorphic
    n_leaf_lins = np.array([demography.n_lineages(l) for l in sorted(demography.leaves)])
    min_freqs = np.array(min_freqs) * np.ones(len(demography.leaves), dtype='i')
    max_freqs = n_leaf_lins - min_freqs
    if np.any(min_freqs < 1) or np.any(min_freqs > n_leaf_lins):
        raise Exception("Minimum frequencies must be in (0,num_lins] for each leaf pop")

    vecs = [np.array([[1.0] * deme_min + [0.0] * (deme_n+1 - deme_min),
                      [0.0] * (deme_max+1) + [1.0] * (deme_n - deme_max)])
            for deme_n, deme_min, deme_max in [(n_leaf_lins[i], min_freqs[i], max_freqs[i])
                                               for i in range(len(demography.leaves))]]

    if error_matrices is not None:
        vecs = _apply_error_matrices(vecs, error_matrices)

    return total - np.sum(expected_sfs_tensor_prod(vecs, demography))

        
def expected_sfs_tensor_prod(vecs, demography):
    """
    Viewing the SFS as a D-tensor (where D is the number of demes), this
    returns a 1d array whose j-th entry is a certain summary statistic of the
    expected SFS, given by the following tensor-vector multiplication:

    res[j] = \sum_{(i0,i1,...)} E[sfs[(i0,i1,...)]] * vecs[0][j,i0] * vecs[1][j, i1] * ...

    where E[sfs[(i0,i1,...)]] is the expected SFS entry for config
    (i0,i1,...), as given by compute_sfs

    Parameters
    ----------
    vecs : sequence of 2-dimensional numpy.ndarray
         length-D sequence, where D = number of demes in the demography.
         vecs[k] is 2-dimensional array, with constant number of rows, and
         with n[k]+1 columns, where n[k] is the number of samples in the
         k-th deme. The row vector vecs[k][j,:] is multiplied against
         the expected SFS along the k-th mode, to obtain res[j].
    demo : Demography
         object constructed by make_demography
    
    Returns
    -------
    res : numpy.ndarray (1-dimensional)
        res[j] is the tensor multiplication of the sfs against the vectors
        vecs[0][j,:], vecs[1][j,:], ... along its tensor modes.

    See Also
    --------
    sfs_tensor_prod : compute the same summary statistics for a specific SFS
    compute_sfs : compute individual SFS entries
    expected_total_branch_len : a summary statistic that uses this function
    """
    leaf_states = dict(zip(sorted(demography.leaves), vecs))
    
    for leaf in leaf_states.keys():
        n = demography.n_lineages(leaf)
        # add states for all ancestral/derived
        leaf_states[leaf] = np.vstack([np.array([1.0] + [0.0]*n), # all ancestral state
                                       np.array([0.0]*n + [1.0]), # all derived state
                                       leaf_states[leaf]])

    non_neg = all([np.all(l >= 0.0) for l in leaf_states.values()])
    is_prob = non_neg and all([np.all(l <= 1.0) for l in leaf_states.values()])
    
    _,res = _partial_likelihood(leaf_states,
                                demography, demography.event_root,
                                non_neg, is_prob)

    # subtract out mass for all ancestral/derived state
    for k in (0,1):
        res = res - res[k] * np.prod([l[:,-k] for l in leaf_states.values()], axis=0)
        assert np.isclose(res[k], 0.0)
    # remove monomorphic states
    res = res[2:]
    return res

def _apply_error_matrices(vecs, error_matrices):
    if not all([np.allclose(np.sum(err, axis=0), 1.0) for err in error_matrices]):
        raise Exception("Columns of error matrix should sum to 1")
    
    return [np.dot(v, err) for v,err in zip(vecs, error_matrices)]

def _partial_likelihood(leaf_states, G, event, non_neg, is_prob):
    ''' 
    Partial likelihood of data at event,
    P(x | n_derived_node, n_ancestral_node)
    with all subpopulation nodes at their initial time.
    '''
    lik_fun = _event_lik_fun(G, event)
    lik,sfs = lik_fun(leaf_states, G, event, non_neg, is_prob)

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

    _check_liks(lik, sfs, non_neg, is_prob)
    return lik,sfs

def _partial_likelihood_top(leaf_states, G, event, popList, non_neg, is_prob):
    ''' 
    Partial likelihood of data at top of nodes in popList,
    P(x | n_derived_top, n_ancestral_top)
    '''       
    lik,sfs = _partial_likelihood(leaf_states, G, event, non_neg, is_prob)
    for pop in popList:
        idx = (_lik_axes(G, event)).index(pop)
        lik = G.apply_transition(pop, lik, idx)

    _check_liks(lik, sfs, non_neg, is_prob)
    return lik,sfs

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

def _leaf_likelihood(leaf_states, G, event, non_neg, is_prob):
    leaf, = G.parent_pops(event)
    return leaf_states[leaf],0.

def _admixture_likelihood(leaf_states, G, event, non_neg, is_prob):
    child_pop, = G.child_pops(event).keys()
    p1,p2 = G.parent_pops(event)

    child_event, = G.event_tree[event]
    lik,sfs = _partial_likelihood_top(leaf_states, G, child_event, [child_pop], non_neg, is_prob)

    admixture_prob, admixture_idxs = G.admixture_prob(child_pop)
    lik = einsum2(lik, _lik_axes(G, child_event),
                  admixture_prob, admixture_idxs,
                  _lik_axes(G, event))

    return lik,sfs

def _merge_subpops_likelihood(leaf_states, G, event, non_neg, is_prob):
    newpop, = G.parent_pops(event)
    child_pops = G[newpop]
    child_event, = G.event_tree[event]

    lik,sfs = _partial_likelihood_top(leaf_states, G, child_event, child_pops,
                                      non_neg, is_prob)

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
    if non_neg:
        # make sure likelihoods are non negative
        lik = truncate0(lik, axis=newidx)
    assert lik.shape[newidx] == G.n_lineages(newpop)+1

    return lik,sfs

def _merge_clusters_likelihood(leaf_states, G, event, non_neg, is_prob):
    newpop, = G.parent_pops(event)
    child_liks = []
    for child_pop, child_event in G.child_pops(event).iteritems():
        axes = _lik_axes(G, child_event)        
        lik,sfs = _partial_likelihood_top(leaf_states, G, child_event, [child_pop], non_neg, is_prob)
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


def _check_liks(lik, sfs, non_neg, is_prob):
    if non_neg:
        assert np.all(lik >= 0.0) and np.all(sfs >= 0.0)

    if is_prob:
        assert np.all(np.logical_or(lik <= 1.0, np.isclose(lik, 1.0)))
