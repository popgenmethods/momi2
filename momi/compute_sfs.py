
import warnings
import autograd.numpy as np
import scipy
from .util import memoize_instance, memoize, make_constant, set0, reversed_configs
from .math_functions import einsum2, sum_antidiagonals, hypergeom_quasi_inverse, convolve_axes, roll_axes, binom_coeffs
from autograd.core import primitive
from autograd import hessian

def expected_sfs(demography, config_list, normalized=False, error_matrices=None, folded=False):
    """
    Expected sample frequency spectrum (SFS) entries for the specified
    demography and configs. The expected SFS is the expected number of
    observed mutations for a configuration when mu=1, or equivalently,
    the expected branch length (in ms-scaled units) subtending a
    configuration.

    Parameters
    ----------
    demography : Demography
         object returned by make_demography
    config_list : list of tuples
         list of the configs to compute the SFS entries for.
         If there are D sampled populations, then each config is
         represented by a D-tuple (i_1,i_2,...,i_D), where i_j is the
         number of derived mutants in deme j.

         WARNING: in python, tuples are indexed starting at 0, whereas
         in ms, populations are indexed starting at 1. So config[j] =
         the number of derived mutants in the deme labeled j+1.
    normalized : optional, bool
         if True, divide the SFS by the expected total branch length.
         The returned values then represent probabilities, that a given
         mutation will segregate according to the specified configurations.
    folded : optional, bool
         if True, return the folded SFS entry for each config

    Returns
    -------
    sfs : 1d numpy.ndarray
         sfs[j] is the SFS entry corresponding to config_list[j]

    Other Parameters
    ----------------
    error_matrices : optional, sequence of 2-dimensional numpy.ndarray
         length-D sequence, where D = number of demes in demography.
         error_matrices[i] describes the sampling error in deme i as:

         error_matrices[i][j,k] = P(observe j mutants in deme i | k mutants in deme i)

         If error_matrices is not None, then the returned value is adjusted
         to account for this sampling error, in particular the effect it
         has on the total number of observed mutations.

    See Also
    --------
    expected_total_branch_len : sum of all expected SFS entries
    expected_sfs_tensor_prod : compute summary statistics of SFS
    """
    if folded:
        rev_configs,symm = reversed_configs(config_list, demography.n_at_leaves, return_is_symmetric=True)
        ret = expected_sfs(demography,
                           list(config_list) + list(rev_configs),
                           normalized=normalized, error_matrices=error_matrices,
                           folded=False)
        ret = ret[:len(config_list)] + ret[len(config_list):]
        ret = ret / (np.array(symm) + 1.0) # symmetric configs need to be divided by 2
        return ret
        
    data = np.array(config_list, ndmin=2)
    if data.ndim != 2 or data.shape[1] != len(demography.leaves):
        raise IOError("Invalid config_list.")

    # the likelihoods at the leaf populations
    leaf_liks = [np.zeros((data.shape[0], demography.n_lineages(leaf)+1))
                 for leaf in sorted(demography.leaves)]
    for i in range(len(leaf_liks)):
        leaf_liks[i][list(zip(*enumerate(data[:,i])))] = 1.0 # likelihoods for config_list
    
    if error_matrices is not None:
        leaf_liks = _apply_error_matrices(leaf_liks, error_matrices)
        
    sfs = expected_sfs_tensor_prod(leaf_liks, demography)
    assert np.all(np.logical_or(sfs >= 0.0, np.isclose(sfs, 0.0)))
    if normalized:
        sfs = sfs / expected_total_branch_len(demography, error_matrices=error_matrices)
        
    return sfs

def expected_total_branch_len(demography, error_matrices=None, min_freqs=1):
    """
    The expected sum of SFS entries for all configs (as given by
    expected_sfs). Equivalently, the expected number of observed mutations
    when mu=1, or the expected total branch length of the sample
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
    expected_sfs : individual SFS entries
    expected_tmrca, expected_deme_tmrca : other interesting statistics
    expected_sfs_tensor_prod : compute general class of summary statistics
    """
    vecs = [np.ones(demography.n_lineages(l)+1) for l in sorted(demography.leaves)]

    if error_matrices is not None:
        vecs = _apply_error_matrices(vecs, error_matrices)

    total = np.squeeze(expected_sfs_tensor_prod(vecs, demography))
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

def expected_tmrca(demography):
    """
    The expected time to most recent common ancestor of the sample,
    in ms-scaled units.

    Parameters
    ----------
    demography : Demography
         object returned by make_demography

    Returns
    -------
    tmrca : float-like

    See Also
    --------
    expected_deme_tmrca : tmrca of subsample within a deme
    expected_sfs_tensor_prod : compute general class of summary statistics
    """
    vecs = [np.ones(demography.n_lineages(l)+1) for l in sorted(demography.leaves)]
    n0 = len(vecs[0])-1.0
    vecs[0] = np.arange(n0+1) / n0
    return np.squeeze(expected_sfs_tensor_prod(vecs, demography))

def expected_deme_tmrca(demography, deme):
    """
    The expected time to most recent common ancestor, of the samples within
    a particular deme. Returned time is in ms-scaled units.

    Parameters
    ----------
    demography : Demography
         object returned by make_demography
    deme : int
         the deme

    Returns
    -------
    tmrca : float

    See Also
    --------
    expected_tmrca : the tmrca of the whole sample
    expected_sfs_tensor_prod : compute general class of summary statistics    
    """
    vecs = [np.ones(demography.n_lineages(l)+1) for l in sorted(demography.leaves)]

    n = len(vecs[deme])-1
    vecs[deme] = np.arange(n+1) / n
    vecs[deme][-1] = 0.0
    
    return np.squeeze(expected_sfs_tensor_prod(vecs, demography))

def expected_sfs_tensor_prod(vecs, demography):
    """
    Viewing the SFS as a D-tensor (where D is the number of demes), this
    returns a 1d array whose j-th entry is a certain summary statistic of the
    expected SFS, given by the following tensor-vector multiplication:

    res[j] = \sum_{(i0,i1,...)} E[sfs[(i0,i1,...)]] * vecs[0][j,i0] * vecs[1][j, i1] * ...

    where E[sfs[(i0,i1,...)]] is the expected SFS entry for config
    (i0,i1,...), as given by expected_sfs

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
    sfs_tensor_prod : compute the same summary statistics for an observed SFS
    expected_sfs : compute individual SFS entries
    expected_total_branch_len, expected_tmrca, expected_deme_tmrca : 
         examples of coalescent statistics that use this function
    """
    leaf_states = dict(list(zip(sorted(demography.leaves), vecs)))
    
    for leaf in list(leaf_states.keys()):
        n = demography.n_lineages(leaf)
        # add states for all ancestral/derived
        leaf_states[leaf] = np.vstack([np.array([1.0] + [0.0]*n), # all ancestral state
                                       np.array([0.0]*n + [1.0]), # all derived state
                                       leaf_states[leaf]])

    _,res = _partial_likelihood(leaf_states,
                                demography, demography.event_root)

    # subtract out mass for all ancestral/derived state
    for k in (0,1):
        res = res - res[k] * np.prod([l[:,-k] for l in list(leaf_states.values())], axis=0)
        assert np.isclose(res[k], 0.0)
    # remove monomorphic states
    res = res[2:]
    return res

def _partial_likelihood(leaf_states, demo, event):
    ''' 
    Partial likelihood of data at event,
    P(x | n_derived_node, n_ancestral_node)
    with all subpopulation nodes at their initial time.
    '''
    lik_fun = _event_lik_fun(demo, event)
    lik,sfs = lik_fun(leaf_states, demo, event)

    # add on sfs entry at this event
    axes = _lik_axes(demo, event)
    for newpop in demo.parent_pops(event):
        # term for mutation occurring at the newpop
        newpop_idx = axes.index(newpop)
        idx = [0] * lik.ndim
        idx[0], idx[newpop_idx] = slice(None), slice(None)

        sub_lik, trunc_sfs = lik[idx], demo.truncated_sfs(newpop)
        sfs = sfs + einsum2(sub_lik, ['',newpop],
                            trunc_sfs, [newpop],
                            [''])

    return lik,sfs

def _partial_likelihood_top(leaf_states, demo, event, popList):
    ''' 
    Partial likelihood of data at top of nodes in popList,
    P(x | n_derived_top, n_ancestral_top)
    '''       
    lik,sfs = _partial_likelihood(leaf_states, demo, event)
    for pop in popList:
        idx = (_lik_axes(demo, event)).index(pop)
        lik = demo.apply_transition(pop, lik, idx)

    return lik,sfs

def _lik_axes(demo, event):
    '''
    Returns axes labels of the partial likelihood tensor
    first axis corresponds to SFS entry
    subsequent axes correspond to subpopulations
    '''
    sub_pops = list(demo.sub_pops(event))
    assert '' not in sub_pops
    return [''] + sub_pops

def _event_lik_fun(demo, event):
    e_type = demo.event_type(event)
    if e_type == 'leaf':
        return _leaf_likelihood
    # elif e_type == 'admixture':
    #     return _admixture_likelihood
    elif e_type == 'merge_subpops':
        return _merge_subpops_likelihood
    elif e_type == 'merge_clusters':
        return _merge_clusters_likelihood
    elif e_type == 'pulse':
        return _pulse_likelihood
    else:
        raise Exception("Unrecognized event type.")

def _leaf_likelihood(leaf_states, demo, event):
    (leaf,_), = demo.parent_pops(event)
    if leaf in leaf_states:
        return leaf_states[leaf],0.
    else:
        return np.ones((iter(leaf_states.values()).next().shape[0],1)),0.

# def _admixture_likelihood(leaf_states, demo, event):
#     child_pop, = list(demo.child_pops(event).keys())
#     p1,p2 = demo.parent_pops(event)

#     child_event, = demo.event_tree[event]
#     lik,sfs = _partial_likelihood_top(leaf_states, demo, child_event, [child_pop])

#     admixture_prob, admixture_idxs = demo.admixture_prob(child_pop)
#     lik = einsum2(lik, _lik_axes(demo, child_event),
#                   admixture_prob, admixture_idxs,
#                   _lik_axes(demo, event))

#     return lik,sfs

def _pulse_likelihood(leaf_states, demo, event):
    parent_pops = demo.parent_pops(event)    
    child_pops_events = demo.child_pops(event)
    assert len(child_pops_events) == 2
    child_pops, child_events = zip(*child_pops_events.items())
    
    if len(set(child_events)) == 2:
        ## in this case, it is more efficient to model the pulse as a split (-es) followed by a join (-ej)
        sfs, child_pops, child_axes, child_liks = _disjoint_children_liks(leaf_states, demo, event)
        
        recipient, non_recipient, donor, non_donor = demo.pulse_nodes(event)
        admixture_prob, admixture_idxs = demo.admixture_prob(recipient)

        child_liks = dict(zip(child_pops, child_liks))
        child_axes = dict(zip(child_pops, child_axes))

        tmp_axes = child_axes[recipient]
        child_axes[recipient] = [x for x in tmp_axes if x != recipient] + list(parent_pops)
        child_liks[recipient] = einsum2(child_liks[recipient], tmp_axes,
                                        admixture_prob, admixture_idxs,
                                        child_axes[recipient])

        tmp_axes = child_axes[recipient]
        child_axes[recipient] = [x if x != donor else recipient for x in tmp_axes]

        child_liks = [child_liks[c] for c in child_pops]
        child_axes = [child_axes[c] for c in child_pops]
        
        lik = _convolve_children_liks(child_pops, child_liks, child_axes, donor,
                                      _lik_axes(demo, event))
        return lik,sfs
    else:
        ## in this case, (typically) more memory-efficient to multiply likelihood by transition 4-tensor
        ## (if only 2 populations, and much fewer SFS entries than samples, it may be more efficient to replace -ep with -es,-ej)
        child_event, = set(child_events)
        lik, sfs = _partial_likelihood_top(leaf_states, demo, child_event, child_pops)
        axes = _lik_axes(demo, child_event)

        pulse_prob, pulse_idxs = demo.pulse_prob(event)

        lik = einsum2(lik, axes,
                      pulse_prob, pulse_idxs,
                      _lik_axes(demo, event))
        return lik,sfs

def _merge_subpops_likelihood(leaf_states, demo, event):
    newpop, = demo.parent_pops(event)
    child_pops = demo.G[newpop]
    child_event, = demo.event_tree[event]

    child_axes = _lik_axes(demo, child_event)
    event_axes = _lik_axes(demo,event)

    lik,sfs = _partial_likelihood_top(leaf_states, demo, child_event, child_pops)
    
    lik = _merge_lik_axes(lik, child_axes, event_axes, child_pops, newpop,
                          {pop: demo.n_lineages(pop)
                           for pop in list(child_pops) + [newpop]})
    
    return lik,sfs

def _merge_lik_axes(lik, child_axes, new_axes, child_pops, newpop, n_lins):
    assert len(child_pops) == 2 and len(child_axes) == len(new_axes)+1
    
    c1,c2 = child_pops
    for c in c1,c2:
        lik = einsum2(lik, child_axes,
                      binom_coeffs(n_lins[c]), [c],
                      child_axes)
    lik,axes = sum_antidiagonals(lik, child_axes, c1, c2, newpop)

    assert set(axes) == set(new_axes)
    newidx = axes.index(newpop)
    lik = einsum2(lik, axes,
                  1.0/binom_coeffs(lik.shape[newidx]-1), [newpop],
                  new_axes)

    # reduce the number of lineages in newpop to only the number necessary
    axes = new_axes
    newidx = axes.index(newpop)
    N,n = lik.shape[newidx]-1, n_lins[newpop]
    assert N >= n
    if N > n:
        lik = einsum2(lik, new_axes[:newidx] + [c1] + axes[(newidx+1):],
                      hypergeom_quasi_inverse(N,n),
                      [c1,newpop], axes)
    assert lik.shape[newidx] == n+1

    return lik

def _merge_clusters_likelihood(leaf_states, demo, event):
    sfs, child_pops, child_axes, child_liks = _disjoint_children_liks(leaf_states, demo, event)
    axes = _lik_axes(demo, event)
    newpop, = demo.parent_pops(event)    
    lik = _convolve_children_liks(child_pops, child_liks, child_axes, newpop, axes)
    return lik, sfs

def _convolve_children_liks(child_pops, child_liks, child_axes, newpop, axes):
    child_liks = list(child_liks)
    for i,(lik,child_axis,child_pop) in enumerate(zip(child_liks, child_axes, child_pops)):
        idx = child_axis.index(child_pop)
        lik = einsum2(lik, child_axis,
                      binom_coeffs(lik.shape[idx]-1), [child_pop],
                      child_axis)
        child_liks[i] = lik

    lik, old_axes = convolve_axes(child_liks[0], child_liks[1],
                                  child_axes, child_pops, newpop)

    idx = old_axes.index(newpop)
    lik = einsum2(lik, old_axes,
                  1.0/binom_coeffs(lik.shape[idx]-1), [newpop],
                  axes)
    return lik   

def _disjoint_children_liks(leaf_states, demo, event):
    child_liks = []
    for child_pop, child_event in demo.child_pops(event).items():
        axes = _lik_axes(demo, child_event)        
        lik,sfs = _partial_likelihood_top(leaf_states, demo, child_event, [child_pop])
        child_liks.append((child_pop,axes,lik,sfs))
        
    child_pops,child_axes,child_liks,child_sfs = list(zip(*child_liks))

    sfs = 0.0
    assert len(child_liks) == 2    
    for freq, other_lik in zip(child_sfs, child_liks[::-1]):
        sfs = sfs + freq * np.squeeze(other_lik[[slice(None)] + [0] * (other_lik.ndim-1)])

    return (sfs, child_pops, child_axes, child_liks)
    
def _apply_error_matrices(vecs, error_matrices):
    if not all([np.allclose(np.sum(err, axis=0), 1.0) for err in error_matrices]):
        raise Exception("Columns of error matrix should sum to 1")
    
    return [np.dot(v, err) for v,err in zip(vecs, error_matrices)]
