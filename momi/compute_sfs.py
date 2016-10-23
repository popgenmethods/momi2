
import warnings
import autograd.numpy as np
import scipy
from .util import memoize_instance, memoize, make_constant, set0
from .data_structure import config_array
from .math_functions import sum_antidiagonals, hypergeom_quasi_inverse, convolve_axes, roll_axes, binom_coeffs, _apply_error_matrices, par_einsum
#from .data_structure import Configs
from .moran_model import moran_action
from autograd.core import primitive
from autograd import hessian

def expected_sfs(demography, configs, mut_rate=1.0, normalized=False, folded=False, error_matrices=None):
    """
    Expected sample frequency spectrum (SFS) entries for the specified
    demography and configs. The expected SFS is the expected number of
    observed mutations for a configuration. If mutation rate=1, it is
    equivalent to the expected branch length subtended by a configuration.

    Parameters
    ----------
    demography : Demography
    configs : ConfigArray
        if configs.folded == True, returns the folded SFS entries
    mut_rate : float
         mutation rate per unit time
    normalized : optional, bool
         if True, mut_rate is ignored, and the SFS is divided by the 
         expected total branch length.
         The returned values then represent probabilities, that a given
         mutation will segregate according to the specified configurations.

    Returns
    -------
    sfs : 1d numpy.ndarray
         sfs[j] is the SFS entry corresponding to configs[j]


    Other Parameters
    ----------------
    folded: optional, bool
         if True, return the folded SFS value for each entry
         Default is False.
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
    sfs, denom = _expected_sfs(demography, configs, folded, error_matrices)
    if normalized:
        sfs = sfs / denom
    else:
        sfs = sfs*mut_rate
    return sfs

def _expected_sfs(demography, configs, folded, error_matrices):    
    if np.any(configs.sampled_n != demography.sampled_n) or np.any(configs.sampled_pops != demography.sampled_pops):
        raise ValueError("configs and demography must have same sampled_n, sampled_pops. Use Demography.copy() or ConfigArray.copy() to make a copy with different sampled_n.")

    vecs, idxs = configs._vecs_and_idxs(folded)
    
    if error_matrices is not None:
        vecs = _apply_error_matrices(vecs, error_matrices)
        
    vals = expected_sfs_tensor_prod(vecs, demography)

    sfs = vals[idxs['idx_2_row']]
    if folded:
        sfs = sfs + vals[idxs['folded_2_row']]
        
    denom = vals[idxs['denom_idx']]
    for i in (0,1):
        denom = denom - vals[idxs[("corrections_2_denom",i)]]
    
    assert np.all(np.logical_or(vals >= 0.0, np.isclose(vals, 0.0)))
    
    return sfs, denom

def expected_total_branch_len(demography, error_matrices=None, ascertainment_pop=None, p_missing=0.0):
    """
    The expected total branch length of the sample genealogy.
    Equivalently, the expected number of observed mutations when 
    mutation rate=1.

    Parameters
    ----------
    demography : Demography

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

    See Also
    --------
    expected_sfs : individual SFS entries
    expected_tmrca, expected_deme_tmrca : other interesting statistics
    expected_sfs_tensor_prod : compute general class of summary statistics
    """
    if ascertainment_pop is None:
        ascertainment_pop = [True]*len(demography.sampled_n)
    ascertainment_pop = np.array(ascertainment_pop)

    p_missing = p_missing * np.ones(len(demography.sampled_n))
    if np.any(np.logical_or(p_missing < 0.0, p_missing >= 1.0)):
        raise ValueError("p_missing (the probability that a sample is missing) must be in [0,1)")
    vecs = [[np.ones(n+1),
             p**np.arange(n+1),
             p**np.arange(n+1)[::-1]]
            if asc else np.ones((3, n+1), dtype=float)
            for p,asc,n in zip(p_missing, ascertainment_pop, demography.sampled_n)]
    if error_matrices is not None:
        vecs = _apply_error_matrices(vecs, error_matrices)

    ret = expected_sfs_tensor_prod(vecs, demography)
    ## inclusion-exclusion: subtract off fraction where all ancestral missing, allderived missing, add back on all missing
    return ret[0] - ret[1] - ret[2] + ret[0] * np.prod((p_missing**demography.sampled_n)[ascertainment_pop])

def expected_tmrca(demography):
    """
    The expected time to most recent common ancestor of the sample.

    Parameters
    ----------
    demography : Demography

    Returns
    -------
    tmrca : float-like

    See Also
    --------
    expected_deme_tmrca : tmrca of subsample within a deme
    expected_sfs_tensor_prod : compute general class of summary statistics
    """
    vecs = [np.ones(n+1) for n in demography.sampled_n]
    n0 = len(vecs[0])-1.0
    vecs[0] = np.arange(n0+1) / n0
    return np.squeeze(expected_sfs_tensor_prod(vecs, demography))

def expected_deme_tmrca(demography, deme):
    """
    The expected time to most recent common ancestor, of the samples within
    a particular deme.

    Parameters
    ----------
    demography : Demography
    deme : the deme

    Returns
    -------
    tmrca : float

    See Also
    --------
    expected_tmrca : the tmrca of the whole sample
    expected_sfs_tensor_prod : compute general class of summary statistics    
    """
    deme = list(demography.sampled_pops).index(deme)
    vecs = [np.ones(n+1) for n in demography.sampled_n]

    n = len(vecs[deme])-1
    vecs[deme] = np.arange(n+1) / (1.0*n)
    vecs[deme][-1] = 0.0
    
    return np.squeeze(expected_sfs_tensor_prod(vecs, demography))

def expected_sfs_tensor_prod(vecs, demography, mut_rate=1.0):
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
    mut_rate : float
         the rate of mutations per unit time
    
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
    ## NOTE cannot use vecs[i] = ... due to autograd issues
    vecs = [np.vstack([np.array([1.0] + [0.0]*n), # all ancestral state
                       np.array([0.0]*n + [1.0]), # all derived state
                       v])
            for v,n in zip(vecs, demography.sampled_n)]
    
    res = _expected_sfs_tensor_prod(vecs, demography, mut_rate=mut_rate)
    
    # subtract out mass for all ancestral/derived state
    for k in (0,1):
        res = res - res[k] * np.prod([l[:,-k] for l in vecs], axis=0)
        assert np.isclose(res[k], 0.0)
    # remove monomorphic states
    res = res[2:]    
    
    return res
    
def _expected_sfs_tensor_prod(vecs, demography, mut_rate=1.0):    
    leaf_states = dict(list(zip(demography.sampled_pops, vecs)))
    
    _,res = _partial_likelihood(leaf_states,
                                demography, demography._event_root)

    return res * mut_rate

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
    for newpop in demo._parent_pops(event):
        # term for mutation occurring at the newpop
        newpop_idx = axes.index(newpop)
        idx = [0] * lik.ndim
        idx[0], idx[newpop_idx] = slice(None), slice(None)

        sub_lik, trunc_sfs = lik[idx], demo._truncated_sfs(newpop)
        sfs = sfs + par_einsum(sub_lik, ['',newpop],
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
        #lik = demo._apply_transition(pop, lik, idx)
        assert lik.shape[idx] == demo._n_at_node(pop)+1
        lik = moran_action(demo._scaled_time(pop), lik, axis=idx)

    return lik,sfs

def _lik_axes(demo, event):
    '''
    Returns axes labels of the partial likelihood tensor
    first axis corresponds to SFS entry
    subsequent axes correspond to subpopulations
    '''
    sub_pops = list(demo._sub_pops(event))
    assert '' not in sub_pops
    return [''] + sub_pops

def _event_lik_fun(demo, event):
    e_type = demo._event_type(event)
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
    (pop,idx), = demo._parent_pops(event)
    if idx == 0:
        return leaf_states[pop],0.
    else:
        return np.ones((next(iter(leaf_states.values())).shape[0],1)),0.

# def _admixture_likelihood(leaf_states, demo, event):
#     child_pop, = list(demo._child_pops(event).keys())
#     p1,p2 = demo._parent_pops(event)

#     child_event, = demo._event_tree[event]
#     lik,sfs = _partial_likelihood_top(leaf_states, demo, child_event, [child_pop])

#     admixture_prob, admixture_idxs = demo._admixture_prob(child_pop)
#     lik = par_einsum(lik, _lik_axes(demo, child_event),
#                   admixture_prob, admixture_idxs,
#                   _lik_axes(demo, event))

#     return lik,sfs

def _pulse_likelihood(leaf_states, demo, event):
    parent_pops = demo._parent_pops(event)    
    child_pops_events = demo._child_pops(event)
    assert len(child_pops_events) == 2
    child_pops, child_events = list(zip(*list(child_pops_events.items())))
    
    if len(set(child_events)) == 2:
        ## in this case, it is more efficient to model the pulse as a split (-es) followed by a join (-ej)
        sfs, child_pops, child_axes, child_liks = _disjoint_children_liks(leaf_states, demo, event)
        
        recipient, non_recipient, donor, non_donor = demo._pulse_nodes(event)
        admixture_prob, admixture_idxs = demo._admixture_prob(recipient)

        child_liks = dict(list(zip(child_pops, child_liks)))
        child_axes = dict(list(zip(child_pops, child_axes)))

        tmp_axes = child_axes[recipient]
        child_axes[recipient] = [x for x in tmp_axes if x != recipient] + list(parent_pops)
        child_liks[recipient] = par_einsum(child_liks[recipient], tmp_axes,
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

        pulse_prob, pulse_idxs = demo._pulse_prob(event)

        lik = par_einsum(lik, axes,
                      pulse_prob, pulse_idxs,
                      _lik_axes(demo, event))
        return lik,sfs

def _merge_subpops_likelihood(leaf_states, demo, event):
    newpop, = demo._parent_pops(event)
    child_pops = demo._G[newpop]
    child_event, = demo._event_tree[event]

    child_axes = _lik_axes(demo, child_event)
    event_axes = _lik_axes(demo,event)

    lik,sfs = _partial_likelihood_top(leaf_states, demo, child_event, child_pops)
    
    lik = _merge_lik_axes(lik, child_axes, event_axes, child_pops, newpop,
                          {pop: demo._n_at_node(pop)
                           for pop in list(child_pops) + [newpop]})
    
    return lik,sfs

def _merge_lik_axes(lik, child_axes, new_axes, child_pops, newpop, n_lins):
    assert len(child_pops) == 2 and len(child_axes) == len(new_axes)+1
    
    c1,c2 = child_pops
    for c in c1,c2:
        lik = par_einsum(lik, child_axes,
                      binom_coeffs(n_lins[c]), [c],
                      child_axes)
    lik,axes = sum_antidiagonals(lik, child_axes, c1, c2, newpop)

    assert set(axes) == set(new_axes)
    newidx = axes.index(newpop)
    lik = par_einsum(lik, axes,
                  1.0/binom_coeffs(lik.shape[newidx]-1), [newpop],
                  new_axes)

    # reduce the number of lineages in newpop to only the number necessary
    axes = new_axes
    newidx = axes.index(newpop)
    N,n = lik.shape[newidx]-1, n_lins[newpop]
    assert N >= n
    if N > n:
        lik = par_einsum(lik, new_axes[:newidx] + [c1] + axes[(newidx+1):],
                      hypergeom_quasi_inverse(N,n),
                      [c1,newpop], axes)
    assert lik.shape[newidx] == n+1

    return lik

def _merge_clusters_likelihood(leaf_states, demo, event):
    sfs, child_pops, child_axes, child_liks = _disjoint_children_liks(leaf_states, demo, event)
    axes = _lik_axes(demo, event)
    newpop, = demo._parent_pops(event)    
    lik = _convolve_children_liks(child_pops, child_liks, child_axes, newpop, axes)
    return lik, sfs

def _convolve_children_liks(child_pops, child_liks, child_axes, newpop, axes):
    child_liks = list(child_liks)
    for i,(lik,child_axis,child_pop) in enumerate(zip(child_liks, child_axes, child_pops)):
        idx = child_axis.index(child_pop)
        lik = par_einsum(lik, child_axis,
                      binom_coeffs(lik.shape[idx]-1), [child_pop],
                      child_axis)
        child_liks[i] = lik

    lik, old_axes = convolve_axes(child_liks[0], child_liks[1],
                                  child_axes, child_pops, newpop)

    idx = old_axes.index(newpop)
    lik = par_einsum(lik, old_axes,
                  1.0/binom_coeffs(lik.shape[idx]-1), [newpop],
                  axes)
    return lik   

def _disjoint_children_liks(leaf_states, demo, event):
    child_liks = []
    for child_pop, child_event in list(demo._child_pops(event).items()):
        axes = _lik_axes(demo, child_event)        
        lik,sfs = _partial_likelihood_top(leaf_states, demo, child_event, [child_pop])
        child_liks.append((child_pop,axes,lik,sfs))
        
    child_pops,child_axes,child_liks,child_sfs = list(zip(*child_liks))

    sfs = 0.0
    assert len(child_liks) == 2    
    for freq, other_lik in zip(child_sfs, child_liks[::-1]):
        sfs = sfs + freq * np.squeeze(other_lik[[slice(None)] + [0] * (other_lik.ndim-1)])

    return (sfs, child_pops, child_axes, child_liks)
