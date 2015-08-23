from __future__ import division
from util import make_constant, check_symmetric, make_function
from autograd import hessian, grad, hessian_vector_product, jacobian
import autograd.numpy as np
import scipy
from sum_product import compute_sfs, expected_total_branch_len
from math_functions import einsum2

def unlinked_log_likelihood(sfs, demo, theta, adjust_probs = 0.0, **kwargs):
    """
    Compute the log likelihood for a collection of SNPs, assuming they are
    unlinked (independent). Calls compute_sfs to compute the individual SFS
    entries.
    
    Parameters
    ----------
    sfs : dict
        maps configs (tuples) to their observed counts (ints or floats)
        See compute_sfs for additional details
    demo : Demography
        object created using the function make_demography
    theta : float or None
        The mutation rate. If None, the number of SNPs is assumed to be
        fixed and a multinomial distribution is used. Otherwise the number
        of SNPs is assumed to be Poisson with parameter theta*E[branch_len]

    Returns
    -------
    log_lik : numpy.float
        Log-likelihood of observed SNPs, under either a multinomial
        distribution or a Poisson random field.

    Other Parameters
    ----------------
    adjust_probs : float, optional
        Added to the probability of each SNP. Default is 0 (no adjustment).
        Setting adjust_probs to a small positive number (e.g. 1e-80)
        prevents the likelihood from being exactly 0, and thus the
        log-likelihood from being -Inf. This is useful for dealing with 
        potential numerical underflow errors, and can improve parameter
        search behavior, especially in regions of very low likelihood.
    **kwargs : optional
        Additional optional arguments to be passed to compute_sfs
        (e.g. error_matrices)

    See Also
    --------
    compute_sfs : compute SFS of individual entries
    unlinked_log_lik_vector : efficiently compute log-likelihoods for each
                              of several loci
    """
    return unlinked_log_lik_vector([sfs], demo, theta, adjust_probs = adjust_probs, **kwargs)


def unlinked_mle_search1(sfs, demo_func, theta, start_params, bounds, sfs_kwargs = {}, adjust_probs = 1e-80):
    """
    Search for the MLE, assuming all sites are unlinked. Uses
    unlinked_log_likelihood to compute log-likelihoods and
    scipy.optimize.minimize to perform parameter search.

    This is a simple wrapper function for unlinked_mle_search. It searches
    for the optimal parameters within a rectangular region, and requires
    the user to specify finite bounds for all parameters. The L-BFGS-B
    search algorithm is used, which uses the gradient and an approximation
    to the inverse Hessian to steer its search. Note this is susceptible
    to local optima, and multiple runs with different starting parameters
    may be required to find the global optimum.

    Parameters
    ----------
    sfs : dict
        maps configs (tuples) to their observed counts (ints or floats)
        See compute_sfs for additional details
    demo_func : function
        a function returning a valid demography object (as created by
        make_demography) for every parameter value within bounds
    theta : None or float or function
        The mutation rate, or a function that takes in the parameters
        and returns the mutation rate. If None, uses a multinomial
        distribution; if a float, uses a Poisson random field. See
        unlinked_log_likelihood for additional details.
    start_params : numpy.ndarray
        The starting point for the parameter search.
    bounds : list of tuples
        list of (min,max) pairs for the parameter values. All parameters
        within the rectangular region described by bounds must return
        a valid demography. For robustness, it is also recommended that
        all parameters just outside the bounds, but within numerical
        precision of the boundary, also return a valid demography.

    Other Parameters
    ----------------
    sfs_kwargs : dict, optional
        additional keyword arguments to pass to unlinked_log_likelihood
    adjust_probs : float, optional
        Added to the probability of each SNP. Default is 1e-80, to
        prevent the likelihood from being exactly 0, and thus the
        log-likelihood from being -Inf. This also improves parameter
        search behavior, especially in regions of very low likelihood.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
         The optimization result represented as a OptimizeResult object.
         Important attributes are: x the solution array, success a Boolean
         flag indicating if the optimizer exited successfully and message 
         which describes the cause of the termination. See scipy
         documentation for a description of other attributes.

    See Also
    --------
    unlinked_log_likelihood : the objective that is optimized here
    unlinked_mle_search : parameter search with more options
    unlinked_mle_search2 : unbounded search with hessian information
    composite_mle_approx_covariance : approximate covariance matrix of the
         composite MLE, used for constructing approximate confidence
         intervals.
    sum_sfs_list : combine SFS's of multiple loci into one SFS, before
         passing into this function
    """
    if bounds is None or any([x is None for bd in bounds for x in bd]):
        raise Exception("Finite bounds required for all parameters")
    
    return unlinked_mle_search(sfs, demo_func, theta, start_params, opt_kwargs = {'bounds': bounds}, sfs_kwargs = sfs_kwargs, adjust_probs = adjust_probs)


def unlinked_mle_search2(sfs, demo_func, theta, start_params, sfs_kwargs = {}, adjust_probs = 1e-80):
    """
    Search for the MLE, assuming all sites are unlinked. Uses
    unlinked_log_likelihood to compute log-likelihoods and
    scipy.optimize.minimize to perform parameter search.

    This is a simple wrapper function for unlinked_mle_search. The
    trust-region Newton conjugate gradient algorithm is used in an
    unbounded search. Gradients and Hessian-vector products are used to
    steer the search. Since the search is unbounded, all possible
    parameters should correspond to valid demographies.

    Note this is susceptible to local optima, and  multiple runs with
    different starting parameters may be required to find the global
    optimum.

    Parameters
    ----------
    sfs : dict
        maps configs (tuples) to their observed counts (ints or floats)
        See compute_sfs for additional details
    demo_func : function
        a function returning a valid demography object (as created by
        make_demography) for every parameter value within R^p, where
        p is the number of parameters.
    theta : None or float or function
        The mutation rate, or a function that takes in the parameters
        and returns the mutation rate. If None, uses a multinomial
        distribution; if a float, uses a Poisson random field. See
        unlinked_log_likelihood for additional details.
    start_params : numpy.ndarray
        The starting point for the parameter search.

    Other Parameters
    ----------------
    sfs_kwargs : dict, optional
        additional keyword arguments to pass to unlinked_log_likelihood
    adjust_probs : float, optional
        Added to the probability of each SNP. Default is 1e-80, to
        prevent the likelihood from being exactly 0, and thus the
        log-likelihood from being -Inf. This also improves parameter
        search behavior, especially in regions of very low likelihood.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
         The optimization result represented as a OptimizeResult object.
         Important attributes are: x the solution array, success a Boolean
         flag indicating if the optimizer exited successfully and message 
         which describes the cause of the termination. See scipy
         documentation for a description of other attributes.

    See Also
    --------
    unlinked_log_likelihood : the objective that is optimized here
    unlinked_mle_search : parameter search with more options
    unlinked_mle_search1 : bounded search with L-BFGS-B
    composite_mle_approx_covariance : approximate covariance matrix of the
         composite MLE, used for constructing approximate confidence
         intervals.
    sum_sfs_list : combine SFS's of multiple loci into one SFS, before
         passing into this function
    """    
    return unlinked_mle_search(sfs, demo_func, theta, start_params, derivs = ['jac', 'hessp'], opt_kwargs = {'method' : 'trust-ncg'}, sfs_kwargs = sfs_kwargs, adjust_probs = adjust_probs)


def composite_mle_approx_covariance(params, sfs_list, demo_func, theta):
    """
    Approximate covariance matrix for the composite MLE, i.e. the inverse
    'Godambe Information'. Under certain conditions, the composite MLE
    will be asymptotically Gaussian with this covariance, and so this
    covariance matrix can be used to construct asymptotically correct
    confidence intervals.

    Parameters
    ----------
    params : numpy.ndarray
         The true parameters, or a reasonable approximation thereof (e.g.,
         the composite MLE). A necessary condition for the covariance
         matrix to be asymptotically correct is that the plugged-in
         parameters converge to the truth.
    sfs_list : list of dicts
         The i-th entry is the SFS of the i-th locus. The estimated
         covariance assumes that the loci are i.i.d., though it may be
         possible to relax the assumption that the loci are identically
         distributed.
         Computing the Godambe information requires certain expectations
         to be empirically estimated from this list, and thus it is
         necessary to have 'enough' independent loci to estimate these
         expectations accurately.
    demo_func : function
         Function that maps parameter values to demographies
    theta : function or numpy.ndarray or float or None
         The mutation rate at each locus, or a function mapping parameters
         to the mutation rates, or None (if using multinomial instead of
         Poisson).
    """    
    theta = make_function(theta)
        
    f_vec = lambda x: unlinked_log_lik_vector(sfs_list, demo_func(x), theta(x))
    # the sum of f_vec
    f_sum = lambda x: np.sum(f_vec(x))

    h = hessian(f_sum)(params)
    
    # g_out = einsum('ij,ik', jacobian(f_vec)(params), jacobian(f_vec)(params))
    # but computed in a roundabout way because jacobian implementation is slow
    def _g_out_antihess(x):
        l = f_vec(x)
        lc = make_constant(l)
        return np.sum(0.5 * (l**2 - l*lc - lc*l))
    g_out = hessian(_g_out_antihess)(params)
    
    h,g_out = (check_symmetric(_) for _ in (h,g_out))

    h_inv = np.linalg.inv(h)
    h_inv = check_symmetric(h_inv)

    ret = np.dot(h_inv, np.dot(g_out,h_inv))
    return check_symmetric(ret)


def unlinked_log_lik_vector(sfs_list, demo, theta, adjust_probs = 0.0, **kwargs):
    """
    Return a vector of log likelihoods for a collection of loci. Equivalent
    to, but much more efficient than, calling unlinked_log_likelihood on
    each locus separately.

    Parameters
    ----------
    sfs_list : list of dicts
        a list whose i-th entry is the SFS at locus i. Each SFS is a dict
        mapping configs (tuples) to observed counts (floats or ints)
    demo : Demography
        object created using the function make_demography
    theta : float or numpy.ndarray or None
        The mutation rate. If None, the number of SNPs is assumed to be
        fixed and a multinomial distribution is used. If a numpy.ndarray,
        the number of SNPs at locus i is assumed to be Poisson with
        parameter theta[i]*E[branch_len]. If a float, the mutation rate at
        all loci are assumed to be equal.

    Returns
    -------
    log_lik : numpy.ndarray
        The i-th entry is the log-likelihood of the observed SNPs at locus i,
        under either a multinomial distribution or a Poisson random field.

    Other Parameters
    ----------------
    adjust_probs : float, optional
        See unlinked_log_likelihood for description.
    **kwargs : optional
        Additional optional arguments to be passed to compute_sfs
        (e.g. error_matrices)


    See Also
    --------
    unlinked_log_likelihood : likelihood for a single locus
    """
    # the list of all observed configs
    config_list = list(set(sum([sfs.keys() for sfs in sfs_list],[])))

    # counts_ij is a matrix whose [i,j]th entry is the count of config j at locus i
    counts_ij = np.zeros((len(sfs_list), len(config_list)))
    for i,sfs in enumerate(sfs_list):
        for j,config in enumerate(config_list):
            try:
                counts_ij[i,j] = sfs[config]
            except KeyError:
                pass
    # counts_i is the total number of SNPs at each locus
    counts_i = np.einsum('ij->i',counts_ij)

    # get the expected counts for each config
    E_counts, E_total = compute_sfs(demo, config_list, **kwargs)
    
    # a function to return the log factorial
    lnfact = lambda x: scipy.special.gammaln(x+1)

    # log likelihood of the multinomial distribution for observed SNPs
    log_lik = np.dot(counts_ij, np.log(E_counts/ E_total + adjust_probs)) - np.einsum('ij->i',lnfact(counts_ij)) + lnfact(counts_i)
    # add on log likelihood of poisson distribution for total number of SNPs
    if theta is not None:
        lambd = theta * E_total
        log_lik = log_lik - lambd + counts_i * np.log(lambd) - lnfact(counts_i)

    return log_lik


def unlinked_mle_search(sfs, demo_func, theta, start_params, derivs = ['jac'], opt_kwargs = {}, sfs_kwargs = {}, adjust_probs = 1e-80):
    """
    Search for the MLE, assuming all sites are unlinked. Uses
    unlinked_log_likelihood to compute log-likelihoods and
    scipy.optimize.minimize to perform parameter search.

    New users are recommended to use the wrapper functions
    unlinked_mle_search1 and unlinked_mle_search2 instead of calling
    this function directly.

    The search may be susceptible to local optima, and multiple runs with
    different starting parameters may be required to find the global
    optimum.

    Parameters
    ----------
    sfs : dict
        maps configs (tuples) to their observed counts (ints or floats)
        See compute_sfs for additional details
    demo_func : function
        a function returning a valid demography object (as created by
        make_demography) for every parameter value within R^p, where
        p is the number of parameters.
    theta : None or float or function
        The mutation rate, or a function that takes in the parameters
        and returns the mutation rate. If None, uses a multinomial
        distribution; if a float, uses a Poisson random field. See
        unlinked_log_likelihood for additional details.
    start_params : numpy.ndarray
        The starting point for the parameter search.
    derivs : collection, optional
        A collection of strings, telling which derivatives the optimizer
        should use. Valid options are:
        'jac' : gradient (first-order derivative)
        'hess' : hessian (second-order derivative)
        'hessp' : hessian-vector product
    opt_kwargs : dict, optional
        Additional arguments to pass to scipy.optimize.minimize.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
         The optimization result represented as a OptimizeResult object.
         Important attributes are: x the solution array, success a Boolean
         flag indicating if the optimizer exited successfully and message 
         which describes the cause of the termination. See scipy
         documentation for a description of other attributes.

    Other Parameters
    ----------------
    sfs_kwargs : dict, optional
        additional keyword arguments to pass to unlinked_log_likelihood
    adjust_probs : float, optional
        Added to the probability of each SNP. Default is 1e-80, to
        prevent the likelihood from being exactly 0, and thus the
        log-likelihood from being -Inf. This also improves parameter
        search behavior, especially in regions of very low likelihood.

    See Also
    --------
    unlinked_log_likelihood : the objective that is optimized here
    unlinked_mle_search1 : bounded search with L-BFGS-B
    unlinked_mle_search2 : unbounded search with hessian information
    composite_mle_approx_covariance : approximate covariance matrix of the
         composite MLE, used for constructing approximate confidence
         intervals.
    sum_sfs_list : combine SFS's of multiple loci into one SFS, before
         passing into this function
    """        
    theta = make_function(theta)
    f = lambda params: -unlinked_log_likelihood(sfs, demo_func(params), theta(params), adjust_probs = adjust_probs, **sfs_kwargs)

    derivs = list(derivs)
    opt_kwargs = dict(opt_kwargs)
    
    for kw, d in [('jac',grad),
                  ('hessp',hessian_vector_product),
                  ('hess',hessian)]:
        if kw in derivs:
            opt_kwargs[kw] = d(f)
            derivs.remove(kw)
    if len(derivs) != 0:
        raise Exception("Unrecognized derivative name")

    return scipy.optimize.minimize(f, start_params, **opt_kwargs)
