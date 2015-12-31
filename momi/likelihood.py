
from .util import make_constant, check_symmetric, make_function, optimize, _npstr, folded_sfs
import autograd.numpy as np
from .compute_sfs import expected_sfs, expected_total_branch_len
from .math_functions import einsum2
import scipy
from autograd import hessian

def unlinked_log_likelihood(sfs, demo, mu, adjust_probs = 0.0, folded=False, **kwargs):
    """
    Compute the log likelihood for a collection of SNPs, assuming they are
    unlinked (independent). Calls expected_sfs to compute the individual SFS
    entries.
    
    Parameters
    ----------
    sfs : dict
        maps configs (tuples) to their observed counts (ints or floats).

        If there are D sampled populations, then each config is
        represented by a D-tuple (i_0,i_1,...,i_{D-1}), where i_j is the
        number of derived mutants in deme j.
    demo : Demography
        object created using the function make_demography
    mu : float or None
        The mutation rate. If None, the number of SNPs is assumed to be
        fixed and a multinomial distribution is used. Otherwise the number
        of SNPs is assumed to be Poisson with parameter mu*E[branch_len]
    folded : optional, bool
        if True, compute likelihoods for folded SFS

    Returns
    -------
    log_lik : numpy.float
        Log-likelihood of observed SNPs, under either a multinomial
        distribution or a Poisson random field.

    Other Parameters
    ----------------
    adjust_probs : float, optional
        Added to the probability of each SNP.
        Setting adjust_probs to a small positive number (e.g. 1e-80)
        prevents the likelihood from being 0 or negative, due to
        precision or underflow errors.
    **kwargs : optional
        Additional optional arguments to be passed to expected_sfs
        (e.g. error_matrices)

    See Also
    --------
    expected_sfs : compute SFS of individual entries
    unlinked_log_lik_vector : efficiently compute log-likelihoods for each
                              of several loci
    """
    return np.squeeze(unlinked_log_lik_vector([sfs], demo, mu, adjust_probs = adjust_probs, folded=folded, **kwargs))


def unlinked_log_lik_vector(sfs_list, demo, mu, adjust_probs = 0.0, folded=False, **kwargs):
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
    mu : float or numpy.ndarray or None
        The mutation rate. If None, the number of SNPs is assumed to be
        fixed and a multinomial distribution is used. If a numpy.ndarray,
        the number of SNPs at locus i is assumed to be Poisson with
        parameter mu[i]*E[branch_len]. If a float, the mutation rate at
        all loci are assumed to be equal.
    folded : optional, bool
        if True, compute likelihoods for folded SFS

    Returns
    -------
    log_lik : numpy.ndarray
        The i-th entry is the log-likelihood of the observed SNPs at locus i,
        under either a multinomial distribution or a Poisson random field.

    Other Parameters
    ----------------
    adjust_probs : float, optional
        Added to the probability of each SNP.
        Setting adjust_probs to a small positive number (e.g. 1e-80)
        prevents the likelihood from being 0 or negative, due to
        precision or underflow errors.
    **kwargs : optional
        Additional optional arguments to be passed to expected_sfs
        (e.g. error_matrices)


    See Also
    --------
    unlinked_log_likelihood : likelihood for a single locus
    """
    # remove 0 entries
    sfs_list = [dict([(k,v) for k,v in sfs.items() if v != 0]) for sfs in sfs_list]
    if folded:
        sfs_list = [folded_sfs(sfs, demo.n_at_leaves) for sfs in sfs_list] # for correct combinatorial factors
    # the list of all observed configs
    config_list = list(set(sum([list(sfs.keys()) for sfs in sfs_list],[])))

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
    E_counts = expected_sfs(demo, config_list, folded=folded, **kwargs)
    E_total = expected_total_branch_len(demo, **kwargs)

    sfs_probs = E_counts/ E_total + adjust_probs
    if np.any(sfs_probs <= 0.0):
        raise FloatingPointError("0 or negative probability leading to non-finite log-likelihood. Try increasing adjust_probs from %g to something larger." % adjust_probs)
    
    # a function to return the log factorial
    lnfact = lambda x: scipy.special.gammaln(x+1)

    # log likelihood of the multinomial distribution for observed SNPs
    log_lik = np.dot(counts_ij, np.log(sfs_probs)) - np.einsum('ij->i',lnfact(counts_ij)) + lnfact(counts_i)
    # add on log likelihood of poisson distribution for total number of SNPs
    if mu is not None:
        lambd = mu * E_total
        log_lik = log_lik - lambd + counts_i * np.log(lambd) - lnfact(counts_i)

    return log_lik


def unlinked_mle_search(sfs, demo_func, mu, start_params,
                        folded = False,
                        jac = True, hess = False, hessp = False,
                        method = 'tnc', maxiter = 100, bounds = None, tol = None, options = {},
                        output_progress = False,                        
                        sfs_kwargs = {}, adjust_probs = 1e-80,
                        **kwargs):
    """
    Find the maximum of unlinked_log_likelihood(), by calling
    scipy.optimize.minimize() on -1*unlinked_log_likelihood().

    See http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    or help(scipy.optimize.minimize) for more details on these parameters:
    (method, bounds, tol, options, **kwargs)

    Parameters
    ----------
    sfs : dict
        maps configs (tuples) to their observed counts (ints or floats)
        See unlinked_log_likelihood for details
    demo_func : function
        maps the parameters (numpy.ndarray) to Demography

        if jac=True, demo_func should work with autograd;
        see examples/tutorial.py (Section 5 FOOTNOTE)
    mu : None or float or function
        The mutation rate, or a function that takes in the parameters
        and returns the mutation rate. If None, uses a multinomial
        distribution; if a float, uses a Poisson random field. See
        unlinked_log_likelihood for additional details.

        if mu is function and jac=True, mu should work with autograd.
    folded : optional, bool
        if True, compute likelihoods for folded SFS
    start_params : numpy.ndarray
        The starting point for the parameter search
    jac : bool, optional
        If True, use autograd to compute the gradient (jacobian)
    hess, hessp : bool, optional
        If True, use autograd for the hessian or hessian-vector-product.
        At most one of hess or hessp should be True. If True, 'method' must
        be one of 'newton-cg','trust-ncg','dogleg', or a custom minimizer
    method : str or callable, optional
        The solver for scipy.optimize.minimize() to use
    maxiter : int, optional
        The maximum number of iterations to use
    bounds : list of (lower,upper), optional
        lower and upper bounds for each parameter. Use None to indicate
        parameter is unbounded in a direction.
        if bounds!=None, 'method' must be one of 'l-bfgs-b','tnc','slsqp'.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific options.
    options : dict, optional
        A dictionary of solver-specific options.
    output_progress : bool or int, optional
        if True, print every function/gradient/hessian evaluation
        if int i, print function evaluation at every i-th iteration

    Returns
    -------
    res : scipy.optimize.OptimizeResult
         Return value of scipy.optimize.minimize.

         Important attributes are: x the solution array, success a Boolean
         flag indicating if the optimizer exited successfully and message 
         which describes the cause of the termination.

         Note function & derivative values are for -1*unlinked_log_likelihood(),
         since that is the function minimized.

    Other Parameters
    ----------------
    sfs_kwargs : dict, optional
        additional keyword arguments to pass to unlinked_log_likelihood
    adjust_probs : float, optional
        Added to the probability of each SNP.
        Setting adjust_probs to a small positive number (e.g. 1e-80)
        prevents the likelihood from being 0 or negative, due to
        precision or underflow errors.
    **kwargs : optional
        additional arguments to pass to scipy.optimize.minimize

    See Also
    --------
    unlinked_log_likelihood : the objective that is optimized here
    unlinked_mle_approx_cov : approximate covariance matrix of the
         composite MLE, used for constructing approximate confidence
         intervals.
    sum_sfs_list : combine SFS's of multiple loci into one SFS, before
         passing into this function
    """
    mu = make_function(mu)
    f = lambda params: -unlinked_log_likelihood(sfs, demo_func(params), mu(params), adjust_probs = adjust_probs, folded=folded, **sfs_kwargs)

    if output_progress is True:
        # print the Demography after every iteration
        ## TODO: this prints after every scipy.minimize.optimize iteration,
        ##       whereas other printing occurs after every function evaluation...
        kwargs = dict(kwargs)
        callback0 = kwargs.get('callback', lambda x: None)
        def callback1(x):
            print("demo_func(%s) = %s" % (_npstr(x), demo_func(x)))
        def callback(x):
            callback1(x)            
            callback0(x)
        kwargs['callback'] = callback
    
    return optimize(f=f, start_params=start_params, jac=jac, hess=hess, hessp=hessp, method=method, maxiter=maxiter, bounds=bounds, tol=tol, options=options, output_progress=output_progress, **kwargs)

def unlinked_mle_approx_cov(params, sfs_list, demo_func, mu_per_locus, **kwargs):
    """
    Approximate covariance matrix for the composite MLE, i.e. the scaled
    inverse 'Godambe Information'.

    Parameters
    ----------
    params : numpy.ndarray
         The true parameters, or a consistent estimate thereof (e.g.,
         the composite MLE).
    sfs_list : list of dicts
         The i-th entry is the SFS of the i-th locus.
         The loci are assumed to be i.i.d. 
         See Notes for violations of this assumption.
    demo_func : function
         Function that maps parameter values to demographies.
         Must be differentiable by autograd.
    mu_per_locus : function or numpy.ndarray or float or None
         The mutation rate at each locus, or a function mapping parameters
         to the mutation rates, or None (if using multinomial instead of
         Poisson).
         If a function, must work with autograd.
    **kwargs: additional arguments for unlinked_log_lik_vector()

    Returns
    -------
    cov : numpy.ndarray
         The approximate covariance matrix

    See Also
    --------
    unlinked_log_likelihood : the composite log likelihood function
    unlinked_log_lik_vector : composite log likelihoods for each locus
    unlinked_mle_search : search for the composite MLE
    
    Notes
    -----
    As n_loci -> infinity, and assuming certain regularity conditions
    (e.g. identifiability), the composite MLE will be asymptotically
    Gaussian with this covariance. This can be used to construct approximate
    confidence intervals (see examples/tutorial.py, Section 6).

    Computing the covariance requries certain expectations to be empirically
    estimated, and thus it is necessary to have 'enough' i.i.d. loci to
    ensure accuracy.

    Mild violations of the i.i.d. assumption may be acceptable. For example,
    if the mutation rates are of roughly the same magnitude for all loci,
    then we can think of them as being drawn from the same hyperdistribution,
    so that the loci are identically distributed.
    """    
    mu_per_locus = make_function(mu_per_locus)
        
    f_vec = lambda x: unlinked_log_lik_vector(sfs_list, demo_func(x), mu_per_locus(x), **kwargs)
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

    cov = np.dot(h_inv, np.dot(g_out,h_inv))
    return check_symmetric(cov)

