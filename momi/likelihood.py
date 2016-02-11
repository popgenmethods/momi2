
from .util import make_constant, check_symmetric, make_function, optimize, _npstr, folded_sfs, truncate0, get_sfs_list
import autograd.numpy as np
from .compute_sfs import expected_sfs, expected_total_branch_len
from .math_functions import einsum2
import scipy
from autograd import hessian
from collections import Counter
from .data_structure import ObservedSfs, ObservedSfsList

def composite_log_likelihood(observed_sfs, demo, mut_rate=None, truncate_probs = 0.0, folded=False, **kwargs):
    """
    Compute the log likelihood for a collection of SNPs, assuming they are
    unlinked (independent). Calls expected_sfs to compute the individual SFS
    entries.
    
    Parameters
    ----------
    observed_sfs : dict, or ObservedSfs
        maps configs (tuples) to their observed counts (ints or floats).

        If there are D sampled populations, then each config is
        represented by a D-tuple (i_0,i_1,...,i_{D-1}), where i_j is the
        number of derived mutants in deme j.
    demo : Demography
    mut_rate : float or None
        The mutation rate. If None (as by default), the number of SNPs is assumed to be
        fixed and a multinomial distribution is used. Otherwise the number
        of SNPs is assumed to be Poisson with parameter mut_rate*E[branch_len]
    folded : optional, bool
        if True, compute likelihoods for folded SFS

    Returns
    -------
    log_lik : numpy.float
        Log-likelihood of observed SNPs, under either a multinomial
        distribution or a Poisson random field.

    Other Parameters
    ----------------
    truncate_probs : float, optional
        Replace log(sfs_probs) with log(max(sfs_probs, truncate_probs)),
        where sfs_probs are the normalized theoretical SFS entries.
        Setting truncate_probs to a small positive number (e.g. 1e-100)
        will avoid taking log(0) due to precision or underflow error.
    **kwargs : optional
        Additional optional arguments to be passed to expected_sfs
        (e.g. error_matrices)

    See Also
    --------
    expected_sfs : compute SFS of individual entries
    composite_log_lik_vector : efficiently compute log-likelihoods for each
                              of several loci
    """
    if not isinstance(observed_sfs, ObservedSfs):
        observed_sfs = ObservedSfs(observed_sfs, demo.sampled_n)
    return np.squeeze(composite_log_lik_vector(observed_sfs._sfs_list(), demo, mut_rate, truncate_probs = truncate_probs, folded=folded, **kwargs))


def composite_log_lik_vector(observed_sfs_list, demo, mut_rate=None, truncate_probs = 0.0, folded=False, **kwargs):
    """
    Return a vector of log likelihoods for a collection of loci. Equivalent
    to, but much more efficient than, calling composite_log_likelihood on
    each locus separately.

    Parameters
    ----------
    observed_sfs_list : list of dicts, or ObservedSfsList
        a list whose i-th entry is the SFS at locus i. Each SFS is a dict
        mapping configs (tuples) to observed counts (floats or ints)
    demo : Demography
    mut_rate : float or numpy.ndarray or None
        The mutation rate. If None (as by default), the number of SNPs is assumed to be
        fixed and a multinomial distribution is used. If a numpy.ndarray,
        the number of SNPs at locus i is assumed to be Poisson with
        parameter mut_rate[i]*E[branch_len]. If a float, the mutation rate at
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
    truncate_probs : float, optional
        Replace log(sfs_probs) with log(max(sfs_probs, truncate_probs)),
        where sfs_probs are the normalized theoretical SFS entries.
        Setting truncate_probs to a small positive number (e.g. 1e-100)
        will avoid taking log(0) due to precision or underflow error.
    **kwargs : optional
        Additional optional arguments to be passed to expected_sfs
        (e.g. error_matrices)


    See Also
    --------
    composite_log_likelihood : likelihood for a single locus
    """
    if not isinstance(observed_sfs_list, ObservedSfsList):
        observed_sfs_list = ObservedSfsList(observed_sfs_list, demo.sampled_n)

    config_list = observed_sfs_list._config_list(folded)
    # get the expected counts for each config
    sfs_probs = np.maximum(expected_sfs(demo, config_list, normalized=True, folded=folded, **kwargs),
                           truncate_probs)
    
    counts_ij = observed_sfs_list._counts_ij(folded)
    # counts_i is the total number of SNPs at each locus
    counts_i = np.einsum('ij->i',counts_ij)
    
    # a function to return the log factorial
    lnfact = lambda x: scipy.special.gammaln(x+1)

    # log likelihood of the multinomial distribution for observed SNPs
    log_lik = np.dot(counts_ij, np.log(sfs_probs)) - np.einsum('ij->i',lnfact(counts_ij)) + lnfact(counts_i)
    # add on log likelihood of poisson distribution for total number of SNPs
    if mut_rate is not None:
        sampled_n = np.sum(config_list.config_array, axis=2)
        if np.any(sampled_n != demo.sampled_n):
            raise NotImplementedError("Poisson model not implemented for missing data.")
        E_total = expected_total_branch_len(demo, **kwargs)        
        lambd = mut_rate * E_total
        log_lik = log_lik - lambd + counts_i * np.log(lambd) - lnfact(counts_i)

    return log_lik


def composite_mle_search(observed_sfs, demo_func, start_params,
                         mut_rate = None,
                         folded = False,
                         jac = True, hess = False, hessp = False,
                         method = 'tnc', maxiter = 100, bounds = None, tol = None, options = {},
                         output_progress = False,                        
                         sfs_kwargs = {}, truncate_probs = 1e-100,
                         **kwargs):
    """
    Find the maximum of composite_log_likelihood(), by calling
    scipy.optimize.minimize() on -1*composite_log_likelihood().

    See http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    or help(scipy.optimize.minimize) for more details on these parameters:
    (method, bounds, tol, options, **kwargs)

    Parameters
    ----------
    observed_sfs : dict, or ObservedSfs
        maps configs (tuples) to their observed counts (ints or floats)
        See composite_log_likelihood for details
    demo_func : function
        function that returns a Demography

        if jac=True, demo_func should work with autograd;
        see examples/tutorial.py (Section 5 FOOTNOTE)
    start_params : list
        The starting point for the parameter search.
        len(start_params) should equal the number of arguments of demo_func
    mut_rate : None or float or function, optional
        The mutation rate, or a function that takes in the parameters
        and returns the mutation rate. If None (the default), uses a multinomial
        distribution; if a float, uses a Poisson random field. See
        composite_log_likelihood for additional details.

        if mut_rate is function and jac=True, mut_rate should work with autograd.
    folded : optional, bool
        if True, compute likelihoods for folded SFS
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

         Note function & derivative values are for -1*composite_log_likelihood(),
         since that is the function minimized.

    Other Parameters
    ----------------
    sfs_kwargs : dict, optional
        additional keyword arguments to pass to composite_log_likelihood
    truncate_probs : float, optional
        Replace log(sfs_probs) with log(max(sfs_probs, truncate_probs)),
        where sfs_probs are the normalized theoretical SFS entries.
        Setting truncate_probs to a small positive number (e.g. 1e-100)
        will avoid taking log(0) due to precision or underflow error.
    **kwargs : optional
        additional arguments to pass to scipy.optimize.minimize

    See Also
    --------
    composite_log_likelihood : the objective that is optimized here
    composite_mle_approx_cov : approximate covariance matrix of the
         composite MLE, used for constructing approximate confidence
         intervals.
    sum_sfs_list : combine SFS's of multiple loci into one SFS, before
         passing into this function
    """
    old_demo_func = demo_func
    demo_func = lambda x: old_demo_func(*x)
    start_params = np.array(start_params)

    mut_rate = make_function(mut_rate)

    # wrap it in ObservedSfs to avoid repeating computation
    if not isinstance(observed_sfs, ObservedSfs):
        observed_sfs = ObservedSfs(observed_sfs, demo_func(start_params).sampled_n)
    f = lambda params: -composite_log_likelihood(observed_sfs, demo_func(params), mut_rate(params), truncate_probs = truncate_probs, folded=folded, **sfs_kwargs)

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

def composite_mle_approx_cov(params, seg_sites, demo_func,
                             mut_rate_per_locus=None,
                             method="iid", **kwargs):
    """
    Approximate covariance matrix for the composite MLE, i.e. the scaled
    inverse 'Godambe Information'.

    Under certain regularity conditions (e.g. identifiability, consistency),
    the composite MLE is asymptotically Gaussian, with covariance given
    by this function.

    Parameters
    ----------
    params : list
         The true parameters, or a consistent estimate thereof (e.g.,
         the composite MLE).
         len(params) should equal number of arguments that demo_func takes
    seg_sites : list of list of tuples, (as returned by momi.read_seg_sites).
         The i-th entry are the segregating sites of the i-th locus.
         Each segregating site is represented as a pair, (position, config).
         Each locus should be independent.

         If has attribute seg_sites.sampled_pops, it must equal demo.sampled_pops
    demo_func : function
         Function that maps parameter values to demographies.
         Must be differentiable by autograd.
    mut_rate_per_locus : function or numpy.ndarray or float or None, optional
         The mutation rate at each locus, or a function mapping parameters
         to the mutation rates, or None (the default).
         If a function, must work with autograd.
         If method='series' (the default), must be None.
    method : str, optional
         The method to compute covariance. Current options are:
         'iid' :  Estimate covariance by treating the loci as iid.
                  Appropriate when there are many loci, roughly identically distributed.
                  Not appropriate for just a few loci.
         'series' : Estimates the covariance by treating the
                    segregating sites as a time series.

                    Appropriate for a few long loci (i.e. chromosomes).
                    Not appropriate for short loci.
                    Only implemented for multinomial likelihood (mut_rate=None)
    **kwargs: additional arguments for composite_log_lik_vector()

    Returns
    -------
    cov : numpy.ndarray
         The approximate covariance matrix

    See Also
    --------
    composite_log_likelihood : the composite log likelihood function
    composite_log_lik_vector : composite log likelihoods for each locus
    composite_mle_search : search for the composite MLE    
    """
    if hasattr(seg_sites, "sampled_pops"):
        if seg_sites.sampled_pops != demo_func(*params).sampled_pops:
            raise Exception("seg_sites.sampled_pops should equal demo.sampled_pops")
    if method == "series":
        if mut_rate_per_locus != None:
            raise NotImplementedError("'series' covariance method not implemented for Poisson approximation")
        return _series_cov(params, seg_sites, demo_func, **kwargs)
    elif method == "iid":
        return _iid_cov(params, get_sfs_list(seg_sites), demo_func,
                        mut_rate_per_locus=mut_rate_per_locus, **kwargs)
    
def _iid_cov(params, observed_sfs_list, demo_func, mut_rate_per_locus=None, **kwargs):    
    old_demo_func = demo_func
    demo_func = lambda x: old_demo_func(*x)
    params = np.array(params)

    mut_rate_per_locus = make_function(mut_rate_per_locus)
        
    f_vec = lambda x: composite_log_lik_vector(observed_sfs_list, demo_func(x), mut_rate_per_locus(x), **kwargs)
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


def _series_cov(params, seg_sites, demo_func, **kwargs):
    """
    Composite MLE covariance, for a few long loci (rather than many short loci)
    """
    if "mut_rate" in kwargs:
        raise NotImplementedError("Currently only implemented for multinomial composite likelihood")    
    params = np.array(params)

    # sort the loci
    seg_sites = [sorted(locus) for locus in seg_sites]
    # discard the positions
    seg_sites = [[config for position,config in locus] for locus in seg_sites]
    
    uniq_snps, snp_counts = zip(*Counter(sum(seg_sites, [])).items())
    snp_counts = np.array(snp_counts)

    snp_log_probs = lambda x: np.log(expected_sfs(demo_func(*x), uniq_snps, normalized=True, **kwargs))
    
    h = hessian(lambda x: np.sum(snp_counts * snp_log_probs(x)))(params)
    h_inv = check_symmetric(np.linalg.inv(h))
    
    uniq_snp_idxs = {snp: i for i,snp in enumerate(uniq_snps)}
    idx_series_list = [np.array([uniq_snp_idxs[snp] for snp in chrom], dtype=int)
                       for chrom in seg_sites]

    # g_out = sum(autocov(einsum("ij,ik->ikj",jacobian(idx_series), jacobian(idx_series))))
    # computed in roundabout way, in case jacobian is slow for many snps
    # autocovariance is truncated at sqrt(len(idx_series)), to avoid statistical/numerical issues
    def g_out_antihess(y):
        lp = snp_log_probs(y)
        ret = 0.0
        for idx_series in idx_series_list:
            L = len(idx_series)
            
            l = lp[idx_series]
            lc = make_constant(l)

            fft = np.fft.fft(l)
            # (assumes l is REAL)
            assert np.all(np.imag(l) == 0.0)
            fft_rev = np.conj(fft) * np.exp(2 * np.pi * 1j * np.arange(L) / float(L))

            curr = 0.5 * (fft * fft_rev - fft * make_constant(fft_rev) - make_constant(fft) * fft_rev)        
            curr = np.fft.ifft(curr)[(L-1)::-1]

            # make real
            assert np.allclose(np.imag(curr / L), 0.0)
            curr = np.real(curr)
            curr = curr[0] + 2.0 * np.sum(curr[1:int(np.sqrt(L))])
            ret = ret + curr
        return ret
    g_out = hessian(g_out_antihess)(params)
    g_out = 0.5 * (g_out + np.transpose(g_out))
    
    return check_symmetric(np.dot(h_inv, np.dot(g_out, h_inv)))
