
from .util import make_constant, optimize, _npstr, truncate0, check_psd, memoize_instance
import autograd.numpy as np
from .compute_sfs import expected_sfs, expected_total_branch_len
from .math_functions import einsum2, inv_psd
from .demography import DemographyError
from .data_structure import _hashable_config
import scipy, scipy.stats
import autograd
from collections import Counter

def composite_log_likelihood(data, demo, mut_rate=None, truncate_probs = 0.0, vector=False, **kwargs):
    """
    Returns the composite log likelihood for the data.

    Parameters
    ----------
    data : SegSites or Sfs
          if data.folded==True, functions returns composite log likelihood for folded SFS
    demo : Demography
    mut_rate : None or float or list of floats
           if None, function returns the multinomial composite log likelihood
           if float or list of floats, it is the mutation rate per locus, 
           and returns Poisson random field approximation.
           Note the Poisson model is not implemented for missing data.
    vector : boolean
           if False, return composite log-likelihood for the whole SFS
           if True, return list of the composite log-likelihood for each locus

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
    """
    try:
        sfs = data.sfs
    except AttributeError:
        sfs = data
    
    # get the expected counts for each config
    sfs_probs = np.maximum(expected_sfs(demo, sfs.configs, normalized=True, **kwargs),
                           truncate_probs)
    
    counts_ij = sfs._counts_ij()
    if not vector:
        counts_ij = np.array(np.sum(counts_ij, axis=0), ndmin=2)
            
    # counts_i is the total number of SNPs at each locus
    counts_i = np.einsum('ij->i',counts_ij)
    
    # a function to return the log factorial
    lnfact = lambda x: scipy.special.gammaln(x+1)

    # log likelihood of the multinomial distribution for observed SNPs
    log_lik = np.dot(counts_ij, np.log(sfs_probs)) - np.einsum('ij->i',lnfact(counts_ij)) + lnfact(counts_i)
    # add on log likelihood of poisson distribution for total number of SNPs
    if mut_rate is not None:
        mut_rate = mut_rate * np.ones(len(sfs.loci))
        if not vector:
            mut_rate = np.sum(mut_rate)
        sampled_n = np.sum(sfs.configs.config_array, axis=2)
        if np.any(sampled_n != demo.sampled_n):
            raise NotImplementedError("Poisson model not implemented for missing data.")
        E_total = expected_total_branch_len(demo, **kwargs)        
        lambd = mut_rate * E_total
        log_lik = log_lik - lambd + counts_i * np.log(lambd) - lnfact(counts_i)

    if not vector:
        log_lik = np.squeeze(log_lik)
    return log_lik

def composite_mle_search(data, demo_func, start_params,
                         mut_rate = None,
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
    data : SegSites or Sfs
    demo_func : function
        function that returns a Demography
        if jac=True, demo_func should work with autograd (see tutorial)
    start_params : list
        The starting point for the parameter search.
        len(start_params) should equal the number of arguments of demo_func
    mut_rate : None or float, optional
        The mutation rate. If None (the default), uses a multinomial
        distribution; if a float, uses a Poisson random field. See
        composite_log_likelihood for additional details.
        Note the Poisson model is not implemented for missing data.
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
    bounds : list of (lower,upper) or float, optional
        lower and upper bounds for each parameter. Use None to indicate
        parameter is unbounded in a direction. Use a float to indicate
        the parameter should be fixed to a specific value.
        if using lower,upper bounds, 'method' must be one of 'l-bfgs-b','tnc','slsqp'.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific options.
    options : dict, optional
        A dictionary of solver-specific options.
    output_progress : int, optional
        print output at every i-th call to the function

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
    """
    start_params = np.array(start_params)

    # old_demo_func = demo_func
    # def demo_func(params):
    #     try:
    #         return old_demo_func(*params)
    #     except DemographyError as err:
    #         raise DemographyError("DemographyError at params %s. Error message:\n\t%s" % (str(params), str(err)))
    
    #mut_rate = make_function(mut_rate)

    f = lambda params: -composite_log_likelihood(data, demo_func(params), mut_rate, truncate_probs = truncate_probs, **sfs_kwargs)
    
    return optimize(f=f, start_params=start_params, jac=jac, hess=hess, hessp=hessp, method=method, maxiter=maxiter, bounds=bounds, tol=tol, options=options, output_progress=output_progress, **kwargs)

class ConfidenceRegion(object):
    """
    Constructs asymptotic confidence regions and hypothesis tests,
    using the Limit of Experiments theory.
    """
    def __init__(self, point_estimate, demo_func, data, regime="long", **kwargs):
        """
        Parameters
        ----------
        point_estimate : array
                 a statistically consistent estimate for the true parameters.
                 confidence regions and hypothesis tests are computed for a (shrinking)
                 neighborhood around this point.
        demo_func : function that returns a Demography from parameters
        data : SegSites (or Sfs, if regime="many")
        regime : the limiting regime for the asymptotic confidence region
              if "long", number of loci is fixed, and the length of the loci -> infinity.
                 * uses time series information to estimate covariance structure
                 * requires isinstance(data, SegSites)
                 * loci should be independent. they don't have to be identically distributed
              if "many", the number of loci -> infinity
                 * loci should be independent, and roughly identically distributed
        **kwargs : additional arguments passed into composite_log_likelihood
        """
        if regime not in ("long","many"):
            raise ValueError("Unrecognized regime '%s'" % regime)
        
        self.point = np.array(point_estimate)
        self.demo_func = demo_func
        self.data = data
        self.regime = regime
        self.kwargs = kwargs

        self.score = autograd.grad(self.lik_fun)(self.point)
        self.score_cov = _observed_score_covariance(self.regime, self.point, self.data,
                                                   self.demo_func, **self.kwargs)
        self.fisher = _observed_fisher_information(self.point, self.data, self.demo_func,
                                                  assert_psd=False, **self.kwargs)
        
    def lik_fun(self, params, vector=False):
        """Returns composite log likelihood from params"""
        return composite_log_likelihood(self.data, self.demo_func(*params), vector=vector, **self.kwargs)
    
    @memoize_instance
    def godambe(self, inverse=False):
        """
        Returns Godambe Information.
        If the true params are in the interior of the parameter space,
        the composite MLE will be approximately Gaussian with mean 0,
        and covariance given by the Godambe information.
        """
        fisher_inv = inv_psd(self.fisher)
        ret = check_psd(np.dot(fisher_inv, np.dot(self.score_cov, fisher_inv)))
        if not inverse:
            ret = inv_psd(ret)
        return ret

    def test(self, null_point, sims=int(1e3), test_type="ratio", alt_point=None, null_cone=None, alt_cone=None, p_only=True):
        """
        Returns p-value for a single or several hypothesis tests.
        By default, does a simple hypothesis test with the log-likelihood ratio.

        Note that for tests on the boundary, the MLE for the null and alternative
        models are often the same (up to numerical precision), leading to a p-value of 1

        Parameters
        ----------
        null_point : array or list of arrays
              the MLE of the null model
              if a list of points, will do a hypothesis test for each point
        sims : the number of Gaussian simulations to use for computing null distribution
              ignored if test_type="wald"
        test_type : "ratio" for likelihood ratio, "wald" for Wald test
              only simple hypothesis tests are implemented for "wald"

              For "ratio" test:
              Note that we set the log likelihood ratio to 0 if the two
              likelihoods are within numerical precision (as defined by numpy.isclose)

              For tests on interior of parameter space, it generally shouldn't happen
              to get a log likelihood ratio of 0 (and hence p-value of 1).
              But this can happen on the interior of the parameter space.

        alt_point : the MLE for the alternative models
              if None, use self.point (the point estimate used for this ConfidenceRegion)
              dimensions should be compatible with null_point
        null_cone, alt_cone : the nested Null and Alternative models
              represented as a list, whose length is the number of parameters
              each entry of the list should be in (None,0,1,-1)
                     None: parameter is unconstrained around the "truth"
                     0: parameter is fixed at "truth"
                     1: parameter can be >= "truth"
                     -1: parameter can be <= "truth"

              if null_cone=None, it is set to (0,0,...,0), i.e. totally fixed
              if alt_cone=None, it is set to (None,None,...), i.e. totally unconstrained
        p_only : bool
              if True, only return the p-value (probability of observing a more extreme statistic)
              if False, return 3 values per test:
                   [0] the p-value: (probability of more extreme statistic)
                   [1] probability of equally extreme statistic (up to numerical precision)
                   [2] probability of less extreme statistic

              [1] should generally be 0 in the interior of the parameter space.
              But on the boundary, the log likelihood ratio will frequently be 0,
              leading to a point mass at the boundary of the null distribution.
        """
        in_shape = np.broadcast(np.array(null_point), np.array(alt_point),
                                np.array(null_cone), np.array(alt_cone)).shape

        null_point = np.array(null_point, ndmin=2)

        if null_cone is None:
            null_cone = [0]*null_point.shape[1]
        null_cone = np.array(null_cone, ndmin=2)
        
        if alt_point is None:
            alt_point = self.point
        alt_point = np.array(alt_point, ndmin=2)
        
        if alt_cone is None:
            alt_cone = [None]*null_point.shape[1]
        alt_cone = np.array(alt_cone, ndmin=2)

        b = np.broadcast_arrays(null_point, null_cone, alt_point, alt_cone)
        try:
            assert all(bb.shape[1:] == (len(self.point),) for bb in b)
        except AssertionError:
            raise ValueError("points, cones have incompatible shapes")
        b = [map(tuple, x) for x in b]
        null_point, null_cone, alt_point, alt_cone = b
        
        if test_type == "ratio":
            sims = np.random.multivariate_normal(self.score, self.score_cov, size=sims)
            
            liks = {}
            for p in list(null_point) + list(alt_point):
                if p not in liks:
                    liks[p] = self.lik_fun(np.array(p))

            sim_mls = {}
            for nc, ac in zip(null_cone, alt_cone):
                if (nc,ac) not in sim_mls:
                    nml, nmle = _project_scores(sims, self.fisher, nc)
                    aml, amle = _project_scores(sims, self.fisher, ac, init_vals=nmle)
                    sim_mls[(nc,ac)] = (nml,aml)

            ret = []
            for n_p,n_c,a_p,a_c in zip(null_point, null_cone, alt_point, alt_cone):
                lr = _trunc_lik_ratio(liks[n_p], liks[a_p])
                lr_distn = _trunc_lik_ratio(*sim_mls[(n_c,a_c)])
                ret += [map(np.mean, [lr > lr_distn,
                                      lr == lr_distn,
                                      lr < lr_distn])]
            ret = np.array(ret)
        elif test_type == "wald":
            if np.any(np.array(null_cone) != 0) or any(a_c != tuple([None]*len(self.point)) for a_c in alt_cone):
                raise NotImplementedError("Only simple tests implemented for wald")

            gdmb = self.godambe(inverse=False)

            resids = np.array(alt_point) - np.array(null_point)
            ret = np.einsum("ij,ij->i", resids,
                            np.dot(resids, gdmb)) 
            ret = 1.-scipy.stats.chi2.cdf(ret, df=len(self.point))           
            ret = np.array([ret, [0]*len(ret), 1.-ret]).T
        else:
            raise NotImplementedError("%s tests not implemented" % test_type)

        if p_only:
            ret = ret[:,0]
        if len(in_shape) == 1:
            ret = np.squeeze(ret)
        return ret        

    def wald_intervals(self, lower=.025, upper=.975):
        """
        Marginal wald-type confidence intervals.
        """
        conf_lower, conf_upper = scipy.stats.norm.interval(.95,
                                                           loc = self.point,
                                                           scale = np.sqrt(np.diag(self.godambe(inverse=True))))
        return np.array([conf_lower, conf_upper]).T

def _trunc_lik_ratio(null, alt):
    return (1-np.isclose(alt,null)) * (null - alt)
    
def _observed_fisher_information(params, data, demo_func, assert_psd=True, **kwargs):
    params = np.array(params)
    f = lambda x: composite_log_likelihood(data, demo_func(*x), **kwargs)
    ret = -autograd.hessian(f)(params)
    if assert_psd:
        try:
            ret = check_psd(ret)
        except AssertionError:
            raise Exception("Observed Fisher Information is not PSD (either due to numerical instability, or because the parameters are not a local maxima in the interior)")
    return ret

def _observed_score_covariance(method, params, seg_sites, demo_func, **kwargs):
    if method == "long":
        if "mut_rate" in kwargs:
            raise NotImplementedError("'long' godambe method not implemented for Poisson approximation")
        ret = _long_score_cov(params, seg_sites, demo_func, **kwargs)
    elif method == "many":
        ret = _many_score_cov(params, seg_sites, demo_func, **kwargs)
    else:
        raise Exception("Unrecognized method")

    try:
        ret = check_psd(ret)
    except AssertionError:
        raise Exception("Numerical instability: score covariance is not PSD")
    return ret
    
def _many_score_cov(params, data, demo_func, **kwargs):    
    params = np.array(params)

    def f_vec(x):
        ret = composite_log_likelihood(data, demo_func(*x), vector=True, **kwargs)
        # centralize
        return ret - np.mean(ret)
    
    # g_out = einsum('ij,ik', jacobian(f_vec)(params), jacobian(f_vec)(params))
    # but computed in a roundabout way because jacobian implementation is slow
    def _g_out_antihess(x):
        l = f_vec(x)
        lc = make_constant(l)
        return np.sum(0.5 * (l**2 - l*lc - lc*l))
    return autograd.hessian(_g_out_antihess)(params)


def _long_score_cov(params, seg_sites, demo_func, **kwargs):
    if "mut_rate" in kwargs:
        raise NotImplementedError("Currently only implemented for multinomial composite likelihood")    
    params = np.array(params)
   
    configs = seg_sites.sfs.configs
    snp_counts = np.sum(seg_sites.sfs._counts_ij(), axis=0)
    weights = snp_counts / float(np.sum(snp_counts))
    
    def snp_log_probs(x):
        ret = np.log(expected_sfs(demo_func(*x), configs, normalized=True, **kwargs))
        return ret - np.sum(weights * snp_counts) # subtract off mean
       
    uniq_snp_idxs = {snp: i for i,snp in enumerate(configs)}
    seg_sites = [map(_hashable_config, chrom) for chrom in seg_sites.config_arrays]
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
    g_out = autograd.hessian(g_out_antihess)(params)
    g_out = 0.5 * (g_out + np.transpose(g_out))
    return g_out


def _project_scores(simulated_scores, fisher_information, polyhedral_cone, init_vals=None, method="tnc"):
    """
    Under usual theory, the score is asymptotically 
    Gaussian, with covariance == Fisher information.
    
    For Gaussian location model w~N(x,Fisher^{-1}),
    with location parameter x, the likelihood ratio
    of x vs. 0 is
    
    LR0 = x*z - x*Fisher*x / 2
    
    where the "score" z=Fisher*w has covariance==Fisher.
    
    This function takes a bunch of simulated scores,
    and returns LR0 for the corresponding Gaussian 
    location MLEs on a polyhedral cone.
    
    Input:
    simulated_scores: list or array of simulated values.
        For a correctly specified model in the interior
        of parameter space, this would be Gaussian with
        mean 0 and cov == Fisher information.
        For composite or misspecified model, or model
        on boundary of parameter space, this may have
        a different distribution.
    fisher information: 2d array
    polyhedral_cone: list
        length is the number of parameters
        entries are 0,1,-1,None
            None: parameter is unconstrained
            0: parameter == 0
            1: parameter is >= 0
            -1: parameter is <= 0
    """    
    if init_vals is None:
        init_vals = np.zeros(simulated_scores.shape)
    
    fixed_params = [c == 0 for c in polyhedral_cone]
    if any(fixed_params):
        if all(fixed_params):
            return np.zeros(init_vals.shape[0]), init_vals
        fixed_params = np.array(fixed_params)
        proj = np.eye(len(polyhedral_cone))[~fixed_params,:]
        fisher_information = np.dot(proj, np.dot(fisher_information, proj.T))
        simulated_scores = np.einsum("ij,kj->ik", simulated_scores, proj)
        polyhedral_cone = [c for c in polyhedral_cone if c != 0 ]
        init_vals = np.einsum("ij,kj->ik", init_vals, proj)
        
        liks, mles = _project_scores(simulated_scores, fisher_information, polyhedral_cone, init_vals, method)
        mles = np.einsum("ik,kj->ij", mles, proj)
        return liks,mles
    else:
        if all(c is None for c in polyhedral_cone):
           # solve analytically
           try:
               fisher_information = check_psd(fisher_information)
           except AssertionError:
               raise Exception("Optimization problem is unbounded and unconstrained")
           mles = np.linalg.solve(fisher_information, simulated_scores.T).T
           liks = np.einsum("ij,ij->i", mles, simulated_scores)
           liks = liks-.5*np.einsum("ij,ij->i", mles, 
                                    np.dot(mles, fisher_information))
           return liks,mles
        
        bounds = []
        for c in polyhedral_cone:
            assert c in (None,-1,1)
            if c == -1:
                bounds += [(None,0)]
            elif c == 1:
                bounds += [(0,None)]
            else:
                bounds += [(None,None)]
        
        assert init_vals.shape == simulated_scores.shape
        def obj(x):
            return -np.dot(z,x) + .5 * np.dot(x, np.dot(fisher_information, x))
        def jac(x):
            return -z + np.dot(fisher_information, x)
        sols = []
        for z,i in zip(simulated_scores,init_vals):
            sols += [scipy.optimize.minimize(obj, i, method=method, jac=jac, bounds=bounds)]
        liks = np.array([-s.fun for s in sols])
        mles = np.array([s.x for s in sols])
        return liks, mles

# def _simulate_log_lik_ratios(n_sims, cone0, coneA, score, score_covariance, fisher_information):
#     """
#     Returns a simulated asymptotic null distribution for the log likelihood
#     ratio.
#     Under the "usual" theory this distribution is known to be chi-square.
#     But for composite or misspecified likelihood, or parameter on the
#     boundary, this will have a nonstandard distribution.
    
#     Input:
#     n_sims : the number of simulated Gaussians
#     cone0, coneA: 
#         the constraints for the null, alternate models,
#         in a small neighborhood around the "truth".
        
#         cone is represented as a list,
#         whose length is the number of parameters,
#         with entries 0,1,-1,None.
#             None: model is unconstrained
#             0: model is fixed at "truth"
#             1: model can be >= "truth"
#             -1: model can be <= "truth"
#     score : the score.
#         Typically 0, but may be non-0 at the boundary
#     score_covariance : the covariance of the score
#         For correct model, this will be Fisher information
#         But this is not true for composite or misspecified likelihood
#     fisher_information : the fisher information
#     """
#     assert len(cone0) == len(score) and len(coneA) == len(score)
#     for c0,cA in zip(cone0,coneA):
#         if not all(c in (0,1,-1,None) for c in (c0,cA)):
#             raise Exception("Invalid cone")
#         if c0 != 0 and cA is not None and c0 != cA:
#             raise Exception("Null and alternative cones not nested")
#     score_sims = np.random.multivariate_normal(score, score_covariance, size=n_sims)
#     null_liks,null_mles = _project_scores(score_sims, fisher_information, cone0)    
#     alt_liks,alt_mles = _project_scores(score_sims, fisher_information, coneA, init_vals=null_mles)
#     return alt_liks, alt_mles, null_liks, null_mles
#     #ret = (1. - np.isclose(alt_liks, null_liks)) * (null_liks - alt_liks)
#     #assert np.all(ret <= 0.0)
#     #return ret
   
# def test_log_lik_ratio(null_lik, alt_lik, n_sims, cone0, coneA, score, score_covariance, fisher_information):
#     if alt_lik > 0 or null_lik > 0:
#         raise Exception("Log likelihoods should be non-positive.")
#     lik_ratio = _trunc_lik_ratio(null_lik, alt_lik)
#     if lik_ratio > 0:
#         raise Exception("Likelihood of full model is less than likelihood of sub-model")
#     alt_lik_distn,_,null_lik_distn,_ = _simulate_log_lik_ratios(n_sims, cone0, coneA, 
#                                                                score, score_covariance, fisher_information)
#     lik_ratio_distn = _trunc_lik_ratio(null_lik_distn, alt_lik_distn)
#     assert np.all(lik_ratio_distn <= 0.0)
#     return tuple(map(np.mean, [lik_ratio > lik_ratio_distn, 
#                                lik_ratio == lik_ratio_distn, 
#                                lik_ratio < lik_ratio_distn]))

# def log_lik_ratio_p(method, n_sims, unconstrained_mle, constrained_mle,
#                     constrained_params, seg_sites, demo_func, **kwargs):
#     unconstrained_mle, constrained_mle = (np.array(x) for x in (unconstrained_mle,
#                                                                 constrained_mle))

#     #sfs_list = seg_sites.sfs_list
#     #combined_sfs = sum_sfs_list(sfs_list)
    
#     log_lik_fun = lambda x: composite_log_likelihood(seg_sites, demo_func(*x))
#     alt_lik = log_lik_fun(unconstrained_mle)
#     null_lik = log_lik_fun(constrained_mle)
    
#     score = autograd.grad(log_lik_fun)(unconstrained_mle)
#     score_cov = observed_score_covariance(method, unconstrained_mle, seg_sites, demo_func)
#     fish = observed_fisher_information(unconstrained_mle, seg_sites, demo_func, assert_psd=False)

#     null_cone = [0 if c else None for c in constrained_params]
#     alt_cone = [None] * len(constrained_params)
    
#     return test_log_lik_ratio(null_lik, alt_lik, n_sims,
#                               null_cone, alt_cone,
#                               score, score_cov, fish)[0]
