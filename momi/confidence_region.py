from .compute_sfs import expected_sfs
from .likelihood import _composite_log_likelihood
from .util import memoize_instance, make_constant, check_psd
from .math_functions import inv_psd
import scipy
import scipy.stats
import autograd
import autograd.numpy as np

# stuff for confidence intervals


class _ConfidenceRegion(object):
    def __init__(self, point, score, score_cov, fisher, lik_fun, psd_rtol=1e-8):
        self.point = point
        self.score = score
        self.score_cov = check_psd(score_cov, tol=psd_rtol)
        self.fisher = fisher
        self.lik_fun = lik_fun
        self.psd_rtol = psd_rtol

    @memoize_instance
    def godambe(self, inverse=False):
        """
        Returns Godambe Information.
        If the true params are in the interior of the parameter space,
        the composite MLE will be approximately Gaussian with mean 0,
        and covariance given by the Godambe information.
        """
        fisher_inv = inv_psd(self.fisher, tol=self.psd_rtol)
        ret = check_psd(np.dot(fisher_inv, np.dot(
            self.score_cov, fisher_inv)), tol=self.psd_rtol)
        if not inverse:
            ret = inv_psd(ret, tol=self.psd_rtol)
        return ret

    def wald_intervals(self, lower=.025, upper=.975):
        """
        Marginal wald-type confidence intervals.
        """
        conf_lower, conf_upper = scipy.stats.norm.interval(.95,
                                                           loc=self.point,
                                                           scale=np.sqrt(np.diag(self.godambe(inverse=True))))
        return np.array([conf_lower, conf_upper]).T

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
            null_cone = [0] * null_point.shape[1]
        null_cone = np.array(null_cone, ndmin=2)

        if alt_point is None:
            alt_point = self.point
        alt_point = np.array(alt_point, ndmin=2)

        if alt_cone is None:
            alt_cone = [None] * null_point.shape[1]
        alt_cone = np.array(alt_cone, ndmin=2)

        b = np.broadcast_arrays(null_point, null_cone, alt_point, alt_cone)
        try:
            assert all(bb.shape[1:] == (len(self.point),) for bb in b)
        except AssertionError:
            raise ValueError("points, cones have incompatible shapes")
        b = [list(map(tuple, x)) for x in b]
        null_point, null_cone, alt_point, alt_cone = b

        if test_type == "ratio":
            sims = np.random.multivariate_normal(
                self.score, self.score_cov, size=sims)

            liks = {}
            for p in list(null_point) + list(alt_point):
                if p not in liks:
                    liks[p] = self.lik_fun(np.array(p))

            sim_mls = {}
            for nc, ac in zip(null_cone, alt_cone):
                if (nc, ac) not in sim_mls:
                    nml, nmle = _project_scores(
                        sims, self.fisher, nc, psd_rtol=self.psd_rtol)
                    aml, amle = _project_scores(
                        sims, self.fisher, ac, psd_rtol=self.psd_rtol, init_vals=nmle)
                    sim_mls[(nc, ac)] = (nml, aml)

            ret = []
            for n_p, n_c, a_p, a_c in zip(null_point, null_cone, alt_point, alt_cone):
                lr = _trunc_lik_ratio(liks[n_p], liks[a_p])
                lr_distn = _trunc_lik_ratio(*sim_mls[(n_c, a_c)])
                ret += [list(map(np.mean, [lr > lr_distn,
                                           lr == lr_distn,
                                           lr < lr_distn]))]
            ret = np.array(ret)
        elif test_type == "wald":
            if np.any(np.array(null_cone) != 0) or any(a_c != tuple([None] * len(self.point)) for a_c in alt_cone):
                raise NotImplementedError(
                    "Only simple tests implemented for wald")

            gdmb = self.godambe(inverse=False)

            resids = np.array(alt_point) - np.array(null_point)
            ret = np.einsum("ij,ij->i", resids,
                            np.dot(resids, gdmb))
            ret = 1. - scipy.stats.chi2.cdf(ret, df=len(self.point))
            ret = np.array([ret, [0] * len(ret), 1. - ret]).T
        else:
            raise NotImplementedError("%s tests not implemented" % test_type)

        if p_only:
            ret = ret[:, 0]
        if len(in_shape) == 1:
            ret = np.squeeze(ret)
        return ret


class ConfidenceRegion(_ConfidenceRegion):
    """
    Constructs asymptotic confidence regions and hypothesis tests,
    using the Limit of Experiments theory.
    """

    def __init__(self, point_estimate, demo_func, data, mut_rate=None, length=1, regime="long", psd_rtol=1e-8, **kwargs):
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
        psd_rtol: for checking if certain matrices (e.g. covariance matrices) are positive semidefinite
              if psd_rtol = epsilon, then we will consider a matrix positive semidefinite if its most
              negative eigenvalue has magnitude less than epsilon * most positive eigenvalue.
        **kwargs : additional arguments passed into composite_log_likelihood
        """
        if regime not in ("long", "many"):
            raise ValueError("Unrecognized regime '%s'" % regime)

        try:
            data = data.seg_sites
        except AttributeError:
            data = data

        if mut_rate is not None:
            mut_rate = mut_rate * length

        self.point = np.array(point_estimate)
        self.demo_func = demo_func
        self.data = data
        self.regime = regime
        self.kwargs = dict(kwargs)
        self.psd_rtol = psd_rtol

        self.score = autograd.grad(self.lik_fun)(self.point)
        self.score_cov = _observed_score_covariance(
            self.regime, self.point, self.data,
            self.demo_func, psd_rtol=self.psd_rtol, mut_rate=mut_rate,
            **self.kwargs)
        self.fisher = _observed_fisher_information(
            self.point, self.data, self.demo_func,
            psd_rtol=self.psd_rtol, assert_psd=False,
            mut_rate=mut_rate, **self.kwargs)

    def lik_fun(self, params, vector=False):
        """Returns composite log likelihood from params"""
        return _composite_log_likelihood(self.data, self.demo_func(*params), vector=vector, **self.kwargs)


def _trunc_lik_ratio(null, alt):
    return (1 - np.isclose(alt, null)) * (null - alt)


def _observed_fisher_information(params, data, demo_func, psd_rtol, assert_psd=True, **kwargs):
    params = np.array(params)
    f = lambda x: _composite_log_likelihood(data, demo_func(*x), **kwargs)
    ret = -autograd.hessian(f)(params)
    if assert_psd:
        try:
            ret = check_psd(ret, tol=psd_rtol)
        except AssertionError:
            raise Exception(
                "Observed Fisher Information is not PSD (either due to numerical instability, or because the parameters are not a local maxima in the interior)")
    return ret


def _observed_score_covariance(method, params, seg_sites, demo_func, psd_rtol, **kwargs):
    if method == "long":
        if "mut_rate" in kwargs:
            raise NotImplementedError(
                "'long' godambe method not implemented for Poisson approximation")
        ret = _long_score_cov(params, seg_sites, demo_func, **kwargs)
    elif method == "many":
        ret = _many_score_cov(params, seg_sites, demo_func, **kwargs)
    else:
        raise Exception("Unrecognized method")

    try:
        ret = check_psd(ret, tol=psd_rtol)
    except AssertionError:
        raise Exception("Numerical instability: score covariance is not PSD")
    return ret


def _many_score_cov(params, data, demo_func, **kwargs):
    params = np.array(params)

    def f_vec(x):
        ret = _composite_log_likelihood(
            data, demo_func(*x), vector=True, **kwargs)
        # centralize
        return ret - np.mean(ret)

    # g_out = einsum('ij,ik', jacobian(f_vec)(params), jacobian(f_vec)(params))
    # but computed in a roundabout way because jacobian implementation is slow
    def _g_out_antihess(x):
        l = f_vec(x)
        lc = make_constant(l)
        return np.sum(0.5 * (l**2 - l * lc - lc * l))
    return autograd.hessian(_g_out_antihess)(params)


def _long_score_cov(params, seg_sites, demo_func, **kwargs):
    if "mut_rate" in kwargs:
        raise NotImplementedError(
            "Currently only implemented for multinomial composite likelihood")
    params = np.array(params)

    configs = seg_sites.sfs.configs
    #_,snp_counts = seg_sites.sfs._idxs_counts(None)
    snp_counts = seg_sites.sfs._total_freqs
    weights = snp_counts / float(np.sum(snp_counts))

    def snp_log_probs(x):
        ret = np.log(expected_sfs(
            demo_func(*x), configs, normalized=True, **kwargs))
        return ret - np.sum(weights * ret)  # subtract off mean

    # g_out = sum(autocov(einsum("ij,ik->ikj",jacobian(idx_series), jacobian(idx_series))))
    # computed in roundabout way, in case jacobian is slow for many snps
    # autocovariance is truncated at sqrt(len(idx_series)), to avoid
    # statistical/numerical issues
    def g_out_antihess(y):
        lp = snp_log_probs(y)
        ret = 0.0
        for l in seg_sites._get_likelihood_sequences(lp):
            L = len(l)
            lc = make_constant(l)

            fft = np.fft.fft(l)
            # (assumes l is REAL)
            assert np.all(np.imag(l) == 0.0)
            fft_rev = np.conj(fft) * np.exp(2 * np.pi *
                                            1j * np.arange(L) / float(L))

            curr = 0.5 * (fft * fft_rev - fft *
                          make_constant(fft_rev) - make_constant(fft) * fft_rev)
            curr = np.fft.ifft(curr)[(L - 1)::-1]

            # make real
            assert np.allclose(np.imag(curr / L), 0.0)
            curr = np.real(curr)
            curr = curr[0] + 2.0 * np.sum(curr[1:int(np.sqrt(L))])
            ret = ret + curr
        return ret
    g_out = autograd.hessian(g_out_antihess)(params)
    g_out = 0.5 * (g_out + np.transpose(g_out))
    return g_out


def _project_scores(simulated_scores, fisher_information, polyhedral_cone, psd_rtol, init_vals=None, method="tnc"):
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
        proj = np.eye(len(polyhedral_cone))[~fixed_params, :]
        fisher_information = np.dot(proj, np.dot(fisher_information, proj.T))
        simulated_scores = np.einsum("ij,kj->ik", simulated_scores, proj)
        polyhedral_cone = [c for c in polyhedral_cone if c != 0]
        init_vals = np.einsum("ij,kj->ik", init_vals, proj)

        liks, mles = _project_scores(
            simulated_scores, fisher_information, polyhedral_cone, psd_rtol, init_vals, method)
        mles = np.einsum("ik,kj->ij", mles, proj)
        return liks, mles
    else:
        if all(c is None for c in polyhedral_cone):
            # solve analytically
            try:
                fisher_information = check_psd(
                    fisher_information, tol=psd_rtol)
            except AssertionError:
                raise Exception(
                    "Fisher information is not PSD (optimization problem is unbounded and unconstrained)")
            mles = np.linalg.solve(fisher_information, simulated_scores.T).T
            liks = np.einsum("ij,ij->i", mles, simulated_scores)
            liks = liks - .5 * np.einsum("ij,ij->i", mles,
                                         np.dot(mles, fisher_information))
            return liks, mles

        bounds = []
        for c in polyhedral_cone:
            assert c in (None, -1, 1)
            if c == -1:
                bounds += [(None, 0)]
            elif c == 1:
                bounds += [(0, None)]
            else:
                bounds += [(None, None)]

        assert init_vals.shape == simulated_scores.shape

        def obj(x):
            return -np.dot(z, x) + .5 * np.dot(x, np.dot(fisher_information, x))

        def jac(x):
            return -z + np.dot(fisher_information, x)
        sols = []
        for z, i in zip(simulated_scores, init_vals):
            sols += [scipy.optimize.minimize(obj, i,
                                             method=method, jac=jac, bounds=bounds)]
        liks = np.array([-s.fun for s in sols])
        mles = np.array([s.x for s in sols])
        return liks, mles
