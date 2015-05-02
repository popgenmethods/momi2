from __future__ import division
from util import aggregate_sfs, make_constant
from autograd import hessian, grad, hessian_vector_product
import autograd.numpy as np
import scipy
from sum_product import compute_sfs
from scipy.stats import norm, chi2
from scipy.optimize import minimize

## TODO: rename to LogLikelihoodSurface
class LogLikelihoodPRF(object):
    ## TODO: add EPSILON parameter for underflow issues
    ## TODO: add comments/documentation
    def __init__(self, get_demo_theta, sfs_list):
        self.raw_get_demo_theta = get_demo_theta
        self.sfs_list = sfs_list
        self.sfs_agg = aggregate_sfs(sfs_list)

        self.config_list, self.counts_agg = zip(*sorted(self.sfs_agg.iteritems()))
        self.counts_agg = np.array(self.counts_agg)
        self.factorials_agg = scipy.special.gammaln(self.counts_agg + 1)

        # (i,j)th coordinate = count of config j in dataset i
        self.counts = np.zeros((len(self.sfs_list), len(self.config_list)))
        for i,sfs in enumerate(self.sfs_list):
            for j,config in enumerate(self.config_list):
                self.counts[i,j] = sfs[config]

        self.factorials_mat = scipy.special.gammaln(self.counts+1)

        # mean across datasets, outer product of the gradient of log-likelihood
        self._E_grad_outer = hessian(self._pre_outer_gradient)
        # mean across datasets, hessian of the log-likelihood
        self._E_hess = hessian(lambda x: self.log_likelihood(x,aggregate=False) / len(self.sfs_list))

    def get_demo_theta(self, x):
        demo,theta = self.raw_get_demo_theta(x)
        if theta is None:
            theta = 1.0
        theta = theta * np.ones(shape=(len(self.sfs_list),))
        assert len(theta) == len(self.sfs_list)
        return demo,theta

    ## TODO: explain what aggregate does in comment
    def log_likelihood(self, x, aggregate=True):
        if not aggregate:
            return np.sum(self.log_likelihood_vector(x))

        demo,theta = self.get_demo_theta(x)
        theta = np.sum(theta)

        sfs_vals, branch_len = compute_sfs(demo, self.config_list)
        ret = -branch_len * theta + np.sum(np.log(sfs_vals * theta) * self.counts_agg - self.factorials_agg)

        assert ret < 0.0
        return ret

    def log_likelihood_vector(self, x):
        demo,theta = self.get_demo_theta(x)

        sfs_vals, branch_len = compute_sfs(demo, self.config_list)
        return -branch_len * theta + np.sum(np.log(np.outer(theta, sfs_vals)) * self.counts 
                                            - self.factorials_mat, axis=1)
    def mle_Sigma_hat(self, true_x):
        g_out = self._E_grad_outer(true_x)
        h = self._E_hess(true_x)

        # g_out,h should be symmetric
        assert np.allclose(h, h.transpose()) and np.allclose(g_out, g_out.transpose())
        g_out,h = (g_out + g_out.transpose())/2, (h + h.transpose())/2

        h_inv = np.linalg.inv(h)
        #h_inv should be symmetric
        assert np.allclose(h_inv, h_inv.transpose())
        h_inv = (h_inv + h_inv.transpose()) / 2

        return h_inv.dot(g_out.dot(h_inv)) / len(self.sfs_list)

    def _pre_outer_gradient(self, x):
        l = self.log_likelihood_vector(x)
        lc = make_constant(l)
        return np.sum(0.5 * (l**2 - l*lc - lc*l)) / len(self.sfs_list)


def fit_log_likelihood_example(get_demo_theta, num_sims, true_x, init_x):
    true_demo,theta = get_demo_theta(true_x)

    print "# Simulating %d trees" % num_sims
    sfs_list = true_demo.simulate_sfs(num_sims, theta=theta)
    log_lik_surface = LogLikelihoodPRF(get_demo_theta, sfs_list)

    f = lambda x: -log_lik_surface.log_likelihood(x)
    g, hp = grad(f), hessian_vector_product(f)
    def f_verbose(x):
        print (x - true_x) / true_x
        return f(x)    

    init_x = np.random.normal(size=len(true_x))
    print "# Start point:"
    print init_x
    print "# Performing optimization"
    optimize_res = minimize(f_verbose, init_x, jac=g, hessp=hp, method='newton-cg')
    print optimize_res
    
    inferred_x = optimize_res.x
    error = (true_x - inferred_x) / true_x
    print "# Max Percent Error: %f" % max(abs(error))
    print "# Percent Error:","\n",error
    print "# True params:", "\n", true_x
    print "# Inferred params:", "\n", inferred_x   

    for x,xname in ((true_x,"TRUTH"), (inferred_x,"PLUGIN")):
        print "\n\n**** Estimating Sigma_hat at %s" % xname

        sigma_hat = log_lik_surface.mle_Sigma_hat(x)
        print "# Z = Sigma_hat^{-1/2} * (inferred - truth)"
        z = scipy.linalg.sqrtm(np.linalg.inv(sigma_hat)).dot( inferred_x - true_x )
        print z
        print "# (1 - Normal_cdf(|Z|)) * 2.0"
        p = (1.0 - norm.cdf(np.abs(z))) * 2.0
        print p
        print "# <Z,Z>, Chi2_cdf(<Z,Z>,df=%d)" % len(z)
        znorm = np.sum(z**2)
        print znorm, chi2.cdf(znorm, df=len(z))
