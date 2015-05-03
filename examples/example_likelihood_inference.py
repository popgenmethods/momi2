from __future__ import division

import scipy
from scipy.stats import norm,chi2
from scipy.optimize import minimize

import autograd.numpy as np
from autograd.numpy import log,exp,dot
from autograd import grad, hessian_vector_product

from util import check_symmetric
from sum_product import compute_sfs
from likelihood_surface import CompositeLogLikelihood

from example_demographies import vectorized_demo_func, simple_human_demo, example_admixture_demo

## TODO: rename file to example_inference.py

def fit_log_likelihood_example(demo_func, num_loci, true_params, init_params, theta=None):
    '''
    Simulate SFS for a demography, then estimate the demography via 
    search over composite likelihood surface using Newton method.

    demo_func: a function that takes a numpy array and returns a demography
    num_loci: number of unlinked loci to simulate
    true_params: true parameters
    init_params: parameters to start gradient descent
    theta: mutation rate per locus. If None, uses the branch lengths ala fastsimcoal.
    '''
    true_demo = demo_func(true_params)

    print "# Simulating %d trees" % num_loci
    sfs_list = true_demo.simulate_sfs(num_loci, theta=theta)
    if theta is None:
        theta = 1.0
    surface = CompositeLogLikelihood(sfs_list, demo_func, theta=theta)

    # construct the function to minimize, and its derivatives
    f = lambda params: -surface.log_likelihood(params)
    g, hp = grad(f), hessian_vector_product(f)
    def f_verbose(params):
        # for verbose output during the gradient descent
        print (params - true_params) / true_params
        return f(params)

    print "# Start point:"
    print init_params
    print "# Performing optimization. Printing relative error."
    optimize_res = minimize(f_verbose, init_params, jac=g, hessp=hp, method='newton-cg')
    print optimize_res
    
    inferred_params = optimize_res.x
    error = (true_params - inferred_params) / true_params
    print "# Max Relative Error: %f" % max(abs(error))
    print "# Relative Error:","\n", error
    print "# True params:", "\n", true_params
    print "# Inferred params:", "\n", inferred_params   

    for params,param_name in ((true_params,"TRUTH"), (inferred_params,"PLUGIN")):
        print "\n\n**** Estimating Sigma_hat at %s" % param_name
        sigma = surface.max_covariance(params)

        # recommend to call check_symmetric on matrix inverse and square root
        # linear algebra routines may not preserve symmetry due to numerical errors
        sigma_inv = check_symmetric(np.linalg.inv(sigma))
        sigma_inv_root = check_symmetric(scipy.linalg.sqrtm(sigma_inv))

        print "# Estimated standard deviation of inferred[i] - truth[i]"
        sd = np.sqrt(np.diag(sigma))
        print sd
        ## TODO: use t-test instead
        print "# p-value of Z-test that params[i]=true_params[i]"
        z = (inferred_params - true_params) / sd
        print (1.0 - norm.cdf(np.abs(z))) * 2.0
        print "# Transformed residuals EPS_hat = Sigma_hat^{-1/2} * (inferred - truth)"
        eps_hat = sigma_inv_root.dot(inferred_params - true_params )
        print eps_hat
        ## TODO: use correct degrees of freedom
        print "# Chi2 test for params=true_params, using transformed residuals"
        print "# <EPS_hat,EPS_hat>, 1-Chi2_cdf(<EPS_hat,EPS_hat>,df=%d)" % len(eps_hat)
        eps_norm = np.sum(eps_hat**2)
        print eps_norm, 1.0 - chi2.cdf(eps_norm, df=len(eps_hat))

## TODO: use an example that works
fit_log_likelihood_example(vectorized_demo_func(simple_human_demo, [5] * 3),
                           num_loci=10000,
                           true_params = np.random.normal(size=8),
                           init_params = np.random.normal(size=8))
# fit_log_likelihood_example(example_admixture_demo,
#                            num_loci=100,
#                            true_params = np.random.normal(size=8),
#                            init_params = np.random.normal(size=8))
