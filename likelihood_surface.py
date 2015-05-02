from __future__ import division
from util import make_constant, check_symmetric
from autograd import hessian, grad, hessian_vector_product
import autograd.numpy as np
import scipy
from sum_product import compute_sfs
from scipy.stats import norm, chi2
from scipy.optimize import minimize
from math_functions import einsum2

class CompositeLogLikelihood(object):
    '''
    Composite log-likelihood surface, where data
    is a list of frequency spectra at unlinked loci.

    Models SNP counts at each locus as a Poisson
    distribution given by mutation rate and the expected SFS.

    Methods:
    log_likelihood: returns the composite log likelihood
    max_covariance: estimates covariance matrix of composite MLE
    '''
    ## TODO: add EPSILON parameter for underflow issues
    def __init__(self, sfs_list, demo_func=lambda demo: demo, theta=None, theta_func=None):
        '''
        sfs_list: [{config : count}], a list of length num_loci,
                  with each entry a dictionary giving the SFS at the locus
        demo_func: a function that returns demography from the parameters.
                   Default is the identity function (so surface is parametrized
                   by space of demographies).
                   Note derivatives and covariance estimation only work if
                   parameters are array-valued.
        theta: a number or list of numbers, giving mutation rate at each locus.
               Exactly one of theta or theta_func must be specified.
        theta_func: a function that returns mutation rate(s) from the parameters
        '''
        self.sfs_list = sfs_list
        self.config_list = list(set(sum([sfs.keys() for sfs in sfs_list],[])))
        self.demo_func = demo_func

        if (theta is None) == (theta_func is None):
            raise Exception("Exactly one of theta,theta_func should be given")
        if theta_func is None:
            theta_func = lambda params: theta
        # make sure theta_func returns theta for each locus
        self.theta_func = lambda params: theta_func(params) * np.ones(len(self.sfs_list))

        # (i,j)th coordinate = count of config j in locus i
        self.counts_matrix = np.zeros((len(self.sfs_list), len(self.config_list)))
        for i,sfs in enumerate(self.sfs_list):
            for j,config in enumerate(self.config_list):
                try:
                    self.counts_matrix[i,j] = sfs[config]
                except KeyError:
                    pass

        self.logfactorial_rows = np.sum(scipy.special.gammaln(self.counts_matrix+1), axis=1)
        self.counts_columns = einsum2(self.counts_matrix, ['locus','config'], ['config'])

    def log_likelihood(self, params, vector=False):
        '''
        Returns the composite log likelihood at parameter values.
        If vector=True, return vector of composite-log-likelihoods at each locus.
        Otherwise, return the sum of this vector.
        '''
        demo,theta = self.demo_func(params), self.theta_func(params)
        sfs_vals, branch_len = compute_sfs(demo, self.config_list)
        log_fact = self.logfactorial_rows

        # dimensions = (locus,config)
        if vector:
            counts = self.counts_matrix
            loc_dim = ['loc']
        else:
            # sum out the locus dimension
            counts = self.counts_columns
            theta, log_fact = np.sum(theta), np.sum(log_fact)
            loc_dim = []

        # Return -theta*branch_length - log_factorial + counts * log(theta*sfs)
        # first multiply theta*sfs along config ('c') dimension
        log_theta_sfs = np.log(einsum2(theta, loc_dim, sfs_vals, ['c'], loc_dim+['c']))
        # then multiply by counts and sum out config dimension
        return -theta*branch_len - log_fact + einsum2(counts, loc_dim+['c'], 
                                                      log_theta_sfs, loc_dim+['c'],
                                                      loc_dim)

    def max_covariance(self, true_params):
        '''
        Given true parameters (or a reasonable esimate thereof),
        estimates the covariance matrix of the surface maximum.

        Based on theory of M- and Z-estimators,
        which generalizes asymptotic results of MLE to
        more general class of estimators, e.g. the composite MLE.

        Note the theory assumes certain regularity conditions
        which may not hold for arbitrary surfaces
        (e.g., non-identifiable surfaces).
        '''
        h = hessian(self.log_likelihood)(true_params)

        #g_out = einsum("ij,ik",Jacobian,Jacobian)
        ## autograd Jacobian implementation is slow; construct g_out in roundabout way
        def g_out_antihess(param):
            l = self.log_likelihood(param, vector=True)
            lc = make_constant(l)
            return np.sum(0.5 * (l**2 - l*lc - lc*l))
        g_out = hessian(g_out_antihess)(true_params)

        h,g_out = (check_symmetric(_) for _ in (h,g_out))

        h_inv = np.linalg.inv(h)
        h_inv = check_symmetric(h_inv)

        return np.dot(h_inv, np.dot(g_out,h_inv))


def fit_log_likelihood_example(demo_func, num_sims, true_params, init_params, theta=None):
    true_demo = demo_func(true_params)

    print "# Simulating %d trees" % num_sims
    sfs_list = true_demo.simulate_sfs(num_sims, theta=theta)
    if theta is None:
        theta = 1.0
    surface = CompositeLogLikelihood(sfs_list, demo_func, theta=theta)

    # construct the function to minimize, and its gradients
    f = lambda params: -surface.log_likelihood(params)
    g, hp = grad(f), hessian_vector_product(f)
    def f_verbose(params):
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
        sigma_hat = surface.max_covariance(params)
        sigma_hat_inv = scipy.linalg.sqrtm(np.linalg.inv(sigma_hat))
        ## TODO: use t-test instead
        print "# p-values of marginal Z-tests, that params[i]=true_params[i]"
        print "# 2.0 * (1.0 - Normal_cdf(|Sigma_hat^{-1/2}[:,i] * (inferred[i]-truth[i])|))"
        z = np.sqrt(np.sum(sigma_hat_inv * sigma_hat_inv, axis=1) * (inferred_params - true_params)**2)
        print (1.0 - norm.cdf(z)) * 2.0
        print "# Transformed residuals EPS_hat = Sigma_hat^{-1/2} * (inferred - truth)"
        eps_hat = sigma_hat_inv.dot(inferred_params - true_params )
        print eps_hat
        ## TODO: use correct degrees of freedom
        print "# Chi2 test on transformed residuals, for params=true_params"
        print "# <EPS_hat,EPS_hat>, 1-Chi2_cdf(<EPS_hat,EPS_hat>,df=%d)" % len(eps_hat)
        eps_norm = np.sum(eps_hat**2)
        print eps_norm, 1.0 - chi2.cdf(eps_norm, df=len(eps_hat))
