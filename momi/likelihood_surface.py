from __future__ import division
from util import make_constant, check_symmetric
from autograd import hessian, grad, hessian_vector_product, jacobian
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

    Depending on whether mutation rate (theta) is specified,
    models SNP counts at each locus as either:
    (1) a Poisson distribution given by mutation rate and the expected SFS
    (2) multinomial distribution, with probabilities given by normalized SFS

    Methods:
    log_likelihood: returns the composite log likelihood
    max_covariance: estimates covariance matrix of composite MLE
    '''
    def __init__(self, sfs_list, theta, demo_func=lambda demo: demo, eps=1e-6, *args, **kwargs):
        '''
        sfs_list: [{config : count}], a list of length num_loci,
                  with each entry a dictionary giving the SFS at the locus
        demo_func: a function that returns demography from the parameters.
                   Default is the identity function (so surface is parametrized
                   by space of demographies).
                   Note derivatives and covariance estimation only work if
                   parameters are array-valued.
        theta: One of the following:
               (a) a number giving the mutation rate at each locus
               (b) a list of numbers giving the mutation rate at each locus
               (c) a function that takes in parameters, and returns (a) or (b)
               (d) None. In this case uses multinomial probability, instead of Poisson Random Field
        eps: add branch_len*eps/num_configs to each SFS entry, to avoid taking log(0) in
             case of underflow
        args, kwargs: additional arguments for compute_sfs (e.g. error model)
        '''
        self.sfs_list = sfs_list
        self.config_list = list(set(sum([sfs.keys() for sfs in sfs_list],[])))
        self.theta = theta
        self.demo_func = demo_func
        self.eps = eps
        self.args, self.kwargs = args, kwargs

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
        demo = self.demo_func(params)
        sfs_vals, branch_len = compute_sfs(demo, self.config_list, *self.args, **self.kwargs)

        # dimensions = (locus,config)
        if vector:
            counts = self.counts_matrix
            loc_dim = ['loc']
        else:
            # sum out the locus dimension
            counts = self.counts_columns
            loc_dim = []

        num_configs = self._num_configs(demo)

        eps = self.eps*branch_len

        if self.theta is not None:
            theta = self._get_theta(params)
            log_fact = self.logfactorial_rows
            if not vector:
                theta, log_fact = np.sum(theta), np.sum(log_fact)
            # Return -theta*branch_length - log_factorial + counts * log(theta*sfs)
            # first multiply theta*sfs along config ('c') dimension
            log_theta_sfs = np.log(einsum2(theta, loc_dim, 
                                           sfs_vals+eps/num_configs, ['c'], loc_dim+['c']))
            # then multiply by counts and sum out config dimension
            return -theta*(branch_len
                           +eps) - log_fact + einsum2(counts, loc_dim+['c'], 
                                                           log_theta_sfs, loc_dim+['c'],
                                                           loc_dim)
        else:
            # Return counts * log(sfs / branch_len)
            return einsum2(counts, loc_dim+['c'],
                           np.log(sfs_vals + eps/num_configs) - np.log(branch_len + eps),
                           ['c'], loc_dim)
            

    def _num_configs(self, demo):
        try:
            self.min_freqs = self.args[1]
        except IndexError:
            try:
                self.min_freqs = self.kwargs['min_freqs']
            except KeyError:
                self.min_freqs = 1
        n_leaf_lins = np.array([demo.n_lineages(l) for l in demo.leaves])
        min_freqs = self.min_freqs * np.ones(len(n_leaf_lins))
        assert np.all(min_freqs > 0) and np.all(min_freqs <= n_leaf_lins)
        return np.prod(n_leaf_lins + 1) - 2 * np.prod(min_freqs)

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

        g_out = self._g_out(true_params)
        #assert g_out == self._g_out_slow(true_params)
        h,g_out = (check_symmetric(_) for _ in (h,g_out))

        h_inv = np.linalg.inv(h)
        h_inv = check_symmetric(h_inv)

        ret = np.dot(h_inv, np.dot(g_out,h_inv))
        return check_symmetric(ret)

    def _get_theta(self, params):
        try:
            theta = self.theta(params)
        except TypeError:
            theta = self.theta
        # make sure there is a theta for each locus
        return theta * np.ones(len(self.sfs_list))

    def _g_out(self, params):
        '''
        Returns einsum("ij,ik", jacobian(params), jacobian(params))
        But in a roundabout way because jacobian implementation is slow
        '''
        def g_out_antihess(x):
            l = self.log_likelihood(x, vector=True)
            lc = make_constant(l)
            return np.sum(0.5 * (l**2 - l*lc - lc*l))
        return hessian(g_out_antihess)(params)

    ## TODO: make unit test of this function
#     def _g_out_slow(self, params):
#         j = jacobian(lambda x: self.log_likelihood(x, vector=True))(params)
#         return np.einsum("ij,ik",j,j)
