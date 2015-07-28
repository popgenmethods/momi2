from __future__ import division
from util import make_constant, check_symmetric
from autograd import hessian, grad, hessian_vector_product, jacobian
import autograd.numpy as np
import scipy
from sum_product import compute_sfs, raw_compute_sfs
from scipy.stats import norm, chi2
from math_functions import einsum2

class MEstimatorSurface(object):
    def __init__(self, theta, demo_func=lambda demo: demo):
        '''
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
        '''
        self.theta = theta
        self.demo_func = demo_func

    def evaluate(self, params, vector=False):
        pass
        
    def _get_theta(self, params):
        ## TODO: require same theta for all loci, until I figure out how to appropriately compute confidence intervals when using different theta
        try:
            ret = self.theta(params)
        except TypeError:
            ret = self.theta
        return np.array(ret)

    def max_covariance(self, true_params):
        '''
        Given true parameters (or a reasonable esimate thereof),
        estimates the covariance matrix of the surface extremum.
        '''
        h = hessian(self.evaluate)(true_params)

        g_out = self._g_out(true_params)
        #assert g_out == self._g_out_slow(true_params)
        h,g_out = (check_symmetric(_) for _ in (h,g_out))

        h_inv = np.linalg.inv(h)
        h_inv = check_symmetric(h_inv)

        ret = np.dot(h_inv, np.dot(g_out,h_inv))
        return check_symmetric(ret)

    def _g_out(self, params):
        '''
        Returns einsum("ij,ik", jacobian(params), jacobian(params))
        But in a roundabout way because jacobian implementation is slow
        '''
        def g_out_antihess(x):
            l = self.evaluate(x, vector=True)
            lc = make_constant(l)
            return np.sum(0.5 * (l**2 - l*lc - lc*l))
        return hessian(g_out_antihess)(params)

    ## TODO: make unit test of this function
#     def _g_out_slow(self, params):
#         j = jacobian(lambda x: self.evaluate(x, vector=True))(params)
#         return np.einsum("ij,ik",j,j)

class L2ErrorSurface(MEstimatorSurface):
    def __init__(self, sfs_list, sfs_directions, theta, demo_func=lambda demo: demo):    
        super(L2ErrorSurface, self).__init__(theta, demo_func)

        ## TODO: allow for error model? or is this already implicitly in sfs_directions?
        
        self.sfs_directions = sfs_directions
        leaves = sorted(sfs_directions.keys())
        
        self.empirical_projections = [] # indexed by locus
        for sfs in sfs_list: # go thru each locus
            projection = 0.
            ## TODO: vectorize for loop?
            for config,val in sfs.iteritems():
                for leaf,i in zip(leaves, config):
                    val = val * sfs_directions[leaf][:,i]
                projection = projection + val                
            self.empirical_projections.append(projection)

        self.empirical_projections = np.transpose(np.array(self.empirical_projections))
        
    def evaluate(self, params, vector=False):
        demo = self.demo_func(params)
        expectations = raw_compute_sfs(self.sfs_directions, demo)

        ## TODO: allow theta = None
        ## TODO: make this all cleaner
        ## TODO: make this faster for vector=False, by using MSE=Bias**2 + Variance
        theta = self._get_theta(params)
        theta = np.ones(self.empirical_projections.shape[1]) * theta # make sure theta has right dims

        ## TODO: divide by number of loci?
        ret = np.outer(expectations, theta) - self.empirical_projections
        ret = np.sum(ret**2,axis=0)
        if not vector:
            ret = np.sum(ret)
        return ret
        
class NegativeLogLikelihood(MEstimatorSurface):
    '''
    Negative of composite log-likelihood surface, where data
    is a list of frequency spectra at unlinked loci.

    Depending on whether mutation rate (theta) is specified,
    models SNP counts at each locus as either:
    (1) a Poisson distribution given by mutation rate and the expected SFS
    (2) multinomial distribution, with probabilities given by normalized SFS
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
        super(NegativeLogLikelihood, self).__init__(theta, demo_func)
        
        self.sfs_list = sfs_list
        self.config_list = list(set(sum([sfs.keys() for sfs in sfs_list],[])))       
        self.eps = eps
        self.args, self.kwargs = args, kwargs

        # (i,j)th coordinate = at locus i, count of config j
        self.counts_ij = np.zeros((len(self.sfs_list), len(self.config_list)))
        for i,sfs in enumerate(self.sfs_list):
            for j,config in enumerate(self.config_list):
                try:
                    self.counts_ij[i,j] = sfs[config]
                except KeyError:
                    pass
        self.counts_i = np.einsum('ij->i',self.counts_ij) 
        self.counts_j = np.einsum('ij->j',self.counts_ij)

        self.log_fact_num_i = scipy.special.gammaln(self.counts_i+1)
        self.log_fact_denom_i = np.einsum('ij->i',scipy.special.gammaln(self.counts_ij+1))

    def evaluate(self, params, vector=False):
        '''
        Returns the negative composite log likelihood at parameter values.
        If vector=True, return vector of composite-log-likelihoods at each locus.
        Otherwise, return the sum of this vector.
        '''
        demo = self.demo_func(params)
        sfs_vals, branch_len = compute_sfs(demo, self.config_list, *self.args, **self.kwargs)

        num_configs = self._num_configs(demo)

        eps = self.eps*branch_len

        # add on normalization & combinatorial terms
        if self.theta is not None:
            # poisson case
            theta = self._get_theta(params)
            ret = -theta*(branch_len+eps) - self.log_fact_denom_i + self.counts_i * np.log(theta)
        else:
            # multinomial case
            ret = -self.counts_i * np.log(branch_len+eps) - self.log_fact_denom_i + self.log_fact_num_i

        # add on unnormalized sfs entries
        if not vector:
            ret = np.sum(ret)
            counts = self.counts_j
        else:
            counts = self.counts_ij
        ret = ret + np.dot(counts,np.log(sfs_vals + eps/num_configs))

        assert np.all(ret <= 0.0)
        return -ret

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
