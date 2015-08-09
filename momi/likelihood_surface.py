from __future__ import division
from util import make_constant, check_symmetric, aggregate_sfs
from autograd import hessian, grad, hessian_vector_product, jacobian
import autograd.numpy as np
import scipy
from sum_product import compute_sfs, raw_compute_sfs
from scipy.stats import norm, chi2
from math_functions import einsum2, symmetric_matrix, log_wishart_pdf, slogdet_pos
from tensor import sfs_eval_dirs

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

class PoisGaussSurface(MEstimatorSurface):
    def __init__(self, sfs_list, sfs_directions, theta, demo_func=lambda demo: demo):    
        super(PoisGaussSurface, self).__init__(theta, demo_func)

        self.n_dirs = list(sfs_directions.values())[0].shape[0]
        self.sfs_directions = {l: np.vstack([[1.0]*s.shape[1], s])
                                         for l,s in sfs_directions.iteritems()}
               
        self.n_loci = len(sfs_list)
        
        self.sfs_aggregated = aggregate_sfs(sfs_list)
        leaves = sfs_directions.keys()
        projection = sfs_eval_dirs(self.sfs_aggregated, self.sfs_directions)

        self.num_muts, projection = projection[0], projection[1:]
        self.means = projection / self.num_muts

    def inv_cov_mat(self, demo, branch_len, expectations):
        pass

    def evaluate(self, params, vector=False):
        if vector:
            raise Exception("Vectorized likelihood not implemented")
        
        demo = self.demo_func(params)        

        expectations = raw_compute_sfs(self.sfs_directions, demo)
        branch_len, expectations = expectations[0], expectations[1:]
        expectations = expectations / branch_len
        
        theta = self._get_theta(params)
        theta = np.ones(self.n_loci) * theta
        theta = np.sum(theta)

        resids =  expectations - self.means
        Sigma_inv = self.inv_cov_mat(demo, branch_len, expectations)

        return self.neg_log_lik(theta * branch_len, self.num_muts, resids, Sigma_inv)

    def neg_log_lik(self, expected_snps, n_snps, resids, Sigma_inv):
        return expected_snps - n_snps * np.log(expected_snps) + 0.5 * n_snps * np.dot(resids, np.dot(Sigma_inv, resids)) - 0.5 * slogdet_pos(Sigma_inv * n_snps)
    
def get_cross_dirs(sfs_directions, n_dirs):
    cross_dirs = {}
    for leaf,dirs in sfs_directions.iteritems():
        assert n_dirs == dirs.shape[0]

        idx0, idx1 = np.triu_indices(n_dirs)
        cross_dirs[leaf] = np.einsum('ik,jk->ijk',
                                     dirs, dirs)[idx0,idx1,:]
    return cross_dirs
    
class PGSurface_Empirical(PoisGaussSurface):
    def __init__(self, sfs_list, sfs_directions, theta, demo_func=lambda demo: demo):    
        super(PGSurface_Empirical, self).__init__(sfs_list, sfs_directions, theta, demo_func)
       
        cross_means = sfs_eval_dirs(self.sfs_aggregated, get_cross_dirs(sfs_directions, self.n_dirs)) / self.num_muts
        cross_means = symmetric_matrix(cross_means, self.n_dirs)
        
        cov_mat = cross_means - np.outer(self.means, self.means)

        self.Sigma_inv = np.linalg.inv(cov_mat)

    def inv_cov_mat(self, demo, branch_len, means):
        return self.Sigma_inv

class PGSurface_Diag(PoisGaussSurface):
    def __init__(self, sfs_list, sfs_directions, theta, demo_func=lambda demo: demo):    
        super(PGSurface_Diag, self).__init__(sfs_list, sfs_directions, theta, demo_func)
       
        self.square_sfs_dirs = {l: s**2 for l,s in sfs_directions.iteritems()}
       
    def inv_cov_mat(self, demo, branch_len, means):
        return np.diag(1./ (raw_compute_sfs(self.square_sfs_dirs, demo) / branch_len - means**2))

class PGSurface_Exact(PoisGaussSurface):
    def __init__(self, sfs_list, sfs_directions, theta, demo_func=lambda demo: demo):    
        super(PGSurface_Exact, self).__init__(sfs_list, sfs_directions, theta, demo_func)

        self.cross_dirs = get_cross_dirs(sfs_directions, self.n_dirs)
        cross_means = sfs_eval_dirs(self.sfs_aggregated, self.cross_dirs) / self.num_muts
        cross_means = symmetric_matrix(cross_means, self.n_dirs)
        
        self.empirical_covariance = cross_means - np.outer(self.means, self.means)

    def inv_cov_mat(self, demo, branch_len, means):
        cross_means = raw_compute_sfs(self.cross_dirs, demo) / branch_len
        cross_means = symmetric_matrix(cross_means, self.n_dirs)
        
        return np.linalg.inv(cross_means - np.outer(means, means))
    
class PoissonWishartSurface(PGSurface_Exact):
    def neg_log_lik(self, expected_snps, n_snps, resids, Sigma_inv):
        ret = super(PoissonWishartSurface, self).neg_log_lik(expected_snps, n_snps, resids, Sigma_inv)
        return ret - log_wishart_pdf(self.empirical_covariance * self.num_muts,
                                     np.linalg.inv(Sigma_inv), self.num_muts-1, self.n_dirs)
