import autograd.numpy as np
import sktensor as skt
from sktensor.tucker import hosvd
import pandas as pd
from util import sum_sfs_list
from math_functions import symmetric_matrix, log_wishart_pdf, slogdet_pos
from sum_product import raw_compute_sfs

def sfs_eval_dirs(sfs, dirs):
    projection = 0.
    ## TODO: vectorize for loop?
    for config,val in sfs.iteritems():
        for leaf,i in zip(sorted(dirs.keys()), config):
            val = val * dirs[leaf][:,i]
        projection = projection + val
    return projection

def get_sfs_tensor(sfs, n_per_pop):
    idx, vals = zip(*(sfs.iteritems()))
    idx = tuple(np.array(x) for x in zip(*idx))
    return skt.sptensor(idx, vals, shape=tuple(n+1 for n in n_per_pop), dtype=np.float)

def greedy_hosvd(sfs_tensor, n_entries, verbose=False):
    U_list = hosvd(sfs_tensor, sfs_tensor.shape, compute_core=False)
    total_energy = sfs_tensor.norm()**2
    curr_entries = [(total_energy, [], [], sfs_tensor)]
    
    for d in range(len(sfs_tensor.shape)):
        prev_entries = curr_entries
        curr_entries = []
        
        for prev_energy, prev_dirs, prev_idxs, prev_tens in prev_entries:

            energy_sum = 0.0
            for next_idx, next_dir in enumerate(U_list[d].T):
                next_tens = prev_tens.ttv((next_dir,) , (0,) )
                try:
                    energy = next_tens.norm()**2
                except AttributeError:
                    energy = next_tens**2
                energy_sum += energy
                curr_entries.append( (energy,
                                      prev_dirs + [next_dir],
                                      prev_idxs + [next_idx],
                                      next_tens))

            curr_entries = sorted(curr_entries, key=lambda x: x[0], reverse=True)[:n_entries]
            assert np.isclose(energy_sum, prev_energy)
    if verbose:
        #print "# Selected components:\n", [idx for _,_,idx,_ in curr_entries]
        to_print = pd.DataFrame([(idx, energy / total_energy) for energy,_,idx,_ in curr_entries],
                                columns=['Component','Percent Energy'])
        print "# Selected components:\n", to_print, "\n# Unselected percent energy:", 1.0-sum(to_print['Percent Energy'])
    return [dirs for _,dirs,_,_ in curr_entries]


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

class PoisGaussSurface(MEstimatorSurface):
    def __init__(self, sfs_list, sfs_directions, theta, demo_func=lambda demo: demo):    
        super(PoisGaussSurface, self).__init__(theta, demo_func)

        self.n_dirs = list(sfs_directions.values())[0].shape[0]
        self.sfs_directions = {l: np.vstack([[1.0]*s.shape[1], s])
                                         for l,s in sfs_directions.iteritems()}
               
        self.n_loci = len(sfs_list)
        
        self.sfs_aggregated = sum_sfs_list(sfs_list)
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
