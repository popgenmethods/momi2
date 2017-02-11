import momi
from momi import expected_sfs, expected_total_branch_len
import random
import itertools
import networkx as nx

from autograd.numpy import log
import autograd.numpy as np

import scipy, scipy.stats

from demo_utils import simple_admixture_demo, random_tree_demo, simple_admixture_3pop
from test_msprime import ms_path

def check_num_snps(demo, num_loci, mut_rate, ascertainment_pop=None, error_matrices=None):
    demo = demo.demo_hist._get_multipop_moran(demo.pops, demo.n)
    demo = demo.rescaled()
    if error_matrices is not None:
        ## TODO
        raise NotImplementedError

    if ascertainment_pop is None:
        ascertainment_pop = np.array([True] * len(demo.sampled_n))

    seg_sites = momi.simulate_ms(ms_path, demo, num_loci=num_loci, mut_rate=mut_rate)
    sfs = seg_sites.sfs
    n_sites = sfs.n_snps(vector=True)
        
    n_sites_mean = np.mean(n_sites)
    n_sites_sd = np.std(n_sites)

    n_sites_theoretical = momi.expected_total_branch_len(demo, ascertainment_pop=ascertainment_pop, error_matrices=error_matrices) * mut_rate

    zscore = -np.abs(n_sites_mean - n_sites_theoretical) / n_sites_sd
    pval = scipy.stats.norm.cdf(zscore)*2.0

    assert pval >= .05

def test_admixture_3pop_numsnps():
    check_num_snps(simple_admixture_3pop(), 1000.0, 1.0, ascertainment_pop=[True,True,False])
    
def test_tree_demo_numsnps():
    lins_per_pop=10
    num_leaf_pops=3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_num_snps(demo, 1000.0, 1.0)

def test_admixture_demo_numsnps():
    check_num_snps(simple_admixture_demo(), 1000.0, 1.0)
    
def check_demo_normalization(demo, ascertainment_pop=None, add_n=None, error_matrices=True, **kwargs):
    demo = demo.demo_hist._get_multipop_moran(demo.pops, demo.n)
    leaves = demo.sampled_pops

    sampled_n = demo.sampled_n
    if add_n is not None: sampled_n = sampled_n + add_n
    
    config_list = momi.data_structure.full_config_array(demo.sampled_pops, sampled_n, ascertainment_pop)
    config_list = config_list._copy(sampled_n=demo.sampled_n)
    
    sfs = expected_sfs(demo, config_list, normalized=True, **kwargs)
    assert np.isclose(np.sum(sfs),1.0)

    sfs = expected_sfs(demo, config_list, **kwargs)
    assert np.isclose(np.sum(sfs), expected_total_branch_len(demo.copy(sampled_n=sampled_n), ascertainment_pop=ascertainment_pop, **kwargs))
    
    # check sums to 1 even with error matrix
    error_matrices = [np.exp(np.random.randn(n+1,n+1)) for n in demo.sampled_n]
    error_matrices = [np.einsum('ij,j->ij', x, 1./np.sum(x, axis=0)) for x in error_matrices]
    
    sfs = expected_sfs(demo, config_list, normalized=True, error_matrices=error_matrices, **kwargs)
    assert np.isclose(np.sum(sfs), 1.0)

    # check sums to total branch len, even with error matrix
    demo = demo.copy(sampled_n=sampled_n)
    config_list = config_list._copy(sampled_n=sampled_n)
    error_matrices=None
    if error_matrices:
        error_matrices = [np.exp(np.random.randn(n+1,n+1)) for n in demo.sampled_n]
        error_matrices = [np.einsum('ij,j->ij', x, 1./np.sum(x, axis=0)) for x in error_matrices]
    else:
        error_matrices = None
    if ascertainment_pop is None:
        ascertainment_pop = [True]*len(demo.sampled_n)
    ascertainment_pop = np.array(ascertainment_pop)

    sfs = expected_sfs(demo, config_list, error_matrices=error_matrices, **kwargs)
    total_len = expected_total_branch_len(demo, error_matrices=error_matrices, ascertainment_pop=ascertainment_pop)
    assert np.isclose(np.sum(sfs), total_len)    

def test_admixture_3pop_ascertainment():
    check_demo_normalization(simple_admixture_3pop(), ascertainment_pop=[True,True,False], add_n=(-2,0,-1))
    
def test_tree_demo_normalization():
    lins_per_pop=10
    num_leaf_pops=3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_demo_normalization(demo)

def test_admixture_demo_normalization():
    check_demo_normalization(simple_admixture_demo(), error_matrices=False)
