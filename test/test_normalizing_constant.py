import momi
from momi import expected_sfs, expected_total_branch_len
import random
import itertools
import networkx as nx

from autograd.numpy import log
import autograd.numpy as np

import scipy, scipy.stats

from demo_utils import simple_admixture_demo, random_tree_demo, simple_admixture_3pop
from test_ms import ms_path

def check_num_snps(demo, num_loci, mut_rate, ascertainment_pop=None, p_missing=0.0, error_matrices=None):
    demo = demo.rescaled()
    if error_matrices is not None:
        ## TODO
        raise NotImplementedError

    if ascertainment_pop is None:
        ascertainment_pop = np.array([True] * len(demo.sampled_n))
    p_missing = p_missing * np.ones(len(demo.sampled_n))

    seg_sites = momi.simulate_ms(ms_path, demo, num_loci=num_loci, mut_rate=mut_rate)
    sfs_missing = momi.data_structure._randomly_drop_alleles(seg_sites, p_missing, ascertainment_pop=ascertainment_pop).sfs
    n_sites = sfs_missing.n_snps(vector=True)
        
    n_sites_mean = np.mean(n_sites)
    n_sites_sd = np.std(n_sites)

    n_sites_theoretical = momi.expected_total_branch_len(demo, ascertainment_pop=ascertainment_pop, p_missing=p_missing, error_matrices=error_matrices) * mut_rate

    zscore = -np.abs(n_sites_mean - n_sites_theoretical) / n_sites_sd
    pval = scipy.stats.norm.cdf(zscore)*2.0

    assert pval >= .05

def test_admixture_3pop_numsnps():
    check_num_snps(simple_admixture_3pop(), 1000.0, 1.0, ascertainment_pop=[True,True,False], p_missing=[.9,.3,.5])
    
def test_tree_demo_numsnps():
    lins_per_pop=10
    num_leaf_pops=3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_num_snps(demo, 1000.0, 1.0, p_missing=.9)

def test_admixture_demo_numsnps():
    check_num_snps(simple_admixture_demo(), 1000.0, 1.0)
    
def check_demo_normalization(demo, ascertainment_pop=None, add_n=None, p_missing=0.0, error_matrices=True, **kwargs):
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

    # check sums to total branch len, even with error matrix and p_missing
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
    p_missing = p_missing * np.ones(len(demo.sampled_n))

    sfs = expected_sfs(demo, config_list, error_matrices=error_matrices, **kwargs)
    anc_missing = np.prod((p_missing**config_list.value[:,:,0])[:,ascertainment_pop], axis=1)
    der_missing = np.prod((p_missing**config_list[:,:,1])[:,ascertainment_pop], axis=1)
    all_missing = np.prod((p_missing**demo.sampled_n)[ascertainment_pop])
    not_missing = 1.0 - anc_missing - der_missing + all_missing
    total_len = expected_total_branch_len(demo, error_matrices=error_matrices, ascertainment_pop=ascertainment_pop, p_missing=p_missing)
    assert np.isclose(np.sum(sfs * not_missing), total_len)    

def test_admixture_3pop_ascertainment():
    check_demo_normalization(simple_admixture_3pop(), ascertainment_pop=[True,True,False], add_n=(-2,0,-1), p_missing = [.1,.3,.5])
    
def test_tree_demo_normalization():
    lins_per_pop=10
    num_leaf_pops=3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_demo_normalization(demo)

def test_admixture_demo_normalization():
    check_demo_normalization(simple_admixture_demo(), p_missing=[0,.4], error_matrices=False)
