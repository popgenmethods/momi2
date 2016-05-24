import momi
from momi import expected_sfs, expected_total_branch_len
import random
import itertools
import networkx as nx

from autograd.numpy import log
import autograd.numpy as np

from demo_utils import simple_admixture_demo, random_tree_demo, simple_admixture_3pop

def check_demo_normalization(demo, ascertainment_pop=None, add_n=None, **kwargs):
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
    error_matrices = [np.exp(np.random.randn(n+1,n+1)) for n in demo.sampled_n]
    error_matrices = [np.einsum('ij,j->ij', x, 1./np.sum(x, axis=0)) for x in error_matrices]

    sfs = expected_sfs(demo, config_list, error_matrices=error_matrices, **kwargs)
    assert np.isclose(np.sum(sfs), expected_total_branch_len(demo, error_matrices=error_matrices, ascertainment_pop=ascertainment_pop))    

def test_admixture_3pop_ascertainment():
    check_demo_normalization(simple_admixture_3pop(), ascertainment_pop=[True,True,False], add_n=(-2,0,-1))
    
def test_tree_demo_normalization():
    lins_per_pop=10
    num_leaf_pops=3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_demo_normalization(demo)

def test_admixture_demo_normalization():
    check_demo_normalization(simple_admixture_demo())
