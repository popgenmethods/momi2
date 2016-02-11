import momi
from momi import expected_sfs, expected_total_branch_len
import random
import itertools
import networkx as nx

from autograd.numpy import log
import autograd.numpy as np

from demo_utils import simple_admixture_demo, random_tree_demo

def check_demo_normalization(demo, **kwargs):
    leaves = demo.sampled_pops
    ranges = [range(n+1) for n in demo.sampled_n]

    config_list = momi.util._configs_from_derived([np.array(x,dtype=int) for x in itertools.product(*ranges)],
                                                  demo.sampled_n)
    # config_list = np.array(config_list)
    # polymorphic = np.all(np.sum(config_list, axis=1) != 0, axis=1)
    # config_list = config_list[polymorphic,:,:]

    sfs = expected_sfs(demo, config_list, normalized=True, **kwargs)
    assert np.isclose(np.sum(sfs),1.0)

    sfs = expected_sfs(demo, config_list, **kwargs)
    assert np.isclose(np.sum(sfs), expected_total_branch_len(demo, **kwargs))

def test_tree_demo_normalization():
    lins_per_pop=2
    num_leaf_pops=3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_demo_normalization(demo)

def test_admixture_demo_normalization():
    check_demo_normalization(simple_admixture_demo())


def test_admixture_demo_normalization():
    check_demo_normalization(simple_admixture_demo())

   
def test_tree_demo_errors_normalization():
    lins_per_pop=10
    num_leaf_pops=3

    error_matrices = [np.exp(np.random.randn(lins_per_pop+1,lins_per_pop+1)) for _ in range(num_leaf_pops)]
    error_matrices = [np.einsum('ij,j->ij', x, 1./np.sum(x, axis=0)) for x in error_matrices]

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_demo_normalization(demo, error_matrices=error_matrices)
