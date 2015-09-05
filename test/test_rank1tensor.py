from __future__ import division
from momi import make_demography, expected_sfs, expected_sfs_tensor_prod, sfs_tensor_prod
import pytest
import random
import autograd.numpy as np
import scipy, scipy.stats
import itertools
import sys

from test_sims import simple_admixture_demo
from test_normalizing_constant import random_tree_demo

def check_random_tensor(demo, *args, **kwargs):
    leaves = sorted(list(demo.leaves))
    ranges = [range(demo.n_lineages(l)+1) for l in demo.leaves]

    config_list = list(itertools.product(*ranges))
    sfs = expected_sfs(demo, config_list, *args, **kwargs)

    tensor_components = [np.random.normal(size=(1,demo.n_lineages(l)+1)) for l in demo.leaves]
    #tensor_components_list = tuple(v[0,:] for _,v in sorted(tensor_components.iteritems()))

    prod1 = sfs_tensor_prod(dict(zip(config_list,sfs)), tensor_components)
    prod2 = expected_sfs_tensor_prod(tensor_components, demo)
    
    assert np.allclose(prod1, prod2)


def test_tree_demo_rank1tensor():
    lins_per_pop=2
    num_leaf_pops=3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_random_tensor(demo)

def test_admixture_demo_normalization():
    demo = simple_admixture_demo(np.random.normal(size=7), {'1':5,'2':4})

    check_random_tensor(demo)
