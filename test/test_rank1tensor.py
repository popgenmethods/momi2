from __future__ import division
from momi import make_demography, compute_sfs, raw_compute_sfs, get_sfs_tensor, sfs_eval_dirs
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
    sfs,_ = compute_sfs(demo, config_list, *args, **kwargs)

    tensor_components = {l: np.random.normal(size=(1,demo.n_lineages(l)+1)) for l in demo.leaves}
    #tensor_components_list = tuple(v[0,:] for _,v in sorted(tensor_components.iteritems()))

    #tensor = get_sfs_tensor(dict(zip(config_list,sfs)), [demo.n_lineages(l) for l in leaves])
    #sfs_tensor_prod = tensor.ttv(tensor_components_list)
    sfs_tensor_prod = sfs_eval_dirs(dict(zip(config_list,sfs)), tensor_components)
    
    # sfs_tensor_prod = 0.0
    # for config, val in zip(config_list, sfs):
    #     for i,j in enumerate(config):
    #         val = val * tensor_components[leaves[i]][0,j]
    #     sfs_tensor_prod += val

    sfs_tensor_prod2 = raw_compute_sfs(tensor_components, demo)
    #print sfs_tensor_prod, sfs_tensor_prod2
    
    assert np.allclose(sfs_tensor_prod, sfs_tensor_prod2)


def test_tree_demo_rank1tensor():
    lins_per_pop=2
    num_leaf_pops=3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_random_tensor(demo)

def test_admixture_demo_normalization():
    demo = simple_admixture_demo(np.random.normal(size=7), {'1':5,'2':4})

    check_random_tensor(demo)
