
import momi
from momi import expected_sfs, expected_sfs_tensor_prod, sfs_tensor_prod
import pytest
import random
import autograd.numpy as np
import scipy
import scipy.stats
import itertools
import sys

from demo_utils import simple_admixture_demo, random_tree_demo


def check_random_tensor(demo, *args, **kwargs):
    leaves = demo.sampled_pops
    ranges = [list(range(n + 1)) for n in demo.sampled_n]

    config_list = momi.data.configurations.build_full_config_list(demo.sampled_pops, demo.sampled_n)

    esfs = expected_sfs(demo, config_list, *args, **kwargs)

    tensor_components = [np.random.normal(size=(1, n + 1)) for n in demo.sampled_n]
    #tensor_components_list = tuple(v[0,:] for _,v in sorted(tensor_components.iteritems()))

    #prod1 = sfs_tensor_prod(dict(list(zip(config_list,esfs))), tensor_components)
    # sfs = momi.site_freq_spectrum(demo.sampled_pops, [dict(zip((tuple(map(tuple,c)) for c in config_list),
    #                                                           esfs))])
    sfs = momi.site_freq_spectrum(demo.sampled_pops,
                                  [{tuple(map(tuple, c)): s for c, s in zip(config_list, esfs)}])
    #assert sfs.get_dict() == {tuple(map(tuple,c)): s for c,s in zip(config_list, esfs)}
    prod1 = sfs_tensor_prod(sfs, tensor_components)
    # prod1 = sfs_tensor_prod({tuple(map(tuple,c)): s for c,s in zip(config_list, esfs)},
    #                        tensor_components)
    prod2 = expected_sfs_tensor_prod(
        tensor_components, demo, sampled_pops=demo.sampled_pops)

    assert np.allclose(prod1, prod2)


def test_tree_demo_rank1tensor():
    lins_per_pop = 2
    num_leaf_pops = 3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_random_tensor(demo._get_demo(dict(zip(demo.leafs, [lins_per_pop]*num_leaf_pops))))


def test_admixture_demo_rank1tensor():
    demo = simple_admixture_demo()
    check_random_tensor(demo._get_demo({"a":4,"b":5}))
