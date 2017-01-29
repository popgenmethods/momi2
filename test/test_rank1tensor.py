
import momi
from momi import expected_sfs, expected_sfs_tensor_prod, sfs_tensor_prod
import pytest
import random
import autograd.numpy as np
import scipy, scipy.stats
import itertools
import sys

from demo_utils import simple_admixture_demo, random_tree_demo

def check_random_tensor(demo, *args, **kwargs):
    leaves = demo.pops
    ranges = [list(range(n+1)) for n in demo.n]

    #config_list = momi.data_structure._configs_from_derived(list(itertools.product(*ranges)),
    #                                                        demo.sampled_n, demo.sampled_pops)
    #config_list = momi.config_array(demo.sampled_pops,
    #                                list(itertools.product(*ranges)),
    #                                demo.sampled_n)
    config_list = momi.data_structure.full_config_array(demo.pops, demo.n)
    
    esfs = expected_sfs(demo.demo_hist, config_list, *args, **kwargs)

    tensor_components = [np.random.normal(size=(1,n+1)) for n in demo.n]
    #tensor_components_list = tuple(v[0,:] for _,v in sorted(tensor_components.iteritems()))

    #prod1 = sfs_tensor_prod(dict(list(zip(config_list,esfs))), tensor_components)
    #sfs = momi.site_freq_spectrum(demo.sampled_pops, [dict(zip((tuple(map(tuple,c)) for c in config_list),
    #                                                           esfs))])
    sfs = momi.site_freq_spectrum(demo.pops,
                                  [{tuple(map(tuple,c)): s for c,s in zip(config_list, esfs)}])
    #assert sfs.get_dict() == {tuple(map(tuple,c)): s for c,s in zip(config_list, esfs)}
    prod1 = sfs_tensor_prod(sfs, tensor_components)
    #prod1 = sfs_tensor_prod({tuple(map(tuple,c)): s for c,s in zip(config_list, esfs)},
    #                        tensor_components)
    prod2 = expected_sfs_tensor_prod(tensor_components, demo.demo_hist, sampled_pops=demo.pops)
    
    assert np.allclose(prod1, prod2)


def test_tree_demo_rank1tensor():
    lins_per_pop=2
    num_leaf_pops=3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_random_tensor(demo)

def test_admixture_demo_rank1tensor():
    demo = simple_admixture_demo(n_lins=(5,4))
    check_random_tensor(demo)
