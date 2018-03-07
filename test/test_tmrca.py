
import pytest

import numpy as np

from momi import expected_sfs_tensor_prod, expected_tmrca
from demo_utils import simple_admixture_demo, simple_five_pop_demo


def check_tmrca(demo):
    tmrca = expected_tmrca(demo)
    print(tmrca)
    for i in range(len(demo.sampled_pops)):
        vecs = [np.ones(n + 1) for n in demo.sampled_n]
        vecs[0] = np.arange(len(vecs[0])) / (len(vecs[0]) - 1.0)
        tmrca2 = expected_sfs_tensor_prod(
            vecs, demo, sampled_pops=demo.sampled_pops)
        print(tmrca2)
        assert np.isclose(tmrca, tmrca2)


def test_five_pop_tmrca():
    check_tmrca(simple_five_pop_demo()._get_demo(dict(zip(range(1,6), [10]*5))))


def test_admixture_tmrca():
    check_tmrca(simple_admixture_demo()._get_demo({"a":3,"b":6}))
