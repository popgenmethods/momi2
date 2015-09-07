from __future__ import division
import pytest

import numpy as np

from momi import expected_sfs_tensor_prod, expected_tmrca
from demo_utils import simple_admixture_demo, simple_five_pop_demo


def check_tmrca(demo):
    tmrca = expected_tmrca(demo)
    print tmrca
    for i in range(len(demo.leaves)):
        vecs = [np.ones(demo.n_lineages(l)+1) for l in sorted(demo.leaves)]
        vecs[0] = np.arange(len(vecs[0])) / (len(vecs[0]) - 1.0)        
        tmrca2 = expected_sfs_tensor_prod(vecs, demo)
        print tmrca2
        assert np.isclose(tmrca, tmrca2)

def test_five_pop_tmrca():
    check_tmrca(simple_five_pop_demo())

def test_admixture_tmrca():
    check_tmrca(simple_admixture_demo(n_lins=(3,6)))
