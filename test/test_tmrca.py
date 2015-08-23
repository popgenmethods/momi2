from __future__ import division
import pytest

import numpy as np

from momi import expected_sfs_tensor_prod, expected_tmrca
from test_sims import simple_admixture_demo
from test_gradient import simple_five_pop_demo

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
    check_tmrca(simple_five_pop_demo(np.random.normal(size=30),
                                     dict(zip("abcde",[1,2,3,4,5]))))

def test_admixture_tmrca():
    check_tmrca(simple_admixture_demo(np.random.normal(size=7),
                                      dict(zip("12", [3,6]))))
