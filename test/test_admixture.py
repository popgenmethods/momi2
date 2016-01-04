from __future__ import division
import pytest

import numpy as np

from momi import expected_sfs_tensor_prod, Demography
from demo_utils import simple_admixture_demo
from momi.math_functions import hypergeom_quasi_inverse

class TestDemography(Demography):
    def __init__(self, *args, **kwargs):
        super(TestDemography, self).__init__(*args, **kwargs)

    def n_lineages(self, node):
        if node in self.leaves:
            node = (node,0)
        if node[0] in self.leaves and node[1] == 0:
            return self.G.node[node]['lineages']
        return np.sum(self.n_lineages(l) for l in self.G[node])

def test_pseudoinverse():
    demo_str = simple_admixture_demo().G.graph['cmd']

    # construct from same demo_str so that nodes have same labels
    demo0 = Demography(demo_str)
    demo1 = TestDemography(demo_str)

    p = 20
    vecs = [np.random.normal(size=(p,demo0.n_lineages(l)+1)) for l in sorted(demo0.leaves)]

    vals0, vals1 = [expected_sfs_tensor_prod(vecs, d)
                    for d in (demo0,demo1)]

    assert np.allclose(vals0, vals1)

    n_lins_diff = np.array([demo0.n_lineages(v) - demo1.n_lineages(v) for v in demo0.G])
    assert all(n_lins_diff <= 0)
    assert any(n_lins_diff < 0)

def test_hypergeom_pinv_eye():
    i = np.random.randint(2,50)
    assert np.allclose(hypergeom_quasi_inverse(i,i),
                       np.eye(i+1,i+1))

def test_P():
    t1 = np.random.exponential(.25)
    t2 = np.random.exponential(.25) + t1
    t3 = np.random.exponential(.5) + t2
    p1 = np.random.uniform(0,1)
    p2 = np.random.uniform(0,1)

    i = np.random.choice([0,1])
    j = 1-i
    
    demo0 = Demography("-d 1.0 -n 5 5 -P $t1 0 1 $p1 -P $t2 %d %d $p2 -J $t3 0 1" % (i,j),
                       t1=t1, t2=t2, t3=t3, p1=p1, p2=p2)
    demo1 = Demography("-d 1.0 -n 5 5 -S $t1 0 $p1 -J $t1 2 1 -S $t2 %d $p2 -J $t2 3 %d -J $t3 0 1" % (i,j),
                       t1=t1, t2=t2, t3=t3, p1=p1, p2=p2)

    p = 20
    vecs = [np.random.normal(size=(p,demo0.n_lineages(l)+1)) for l in sorted(demo0.leaves)]

    vals0, vals1 = [expected_sfs_tensor_prod(vecs, d)
                    for d in (demo0,demo1)]
    
    assert np.allclose(vals0, vals1)
    
## TODO:
# implement -P (pulse migration)
# construct random tree demographies with random pulses
