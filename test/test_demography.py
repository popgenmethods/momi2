from __future__ import division
import pytest

import numpy as np

from momi import expected_sfs_tensor_prod, Demography
from demo_utils import simple_admixture_demo
from momi.math_functions import hypergeom_quasi_inverse

class TestDemography(Demography):
    def __init__(self, *args, **kwargs):
        super(TestDemography, self).__init__(*args, **kwargs)

    def _n_at_node(self, node):
        if node[0] in self.sampled_pops and node[1] == 0:
            return self._G.node[node]['lineages']
        return np.sum(self._n_at_node(l) for l in self._G[node])

def test_pseudoinverse():
    demo = simple_admixture_demo()

    # construct from same event_list so that nodes have same labels
    demo0 = Demography(demo.events, demo.sampled_pops, demo.sampled_n, demo.sampled_t, demo.default_N)
    demo1 = TestDemography(demo.events, demo.sampled_pops, demo.sampled_n, demo.sampled_t, demo.default_N)

    p = 20
    vecs = [np.random.normal(size=(p,n+1)) for n in demo0.sampled_n]

    vals0, vals1 = [expected_sfs_tensor_prod(vecs, d)
                    for d in (demo0,demo1)]

    assert np.allclose(vals0, vals1)

    n_lins_diff = np.array([demo0._n_at_node(v) - demo1._n_at_node(v) for v in demo0._G])
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

    root_event = ('-ej',t3,0,1)
    pulse_events0 = [('-ep',t1,0,1,p1),
                     ('-ep',t2,i,j,p2)]
    pulse_events1 = [('-ep',t1,0,'x',p1),('-ej',t1,'x',1),
                     ('-ep',t2,i,'y',p2),('-ej',t2,'y',j)]

    demo0 = Demography(pulse_events0 + [root_event],
                       (0,1), (5,5))
    demo1 = Demography(pulse_events1 + [root_event],
                       (0,1), (5,5))    
    
    p = 20
    vecs = [np.random.normal(size=(p,n+1)) for n in demo0.sampled_n]

    vals0, vals1 = [expected_sfs_tensor_prod(vecs, d)
                    for d in (demo0,demo1)]
    
    assert np.allclose(vals0, vals1)


def test_events_before_sample():
    n_events = 4
    t = [0.0]
    for i in range(n_events):
        t += [np.random.exponential(1./float(n_events)) + t[-1]]
    t = t[1:]
    
    events = [('-ep',t[0],'a','b',np.random.uniform(0,1))]
    
    demo0 = Demography(events + [('-en', 0.0, 'c', 10.0), ('-eg', 0.0, 'c', 1.0),
                                 ('-ej', t[1], 'a','c'),
                                 ('-ej',t[2], 'c', 'b')],
                       sampled_pops=('a','b'), sampled_n=(7,5),
                       sampled_t=(0.,t[3]))

    demo1 = Demography(events + [('-en',t[1],'a',10.0*np.exp(-t[1])), ('-eg',t[1],'a',1.0),
                                 ('-ej', t[2], 'a','b')],
                       sampled_pops=('a','b'), sampled_n=(7,5),
                       sampled_t=(0.,t[3]))                      
    
    vecs = [np.random.normal(size=(10,n+1)) for n in demo0.sampled_n]
    val0,val1 = [expected_sfs_tensor_prod(vecs, d) for d in (demo0,demo1)]

    assert np.allclose(val0,val1)

def test_time_scale():
    n_events = 3
    t = [0.0]
    for i in range(n_events):
        t += [np.random.exponential(1./float(n_events)) + t[-1]]
    t = t[1:]

    demo0 = Demography([('-en',t[0],'a',.3),
                        ('-eg',t[0],'a',0.5),
                        ('-ej',t[2],'a','b')],
                       sampled_pops=('a','b'), sampled_n=(7,5),
                       sampled_t=(0.,t[1]),
                       time_scale='ms')

    demo1 = Demography([('-en',2.0*t[0],'a',.3),
                        ('-eg',2.0*t[0],'a',0.5/2.0),
                        ('-ej',2.0*t[2],'a','b')],
                       sampled_pops=('a','b'), sampled_n=(7,5),
                       sampled_t=(0.,2.0*t[1]),
                       time_scale='standard')

    vecs = [np.random.normal(size=(10,n+1)) for n in demo0.sampled_n]
    val0,val1 = [expected_sfs_tensor_prod(vecs, d) for d in (demo0,demo1)]

    assert np.allclose(val0,val1/2.)
    
