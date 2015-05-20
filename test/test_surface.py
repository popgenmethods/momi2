from __future__ import division
from momi import CompositeLogLikelihood, make_demography, aggregate_sfs, simulate_ms, sfs_list_from_ms
import pytest
import random
import autograd.numpy as np
from test_sims import simple_admixture_demo

default_theta = 10.0
default_num_ms_samples = 1000

def test_admixture():
    n = {'1':5,'2':5}
    check_surface_sum(simple_admixture_demo(np.random.normal(size=7),n))

def test_exp_growth():
    n = 10
    growth_rate = random.uniform(-50,50)
    N_bottom = random.uniform(0.1,10.0)
    tau = .01
    demo = make_demography("-I 1 %d -G $0 -eG $1 0.0" % n,
                           growth_rate,
                           tau)
    check_surface_sum(demo)


def test_tree_demo_2():
    n = [4,4]
    demo = make_demography("-I %d %s -ej $0 2 1" % (len(n), " ".join(map(str,n))), 
                           2 * np.random.random() + 0.1)
    check_surface_sum(demo)

def check_surface_sum(demo, theta=default_theta, num_ms_samples=default_num_ms_samples):
    print demo.graph['cmd']

    sfs_list = sfs_list_from_ms(simulate_ms(demo, num_ms_samples, theta=theta),
                                demo.n_at_leaves)
    config_list = sorted(set(sum([sfs.keys() for sfs in sfs_list],[])))

    surface0 = CompositeLogLikelihood(sfs_list, theta=None)
    surface1 = CompositeLogLikelihood(sfs_list, theta=theta)

    for surface in (surface0,surface1):
        assert np.allclose(np.sum(surface.log_likelihood(demo, vector=True)),
                           surface.log_likelihood(demo))
