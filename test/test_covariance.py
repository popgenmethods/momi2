import pytest
import os, random
import autograd.numpy as np

from momi import Demography, simulate_ms, composite_mle_approx_cov
import momi
from test_ms import ms_path

from demo_utils import simple_three_pop_demo, simple_admixture_demo

def check_cov(method, params, demo_func, num_runs, theta):
    true_demo = demo_func(*params)
    seg_sites = momi.seg_sites_from_ms(simulate_ms(ms_path, true_demo,
                                                   num_loci=num_runs, mut_rate=theta,
                                                   additional_ms_params="-r %f 1000" % theta))
    composite_mle_approx_cov(method, params, seg_sites, demo_func)    


def check_jointime_cov(method, num_runs, theta):
    t0 = random.uniform(.25,2.5)
    t1 = t0 + random.uniform(.5,5.0)
    def demo_func(t):
        return simple_three_pop_demo(t,t1)
    check_cov(method, [t0], demo_func, num_runs, theta)
    
def test_jointime_cov_iid():
    check_jointime_cov("iid", 1000, 1.)
    
def test_jointime_cov_series():
    check_jointime_cov("series", 10, 100.)

def check_admixture_cov(method, num_runs, theta):
    def demo_func(*params):
        return simple_admixture_demo(x=np.array(params)).rescaled()
    check_cov(method, np.random.normal(size=7), demo_func, num_runs, theta)

def test_admixture_cov_iid():
    check_admixture_cov("iid", 1000, 1.)

def test_admixture_cov_series():
    check_admixture_cov("series", 10, 100.)    
