import pytest
import os, random
import autograd.numpy as np

from momi import simulate_ms
import momi
from test_ms import scrm_path

from demo_utils import simple_three_pop_demo, simple_nea_admixture_demo

import scipy

def check_cov(method, params, demo_func, num_runs, theta, bounds=None, subsample_inds=False, **kwargs):
    true_demo = demo_func(*params)
    seg_sites = simulate_ms(scrm_path, true_demo,
                            num_loci=num_runs, mut_rate=theta,
                            additional_ms_params="-r %f 1000" % theta)
    if subsample_inds:
        seg_sites = seg_sites.subsample_inds(subsample_inds)
        old_demo_func = demo_func
        demo_func = lambda *x: old_demo_func(*x, sampled_n = seg_sites.sampled_n)
    
    cmle_search_res = momi.SfsLikelihoodSurface(seg_sites, demo_func).find_mle(params, options={'maxiter':1000}, bounds=bounds, **kwargs)
    est_params = cmle_search_res.x

    cr = momi.ConfidenceRegion(est_params, demo_func, seg_sites, regime=method, **kwargs)
    cov = cr.godambe(inverse=True)

    #print( cr.test(params,sims=100) )
    #print( cr.test([params,est_params],sims=100) )
    print( cr.test(params) )
    print( cr.test([params,est_params]) )

def check_jointime_cov(method, num_runs, theta):
    t0 = random.uniform(.25,2.5)
    t1 = t0 + random.uniform(.5,5.0)
    def demo_func(t):
        return simple_three_pop_demo(t,t1)
    check_cov(method, [t0], demo_func, num_runs, theta)
    
def test_jointime_cov_many():
    check_jointime_cov("many", 1000, 1.)
    
def test_jointime_cov_long():
    check_jointime_cov("long", 10, 100.)

def check_admixture_cov(method, num_runs, theta, **kwargs):
    check_cov(method, simple_nea_admixture_demo.true_params, simple_nea_admixture_demo, num_runs, theta, bounds = simple_nea_admixture_demo.bounds, **kwargs)

def test_admixture_cov_many():
    check_admixture_cov("many", 1000, 1.)

def test_admixture_cov_many_subsample():
    check_admixture_cov("many", 1000, 1., subsample_inds=4)

def test_admixture_cov_long():
    check_admixture_cov("long", 5, 200.)    

def test_admixture_cov_long_subsample():
    check_admixture_cov("long", 2, 1000., subsample_inds=6)
