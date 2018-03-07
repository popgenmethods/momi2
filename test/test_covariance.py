import pytest
import os
import random
import pickle
import functools
import autograd.numpy as np

import momi

from demo_utils import simple_three_pop_demo, simple_nea_admixture_demo

import scipy

# TODO: get these tests working if we reactivate dead Godambe code

#def check_cov(method, params, demo_func, num_runs, theta, bounds=None, subsample_inds=False, p_missing=None, **kwargs):
#    true_demo = demo_func(*params)
#    #seg_sites = simulate_ms(scrm_path, true_demo.demo_hist,
#    #                        sampled_pops=true_demo.pops, sampled_n=true_demo.n,
#    #                        num_loci=num_runs, mut_rate=theta,
#    #                        additional_ms_params="-r %f 1000" % theta)
#
#    # TODO fix this test after removing SegSites object
#
#    if p_missing:
#        seg_sites = momi.data.seg_sites._randomly_drop_alleles(
#            seg_sites, p_missing)
#
#    if subsample_inds:
#        seg_sites = seg_sites.subsample_inds(subsample_inds)
#        old_demo_func = demo_func
#        demo_func = functools.partial(
#            old_demo_func, sampled_n=seg_sites.sampled_n)
#
#    cmle_search_res = momi.SfsLikelihoodSurface(seg_sites, lambda *x: demo_func(
#        *x).demo_hist).find_mle(params, options={'maxiter': 1000}, bounds=bounds, **kwargs)
#    est_params = cmle_search_res.x
#
#    cr = momi.ConfidenceRegion(
#        est_params, lambda *x: demo_func(*x).demo_hist, seg_sites, regime=method, **kwargs)
#    cov = cr.godambe(inverse=True)
#
#    #print( cr.test(params,sims=100) )
#    #print( cr.test([params,est_params],sims=100) )
#    print(cr.test(params))
#    print(cr.test([params, est_params]))
#    return cr
#
#
#def check_jointime_cov(method, num_runs, theta):
#    t0 = random.uniform(.25, 2.5)
#    t1 = t0 + random.uniform(.5, 5.0)
#
#    def demo_func(t):
#        return simple_three_pop_demo(t, t1)
#    return check_cov(method, [t0], demo_func, num_runs, theta)
#
#
#def test_jointime_cov_many():
#    check_jointime_cov("many", 1000, 1.)
#
#
#def test_jointime_cov_long():
#    check_jointime_cov("long", 10, 100.)
#
#
#def check_admixture_cov(method, num_runs, theta, **kwargs):
#    return check_cov(method, simple_nea_admixture_demo.true_params, simple_nea_admixture_demo, num_runs, theta, bounds=simple_nea_admixture_demo.bounds, **kwargs)
#
#
#def test_admixture_cov_many():
#    check_admixture_cov("many", 1000, 1.)
#
#
#def test_admixture_cov_many_subsample():
#    check_admixture_cov("many", 1000, 1., subsample_inds=4)
#
#
#def test_admixture_cov_long():
#    check_admixture_cov("long", 5, 200.)
#
#
#def test_admixture_cov_long_subsample():
#    cr = check_admixture_cov("long", 2, 1000., subsample_inds=6, p_missing=.1)
#
#    # commented because pickling confidence region is too large in filesize
#    # test that pickling the confidence region works
#    #fname = "cr.tmp.pickle"
#    # try:
#    #    with open(fname,"wb") as f:
#    #        pickle.dump(cr, f)
#
#    #    with open(fname, "rb") as f:
#    #        cr2 = pickle.load(f)
#
#    #    assert cr.data == cr2.data
#    #    assert np.all(cr.score == cr2.score)
#    #    assert np.all(cr.score_cov == cr2.score_cov)
#    #    assert np.all(cr.fisher == cr2.fisher)
#    #    assert np.allclose(cr.godambe(), cr2.godambe())
#    # except:
#    #    if os.path.exists(fname):
#    #        os.remove(fname)
#    #    raise
#    # else:
#    #    os.remove(fname)
