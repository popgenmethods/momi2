import pytest
import os, random
import autograd.numpy as np
from autograd import grad

from momi import Demography, simulate_ms, seg_sites_from_ms, composite_mle_search
import momi
from test_ms import ms_path

@pytest.mark.parametrize("folded,add_n",
                         ((f,n) for f in (True,False) for n in (0,3)))
def test_jointime_inference(folded, add_n):
    theta=.1
    t0=random.uniform(.25,2.5)
    t1= t0 + random.uniform(.5,5.0)
    num_runs = 10000

    def get_demo(join_time):
        return Demography([('-ej', join_time, 1, 2), ('-ej', t1, 2, 3)],
                          (1,2,3), (5,5,5))

    true_demo = get_demo(t0)
    #true_demo = Demography(true_demo.events,
    #                       true_demo.sampled_pops,
    #                       np.array(true_demo.sampled_n) - add_n)
    true_demo = true_demo.copy(sampled_n = np.array(true_demo.sampled_n) - add_n)
    sfs = seg_sites_from_ms(simulate_ms(ms_path, true_demo.rescaled(),
                                        num_loci=num_runs, mut_rate=theta), true_demo.sampled_pops).sfs
    sfs = sfs.copy(sampled_n=np.array(true_demo.sampled_n)+add_n)
    if folded:
        sfs = sfs.copy(fold=True)
    
    print(t0,t1)
    
    x0 = np.array([random.uniform(0,t1)])
    #res = composite_mle_search(sfs, get_demo, x0, mu * num_runs, bounds=[(0,t1)], folded=folded)
    res = composite_mle_search(sfs, get_demo, x0, None, bounds=[(0,t1)], folded=folded)
    
    print res.jac
    assert abs(res.x - t0) / t0 < .05

@pytest.mark.parametrize("folded",(True,False))
def test_underflow_robustness(folded):
    num_runs = 1000
    mu=1e-3
    def get_demo(t0, t1):
        return Demography([('-ej', np.exp(t0), 1, 2), ('-ej', np.exp(t0) + np.exp(t1), 2, 3)],
                          (1,2,3), (5,5,5)).rescaled(1e4)
    true_x = np.array([np.log(.5),np.log(.2)])
    true_demo = get_demo(*true_x)

    sfs = seg_sites_from_ms(simulate_ms(ms_path, true_demo.rescaled(),
                                        num_loci=num_runs, mut_rate=mu*true_demo.default_N), true_demo.sampled_pops).sfs
    if folded:
        sfs = sfs.copy(fold=True)
    
    optimize_res = composite_mle_search(sfs, get_demo, np.array([np.log(0.1),np.log(100.0)]), mu, hessp=True, method='newton-cg', folded=folded)
    print optimize_res
    
    inferred_x = optimize_res.x
    error = (true_x - inferred_x) / true_x
    print "# Truth:\n", true_x
    print "# Inferred:\n", inferred_x
    print "# Max Relative Error: %f" % max(abs(error))
    print "# Relative Error:","\n", error

    assert max(abs(error)) < .1
