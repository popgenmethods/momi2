import pytest
import os, random
import autograd.numpy as np
from autograd import grad

from momi import Demography, simulate_ms, composite_mle_search
import momi
from test_ms import ms_path, scrm_path

def test_archaic_sample():
    theta=.1
    join_time = 1.0
    num_runs = 10000
    true_sample_t=random.uniform(0,join_time)
    def get_demo(sample_t):
        return Demography([('-ej',join_time,'a','b')],
                          sampled_pops=['a','b'],
                          sampled_n=[2,2],
                          sampled_t=[0,sample_t])
    true_demo = get_demo(true_sample_t)

    sfs = simulate_ms(scrm_path, true_demo,
                      num_loci=num_runs, mut_rate=theta, cmd_format='scrm').sfs
    
    x0 = np.array([random.uniform(0,join_time)])
    res = composite_mle_search(sfs, get_demo, x0, None, bounds=[(0,join_time)])
    
    print res.jac
    assert abs(res.x - true_sample_t) < .1

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
    sfs = simulate_ms(ms_path, true_demo.rescaled(),
                      num_loci=num_runs, mut_rate=theta).sfs
    sfs = sfs._copy(sampled_n=np.array(true_demo.sampled_n)+add_n)
    if folded:
        sfs = sfs.fold()
    
    print(t0,t1)
    
    x0 = np.array([random.uniform(0,t1)])
    res = composite_mle_search(sfs, get_demo, x0, None, bounds=[(0,t1),], sfs_kwargs={'folded':folded})
    
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

    sfs = simulate_ms(ms_path, true_demo.rescaled(),
                      num_loci=num_runs, mut_rate=mu*true_demo.default_N).sfs
    if folded:
        sfs = sfs.fold()
    
    optimize_res = composite_mle_search(sfs, get_demo, np.array([np.log(0.1),np.log(100.0)]), mu, hessp=True, method='newton-cg', sfs_kwargs={'folded':folded})
    print optimize_res
    
    inferred_x = optimize_res.x
    error = (true_x - inferred_x) / true_x
    print "# Truth:\n", true_x
    print "# Inferred:\n", inferred_x
    print "# Max Relative Error: %f" % max(abs(error))
    print "# Relative Error:","\n", error

    assert max(abs(error)) < .1
