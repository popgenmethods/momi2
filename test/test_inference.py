import pytest
import os, random
import autograd.numpy as np
from autograd import grad

from momi import make_demography, simulate_ms, SfsLikelihoodSurface
import momi
from test_ms import ms_path, scrm_path

def test_archaic_sample():
    theta=.1
    join_time = 1.0
    num_runs = 10000
    true_sample_t=random.uniform(0,join_time)
    def get_demo(sample_t):
        return make_demography([('-ej',join_time,'a','b')],
                          sampled_pops=['a','b'],
                          sampled_n=[2,2],
                          sampled_t=[0,sample_t])
    true_demo = get_demo(true_sample_t)

    sfs = simulate_ms(scrm_path, true_demo,
                      num_loci=num_runs, mut_rate=theta, cmd_format='scrm').sfs
    
    x0 = np.array([random.uniform(0,join_time)])
    res = SfsLikelihoodSurface(sfs, demo_func=get_demo, mut_rate=theta).find_optimum(x0, bounds=[(0,join_time)], subsample_steps=2, output_progress=True, opt_method='L-BFGS-B')
    
    print(res.jac)
    assert abs(res.x - true_sample_t) < .1
    for i,subres in enumerate(res.subsample_results):
        assert abs(subres.x - true_sample_t) < .15, "subsample %d did not fit truth well" % i

    #assert False

@pytest.mark.parametrize("folded,add_n",
                         ((f,n) for f in (True,False) for n in (0,3)))
def test_jointime_inference(folded, add_n):
    check_jointime_inference(folded=folded, add_n=add_n)

def test_nodiff():
    return check_jointime_inference(finite_diff_eps=1e-8)

def test_missing_data():
    return check_jointime_inference(missing_p=.5, folded=True, num_runs=100, theta=10.)

def check_jointime_inference(folded=False, add_n=0, finite_diff_eps=0, missing_p=0,
                             use_theta=False, theta=.1, num_runs = 10000):
    t0=random.uniform(.25,2.5)
    t1= t0 + random.uniform(.5,5.0)

    def get_demo(join_time):
        return make_demography([('-ej', join_time, 1, 2), ('-ej', t1, 2, 3)],
                          (1,2,3), (5,5,5))

    true_demo = get_demo(t0)
    #true_demo = make_demography(true_demo.events,
    #                       true_demo.sampled_pops,
    #                       np.array(true_demo.sampled_n) - add_n)
    true_demo = true_demo.copy(sampled_n = np.array(true_demo.sampled_n) - add_n)
    data = simulate_ms(ms_path, true_demo.rescaled(),
                       num_loci=num_runs, mut_rate=theta)
    if missing_p:
        subsampled_data = []
        for loc in range(data.n_loci):
            curr = np.array(list(data[loc]), dtype=int)
            if len(curr) == 0:
                subsampled_data.append(list(curr))
                continue
            curr = np.random.binomial(curr, 1.0-missing_p)
            curr_polymorphic = np.all(np.sum(curr, axis=1) != 0, axis=1)
            curr = curr[curr_polymorphic,:,:]
            subsampled_data.append(curr)
        assert any([not np.array_equal(subsampled_data[loc],
                                       np.array(list(data[loc]), dtype=int))
                    for loc in range(data.n_loci)])
        data = momi.seg_site_configs(data.sampled_pops, subsampled_data)

    sfs = data.sfs
    assert sfs.n_snps() > 0
    sfs = sfs._copy(sampled_n=np.array(true_demo.sampled_n)+add_n)
    if folded:
        sfs = sfs.fold()
    
    print((t0,t1))
    
    prim_log_lik = momi.likelihood._log_lik_diff
    prim_log_lik.reset_grad_count()
    assert not prim_log_lik.num_grad_calls()

    if not use_theta:
        theta = None
    
    x0 = np.array([random.uniform(0,t1)])
    bound_eps = finite_diff_eps + 1e-12
    res = SfsLikelihoodSurface(sfs, get_demo, mut_rate=theta, folded=folded).find_optimum(x0, bounds=[(bound_eps,t1-bound_eps),], finite_diff_eps=finite_diff_eps)
    
    #res = SfsLikelihoodSurface(sfs, get_demo, folded=folded).find_optimum(x0, bounds=[(0,t1),])

    # make sure autograd is calling the rearranged gradient function
    assert bool(prim_log_lik.num_grad_calls()) != bool(finite_diff_eps)
    
    print(res.jac)
    assert abs(res.x - t0) / t0 < .05

@pytest.mark.parametrize("folded",(True,False))
def test_underflow_robustness(folded):
    num_runs = 1000
    mu=1e-3
    def get_demo(t0, t1):
        return make_demography([('-ej', np.exp(t0), 1, 2), ('-ej', np.exp(t0) + np.exp(t1), 2, 3)],
                          (1,2,3), (5,5,5)).rescaled(1e4)
    true_x = np.array([np.log(.5),np.log(.2)])
    true_demo = get_demo(*true_x)

    sfs = simulate_ms(ms_path, true_demo.rescaled(),
                      num_loci=num_runs, mut_rate=mu*true_demo.default_N).sfs
    if folded:
        sfs = sfs.fold()
    
    #optimize_res = composite_mle_search(sfs, get_demo, np.array([np.log(0.1),np.log(100.0)]), mu, hessp=True, method='newton-cg', sfs_kwargs={'folded':folded})
    optimize_res = SfsLikelihoodSurface(sfs, get_demo, mut_rate=mu, folded=folded).find_optimum(np.array([np.log(0.1),np.log(100.0)]))
    print(optimize_res)
    
    inferred_x = optimize_res.x
    error = (true_x - inferred_x) / true_x
    print("# Truth:\n", true_x)
    print("# Inferred:\n", inferred_x)
    print("# Max Relative Error: %f" % max(abs(error)))
    print("# Relative Error:","\n", error)

    assert max(abs(error)) < .1

def test_validation():
    x1 = np.zeros(2)
    x2 = np.array([10,10])

    f1 = lambda x: np.sum((x-x1)**2)**(.25)
    f2 = lambda x: np.sum((x-x2)**2)**(.25)    

    x0 = x2
    res = momi.util._minimize(f1, x0, maxiter=100, bounds=None, f_validation=f2)
    print(res)
    assert res.message == 'Validation function stopped improving'
