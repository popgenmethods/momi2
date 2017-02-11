import pytest
import os
import random
import sys
import autograd.numpy as np
from autograd import grad
import logging
from momi import SfsLikelihoodSurface, demographic_history
import momi
from test_msprime import ms_path, scrm_path


def test_archaic_and_pairwisediffs():
    theta = .1
    join_time = 1.0
    #num_runs = 10000
    num_runs = 1000

    logit = lambda p: np.log(p / (1. - p))
    expit = lambda x: 1. / (1. + np.exp(-x))

    true_sample_t = logit(random.uniform(0, join_time) / join_time)

    sampled_pops = ['a', 'b']
    sampled_n = [2, 2]

    def get_demo(sample_t, log_N):
        return demographic_history([('-ej', join_time, 'a', 'b')],
                                   archaic_times_dict={"b": expit(sample_t) * join_time}, default_N=np.exp(log_N))
    true_demo = get_demo(true_sample_t, 0)

    # sfs = simulate_ms(scrm_path, true_demo,
    #                  sampled_pops=sampled_pops, sampled_n=sampled_n,
    # num_loci=num_runs, mut_rate=theta, cmd_format='scrm').sfs
    num_bases = 1e3
    sfs = true_demo.simulate_data(
        sampled_pops, sampled_n,
        mutation_rate=theta / num_bases,
        length=num_bases,
        num_replicates=num_runs,
    ).sfs

    log_prior = lambda x: -x[0] / float(true_sample_t)

    x0 = np.array(
        [logit(random.uniform(0, join_time) / join_time), random.uniform(-1, 1)])
    res = SfsLikelihoodSurface(sfs, demo_func=get_demo, mut_rate=theta, log_prior=log_prior,
                               batch_size=-1, use_pairwise_diffs=True).find_mle(x0, method='trust-ncg', hessp=True)
    #res = SfsLikelihoodSurface(sfs, demo_func=get_demo, mut_rate=theta, log_prior=log_prior, batch_size=-1).find_mle(x0, method='trust-ncg', hessp=True)

    print(res.jac)
    assert abs(expit(res.x[0]) - expit(true_sample_t)
               ) < .1 and abs(res.x[1]) < .1
    # for i,subres in enumerate(res.subsample_results):
    #    assert abs(subres.x - true_sample_t) < .15, "subsample %d did not fit truth well" % i

    #assert False


@pytest.mark.parametrize("folded,add_n",
                         ((f, n) for f in (True, False) for n in (0, 3)))
def test_jointime_inference(folded, add_n):
    check_jointime_inference(folded=folded, add_n=add_n)


def test_nodiff():
    return check_jointime_inference(finite_diff_eps=1e-8, use_prior=True)


def test_missing_data():
    return check_jointime_inference(missing_p=.75, theta=1., num_runs=1000, use_theta=True, sampled_n=(4, 4, 4))


def test_subsample_4():
    check_jointime_inference(sampled_n=(
        10, 4, 2), subsample_n=3, missing_p=.25)


def check_jointime_inference(sampled_n=(5, 5, 5), folded=False, add_n=0, finite_diff_eps=0, missing_p=0,
                             use_prior=False,
                             subsample_n=False,
                             use_theta=False, theta=.1, num_runs=10000):
    t0 = random.uniform(.25, 2.5)
    t1 = t0 + random.uniform(.5, 5.0)

    sampled_pops = (1, 2, 3)

    def get_demo(join_time):
        return demographic_history([('-ej', join_time, 1, 2), ('-ej', t1, 2, 3)])

    true_demo = get_demo(t0)
    # true_demo = make_demography(true_demo.events,
    #                       true_demo.sampled_pops,
    #                       np.array(true_demo.sampled_n) - add_n)
    #true_demo = true_demo.copy(sampled_n = np.array(true_demo.sampled_n) - add_n)
    # data = simulate_ms(ms_path, true_demo.rescaled(),
    #                   sampled_pops = sampled_pops,
    #                   sampled_n = sampled_n,
    #                   num_loci=num_runs, mut_rate=theta)
    num_bases = 1e3
    data = true_demo.rescaled().simulate_data(
        sampled_pops, sampled_n,
        mutation_rate=theta / num_bases,
        length=num_bases,
        num_replicates=num_runs,
    )

    if missing_p:
        data = momi.data_structure._randomly_drop_alleles(data, missing_p)
    if subsample_n:
        data = data.subsample_inds(subsample_n)

    sfs = data.sfs
    assert sfs.n_snps() > 0
    sfs = sfs._copy(sampled_n=np.array(sampled_n) + add_n)
    if folded:
        sfs = sfs.fold()

    print((t0, t1))

    #prim_log_lik = momi.likelihood._log_lik_diff
    prim_log_lik = momi.likelihood._raw_log_lik
    prim_log_lik.reset_grad_count()
    assert not prim_log_lik.num_grad_calls()

    if not use_theta:
        theta = None

    x0 = np.array([random.uniform(0, t1)])

    bound_eps = 1e-12
    if finite_diff_eps:
        bound_eps += finite_diff_eps
        jac = False
    else:
        jac = True

    if use_prior:
        log_prior = lambda t: -t / float(t0)
    else:
        log_prior = None
    res = SfsLikelihoodSurface(sfs, get_demo, mut_rate=theta, folded=folded, log_prior=log_prior, p_missing=missing_p,
                               use_pairwise_diffs=True).find_mle(x0, bounds=[(bound_eps, t1 - bound_eps), ], jac=jac)
    #res = SfsLikelihoodSurface(sfs, get_demo, mut_rate=theta, folded=folded, log_prior=log_prior, use_pairwise_diffs=True).find_mle(x0, bounds=[(bound_eps,t1-bound_eps),], jac=jac)

    #res = SfsLikelihoodSurface(sfs, get_demo, folded=folded).find_mle(x0, bounds=[(0,t1),])

    # make sure autograd is calling the rearranged gradient function
    assert bool(prim_log_lik.num_grad_calls()) != bool(finite_diff_eps)

    print(res.jac)
    assert abs(res.x - t0) / t0 < .05


@pytest.mark.parametrize("folded", (True, False))
def test_underflow_robustness(folded):
    num_runs = 1000
    mu = 1e-3
    sampled_pops = (1, 2, 3)
    sampled_n = (5, 5, 5)

    def get_demo(t0, t1):
        return demographic_history([('-ej', np.exp(t0), 1, 2), ('-ej', np.exp(t0) + np.exp(t1), 2, 3)]).rescaled(1e4)
    true_x = np.array([np.log(.5), np.log(.2)])
    true_demo = get_demo(*true_x)

    # sfs = simulate_ms(ms_path, true_demo.rescaled(),
    #                  sampled_pops = sampled_pops, sampled_n = sampled_n,
    #                  num_loci=num_runs, mut_rate=mu*true_demo.default_N).sfs
    num_bases = 1e3
    sfs = true_demo.rescaled().simulate_data(
        sampled_pops, sampled_n,
        mutation_rate=mu * true_demo.default_N / num_bases,
        length=num_bases,
        num_replicates=num_runs,
    ).sfs
    if folded:
        sfs = sfs.fold()

    # logging.basicConfig(level=logging.INFO)
    optimize_res = SfsLikelihoodSurface(sfs, get_demo, mut_rate=mu, folded=folded).find_mle(
        np.array([np.log(0.1), np.log(100.0)]))
    #optimize_res = SfsLikelihoodSurface(sfs, get_demo, mut_rate=mu, folded=folded).find_mle(np.array([np.log(0.1),np.log(0.1)]))
    print(optimize_res)

    inferred_x = optimize_res.x
    error = (true_x - inferred_x) / true_x
    print("# Truth:\n", true_x)
    print("# Inferred:\n", inferred_x)
    print("# Max Relative Error: %f" % max(abs(error)))
    print("# Relative Error:", "\n", error)

    assert max(abs(error)) < .1
