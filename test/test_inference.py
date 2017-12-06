import pytest
import os
import random
import sys
import autograd.numpy as np
from autograd import grad
import logging
from momi import SfsLikelihoodSurface, demographic_history
import momi
#from test_msprime import ms_path, scrm_path


def test_archaic_and_pairwisediffs():
    #logging.basicConfig(level=logging.DEBUG)
    theta = 1
    N_e = 1.0
    join_time = 1.0
    num_runs = 1000

    logit = lambda p: np.log(p / (1. - p))
    expit = lambda x: 1. / (1. + np.exp(-x))

    model = momi.demographic_model(N_e)
    model.add_param("sample_t", x0=logit(random.uniform(0.001, join_time-.001) / join_time),
                    lower_x=None, upper_x=None,
                    transform_x=lambda x, p: expit(x)*join_time)
    model.add_param("N", x0=0, lower_x=None, upper_x=None,
                    transform_x=lambda x, p: np.exp(x))
    model.add_leaf("a", N="N")
    model.add_leaf("b", t="sample_t", N="N")
    model.move_lineages("a", "b", join_time)

    n_bases = 1000
    data = model.simulate_data(length=n_bases,
                               recombination_rate=0,
                               mutation_rate=theta/4./N_e/n_bases,
                               num_replicates=num_runs,
                               sampled_n_dict={"a": 2, "b": 2})

    model.set_data(data, muts_per_gen=theta/4./N_e*num_runs,
                   use_pairwise_diffs=False,
                   mem_chunk_size=-1)

    true_params = np.array(list(model.get_params().values()))
    model.set_x([logit(random.uniform(.001, join_time-.001) / join_time),
                 random.uniform(-1, 1)])
    res = model.optimize(method="trust-ncg", hessp=True)
    inferred_params = np.array(list(model.get_params().values()))

    assert np.max(np.abs(np.log(true_params / inferred_params))) < .2


@pytest.mark.parametrize("folded,add_n",
                         ((f, n) for f in (True, False) for n in (0, 3)))
def test_jointime_inference(folded, add_n):
    check_jointime_inference(folded=folded, add_n=add_n)

def check_jointime_inference(
        sampled_n=(5, 5, 5), folded=False, add_n=0,
        use_theta=False, theta=.1, num_runs=10000):
    t0 = random.uniform(.25, 2.5)
    t1 = t0 + random.uniform(.5, 5.0)

    model = momi.demographic_model(1)
    model.add_leaf(1)
    model.add_leaf(2)
    model.add_leaf(3)
    model.add_param("join_time", x0=t0, upper_x=t1)
    model.move_lineages(1, 2, t="join_time")
    model.move_lineages(2, 3, t=t1)

    sampled_pops = (1, 2, 3)

    num_bases = 1e3
    data = model.simulate_data(num_bases, 0, theta / num_bases, num_runs, dict(zip(sampled_pops, sampled_n)))

    sfs = data.sfs
    assert sfs.n_snps() > 0
    sfs = sfs._copy(sampled_n=np.array(sampled_n) + add_n)
    if folded:
        sfs = sfs.fold()

    print((t0, t1))

    #prim_log_lik = momi.likelihood._raw_log_lik
    #prim_log_lik.reset_grad_count()
    #assert not prim_log_lik.num_grad_calls()

    if not use_theta:
        theta = None

    model.set_x(random.uniform(0, t1), "join_time")

    model.set_data(data, theta, folded=folded)
    res = model.optimize()

    # make sure autograd is calling the rearranged gradient
    #assert bool(prim_log_lik.num_grad_calls())

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
