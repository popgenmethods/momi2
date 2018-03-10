import pytest
import random
import autograd.numpy as np
from momi import SfsLikelihoodSurface
import momi

def test_archaic_and_pairwisediffs():
    #logging.basicConfig(level=logging.DEBUG)
    theta = 1
    N_e = 1.0
    join_time = 1.0
    num_runs = 1000

    def logit(p):
        return np.log(p / (1. - p))

    def expit(x):
        return 1. / (1. + np.exp(-x))

    n_bases = 1000
    model = momi.DemographicModel(
        N_e, muts_per_gen=theta/4./N_e/n_bases)

    model.add_time_param(
        "sample_t", random.uniform(0.001, join_time-.001) / join_time,
        upper=join_time)
    model.add_size_param("N", 1.0)
    model.add_leaf("a", N="N")
    model.add_leaf("b", t="sample_t", N="N")
    model.move_lineages("a", "b", join_time)

    data = model.simulate_data(length=n_bases,
                               recoms_per_gen=0,
                               num_replicates=num_runs,
                               sampled_n_dict={"a": 2, "b": 2})

    model.set_data(data.extract_sfs(1),
                   use_pairwise_diffs=False,
                   mem_chunk_size=-1)

    true_params = np.array(list(model.get_params().values()))
    #model.set_x([logit(random.uniform(.001, join_time-.001) / join_time),
    model.set_params([
        logit(random.uniform(.001, join_time-.001) / join_time),
        random.uniform(-1, 1)],
                     scaled=True)
    res = model.optimize(method="trust-ncg", hessp=True)
    inferred_params = np.array(list(model.get_params().values()))

    assert np.max(np.abs(np.log(true_params / inferred_params))) < .2


@pytest.mark.parametrize("folded,add_n",
                         ((f, n) for f in (True, False) for n in (0, 3)))
def test_jointime_inference(folded, add_n):
    check_jointime_inference(folded=folded, add_n=add_n)

def test_ascertainment_inference():
    check_jointime_inference(non_ascertained_pops=[3], use_theta=True,
                             sampled_n=(5,5,1))

def check_jointime_inference(
        sampled_n=(5, 5, 5), folded=False, add_n=0,
        use_theta=False, theta=.1, num_runs=10000,
        non_ascertained_pops=None):
    t0 = random.uniform(.25, 2.5)
    t1 = t0 + random.uniform(.5, 5.0)

    num_bases = 1e3
    theta = theta / num_bases

    model = momi.DemographicModel(1, muts_per_gen=theta)
    model.add_leaf(1)
    model.add_leaf(2)
    model.add_leaf(3)
    #model.add_parameter("join_time", t0, scaled_lower=0.0, scaled_upper=t1)
    #model.add_param("join_time", x0=t0, upper_x=t1)
    model.add_time_param("join_time", t0, upper=t1)
    model.move_lineages(1, 2, t="join_time")
    model.move_lineages(2, 3, t=t1)

    sampled_pops = (1, 2, 3)

    data = model.simulate_data(
        num_bases, 0, num_runs,
        sampled_n_dict=dict(zip(sampled_pops, sampled_n)))

    sfs = data.extract_sfs(1)
    assert sfs.n_snps() > 0
    sfs = sfs._copy(sampled_n=np.array(sampled_n) + add_n)
    if folded:
        sfs = sfs.fold()

    print((t0, t1))

    #prim_log_lik = momi.likelihood._raw_log_lik
    #prim_log_lik.reset_grad_count()
    #assert not prim_log_lik.num_grad_calls()

    if not use_theta:
        model.set_mut_rate(None)

    #model.set_x(random.uniform(0, t1), "join_time")
    model.set_params({"join_time": random.uniform(0,t1)}, scaled=True)

    model.set_data(sfs, non_ascertained_pops=non_ascertained_pops)
    res = model.optimize()

    # make sure autograd is calling the rearranged gradient
    #assert bool(prim_log_lik.num_grad_calls())

    print(res.jac)
    #assert abs(res.x - t0) / t0 < .05
    #assert (model.get_params()["join_time"] - t0) / t0 < .05
    assert (res.parameters["join_time"] - t0) / t0 < .05


@pytest.mark.parametrize("folded", (True, False))
def test_underflow_robustness(folded):
    num_runs = 1000
    sampled_pops = (1, 2, 3)
    sampled_n = (5, 5, 5)

    n_bases = int(1e3)
    demo = momi.DemographicModel(1.0, .25, muts_per_gen=2.5 / n_bases)
    for p in sampled_pops:
        demo.add_leaf(p)
    demo.add_time_param("t0")
    demo.add_time_param("t1", lower_constraints=["t0"])
    demo.move_lineages(1, 2, "t0")
    demo.move_lineages(2, 3, "t1")

    true_params = np.array([0.5, 0.7])
    demo.set_params(true_params)

    data = demo.simulate_data(
        length=n_bases,
        recoms_per_gen=0.0,
        num_replicates=num_runs,
        sampled_n_dict=dict(zip(sampled_pops, sampled_n)))

    sfs = data.extract_sfs(1)
    if folded:
        sfs = sfs.fold()

    demo.set_data(sfs)
    demo.set_params({"t0": 0.1, "t1": 100.0})
    optimize_res = demo.optimize()

    print(optimize_res)
    inferred_params = np.array(list(demo.get_params().values()))

    error = (true_params - inferred_params) / true_params
    print("# Truth:\n", true_params)
    print("# Inferred:\n", inferred_params)
    print("# Max Relative Error: %f" % max(abs(error)))
    print("# Relative Error:", "\n", error)

    assert max(abs(error)) < .1
