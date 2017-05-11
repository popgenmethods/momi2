
import pytest
import momi
from momi import expected_sfs
import momi.likelihood
from demo_utils import simple_admixture_demo
import autograd.numpy as np
import itertools
import random
import sys
from collections import Counter
import logging
from test_msprime import ms_path, scrm_path

# test subsampling of SNPs


@pytest.mark.parametrize("fold,normalized",
                         ((random.choice((True, False)), random.choice((True, False))),))
def test_subconfigs(fold, normalized):
    demo = simple_admixture_demo()
    demo.demo_hist = demo.demo_hist.rescaled()

    configs = momi.simulate_ms(scrm_path, demo.demo_hist,
                               sampled_pops=demo.pops, sampled_n=demo.n,
                               num_loci=1000, mut_rate=1.).sfs.configs

    if fold:
        configs = momi.site_freq_spectrum(
            configs.sampled_pops, [configs]).fold().configs
        #configs = configs.copy(fold=True)

    sub_idxs = np.array(random.sample(
        list(range(len(configs))), int(len(configs) / 2) + 1))
    assert len(sub_idxs) > 0 and len(sub_idxs) < len(configs)

    val1 = momi.expected_sfs(demo.demo_hist, configs,
                             normalized=normalized, folded=fold)[sub_idxs]

    sub_configs = momi.data_structure._ConfigArray_Subset(configs, sub_idxs)
    val2 = momi.expected_sfs(demo.demo_hist, sub_configs,
                             normalized=normalized, folded=fold)

    assert np.allclose(val1, val2)


@pytest.mark.parametrize("fold,use_mut",
                         ((random.choice((True, False)), random.choice((True, False))),))
def test_subsfs(fold, use_mut):
    demo = simple_admixture_demo()
    demo.demo_hist = demo.demo_hist.rescaled()

    n_loci = 10
    mut_rate = 100.
    sfs = momi.simulate_ms(scrm_path, demo.demo_hist,
                           sampled_pops=demo.pops, sampled_n=demo.n,
                           num_loci=n_loci, mut_rate=mut_rate).sfs

    if fold:
        sfs = sfs.fold()

    locus = random.choice(list(range(n_loci)))
    #subsfs = momi.likelihood._SubSfs(sfs.configs, sfs._counts_ij[locus,:])
    counts = np.zeros(sfs.n_nonzero_entries)
    loc_idxs, loc_counts = sfs.loc_idxs[locus], sfs.loc_counts[locus]
    counts[loc_idxs] = loc_counts
    subsfs = momi.data_structure._sub_sfs(sfs.configs, counts)

    if not use_mut:
        mut_rate = None

    val1 = momi.likelihood._composite_log_likelihood(
        sfs, demo.demo_hist, mut_rate=mut_rate, vector=True, folded=fold)[locus]
    #val2 = momi.CompositeLogLikelihood(subsfs, mut_rate=mut_rate, folded=fold).evaluate(demo)
    val2 = momi.likelihood._composite_log_likelihood(
        subsfs, demo.demo_hist, mut_rate=mut_rate, folded=fold)

    assert np.isclose(val1, val2)


@pytest.mark.parametrize("fold",
                         (random.choice((True, False)),))
def test_subsfs2(fold):
    demo = simple_admixture_demo()
    demo.demo_hist = demo.demo_hist.rescaled()

    n_loci = 10
    mut_rate = 100.
    sfs = momi.simulate_ms(scrm_path, demo.demo_hist,
                           sampled_pops=demo.pops, sampled_n=demo.n,
                           num_loci=n_loci, mut_rate=mut_rate).sfs

    if fold:
        sfs = sfs.fold()

    subsfs_list = momi.likelihood._subsfs_list(sfs, 10, np.random)
    total = [([tuple(map(tuple, sfs.configs[i])) for i in subsfs.configs.sub_idxs],
              subsfs._total_freqs)
             for subsfs in subsfs_list]
    total = [Counter(dict(list(zip(cnfs, cnts)))) for cnfs, cnts in total]
    total = dict(sum(total, Counter()))

    assert total == sfs.to_dict()


@pytest.mark.parametrize("fold",
                         (random.choice((True, False)),))
def test_subliks(fold):
    pre_demo_func = lambda *x: simple_admixture_demo(x)
    demo_func = lambda *x: pre_demo_func(*x).demo_hist.rescaled()
    rnd = np.random.RandomState()
    x0 = rnd.normal(size=7)

    demo = pre_demo_func(*x0)

    n_loci = 100
    mut_rate = 10.
    sfs = momi.simulate_ms(scrm_path, demo.demo_hist.rescaled(),
                           sampled_pops=demo.pops, sampled_n=demo.n,
                           num_loci=n_loci, mut_rate=mut_rate).sfs

    if fold:
        sfs = sfs.fold()

    n_chunks = 10

    surface = momi.SfsLikelihoodSurface(
        sfs, demo_func=demo_func, mut_rate=None, folded=fold)
    subsurfaces = surface._get_stochastic_pieces(n_chunks, np.random)

    val0 = [s.log_lik(x0) for s in subsurfaces]
    val1 = surface.log_lik(x0)

    assert np.isclose(np.sum(val0), val1)


@pytest.mark.parametrize("folded,use_pairwise_diffs",
                         #itertools.product((True, False), repeat=2))
                         (tuple(random.choice((True, False)) for _ in range(2)),))
def test_stochastic_inference(folded, use_pairwise_diffs):
    num_runs = 1000
    mu = 1.0
    sampled_pops = (1, 2, 3)
    sampled_n = (5, 5, 5)

    def get_demo(t0, t1):
        return momi.demographic_history([('-ej', t0, 1, 2), ('-ej', t0 + t1, 2, 3)])
    true_x = np.array([.5, .2])
    true_demo = get_demo(*true_x)

    sfs = momi.simulate_ms(ms_path, true_demo,
                           sampled_pops=sampled_pops, sampled_n=sampled_n,
                           num_loci=num_runs, mut_rate=mu).sfs
    if folded:
        sfs = sfs.fold()

    log_prior = lambda x: np.sum(-x / true_x)

    def callback(x):
        if x.iteration % 10 == 0:
            print(x.iteration, x.fun, x)
    optimize_res = momi.SfsLikelihoodSurface(
        sfs, demo_func=get_demo, mut_rate=mu, folded=folded,
        log_prior=log_prior, use_pairwise_diffs=use_pairwise_diffs
    ).stochastic_surfaces(n_minibatches=10).find_mle(
        np.array([.1, .9]), bounds=[(1e-100, None), (1e-100, None)],
        method="adam", svrg_epoch=10, num_iters=1000, callback=callback)
    print(optimize_res)

    inferred_x = optimize_res.x
    error = (true_x - inferred_x) / true_x
    print("# Truth:\n", true_x)
    print("# Inferred:\n", inferred_x)
    print("# Max Relative Error: %f" % max(abs(error)))
    print("# Relative Error:", "\n", error)

    assert max(abs(error)) < .1
    #assert False


# def test_complex_stochastic_inference():
##     seed = np.random.randint(2**32-1)
# seed = 270543553
# seed = 2300087928
# seed = 3443948355
# seed = 2092909111
# seed = 2470901291
# seed = 2173658782
# print("SEED",seed)
# np.random.seed(seed)
##
##     true_N_chb_bottom, true_N_chb_top, true_pulse_t, true_pulse_p, true_ej_chb, true_ej_yri = 10.,.1,.25,.03,.5,1.5
# def demo_func(log_N_chb_bottom, log_N_chb_top, log_pulse_t, log_pulse_p, log_ej_chb, log_ej_yri):
##         pulse_t = 2**(log_pulse_t) * true_pulse_t
##         ej_chb = pulse_t + 2**(log_ej_chb) * (true_ej_chb - true_pulse_t)
##         ej_yri = ej_chb + 2**(log_ej_yri) * (true_ej_yri - true_ej_chb)
##
##         N_chb_top = 2**(log_N_chb_top) * true_N_chb_top
##         N_chb_bottom = 2**(log_N_chb_bottom) * true_N_chb_bottom
##
##         G_chb = -np.log(N_chb_top / N_chb_bottom) / ej_chb
##
# events = [('-en', 0., 'chb', N_chb_bottom),
##                   ('-eg', 0, 'chb' , G_chb),
##                   ('-ep', pulse_t, 'chb', 'nea', .25*(4.0*true_pulse_p)**((2**(log_pulse_p))**.5)),
##                   ('-ej', ej_chb, 'chb', 'yri'),
##                   ('-ej', ej_yri, 'yri', 'nea'),
# ]
# return momi.make_demography(events, ('yri','chb'), (14,10))
##
##     true_params = np.zeros(6)
##     true_demo = demo_func(*true_params)
##
# sfs = momi.simulate_ms(ms_path, true_demo,
# 1000, mut_rate=10.0,
# seeds=[np.random.randint(2**32-1) for _ in range(3)]).sfs
##
##
# define (lower,upper) bounds on the parameter space
##     lower_bounds = true_params + np.log(.01)/np.log(2)
##     upper_bounds = true_params + np.log(100.)/np.log(2)
##     bounds = list(zip(lower_bounds, upper_bounds))
##
# pick a random start value for the parameter search
##     start_params = np.array([np.random.uniform(l,h) for (l,h) in bounds])
##
# def callback(x):
# if x.iteration % 10 == 0:
##             print(x.iteration, x.fun, x)
# try:
# optimize_res = momi.SfsLikelihoodSurface(sfs, demo_func=demo_func, mut_rate=None).find_mle(start_params, bounds=bounds, method="tnc", maxiter=500, callback=callback)
# optimize_res = momi.SfsLikelihoodSurface(sfs, demo_func=demo_func, mut_rate=None).find_mle(start_params, bounds=bounds, method="L-BFGS-B", callback=callback)
# optimize_res = momi.SfsLikelihoodSurface(sfs, demo_func=demo_func, mut_rate=None).stochastic_surfaces(n_minibatches=100, exact=50).find_mle(start_params, bounds=bounds, method="svrg", stepsize=.1, iter_per_epoch=10, max_epochs=100, callback=callback)
##         optimize_res = momi.SfsLikelihoodSurface(sfs, demo_func=demo_func, mut_rate=None).stochastic_surfaces(n_minibatches=100, exact=50).find_mle(start_params, bounds=bounds, method="adam", svrg_epoch=10, num_iters=1000, callback=callback)
# except:
# print("SEED",seed)
# raise
##
##     true_x = true_params
##     inferred_x = optimize_res.x
##     error = inferred_x-true_x
# print("# Gradient:\n", optimize_res.jac)
# print("# Truth:\n", true_x)
# print("# Inferred:\n", inferred_x)
# print("# Ratio Inferred/Truth:","\n", 2.**error)
# print("# Max Log2 Ratio: %f" % max(abs(error)))
# try:
# print ("# Epochs: ", optimize_res.nepoch)
# except: pass
# print("SEED",seed)
# assert max(abs(np.log(error))) < .1
