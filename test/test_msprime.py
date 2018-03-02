from momi import expected_sfs, expected_total_branch_len
import momi
import pytest
import random
import autograd.numpy as np
import scipy
import scipy.stats
import itertools
import sys
import shutil
from collections import Counter

from demo_utils import *

import os

#demo_funcs = {f.__name__: f for f in [simple_admixture_demo, simple_two_pop_demo,
#                                      piecewise_constant_demo, exp_growth_model, simple_admixture_3pop]}

n_lineages = {simple_admixture_demo: (2, 3),
              simple_two_pop_demo: (5, 6),
              piecewise_constant_demo: (10,),
              exp_growth_model: (10,),
              simple_admixture_3pop: (4, 4, 4)}


@pytest.mark.parametrize("k,folded",
                         ((f.__name__, bool(b))
                          for f, b in itertools.product(list(n_lineages.keys()),
                                                            [True, False])))
def test_sfs_counts(k, folded):
    """Test to make sure converting momi demography to ms cmd works"""
    #demo = demo_funcs[k]()
    f = globals()[k]
    demo = f()
    n = n_lineages[f]
    check_sfs_counts(demo, demo.leafs, n, folded=folded)


def check_sfs_counts(demo, sampled_pops, sampled_n, theta=2.5, rho=2.5, num_loci=1000, num_bases=1e5, folded=False):
    #seg_sites = demo.simulate_data(
    #    sampled_pops, sampled_n,
    #    mutation_rate=theta / num_bases,
    #    recombination_rate=rho / num_bases,
    #    length=num_bases,
    #    num_replicates=num_loci,
    #)
    #sfs_list = seg_sites.sfs
    demo.set_mut_rate(muts_per_gen=theta / num_bases)
    data = demo.simulate_data(
        length=num_bases, recoms_per_gen=rho / num_bases,
        num_replicates=num_loci,
        sampled_n_dict=dict(zip(sampled_pops, sampled_n)))
    sfs_list = data.extract_sfs(None)

    if folded:
        # pass
        sfs_list = sfs_list.fold()

    #config_list = sorted(set(sum([sfs.keys() for sfs in sfs_list.loci],[])))

    #sfs_vals, branch_len = expected_sfs(demo, sfs_list.configs, folded=folded), expected_total_branch_len(
    #    demo, sampled_pops=sfs_list.sampled_pops, sampled_n=sfs_list.sampled_n)
    demo.set_data(sfs_list)
    sfs_vals = np.array(list(demo.expected_sfs().values()))
    theoretical = sfs_vals / num_loci

    # observed = np.zeros((len(sfs_list.configs), len(sfs_list.loci)))
    # for j in range(sfs_list.n_loci):
    #     for i,config in enumerate(sfs_list.configs):
    #         observed[i,j] = sfs_list.freq(config,locus=j)
    observed = np.array(sfs_list.freqs_matrix.todense())

    labels = list(sfs_list.configs)

    p_val = my_t_test(labels, theoretical, observed)
    print("p-value of smallest p-value under beta(1,num_configs)\n", p_val)
    cutoff = 0.05
    #cutoff = 1.0
    assert p_val > cutoff


def my_t_test(labels, theoretical, observed, min_samples=25):

    assert theoretical.ndim == 1 and observed.ndim == 2
    assert len(theoretical) == observed.shape[
        0] and len(theoretical) == len(labels)

    n_observed = np.sum(observed > 0, axis=1)
    theoretical, observed = theoretical[
        n_observed > min_samples], observed[n_observed > min_samples, :]
    labels = np.array(list(map(str, labels)))[n_observed > min_samples]
    n_observed = n_observed[n_observed > min_samples]

    runs = observed.shape[1]
    observed_mean = np.mean(observed, axis=1)
    bias = observed_mean - theoretical
    variances = np.var(observed, axis=1)

    t_vals = bias / np.sqrt(variances) * np.sqrt(runs)

    # get the p-values
    abs_t_vals = np.abs(t_vals)
    p_vals = 2.0 * scipy.stats.t.sf(abs_t_vals, df=runs - 1)
    print("# labels, p-values, empirical-mean, theoretical-mean, nonzero-counts")
    toPrint = np.array([labels, p_vals, observed_mean,
                        theoretical, n_observed]).transpose()
    toPrint = toPrint[np.array(toPrint[:, 1], dtype='float').argsort()[
        ::-1]]  # reverse-sort by p-vals
    print(toPrint)

    print("Note p-values are for t-distribution, which may not be a good approximation to the true distribution")

    # p-values should be uniformly distributed
    # so then the min p-value should be beta distributed
    return scipy.stats.beta.cdf(np.min(p_vals), 1, len(p_vals))


if __name__ == "__main__":
    #     demo = Demography.from_ms(1.0," ".join(sys.argv[3:]))
    #     check_sfs_counts(demo, mu=float(sys.argv[2]), num_loci=int(sys.argv[1]))
    demo = simple_admixture_3pop()
    demo = demo.demo_hist._get_multipop_moran(demo.pops, demo.n)
    treeseq = demo.simulate_trees(mutation_rate=1)
    seg_sites = demo.simulate_data(mutation_rate=1)
