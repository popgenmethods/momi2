from __future__ import division
from momi import Demography, expected_sfs, expected_total_branch_len, simulate_ms, sfs_list_from_ms, run_ms
import momi
import pytest
import random
import autograd.numpy as np
import scipy, scipy.stats
import itertools
import sys

from demo_utils import *

import os
try:
    ms_path = os.environ["MS_PATH"]
except KeyError:
    raise Exception("Need to set system variable $MS_PATH, or run py.test with 'MS_PATH=/path/to/ms py.test ...'")


demo_funcs = {f.__name__ : f for f in [simple_admixture_demo, simple_two_pop_demo, piecewise_constant_demo, exp_growth_model]}


@pytest.mark.parametrize("k,folded",
                         ((fname, bool(b))
                          for fname,b in zip(demo_funcs.keys(),
                                             np.random.choice([True,False],len(demo_funcs)))))
def test_sfs_counts(k,folded):
    """Test to make sure converting momi demography to ms cmd works"""
    check_sfs_counts(demo=demo_funcs[k]().rescaled(), folded=folded)


def check_sfs_counts(demo, theta=10.0, rho=10.0, num_loci=1000, folded=False):
    sfs_list = sfs_list_from_ms(simulate_ms(ms_path, demo, num_loci=num_loci, mut_rate=theta, additional_ms_params='-r %f %d' % (rho, num_loci)))

    if folded:
        sfs_list = [momi.util.folded_sfs(sfs, demo.sampled_n) for sfs in sfs_list]
        
    config_list = sorted(set(sum([sfs.keys() for sfs in sfs_list],[])))
    
    sfs_vals,branch_len = expected_sfs(demo, config_list, folded=folded), expected_total_branch_len(demo)
    theoretical = sfs_vals * theta

    observed = np.zeros((len(config_list), len(sfs_list)))
    for j,sfs in enumerate(sfs_list):
        for i,config in enumerate(config_list):
            try:
                observed[i,j] = sfs[config]
            except KeyError:
                pass

    labels = list(config_list)

    p_val = my_t_test(labels, theoretical, observed)
    print "p-value of smallest p-value under beta(1,num_configs)\n",p_val
    cutoff = 0.05
    #cutoff = 1.0
    assert p_val > cutoff


def my_t_test(labels, theoretical, observed, min_samples=25):

    assert theoretical.ndim == 1 and observed.ndim == 2
    assert len(theoretical) == observed.shape[0] and len(theoretical) == len(labels)

    n_observed = np.sum(observed > 0, axis=1)
    theoretical, observed = theoretical[n_observed > min_samples], observed[n_observed > min_samples, :]
    labels = np.array(map(str,labels))[n_observed > min_samples]
    n_observed = n_observed[n_observed > min_samples]

    runs = observed.shape[1]
    observed_mean = np.mean(observed,axis=1)
    bias = observed_mean - theoretical
    variances = np.var(observed,axis=1)

    t_vals = bias / np.sqrt(variances) * np.sqrt(runs)

    # get the p-values
    abs_t_vals = np.abs(t_vals)
    p_vals = 2.0 * scipy.stats.t.sf(abs_t_vals, df=runs-1)
    print("# labels, p-values, empirical-mean, theoretical-mean, nonzero-counts")
    toPrint = np.array([labels, p_vals, observed_mean, theoretical, n_observed]).transpose()
    toPrint = toPrint[np.array(toPrint[:,1],dtype='float').argsort()[::-1]] # reverse-sort by p-vals
    print(toPrint)

    print("Note p-values are for t-distribution, which may not be a good approximation to the true distribution")
    
    # p-values should be uniformly distributed
    # so then the min p-value should be beta distributed
    return scipy.stats.beta.cdf(np.min(p_vals), 1, len(p_vals))


if  __name__=="__main__":
    demo = Demography.from_ms(1.0," ".join(sys.argv[3:]))
    check_sfs_counts(demo, mu=float(sys.argv[2]), num_loci=int(sys.argv[1]))
