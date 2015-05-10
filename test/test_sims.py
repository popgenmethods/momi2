from __future__ import division
from momi import make_demography, compute_sfs, aggregate_sfs, simulate_ms, sfs_list_from_ms
import pytest
import random
import autograd.numpy as np
import scipy, scipy.stats
import itertools
import sys

default_theta = 10.0
default_num_ms_samples = 1000

def simple_admixture_demo(x, n_lins):
    t = np.cumsum(np.exp(x[:5]))
    #for i in range(1,len(t)):
    #    t[i] += t[i-1]
    p = 1.0 / (1.0 + np.exp(x[5:]))
    return make_demography("-I 2 %d %d -es $2 2 $3 -es $0 2 $1 -ej $4 #3 #4 -ej $t3 #4 1 -ej $t4 2 1" % (n_lins['1'], n_lins['2']), 
                           t[0], p[0], t[1], p[1], t[2], t3=t[3], t4=t[4])

def test_admixture():
    n = {'1':5,'2':5}
    check_sfs_counts(simple_admixture_demo(np.random.normal(size=7),n))

def test_exp_growth():
    n = 10
    growth_rate = random.uniform(-50,50)
    N_bottom = random.uniform(0.1,10.0)
    tau = .01
    demo = make_demography("-I 1 %d -G $0 -eG $1 0.0" % n,
                           growth_rate,
                           tau)
    check_sfs_counts(demo)


def test_tree_demo_2():
    n = [4,4]
    demo = make_demography("-I %d %s -ej $0 2 1" % (len(n), " ".join(map(str,n))), 
                           2 * np.random.random() + 0.1)
    check_sfs_counts(demo)

def test_tree_demo_4():
    n = [2,2,2,2]

    times = np.random.random(len(n)-1) * 2.0 + 0.1
    for i in range(1,len(times)):
        times[i] += times[i-1]

    demo = make_demography("-I %d %s -ej $0 2 1 -ej $1 3 1 -ej $2 4 1" % (len(n), " ".join(map(str,n))), *times)
    check_sfs_counts(demo)


def check_sfs_counts(demo, theta=default_theta, num_ms_samples=default_num_ms_samples):
    print demo.graph['cmd']

    sfs_list = sfs_list_from_ms(simulate_ms(demo, num_ms_samples, theta=theta),
                                demo.n_at_leaves)
    config_list = sorted(set(sum([sfs.keys() for sfs in sfs_list],[])))
    
    sfs_vals,branch_len = compute_sfs(demo, config_list)
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
    
    # p-values should be uniformly distributed
    # so then the min p-value should be beta distributed
    return scipy.stats.beta.cdf(np.min(p_vals), 1, len(p_vals))


if  __name__=="__main__":
    demo = make_demography(" ".join(sys.argv[3:]))
    check_sfs_counts(demo, theta=float(sys.argv[2]), num_ms_samples=int(sys.argv[1]))
