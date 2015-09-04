from __future__ import division
from momi import make_demography, expected_sfs, expected_total_branch_len, simulate_ms, sfs_list_from_ms, run_ms
import pytest
import random
import autograd.numpy as np
import scipy, scipy.stats
import itertools
import sys

def simple_admixture_demo(x, n_lins):
    t = np.cumsum(np.exp(x[:5]))
    p = 1.0 / (1.0 + np.exp(x[5:]))
    return make_demography("-I 2 %d %d -es $2 2 $3 -es $0 2 $1 -ej $4 #3 #4 -ej $t3 #4 1 -ej $t4 2 1" % (n_lins['1'], n_lins['2']), 
                           t[0], p[0], t[1], p[1], t[2], t3=t[3], t4=t[4])

def admixture_demo_str():
    x = np.random.normal(size=7)
    t = np.cumsum(np.exp(x[:5]))
    p = 1.0 / (1.0 + np.exp(x[5:]))

    ms_cmd = "-I 2 5 5 -es %f 2 %f -es %f 2 %f -ej %f 3 4 -ej %f 4 1 -ej %f 2 1" % (t[0], p[0], t[1], p[1], t[2], t[3], t[4])
    return ms_cmd
    
def exp_growth_str():
    n = 10
    growth_rate = random.uniform(-50,50)
    N_bottom = random.uniform(0.1,10.0)
    tau = .01
    demo_cmd = "-I 1 %d -G %f -eG %f 0.0" % (n, growth_rate, tau)
    return demo_cmd


def tree_demo_2_str():
    demo_cmd = "-I 2 4 4 -ej %f 2 1" % (2 * np.random.random() + 0.1,)
    return demo_cmd

def tree_demo_4_str():
    n = [2,2,2,2]

    times = np.random.random(len(n)-1) * 2.0 + 0.1
    for i in range(1,len(times)):
        times[i] += times[i-1]

    demo_cmd = "-I %d %s -ej %f 2 1 -ej %f 3 1 -ej %f 4 1" % tuple([len(n), " ".join(map(str,n))] + list(times))
    return demo_cmd

demo_funcs = (admixture_demo_str, exp_growth_str, tree_demo_2_str, tree_demo_4_str)
demo_funcs = {f.__name__ : f for f in demo_funcs}

@pytest.mark.parametrize("k", 
                         demo_funcs.keys())
def test_sfs_counts1(k):
    """Test to make sure converting ms cmd to momi demography works"""
    check_sfs_counts(demo_ms_cmd=demo_funcs[k]())

@pytest.mark.parametrize("k", 
                         demo_funcs.keys())    
def test_sfs_counts2(k):
    """Test to make sure converting momi demography to ms cmd works"""
    check_sfs_counts(demo=make_demography(demo_funcs[k]()))


def check_sfs_counts(demo_ms_cmd=None, demo=None, mu=10.0, num_ms_samples=1000):
    assert (demo_ms_cmd is None) != (demo is None)

    if demo is None:
        demo = make_demography(demo_ms_cmd)
        n = sum(demo.n_at_leaves)

        ms_cmd = "%d %d -t %f %s -seed %s" % (n, num_ms_samples, mu, demo_ms_cmd,
                                              " ".join([str(random.randint(0,999999999))
                                                        for _ in range(3)]))

        sfs_list = sfs_list_from_ms(run_ms(ms_cmd))
    elif demo_ms_cmd is None:        
        sfs_list = sfs_list_from_ms(simulate_ms(demo, num_ms_samples, mu=mu))
    
    config_list = sorted(set(sum([sfs.keys() for sfs in sfs_list],[])))
    
    sfs_vals,branch_len = expected_sfs(demo, config_list), expected_total_branch_len(demo)
    theoretical = sfs_vals * mu

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
    check_sfs_counts(demo, mu=float(sys.argv[2]), num_ms_samples=int(sys.argv[1]))
