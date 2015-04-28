from demography import Demography
from sum_product import compute_sfs
import pytest
import networkx as nx
import random
import autograd.numpy as np
import scipy, scipy.stats
import itertools

#theta = 1.0
#num_scrm_samples = 10000
#num_scrm_samples = 5000
num_scrm_samples = 1000
#num_scrm_samples = 10
#theta = .01
#num_scrm_samples = 100000


def simple_admixture_demo(x, n_lins):
    t = np.cumsum(np.exp(x[:5]))
    #for i in range(1,len(t)):
    #    t[i] += t[i-1]
    p = 1.0 / (1.0 + np.exp(x[5:]))
    return Demography.from_ms("-I 2 %d %d -es $0 2 $1 -es $2 2 $3 -ej $4 4 3 -ej $5 3 1 -ej $6 2 1" % (n_lins['1'], n_lins['2']), 
                              t[0], p[0], t[1], p[1], t[2], t[3], t[4])

def test_admixture():
    n = {'1':2,'2':2}
    check_sfs_counts(simple_admixture_demo(np.random.normal(size=7),n))

def test_exp_growth():
    n = 10
    tau = .01
    growth_rate = random.uniform(-500,500)
    N_top=random.uniform(0.1,10.0)
    N_bottom = N_top * np.exp(growth_rate * tau)
   
    demo = Demography.from_ms("-I 1 %d -G $0 -eN $1 $2" % n,
                              growth_rate * N_bottom * 2.0,
                              tau / N_bottom / 2.0,
                              N_top / N_bottom)
    check_sfs_counts(demo)


def test_tree_demo_2():
    n = [4,4]
    demo = Demography.from_ms("-I %d %s -ej $0 2 1" % (len(n), " ".join(map(str,n))), 
                              2 * np.random.random() + 0.1)
    check_sfs_counts(demo)

def test_tree_demo_4():
    n = [2,2,2,2]

    times = np.random.random(len(n)-1) * 2.0 + 0.1
    for i in range(1,len(times)):
        times[i] += times[i-1]

    demo = Demography.from_ms("-I %d %s -ej $0 2 1 -ej $1 3 1 -ej $2 4 1" % (len(n), " ".join(map(str,n))),
                              *times)
    check_sfs_counts(demo)


def check_sfs_counts(demo):
    leaf_lins = {l : demo.n_lineages_at_node[l] for l in demo.leaves}
    leaf_pops = sorted(list(leaf_lins.keys()))

    empirical_sfs,sqCounts,nonzeroCounts = demo.simulate_sfs(num_scrm_samples)
    
    theoretical_sfs = {}
    ranges = [range(leaf_lins[v]+1) for v in leaf_pops]
    total_lins = sum([leaf_lins[v] for v in leaf_pops])
    config_list = sorted(empirical_sfs.keys())

    sfs_vals,_ = compute_sfs(demo, config_list)
    theoretical_sfs = {k:v for k,v in zip(config_list, sfs_vals)}

    p_val = sfs_p_value(nonzeroCounts, empirical_sfs, sqCounts, theoretical_sfs, num_scrm_samples)
    print(p_val)
    cutoff = 0.05
    #cutoff = 1.0
    assert p_val > cutoff

    #configs = sorted(empirical_sfs.keys())
    #assert scipy.stats.chisquare(sfsArray(empirical_sfs), sfsArray(theoretical_sfs))[1] >= .05
    #assert theoretical_sfs == empirical_sfs

# approximate empirical_sfs - theoretical_sfs / sd by standard normal
# use theta=2.0 if simulating trees instead of mutations
def sfs_p_value(nonzeroCounts, empirical_sfs, squaredCounts, theoretical_sfs, runs, theta=2.0, minSamples=25):
    configs = theoretical_sfs.keys()
    # throw away all the entries with too few observations (they will not be very Gaussian)
    configs = [x for x in configs if nonzeroCounts[x] > minSamples]
    def sfsArray(sfs):
        return np.array([float(sfs[x]) for x in configs])
    
    empirical_sfs = sfsArray(empirical_sfs)
    squaredCounts = sfsArray(squaredCounts)
    theoretical_sfs = sfsArray(theoretical_sfs)
    nonzeroCounts = sfsArray(nonzeroCounts)

    means = empirical_sfs / float(runs)
    sqMeans = squaredCounts / float(runs)
    bias = theoretical_sfs * theta / 2.0 - means
    # estimated variance = empirical variance + bias^2
    variances = bias**2 + sqMeans - np.square(means)
    variances *= runs / float(runs-1)

    # observed counts are Gaussian by CLT
    # empirical_mean - theoretical mean / estimated variance ~ t distribution with df runs-1
    t_vals = bias / np.sqrt(variances) * np.sqrt(runs)

    # get the p-values
    abs_t_vals = np.abs(t_vals)
    p_vals = 2.0 * scipy.stats.t.sf(abs_t_vals, df=runs-1)
    # print some stuff
    print("# configs, p-values, empirical-sfs, theoretical-sfs, nonzeroCounts")
    toPrint = np.array([configs, p_vals, empirical_sfs, theoretical_sfs * theta / 2.0 * runs, nonzeroCounts]).transpose()
    toPrint = toPrint[toPrint[:,1].argsort()[::-1]] # reverse-sort by p-vals
    #toPrint = toPrint[toPrint[:,0].argsort()] # sort by config
    print(toPrint)
    
    # p-values should be uniformly distributed
    # so then the min p-value should be beta distributed
    return scipy.stats.beta.cdf(np.min(p_vals), 1, len(p_vals))
