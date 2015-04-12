from size_history import ConstantTruncatedSizeHistory, PiecewiseHistory, ExponentialTruncatedSizeHistory
from demography import Demography
import pytest
import networkx as nx
import random
from sum_product import SumProduct
import numpy as np
import scipy, scipy.stats
import itertools

#theta = 1.0
#num_scrm_samples = 10000
#num_scrm_samples = 5000
num_scrm_samples = 1000
#num_scrm_samples = 10
#theta = .01
#num_scrm_samples = 100000

def test_exp_growth():
    n = 10
    tau = .01
    growth_rate = random.uniform(-500,500)
    N_top=random.uniform(0.1,10.0)
    N_bottom = N_top * np.exp(growth_rate * tau)

    scrm_args = "-I 1 %d -G %f -eN %f %f" % (n, growth_rate * N_bottom * 2.0, tau / N_bottom / 2.0, N_top / N_bottom)
    
    check_sfs_counts(scrm_args)


def test_tree_demo_2():
    n = [4,4]
    scrm_args = "-I %d %s -ej %f 2 1" % (len(n), " ".join(map(str,n)), 2 * np.random.random() + 0.1)
    check_sfs_counts(scrm_args)

def test_tree_demo_4():
    n = [2,2,2,2]

    times = np.random.random(len(n)-1) * 2.0 + 0.1
    for i in range(1,len(times)):
        times[i] += times[i-1]

    scrm_args = "-I %d %s -ej %f 2 1 -ej %f 3 1 -ej %f 4 1" % (len(n), " ".join(map(str,n)), times[0], times[1], times[2])
    check_sfs_counts(scrm_args)


def check_sfs_counts(scrm_args):
    demo = Demography.from_ms(scrm_args)
    leaf_lins = {l : demo.n_lineages_subtended_by[l] for l in demo.leaves}
    leaf_pops = sorted(list(leaf_lins.keys()))

#     scrm_args = scrm_args.split()
    #scrm_args += ['-seed', random.randint(0,999999999)]

    #empirical_sfs,sqCounts,nonzeroCounts = run_scrm(scrm_args, tuple([leaf_lins[v] for v in leaf_pops]))
    #empirical_sfs,sqCounts,nonzeroCounts = run_scrm(demo, num_scrm_samples)
    empirical_sfs,sqCounts,nonzeroCounts = demo.simulate_sfs(num_scrm_samples)
    
    theoretical_sfs = {}
    ranges = [range(leaf_lins[v]+1) for v in leaf_pops]
    total_lins = sum([leaf_lins[v] for v in leaf_pops])
    #for sfs_entry in empirical_sfs:
    for sfs_entry in itertools.product(*ranges):
        sfs_entry = tuple(sfs_entry)
        if sum(sfs_entry) == 0 or sum(sfs_entry) == total_lins:
            continue # skip polymorphic sites
        state = {leaf_pops[i] : {'derived' : sfs_entry[i]} for i in range(len(sfs_entry))}
        for v in state:
            state[v].update({'ancestral' : leaf_lins[v] - state[v]['derived']})
        demo.update_state(state)
        theoretical_sfs[sfs_entry] = SumProduct(demo).p()
        #theoretical_sfs[sfs_entry] = SumProduct(demo).p() * float(num_scrm_samples) * theta / 2.0
    #p_val = sfs_p_value(empirical_sfs, sqCounts, theoretical_sfs, num_scrm_samples, theta)
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
        return np.array([sfs[x] for x in configs])
    
    empirical_sfs = sfsArray(empirical_sfs)
    squaredCounts = sfsArray(squaredCounts)
    theoretical_sfs = sfsArray(theoretical_sfs)
    nonzeroCounts = sfsArray(nonzeroCounts)

    means = empirical_sfs / float(runs)
    sqMeans = squaredCounts / float(runs)
    bias = means - theta / 2.0 * theoretical_sfs
    # estimated variance = empirical variance + bias^2
    variances = sqMeans - np.square(means) + np.square(bias)
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
