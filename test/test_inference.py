from __future__ import division
import sh
import os
import scipy.optimize
import itertools
import math
import re
from collections import Counter, defaultdict
from pprint import pprint
import random
import numpy as np
import newick
from msdemo import get_demo

from sum_product import SumProduct
from demography import Demography

scrm = lambda x: sh.Command(os.environ["SCRM_PATH"])(*(x.split()))
# _scrm = sh.Command(os.environ["MSPATH"])
# def scrm(*x):
#     return _scrm(*x,_ok_code=[0,16,17,18])

def test_joint_sfs_inference():
    N0=1.0
    theta=1.0
    t0=random.uniform(.25,2.5)
    t1= t0 + random.uniform(.5,5.0)
    num_runs = 10000

    def scrm_cmd(join_time):
        return "-I 3 1 1 1 -ej %f 1 2 -ej %f 2 3" % (join_time / 2. * N0, t1 / 2. * N0)

    true_demo = get_demo(scrm_cmd(t0))

    jsfs,sqCounts,nonzero = run_scrm(true_demo, num_runs, theta=theta)
    totalSnps = sum([v for k,v in jsfs.items()])
    logFactorialTotalSnps = sum([math.log(x) for x in range(1,totalSnps)])

    pprint(dict(jsfs))
    print(t0,t1)
    def f(join_time):
        demo = get_demo(scrm_cmd(join_time))
        lambd = theta / 2.0 * num_runs * demo.totalSfsSum
        # poisson probability for total # snps is e^-lambd * lambd^totalSnps / totalSnps!
        ret = lambd + logFactorialTotalSnps - totalSnps * math.log(lambd)
        for states, weight in sorted(jsfs.items()):
            st = {a: {'derived': b, 'ancestral': 1 - b} for a, b in zip("123", states)}
            demo.update_state(st)
            sp = SumProduct(demo)
            #print(weight, states, sp.p(normalized=True))
            ret -= weight * math.log(sp.p(normalized=True))
            #ret -= weight * math.log(sp.p())
        return ret
    #f(t0)
    res = scipy.optimize.fminbound(f, 0, t1, xtol=.01)
    #res = scipy.optimize.minimize(f, random.uniform(0,t1), bounds=((0,t1),))
    #assert res == t0
    assert abs(res - t0) / t0 < .05
#    print (res, t0, t1)
#    print(res.x, t0, t1)

def run_scrm_example(N0, theta, t0, t1, num_runs):
    t0 /= 2. * N0
    t1 /= 2. * N0
    scrm_args = [3, num_runs, '-t', theta, '-I', 3, 1, 1, 1, '-ej', t1, 2, 3, '-ej', t0, 1, 2]
    return run_scrm(scrm_args, (1,1,1))

def run_scrm(demo, num_sims, theta=None, seed=None, additionalParams=""):
    if any([(x in additionalParams) for x in "-t","-T","seed"]):
        raise IOError("additionalParams should not contain -t,-T,-seed,-seeds")

    lins_per_pop = [demo.n_lineages_subtended_by[l] for l in sorted(demo.leaves)]
    n = sum(lins_per_pop)
    pops_by_lin = []
    for pop in range(len(lins_per_pop)):
        for i in range(lins_per_pop[pop]):
            pops_by_lin.append(pop)
    assert len(pops_by_lin) == int(n)

    scrm_args = demo.graph['cmd']
    if additionalParams:
        scrm_args = "%s %s" % (scrm_args, additionalParams)

    if seed is None:
        seed = random.randint(0,999999999)
    scrm_args = "%s --seed %s" % (scrm_args, str(seed))

    assert scrm_args.startswith("-I ")
    if not theta:
        scrm_args = "-T %s" % scrm_args
    else:
        scrm_args = "-t %f %s" % (theta, scrm_args)
    scrm_args = "%d %d %s" % (n, num_sims, scrm_args)

    def f(x):
        if x == "//":
            f.i += 1
        return f.i
    f.i = 0
    runs = itertools.groupby((line.strip() for line in scrm(scrm_args)), f)
    next(runs)
    sumCounts = Counter()
    sumSqCounts = Counter()
    nonzeroCounts = Counter()
    for i, lines in runs:
        lines = list(lines)
        if theta:
            currCounts = read_empirical_sfs(lines, len(lins_per_pop), pops_by_lin)
        else:
            currCounts = read_tree_lens(lines, len(lins_per_pop), pops_by_lin)

        for config in currCounts:
            sumCounts[config] += currCounts[config]
            sumSqCounts[config] += currCounts[config]**2
            nonzeroCounts[config] += 1
    return sumCounts,sumSqCounts,nonzeroCounts

def read_empirical_sfs(lines, num_pops, pops_by_lin):
        currCounts = Counter()
        n = len(pops_by_lin)

        assert lines[0] == "//"
        nss = int(lines[1].split(":")[1])
        if nss == 0:
            return currCounts
        # remove header
        lines = lines[3:]
        # remove trailing line if necessary
        if len(lines) == n+1:
            assert lines[n] == ''
            lines = lines[:-1]
        # number of lines == number of haplotypes
        assert len(lines) == n
        # columns are snps
        for column in range(len(lines[0])):
            dd = [0] * num_pops
            for i, line in enumerate(lines):
                dd[pops_by_lin[i]] += int(line[column])
            assert sum(dd) > 0
            currCounts[tuple(dd)] += 1
        return currCounts

def read_tree_lens(lines, num_pops, pops_by_lin):
    assert lines[0] == "//"
    return NewickSfs(lines[1], num_pops, pops_by_lin).sfs

class NewickSfs(newick.tree.TreeVisitor):
    def __init__(self, newick_str, num_pops, pops_by_lin):
        self.tree = newick.parse_tree(newick_str)
        self.sfs = Counter()
        self.num_pops = num_pops
        self.pops_by_lin = pops_by_lin

        self.tree.dfs_traverse(self)

    def pre_visit_edge(self,src,b,len,dst):
        dd = [0] * self.num_pops
        # get the # lineages in each pop below edge
        for leaf in dst.get_leaves_identifiers():
            dd[self.pops_by_lin[int(leaf)-1]] += 1
        # add length to the sfs entry. multiply by 2 cuz of ms format
        self.sfs[tuple(dd)] += len * 2.0


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

if __name__ == "__main__":
    # test_jeff()
    test_joint_sfs_inference()
