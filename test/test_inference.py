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
from adarray import gh, adnumber, admath, array, sum
import networkx as nx
from size_history import ConstantTruncatedSizeHistory

from sum_product import SumProduct
from demography import Demography

scrm = sh.Command(os.environ["SCRM_PATH"])
# _scrm = sh.Command(os.environ["MSPATH"])
# def scrm(*x):
#     return _scrm(*x,_ok_code=[0,16,17,18])

def get_exact_jsfs(demo, theta=2.0):
    leafs = sorted(list(demo.leaves))
    n_ders = [range(demo.node_data[l]['lineages']+1) for l in leafs]
    total_n = sum([demo.node_data[l]['lineages'] for l in leafs])
    jsfs = {}
    for comb in itertools.product(*n_ders):
        assert len(comb) == len(leafs)
        state = {}
        total_der = 0
        for l,n_d in zip(leafs,comb):
            total_der += n_d
            state[l] = {'derived' : n_d, 'ancestral' : demo.node_data[l]['lineages'] - n_d}
        if total_der == 0 or total_der == total_n:
            continue
        demo.update_state(state)
        weight = SumProduct(demo).p(normalized=False) * theta / 2.0
        state = tuple([state[l]['derived'] for l in leafs])
        jsfs[state] = weight
    return jsfs

def test_joint_sfs_inference():
    def get_demo(t0, N0, t1, rt0):
        demo = nx.DiGraph([('ab','a'),('ab','b'),('abc','ab'),('abc','c')])
        nd = dict(demo.nodes(data=True))
        nd['a']['lineages'] = 1
        nd['b']['lineages'] = 1
        nd['c']['lineages'] = 1
        nd['a']['model'] = nd['b']['model'] = ConstantTruncatedSizeHistory(N=N0,
                                                                           tau=t0,
                                                                           n_max=1)
        nd['ab']['model'] = ConstantTruncatedSizeHistory(N=N0, tau=rt0, n_max=2)
        nd['c']['model'] = ConstantTruncatedSizeHistory(N=N0, tau=t1, n_max=1)
        nd['abc']['model'] = ConstantTruncatedSizeHistory(N=N0, tau=float('inf'), n_max=3)
        return Demography(demo)
    N0=1.0
    theta=1.0
    t0=random.uniform(.25,2.5)
    t1= t0 + random.uniform(.5,5.0)
    num_runs = 1000
    jsfs,sqCounts,nonzero = run_scrm_example(N0, theta, t0, t1, num_runs)
    pprint(dict(jsfs))
    true_demo = get_demo(t0=t0,N0=N0,t1=t1,rt0=t1-t0)
    jsfs_exact = get_exact_jsfs(true_demo, theta=theta*num_runs)
    pprint(dict(jsfs_exact))

    jsfs = jsfs_exact
    totalSnps = sum(array([v for k,v in jsfs.items()]))

    print(t0,t1)
    def f(join_time):
        print(join_time[0],N0,t1,-join_time[0]+t1)
        #tree = newick_tpl.format(t0=join_time, N0=N0, t1=t1, rt0=t1 - join_time)
        #demo = Demography.from_newick(tree)
        demo = get_demo(t0=join_time[0], N0=N0, t1=t1, rt0= -join_time[0]+t1)
        #totalSum = NormalizingConstant(demo).normalizing_constant()
        #ret = 0.0
        lambd = demo.totalSfsSum * theta / 2.0 * num_runs
        # poisson probability for total # snps is e^-lambd * lambd^totalSnps / totalSnps!
        ret = lambd - totalSnps * admath.log(lambd)
        for states, weight in sorted(jsfs.items()):
            st = {a: {'derived': b, 'ancestral': 1 - b} for a, b in zip("abc", states)}
            demo.update_state(st)
            #true_demo.update_state(st)
            #weight = SumProduct(true_demo).p(normalized=True)
            sp = SumProduct(demo)
            #print(weight, states, sp.p(normalized=True))
            ret -= weight * admath.log(sp.p(normalized=True))
            #ret -= weight * math.log(sp.p())
        print ret
        return ret
    #f(t0)
    #res = scipy.optimize.fminbound(f, 0, t1, xtol=.01)
    x0 = random.uniform(0,t1)
    #res = scipy.optimize.minimize(f, x0, bounds=((0,t1),), method='L-BFGS-B')
    #assert res == t0
    grad, hess = gh(f)
    #res = scipy.optimize.minimize(f, x0, method='L-BFGS-B', jac=grad, bounds=((0,t1),), options={'ftol': 1e-8, 'disp': False})
    res = scipy.optimize.minimize(f, x0, method='L-BFGS-B', jac=grad, bounds=((0,t1),))
    print res.jac
    assert abs(res.x - t0) / t0 < .05
#    print (res, t0, t1)
#    print(res.x, t0, t1)

def test_jeff():
    return True
    states = """170.00000       A:nean human chimp      D:sima deni
    124.00000       A:deni nean chimp       D:human sima
    231.00000       A:chimp sima    D:human nean deni
    1055.00000      A:chimp deni human sima D:nean
    1300.00000      A:chimp D:human sima nean deni
    136.00000       A:deni human chimp      D:sima nean
    157.00000       A:chimp nean sima       D:human deni
    1094.00000      A:chimp deni nean sima  D:human
    202.00000       A:chimp deni sima       D:human nean
    121.00000       A:nean chimp    D:sima human deni
    1178.00000      A:chimp nean human sima D:deni
    181.00000       A:chimp human sima      D:nean deni
    129.00000       A:human chimp   D:sima nean deni
    142.00000       A:deni chimp    D:human sima nean"""
    linere = re.compile(r"([0-9.]+)\s+A:([^D]*)D:(.*)$")
    st = []
    for line in states.split("\n"):
        weight, ancestral, derived = linere.match(line.strip()).groups()
        weight = float(weight)
        ancestral = ancestral.strip().split()
        derived = derived.strip().split()
        d = {k: {'ancestral': 1, 'derived': 0} for k in ancestral}
        d.update({k: {'ancestral': 0, 'derived': 1} for k in derived})
        if len(derived) > 1:
            st.append((weight, d, line))
    def f(j):
        newick_tpl = """((((nean:0.600000{params},deni:0.600000{params}):{nean_deni_j:f},sima:{sima_j:f}{params}):{j:f},human:0.800000{params}):7.200000,chimp:8.000000{params});"""
        newick = newick_tpl.format(params="[&&momi:model=constant:N=1:lineages=1]", j=j, sima_j=.8 - j, nean_deni_j=.2 - j)
        demo = Demography.from_newick(newick)
        ret = 0.0
        for weight, states, line in sorted(st):
            demo.update_state(states)
            sp = SumProduct(demo)
            ret -= weight * math.log(sp.p())
            print(line, math.log(sp.p()))
        print(j)
        return ret
    res = scipy.optimize.minimize_scalar(f, method="bounded", bounds=(0, 0.2))
    print(res.x)

def run_scrm_example(N0, theta, t0, t1, num_runs):
    t0 /= 2. * N0
    t1 /= 2. * N0
    scrm_args = [3, num_runs, '-t', theta, '-I', 3, 1, 1, 1, '-ej', t1, 2, 3, '-ej', t0, 1, 2]
    return run_scrm(scrm_args, (1,1,1))

def run_scrm(scrm_args, lins_per_pop):
    n = scrm_args[0]
    assert sum(lins_per_pop) == n
    pops_by_lin = []
    for pop in range(len(lins_per_pop)):
        for i in range(lins_per_pop[pop]):
            pops_by_lin.append(pop)
    assert len(pops_by_lin) == n

    print(scrm_args)
    assert scrm_args[2] in ['-t','-T']
    trees = scrm_args[2] == '-T'
    def f(x):
        if x == "//":
            f.i += 1
        return f.i
    f.i = 0
    runs = itertools.groupby((line.strip() for line in scrm(*scrm_args)), f)
    next(runs)
    sumCounts = Counter()
    sumSqCounts = Counter()
    nonzeroCounts = Counter()
    for i, lines in runs:
        lines = list(lines)
        if not trees:
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

if __name__ == "__main__":
    # test_jeff()
    test_joint_sfs_inference()
