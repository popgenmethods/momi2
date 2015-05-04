from momi import make_demography, compute_sfs
import random
import itertools
import networkx as nx

from autograd.numpy import log
import autograd.numpy as np

from test_sims import simple_admixture_demo

def random_tree_demo(num_leaf_pops, lins_per_pop):
    cmd = "-I %d %s" % (num_leaf_pops, " ".join([str(lins_per_pop)] * num_leaf_pops))
    for i in range(num_leaf_pops):
        cmd += " -n %d %f" % (i+1, random.expovariate(1.0))
    roots = set([i+1 for i in range(num_leaf_pops)])
    t = 0.0
    while len(roots) > 1:
        i,j = random.sample(roots, 2)
        t += random.expovariate(1.0)
        cmd += " -ej %f %d %d" % (t, i, j)
        roots.remove(i)
        cmd += " -en %f %d %f" % (t, j, random.expovariate(1.0))
    return make_demography(cmd)

def check_demo_normalization(demo):
    leaves = sorted(list(demo.leaves))
    ranges = [range(demo.n_lineages(l)+1) for l in demo.leaves]

    #totalSum = 0.0
    config_list = []
    for n_derived in itertools.product(*ranges):
        if sum(n_derived) == 0 or sum(n_derived) == sum(map(lambda x: len(x) - 1, ranges)):
            continue #skip monomorphic sites
        config_list.append(n_derived)
    sfs, branch_len = compute_sfs(demo, config_list)
    assert abs(log(np.sum(sfs) / branch_len)) < 1e-12

def test_tree_demo_normalization():
    lins_per_pop=2
    num_leaf_pops=3

    seed = random.randint(0,999999999)
    print("seed",seed)
    random.seed(seed)

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_demo_normalization(demo)

def test_admixture_demo_normalization():
    demo = simple_admixture_demo(np.random.normal(size=7), {'1':2,'2':2})

    check_demo_normalization(demo)
