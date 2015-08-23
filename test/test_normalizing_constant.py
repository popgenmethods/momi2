from momi import make_demography, expected_sfs, expected_total_branch_len
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

def check_demo_normalization(demo, min_freqs=1, **kwargs):
    leaves = sorted(list(demo.leaves))
    ranges = [range(demo.n_lineages(l)+1) for l in demo.leaves]

    config_list = list(itertools.product(*ranges))

    n_leaf_lins = np.array([demo.n_lineages(l) for l in demo.leaves])
    min_freqs = np.array(min_freqs) * np.ones(len(demo.leaves), dtype='i')
    if np.any(min_freqs < 1) or np.any(min_freqs > n_leaf_lins):
        raise Exception("Minimum frequencies must be in (0,num_lins] for each leaf pop")
    max_freqs = n_leaf_lins - min_freqs

    data = np.array(config_list, ndmin=2)
    attain_min_freq = np.logical_and(np.sum(data >= min_freqs, axis=1) > 0, # at least one entry above min_freqs
                                     np.sum(data <= max_freqs, axis=1) > 0 # at least one entry below max_freqs
                                     )
    data = data[attain_min_freq,:]
    
    config_list = list(tuple(x) for x in data)

    sfs, branch_len = expected_sfs(demo, config_list, **kwargs), expected_total_branch_len(demo, min_freqs=min_freqs, **kwargs)
    assert abs(log(np.sum(sfs) / branch_len)) < 1e-10

def test_tree_demo_normalization():
    lins_per_pop=2
    num_leaf_pops=3

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_demo_normalization(demo)

def test_admixture_demo_normalization():
    demo = simple_admixture_demo(np.random.normal(size=7), {'1':2,'2':2})

    check_demo_normalization(demo)

def test_tree_demo_errors_normalization():
    check_tree_demo_errors_normalization([1,2,3])

def test_tree_demo_errors_normalization2():
    check_tree_demo_errors_normalization(1)    
    
def check_tree_demo_errors_normalization(min_freqs):
    lins_per_pop=10
    num_leaf_pops=3

    error_matrices = [np.exp(np.random.randn(lins_per_pop+1,lins_per_pop+1)) for _ in range(num_leaf_pops)]
    error_matrices = [np.einsum('ij,j->ij', x, 1./np.sum(x, axis=0)) for x in error_matrices]

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    check_demo_normalization(demo, error_matrices=error_matrices, min_freqs=min_freqs)
