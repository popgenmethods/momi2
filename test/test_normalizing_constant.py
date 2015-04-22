from demography import Demography
import random
import itertools
import networkx as nx
from sum_product import SumProduct
from autograd.numpy import log

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
    return Demography.from_ms(cmd)

def test_tree_demo_normalization():
    lins_per_pop=2
    num_leaf_pops=3

    seed = random.randint(0,999999999)
    print("seed",seed)
    random.seed(seed)

    demo = random_tree_demo(num_leaf_pops, lins_per_pop)
    print(demo)
    #print(demo.to_newick())
    
    leaves = sorted(list(demo.leaves))

    ranges = [range(lins_per_pop+1)] * num_leaf_pops
    totalSum = 0.0
    for n_derived in itertools.product(*ranges):
        if sum(n_derived) == 0 or sum(n_derived) == num_leaf_pops * lins_per_pop:
            continue #skip monomorphic sites
        state = {}
        for i in range(len(leaves)):
            state[leaves[i]] = {'derived' : n_derived[i], 'ancestral' : lins_per_pop - n_derived[i]}
        demo.update_state(state)

        totalSum = SumProduct(demo).p(normalized=True) + totalSum

    assert abs(log(totalSum / 1.0)) < 1e-12    
    #assert totalSum == 1.0
