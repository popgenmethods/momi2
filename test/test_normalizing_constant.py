from demography import Demography
import random
import itertools
import networkx as nx
from sum_product import SumProduct
from size_history import ConstantTruncatedSizeHistory
import math
from adarray import admath

def random_tree_demo(num_leaf_pops, lins_per_pop):
    currPop = 0
    clades = set([currPop])
    edges = []
    while len(clades) < num_leaf_pops:
        parentPop = random.choice(list(clades))
        clades.remove(parentPop)
        for i in range(2):
            currPop += 1
            clades.add(currPop)
            edges.append( (parentPop, currPop) )

    demo = nx.DiGraph(edges)
    demo.add_nodes_from(range(currPop+1))

    nd = dict(demo.nodes(data=True))
    for leaf in clades:
        nd[leaf]['lineages'] = lins_per_pop

    demo = Demography(demo) 
    for v in demo:
        if v == demo.root:
            tau = float('inf')
        else:
            tau = random.expovariate(1.0)
        nd = demo.node_data[v]
        n_sub = demo.n_lineages_subtended_by[v]
        nd['model'] = ConstantTruncatedSizeHistory(N=random.expovariate(1.0),
                                                   tau=tau,
                                                   n_max=n_sub)
    return demo


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

    assert abs(admath.log(totalSum / 1.0)) < 1e-12    
    #assert totalSum == 1.0
