from size_history import ConstantTruncatedSizeHistory
from demography import Demography
import pytest
import networkx as nx
import random
from sum_product import SumProduct

@pytest.fixture
def demo():
    test_newick = """
        (a:1[&&momi:model=constant:N=2.0:lineages=10],
         b:1[&&momi:model=constant:N=1.5:lineages=8]):3[&&momi:model=constant:N=10.0];
         """
    return Demography.from_newick(test_newick)


def test_from_newick(demo):
    assert demo.n_lineages_at_node[demo.root] == 18

def test_update_state(demo):
    demo.update_state({'a': {'derived': 5, 'ancestral': 5},
                       'b': {'derived': 8, 'ancestral': 0}})
    # Partial updates are allowed
    demo.update_state({'a': {'derived': 5, 'ancestral': 5}})
    with pytest.raises(Exception):
        demo.update_state({'a': {'derived': 5, 'ancestral': 5},
                           'b': {'derived': 4, 'ancestral': 0}})

def test_unsupported_model():
    test_newick = """
        (a:1[&&momi:model=exponential:N=2.0:lineages=10],
         b:1[&&momi:model=constant:N=1.5:lineages=8]):3[&&momi:model=constant:N=10.0];
         """
    with pytest.raises(Exception):
        demo = Demography.from_newick(test_newick)

def test_requires_lineages():
    with pytest.raises(Exception):
        Demography.from_newick("(a:1,b:1)") 

def test_admixture_demo():
    leaf_pops = ['a','bcd']
    # events: ('bc','d'), ('b','c'), 'bd', 'abd', 'abcd'
    eventList = [(('bc','bcd'), ('d','bcd')), # 'bcd' splits into 'bc','d'
                 (('b','bc'), ('c','bc')), # 'bc' splits into 'b','c'
                 (('bd','b'), ('bd','d')), # b,d coalesces into bd
                 (('abd','bd'),('abd','a')), # a,bd coalesce into abd
                 (('abcd','abd'),('abcd','c')), # abd,c coalesce into abcd
                 ]
    demoEdgeList = []
    for e1,e2 in eventList:
        demoEdgeList += [e1,e2]
    demo = nx.DiGraph(demoEdgeList)
    nd = dict(demo.nodes(data=True))
    nd['a']['lineages'] = 2
    nd['bcd']['lineages'] = 2
    bcdProb,bcProb = random.uniform(0,1), random.uniform(0,1)
    nd['bcd']['splitprobs'] = {'bc' : bcdProb, 'd' : 1-bcdProb}
    nd['bc']['splitprobs'] = {'b' : bcProb, 'c' : 1-bcProb}
    demo = Demography(demo) 
    for v in demo:
        nd = demo.node_data[v]
        n_sub = demo.n_lineages_at_node[v]
        if v == demo.root:
            tau = float('inf')
        else:
            tau = random.uniform(.1,10.)
        nd['model'] = ConstantTruncatedSizeHistory(N=random.uniform(.1,10.),
                                                   tau= tau,
                                                   n_max=n_sub)
        
    demo.update_state({'a' : {'derived' : 1, 'ancestral' : 1},
                       'bcd' : {'derived' : 1, 'ancestral' : 1}})
    assert SumProduct(demo).p() >= 0.0
