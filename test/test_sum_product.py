from __future__ import division
import pytest
from sum_product import SumProduct
from demography import Demography
from util import H, grouper

@pytest.fixture
def demo():
    test_newick = """
        (a:1[&&momi:model=constant:N=1.0:lineages=10],
         b:1[&&momi:model=constant:N=1.0:lineages=8]):3[&&momi:model=constant:N=10.0];
    """
    demo = Demography.from_newick(test_newick)
    demo.update_state({'a': {'derived': 5, 'ancestral': 5},
                       'b': {'derived': 8, 'ancestral': 0}})
    return demo

def test_sfs(demo):
    sp = SumProduct(demo)
    sp.p()

def test_generated_cases(demo):
    with open("test_cases.txt", "rt") as f:
        for lines in grouper(3, f):
            newick, node_states, ret = [l.strip() for l in lines]
            node_states = eval(node_states)
            ret = float(ret)
            dl = node_states.values()[0]['lineages']
            # N0 = 20000 in generate data due to haploid/diploid thing
            demo = Demography.from_newick(newick, default_lineages=dl, default_N=20000.)
            demo.update_state(node_states)
            sp = SumProduct(demo)
            p = sp.p()
            assert abs(p - ret) / ret < 1e-2
