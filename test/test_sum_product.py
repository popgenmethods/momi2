from __future__ import division
import pytest
from sum_product import SumProduct
from demography import Demography
from util import H

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
        while f:
            newick = next(f).strip()
            node_states = eval(next(f).strip())
            ret = float(next(f).strip())
            dl = node_states.values()[0]['lineages']
            demo = Demography.from_newick(newick, default_lineages=dl, default_N=10000.)
            demo.update_state(node_states)
            sp = SumProduct(demo)
            p = sp.p()
            assert abs(p - ret) < 1e-8
