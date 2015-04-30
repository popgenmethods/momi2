from __future__ import division
import pytest
from sum_product import compute_sfs
from demography import Demography
import itertools

import os
TEST_CASES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_cases.txt")

@pytest.fixture
def demo():
    test_newick = """
        (a:1[&&momi:model=constant:N=1.0:lineages=10],
         b:1[&&momi:model=constant:N=1.0:lineages=8]):3[&&momi:model=constant:N=10.0];
    """
    demo = Demography.from_newick(test_newick)
    return demo

def test_sfs(demo):
    st = {'a': {'derived': 5, 'ancestral': 5},
          'b': {'derived': 8, 'ancestral': 0}}
    compute_sfs(demo,convert_states(st))
#     sp = SumProduct(demo, st)
#     sp.p()

def demo_generator(fn=TEST_CASES):
    with open(fn, "rt") as f:
        for lines in grouper(3, f):
            newick, node_states, ret = [l.strip() for l in lines]
            node_states = eval(node_states)
            ret = float(ret)
            dl = node_states.values()[0]['lineages']
            # N0 = 20000 in generate data due to haploid/diploid thing
            dm = Demography.from_newick(newick, default_lineages=dl, default_N=20000.)
            #dm.update_state(node_states)
            yield dm, convert_states(node_states), ret

@pytest.mark.parametrize("demo,data,ret", demo_generator())
def test_generated_cases(demo, data, ret):
    p, branch_len = compute_sfs(demo, data)
    assert abs(2.0*p - ret) / ret < 1e-4

def convert_states(node_states):
    ret = []
    leaves = sorted(node_states.keys())
    for leaf in leaves:
        ret.append(node_states[leaf]['derived'])
    return [ret]

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)
    
if __name__=="__main__":
    for dm, data, ret in demo_generator("test_cases.txt"):
        print("Me: %f\tTest case: %f" % (compute_sfs(dm, data), ret))
