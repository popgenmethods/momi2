import pytest
from momi import simulate_ms
import momi
from test_ms import ms_path, scrm_path
from io import StringIO

from demo_utils import simple_five_pop_demo

def test_readwrite_segsites_parse_ms_equal():
    demo = simple_five_pop_demo(n_lins=(10,10,10,10,10)).rescaled()
    n_loci = 1000
    
    data = momi.simulate_ms(scrm_path, demo,
                            num_loci=n_loci, mut_rate=.1)
    assert data.n_loci == n_loci

    strio = StringIO()    
    momi.write_seg_sites(strio, data)

    strio = StringIO(strio.getvalue())
    data2 = momi.read_seg_sites(strio)

    assert data == data2
    assert data.sfs == data2.sfs



