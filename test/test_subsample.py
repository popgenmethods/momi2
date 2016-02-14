from __future__ import division
import pytest
import momi
from momi import expected_sfs
from demo_utils import simple_admixture_demo
import autograd.numpy as np
import itertools

@pytest.mark.parametrize("folded,n_lins",
                         ((f,n) for f in (True,False) for n in ((2,3),(0,3))))
def test_simple_admixture_subsampling(folded,n_lins):
    check_subsampling(simple_admixture_demo(),3,folded=folded)
    
def check_subsampling(demo, add_n, **kwargs):
    leaves = demo.sampled_pops
    ranges = [range(n+1) for n in demo.sampled_n]

    demo = demo.rescaled()
    demo2 = momi.Demography(demo.events, demo.sampled_pops,
                            np.array(demo.sampled_n, dtype=int) + add_n,
                            sampled_t = demo.sampled_t)
    
    config_list = momi.util._configs_from_derived([np.array(x,dtype=int) for x in itertools.product(*ranges)],
                                                  demo.sampled_n)

    sfs1 = expected_sfs(demo, config_list, normalized=True, **kwargs)
    sfs2 = expected_sfs(demo2, config_list, normalized=True, **kwargs)

    assert np.allclose(sfs1, sfs2)

    # check sums to 1 even with error matrix
    error_matrices = [np.exp(np.random.randn(n+1,n+1)) for n in demo2.sampled_n]
    error_matrices = [np.einsum('ij,j->ij', x, 1./np.sum(x, axis=0)) for x in error_matrices]
    
    sfs3 = expected_sfs(demo2, config_list, normalized=True, error_matrices=error_matrices)
    assert np.isclose(np.sum(sfs3), 1.0)
