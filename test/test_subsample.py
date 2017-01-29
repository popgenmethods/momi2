
import pytest
import momi
from momi import expected_sfs
import momi.likelihood
from demo_utils import simple_admixture_demo
import autograd.numpy as np
import itertools, random
from collections import Counter

from test_ms import ms_path, scrm_path

def test_subsample_inds():
    demo = simple_admixture_demo()
    demo.demo_hist = demo.demo_hist.rescaled()
    data = momi.simulate_ms(ms_path, demo.demo_hist,
                            sampled_pops = demo.pops,
                            sampled_n = demo.n,
                            num_loci=1000, mut_rate=1.0)
    assert data.sfs.n_snps() > 0
    assert data.subsample_inds(4).sfs == data.sfs.subsample_inds(4)

@pytest.mark.parametrize("folded,n_lins",
                         ((f,n) for f in (True,False) for n in ((2,3),(0,3))))
def test_simple_admixture_subsampling(folded,n_lins):
    check_subsampling(simple_admixture_demo(),3,folded=folded)
    
def check_subsampling(demo, add_n, folded=False, **kwargs):
    leaves = demo.pops
    ranges = [list(range(n+1)) for n in demo.n]

    demo.demo_hist = demo.demo_hist.rescaled()
    #demo2 = momi.make_demography(demo.events, demo.sampled_pops,
    #                        np.array(demo.sampled_n, dtype=int) + add_n,
    #                        sampled_t = demo.sampled_t)
    
    #config_list = momi.data_structure._configs_from_derived([np.array(x,dtype=int) for x in itertools.product(*ranges)],
    #                                                        demo.sampled_n, demo.sampled_pops)
    #config_list = momi.config_array(demo.sampled_pops,
    #                                [np.array(x,dtype=int) for x in itertools.product(*ranges)],
    #                                demo.sampled_n)
    config_list = momi.data_structure.full_config_array(demo.pops, demo.n)
    if folded:
        config_list = momi.site_freq_spectrum(config_list.sampled_pops, [config_list]).fold().configs

    sfs1 = expected_sfs(demo.demo_hist, config_list, normalized=True, folded=folded, **kwargs)

    configs2 = config_list._copy(sampled_n=np.array(demo.n) + add_n)
    sfs2 = expected_sfs(demo.demo_hist, configs2, normalized=True, folded=folded, **kwargs)

    assert np.allclose(sfs1, sfs2)

    # check sums to 1 even with error matrix
    error_matrices = [np.exp(np.random.randn(n+1,n+1)) for n in configs2.sampled_n]
    error_matrices = [np.einsum('ij,j->ij', x, 1./np.sum(x, axis=0)) for x in error_matrices]
    
    sfs3 = expected_sfs(demo.demo_hist, configs2, normalized=True, folded=folded, error_matrices=error_matrices)
    assert np.isclose(np.sum(sfs3), 1.0)
