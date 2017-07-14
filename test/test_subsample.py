
import pytest
import momi
from momi import expected_sfs
import momi.likelihood
from demo_utils import simple_admixture_demo
import autograd.numpy as np
import itertools
import random
from collections import Counter
import scipy

from test_msprime import ms_path, scrm_path


def test_subsample_inds():
    demo = simple_admixture_demo()
    demo.demo_hist = demo.demo_hist.rescaled()
    data = momi.simulate_ms(ms_path, demo.demo_hist,
                            sampled_pops=demo.pops,
                            sampled_n=demo.n,
                            num_loci=1000, mut_rate=1.0)
    assert data.sfs.n_snps() > 0
    assert data.subsample_inds(4).sfs == data.sfs.subsample_inds(4)


def test_count_subsets():
    demo = simple_admixture_demo(n_lins=(10,10))
    demo.demo_hist = demo.demo_hist.rescaled()
    data = momi.simulate_ms(ms_path, demo.demo_hist,
                            sampled_pops=demo.pops,
                            sampled_n=demo.n,
                            num_loci=1000, mut_rate=1.0)

    subconfig = []
    for n in data.sampled_n:
        sub_n = random.randrange(n+1)
        d = random.randrange(sub_n+1)
        subconfig.append([sub_n-d, d])

    subconfig_probs = np.ones(len(data.configs))
    for i, (a, d) in enumerate(subconfig):
        #subconfig_probs *= scipy.misc.comb(
        #    data.configs.value[:, i, :], [a, d]).prod(axis=1)
        #subconfig_probs /= scipy.misc.comb(
        #    data.configs.value[:, i, :].sum(axis=1), a+d)
        subconfig_probs *= scipy.stats.hypergeom.pmf(
            d, data.configs.value[:, i, :].sum(axis=1),
            data.configs.value[:, i, 1], a+d)

    assert np.allclose(subconfig_probs,
                       data.configs.subsample_probs(subconfig))


@pytest.mark.parametrize("folded,n_lins",
                         ((f, n) for f in (True, False) for n in ((2, 3), (0, 3))))
def test_simple_admixture_subsampling(folded, n_lins):
    check_subsampling(simple_admixture_demo(), 3, folded=folded)


def check_subsampling(demo, add_n, folded=False, **kwargs):
    leaves = demo.pops
    ranges = [list(range(n + 1)) for n in demo.n]

    demo.demo_hist = demo.demo_hist.rescaled()

    config_list = momi.data.config_array.full_config_array(demo.pops, demo.n)
    if folded:
        config_list = momi.site_freq_spectrum(
            config_list.sampled_pops, [config_list]).fold().configs

    sfs1 = expected_sfs(demo.demo_hist, config_list,
                        normalized=True, folded=folded, **kwargs)

    configs2 = config_list._copy(sampled_n=np.array(demo.n) + add_n)
    sfs2 = expected_sfs(demo.demo_hist, configs2,
                        normalized=True, folded=folded, **kwargs)

    assert np.allclose(sfs1, sfs2)

    # check sums to 1 even with error matrix
    error_matrices = [np.exp(np.random.randn(n + 1, n + 1))
                      for n in configs2.sampled_n]
    error_matrices = [
        np.einsum('ij,j->ij', x, 1. / np.sum(x, axis=0)) for x in error_matrices]

    sfs3 = expected_sfs(demo.demo_hist, configs2, normalized=True,
                        folded=folded, error_matrices=error_matrices)
    assert np.isclose(np.sum(sfs3), 1.0)
