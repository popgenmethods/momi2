
import pytest
import momi
from momi import expected_sfs
import momi.likelihood
from demo_utils import simple_admixture_demo, simple_five_pop_demo
import autograd.numpy as np
import itertools
import random
from collections import Counter
import scipy

#def test_subsample_inds():
#    demo = simple_admixture_demo()
#    #demo.demo_hist = demo.demo_hist.rescaled()
#    #data = momi.simulate_ms(ms_path, demo.demo_hist,
#    #                        sampled_pops=demo.pops,
#    #                        sampled_n=demo.n,
#    #                        num_loci=1000, mut_rate=1.0)
#    num_bases = 1000
#    mu = 1.0
#    num_replicates = 1000
#    data = demo.simulate_data(
#        length=num_bases, recoms_per_gen=0,
#        muts_per_gen=1.0 / num_bases,
#        num_replicates=num_replicates,
#        sampled_n_dict={"b":2,"a":3})
#    #data = demo.demo_hist.simulate_data(
#    #    demo.pops, demo.n,
#    #    mutation_rate=mu/num_bases,
#    #    recombination_rate=0,
#    #    length=num_bases,
#    #    num_replicates=num_replicates)
#    assert data.sfs.n_snps() > 0
#    assert data.subsample_inds(4).sfs == data.sfs.subsample_inds(4)


def test_count_subsets():
    demo = simple_admixture_demo()
    #data = momi.simulate_ms(ms_path, demo.demo_hist,
    #                        sampled_pops=demo.pops,
    #                        sampled_n=demo.n,
    #                        num_loci=1000, mut_rate=1.0)
    num_bases = 1000
    mu = 1.0
    num_replicates = 100
    data = demo.simulate_data(
        muts_per_gen=mu/num_bases,
        recoms_per_gen=0,
        length=num_bases,
        num_replicates=num_replicates,
        sampled_n_dict={"b":2,"a":3}).extract_sfs(None)

    subconfig = []
    for n in data.sampled_n:
        sub_n = random.randrange(n+1)
        d = random.randrange(sub_n+1)
        subconfig.append([sub_n-d, d])

    subconfig_probs = np.ones(len(data.configs))
    for i, (a, d) in enumerate(subconfig):
        #subconfig_probs *= scipy.special.comb(
        #    data.configs.value[:, i, :], [a, d]).prod(axis=1)
        #subconfig_probs /= scipy.special.comb(
        #    data.configs.value[:, i, :].sum(axis=1), a+d)
        subconfig_probs *= scipy.stats.hypergeom.pmf(
            d, data.configs.value[:, i, :].sum(axis=1),
            data.configs.value[:, i, 1], a+d)

    assert np.allclose(subconfig_probs,
                       data.configs.subsample_probs(subconfig))


@pytest.mark.parametrize("folded,n_lins",
                         ((f, n) for f in (True, False) for n in ((2, 3), (0, 3))))
def test_simple_admixture_subsampling(folded, n_lins):
    check_subsampling(simple_admixture_demo(), {"b":2,"a":3},
                      3, folded=folded)


def check_subsampling(demo, sampled_n_dict, add_n, folded=False):
    config_list = momi.data.configurations.build_full_config_list(*zip(
        *sampled_n_dict.items()))
    if folded:
        config_list = momi.site_freq_spectrum(
            config_list.sampled_pops, [config_list]).fold().configs

    sfs1 = demo.expected_sfs(config_list, normalized=True, folded=folded)
    #sfs1 = expected_sfs(demo._get_demo(sampled_n_dict), config_list,
    #                    normalized=True, folded=folded, **kwargs)

    configs2 = config_list._copy(sampled_n=config_list.sampled_n + add_n)
    sfs2 = demo.expected_sfs(configs2, normalized=True, folded=folded)
    #sfs2 = expected_sfs(demo._get_demo(dict(zip(configs2.sampled_pops, configs2.sampled_n))),
    #                    configs2,
    #                    normalized=True, folded=folded, **kwargs)

    sfs1 = np.array(list(sfs1.values()))
    sfs2 = np.array(list(sfs2.values()))
    assert np.allclose(sfs1, sfs2)

    ## check sums to 1 even with error matrix
    #error_matrices = [np.exp(np.random.randn(n + 1, n + 1))
    #                  for n in configs2.sampled_n]
    #error_matrices = [
    #    np.einsum('ij,j->ij', x, 1. / np.sum(x, axis=0)) for x in error_matrices]

    #sfs3 = demo.expected_sfs(configs2, normalized=True, folded=folded,
    #                         error_matrices=error_matrices)
    #assert np.isclose(sum(sfs3.values()), 1.0)


def test_subsample_pops():
    demo = simple_five_pop_demo()
    num_bases = 1000
    mu = 1.0
    num_replicates = 100
    data = demo.simulate_data(
        muts_per_gen=mu/num_bases,
        recoms_per_gen=0,
        length=num_bases,
        num_replicates=num_replicates,
        sampled_n_dict={i: 5 for i in range(1,6)})

    sfs1 = data.subset_populations([1,2,3], [3]).extract_sfs(None)
    sfs2 = data.extract_sfs(None).subset_populations([1,2,3], [3])
    assert sfs1 == sfs2
