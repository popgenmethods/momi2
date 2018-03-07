
import pytest
import momi
import momi.likelihood
from momi import SfsLikelihoodSurface
from demo_utils import simple_five_pop_demo

import autograd.numpy as np
from autograd import grad, hessian, hessian_vector_product, jacobian
import autograd

def test_batches():
    demo = simple_five_pop_demo()

    sampled_n_dict = dict(zip(demo.leafs, [10]*5))
    num_bases=1000
    sfs = demo.simulate_data(
        length=num_bases,
        muts_per_gen=.1/num_bases,
        recoms_per_gen=0,
        num_replicates=1000,
        sampled_n_dict=sampled_n_dict)._sfs


    sfs_len = sfs.n_nonzero_entries

    print("total entries", sfs_len)
    print("total snps", sfs.n_snps())

    assert sfs_len > 30

    demo = demo._get_demo(sampled_n_dict)
    assert np.isclose(SfsLikelihoodSurface(sfs, batch_size=5).log_lik(demo),
                      momi.likelihood._composite_log_likelihood(sfs, demo))

    assert np.allclose(SfsLikelihoodSurface(sfs, batch_size=5).log_lik(demo, vector=True),
                      momi.likelihood._composite_log_likelihood(sfs, demo, vector=True))



# TODO does this test still make sense?
# uses obsolete style of demo functions which I think makes it slow
def test_batches_grad():
    x0 = np.random.normal(size=30)
    pre_demo_func = lambda *x: simple_five_pop_demo(
        x=np.array(x))
    demo = pre_demo_func(*x0)
    pops = demo.leafs

    sampled_n_dict = dict(zip(demo.leafs, [10]*5))
    num_bases=1000
    sfs = demo.simulate_data(
        length=num_bases,
        muts_per_gen=.1/num_bases,
        recoms_per_gen=0,
        num_replicates=1000,
        sampled_n_dict=sampled_n_dict)._sfs


    sfs_len = sfs.n_nonzero_entries

    print("total entries", sfs_len)
    print("total snps", sfs.n_snps())

    assert sfs_len > 30

    sampled_n_dict = dict(zip(pops, [10]*5))
    demo_func = lambda *x: pre_demo_func(*x)._get_demo(sampled_n_dict)

    jac1 = -sfs.n_snps() * grad(SfsLikelihoodSurface(sfs, batch_size=5, demo_func=demo_func, mut_rate=1.).kl_div)(x0)
    jac2 = grad(lambda x: momi.likelihood._composite_log_likelihood(sfs, demo_func(*x), mut_rate=1.))(x0)
    assert np.allclose(jac1, jac2)

# TODO reenable these tests?
#def test_batches_jac():
#    x0 = np.random.normal(size=30)
#    pre_demo_func = lambda *x: simple_five_pop_demo(
#        x=np.array(x), n_lins=(10, 10, 10, 10, 10))
#    demo_func = lambda *x: pre_demo_func(*x).demo_hist.rescaled()
#    pre_demo = pre_demo_func(*x0)
#
#    mu = 10
#    num_bases = 1000
#    sfs = pre_demo.demo_hist.simulate_data(
#        pre_demo.pops, pre_demo.n,
#        mutation_rate=mu/num_bases,
#        recombination_rate=0,
#        length=num_bases,
#        num_replicates=10).sfs
#
#
#    sfs_len = sfs.n_nonzero_entries
#
#    print("total entries", sfs_len)
#    print("total snps", sfs.n_snps())
#
#    assert sfs_len > 30
#
#    jac1 = jacobian(lambda x: SfsLikelihoodSurface(sfs, batch_size=5, demo_func=demo_func, mut_rate=mu).log_lik(x, vector=True))(x0)
#    jac2 = jacobian(lambda x: momi.likelihood._composite_log_likelihood(sfs, demo_func(*x), mut_rate=mu, vector=True))(x0)
#    assert np.allclose(jac1, jac2)
#    print(jac1)
#
#
#
#def test_batches_hess():
#    x0 = np.random.normal(size=30)
#    pre_demo_func = lambda *x: simple_five_pop_demo(
#        x=np.array(x), n_lins=(10, 10, 10, 10, 10))
#    demo_func = lambda *x: pre_demo_func(*x).demo_hist.rescaled()
#    pre_demo = pre_demo_func(*x0)
#
#    mu = .05
#    num_bases = 1000
#    sfs = pre_demo.demo_hist.simulate_data(
#        pre_demo.pops, pre_demo.n,
#        mutation_rate=mu/num_bases,
#        recombination_rate=0,
#        length=num_bases,
#        num_replicates=2000).sfs
#
#    sfs_len = sfs.n_nonzero_entries
#
#    print("total entries", sfs_len)
#    print("total snps", sfs.n_snps())
#
#    assert sfs_len > 30
#
#    v = np.random.normal(size=len(x0))
#
#    hess1 = hessian_vector_product(SfsLikelihoodSurface(
#        sfs, batch_size=-1, demo_func=demo_func, mut_rate=mu).log_lik)(x0, v)
#    hess2 = hessian_vector_product(lambda x: momi.likelihood._composite_log_likelihood(
#        sfs, demo_func(*x), mut_rate=mu))(x0, v)
#    #hess1 = hessian(SfsLikelihoodSurface(
#    #    sfs, batch_size=-1, demo_func=demo_func, mut_rate=mu).log_lik)(x0)
#    #hess2 = hessian(lambda x: momi.likelihood._composite_log_likelihood(
#    #    sfs, demo_func(*x), mut_rate=mu))(x0)
#    assert np.allclose(hess1, hess2)
