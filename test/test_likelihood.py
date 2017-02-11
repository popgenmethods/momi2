
import pytest
import momi
import momi.likelihood
from momi import SfsLikelihoodSurface
from demo_utils import simple_five_pop_demo

import autograd.numpy as np
from autograd import grad, hessian, hessian_vector_product
import autograd

from test_msprime import ms_path, scrm_path

def test_batches():
    demo = simple_five_pop_demo(n_lins=(10,10,10,10,10))
    demo.demo_hist = demo.demo_hist.rescaled()

    sfs = momi.simulate_ms(scrm_path, demo.demo_hist,
                           sampled_pops = demo.pops, sampled_n = demo.n,
                           num_loci=1000, mut_rate=.1).sfs

    sfs_len = sfs.n_nonzero_entries
    
    print("total entries", sfs_len)
    print("total snps", sfs.n_snps())

    assert sfs_len > 30

    assert np.isclose(SfsLikelihoodSurface(sfs, batch_size=5).log_lik(demo.demo_hist),
                      momi.likelihood._composite_log_likelihood(sfs, demo.demo_hist))

def test_batches_grad():
    x0 = np.random.normal(size=30)
    pre_demo_func = lambda *x: simple_five_pop_demo(x=np.array(x), n_lins=(10,10,10,10,10))
    demo_func = lambda *x: pre_demo_func(*x).demo_hist.rescaled()
    pre_demo = pre_demo_func(*x0)

    mu = .05
    sfs = momi.simulate_ms(scrm_path, pre_demo.demo_hist,
                           sampled_pops = pre_demo.pops,
                           sampled_n = pre_demo.n,
                           num_loci=2000, mut_rate=mu).sfs

    sfs_len = sfs.n_nonzero_entries
    
    print("total entries", sfs_len)
    print("total snps", sfs.n_snps())

    assert sfs_len > 30

    momi.likelihood._raw_log_lik.reset_grad_count()
    assert not momi.likelihood._raw_log_lik.num_grad_calls()
    assert np.allclose(-sfs.n_snps() * grad(SfsLikelihoodSurface(sfs, batch_size=5, demo_func=demo_func, mut_rate=mu).kl_div)(x0),
                       grad(lambda x: momi.likelihood._composite_log_likelihood(sfs, demo_func(*x), mut_rate=mu))(x0))
    assert momi.likelihood._raw_log_lik.num_grad_calls()


def test_batches_hess():
    x0 = np.random.normal(size=30)
    pre_demo_func = lambda *x: simple_five_pop_demo(x=np.array(x), n_lins=(10,10,10,10,10))
    demo_func = lambda *x: pre_demo_func(*x).demo_hist.rescaled()
    pre_demo = pre_demo_func(*x0)

    mu = .05
    sfs = momi.simulate_ms(scrm_path, pre_demo.demo_hist,
                           sampled_pops = pre_demo.pops,
                           sampled_n = pre_demo.n,
                           num_loci=2000, mut_rate=mu).sfs

    sfs_len = sfs.n_nonzero_entries
    
    print("total entries", sfs_len)
    print("total snps", sfs.n_snps())

    assert sfs_len > 30

    v = np.random.normal(size=len(x0))
    try:
        hessian_vector_product(SfsLikelihoodSurface(sfs, batch_size=5, demo_func=demo_func, mut_rate=mu).log_lik)(x0, v)
    except momi.util.HessianDisabledError:
        pass
    else:
        assert False
    hess1 = hessian_vector_product(SfsLikelihoodSurface(sfs, batch_size=-1, demo_func=demo_func, mut_rate=mu).log_lik)(x0, v)
    hess2 = hessian_vector_product(lambda x: momi.likelihood._composite_log_likelihood(sfs, demo_func(*x), mut_rate=mu))(x0, v)
    assert np.allclose(hess1, hess2)
