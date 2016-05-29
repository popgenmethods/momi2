
import pytest
import momi
import momi.likelihood
from momi import SfsLikelihoodSurface
from demo_utils import simple_five_pop_demo

import autograd.numpy as np
from autograd import grad
import autograd

from test_ms import ms_path, scrm_path

def test_batches():
    demo = simple_five_pop_demo(n_lins=(10,10,10,10,10)).rescaled()

    sfs = momi.simulate_ms(scrm_path, demo,
                           num_loci=1000, mut_rate=.1).sfs

    sfs_len = sfs.n_nonzero_entries
    
    print("total entries", sfs_len)
    print("total snps", sfs.n_snps())

    assert sfs_len > 30

    assert np.isclose(SfsLikelihoodSurface(sfs, batch_size=5).log_likelihood(demo),
                      momi.likelihood._composite_log_likelihood(sfs, demo))

def test_batches_grad():
    x0 = np.random.normal(size=30)
    demo_func = lambda *x: simple_five_pop_demo(x=np.array(x), n_lins=(10,10,10,10,10)).rescaled()
    demo = demo_func(*x0)

    mu = .05
    sfs = momi.simulate_ms(scrm_path, demo,
                           num_loci=2000, mut_rate=mu).sfs

    sfs_len = sfs.n_nonzero_entries
    
    print("total entries", sfs_len)
    print("total snps", sfs.n_snps())

    assert sfs_len > 30

    assert np.allclose(-sfs.n_snps() * grad(SfsLikelihoodSurface(sfs, batch_size=5, demo_func=demo_func, mut_rate=mu).kl_divergence)(x0),
                       grad(lambda x: momi.likelihood._composite_log_likelihood(sfs, demo_func(*x), mut_rate=mu))(x0))

def test_no_hessian():
    x0 = np.random.normal(size=30)
    demo_func = lambda *x: simple_five_pop_demo(x=np.array(x), n_lins=(10,10,10,10,10)).rescaled()
    demo = demo_func(*x0)

    sfs = momi.simulate_ms(scrm_path, demo,
                           num_loci=1000, mut_rate=.1).sfs

    sfs_len = sfs.n_nonzero_entries
    
    print("total entries", sfs_len)
    print("total snps", sfs.n_snps())

    assert sfs_len > 30

    try:
        autograd.hessian(SfsLikelihoodSurface(sfs, batch_size=5, demo_func=demo_func).kl_divergence)(x0)
    except NotImplementedError:
        pass
    else:
        assert False
