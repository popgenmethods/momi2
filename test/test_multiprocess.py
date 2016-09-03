import momi
import autograd.numpy as np
import autograd as ag
from demo_utils import simple_five_pop_demo
from test_ms import ms_path, scrm_path

def powfun(i):
    return lambda x: np.sum(x**i)

def test_autograd_process():
    n = 5
    funlist = [powfun(i) for i in range(n)]
    powsum = lambda x: sum([f(x) for f in funlist])

    proclist = [momi.util.AutogradProcess(powfun, i) for i in range(n)]
    powsum2 = lambda x: momi.util.parsum(x, proclist)

    x = np.random.normal(size=10)
    y = np.random.normal(size=10)

    assert np.allclose(powsum(x), powsum2(x))
    assert np.allclose(ag.grad(powsum)(x), ag.grad(powsum2)(x))
    assert np.allclose(ag.hessian_vector_product(powsum)(x,y),
                       ag.hessian_vector_product(powsum2)(x,y))

    for proc in proclist: proc.join()

def test_make_likelihood():
    x0 = np.random.normal(size=30)
    demo_func = lambda *x: simple_five_pop_demo(x=np.array(x), n_lins=(10,10,10,10,10)).rescaled()
    demo = demo_func(*x0)

    mu = .05
    sfs = momi.simulate_ms(scrm_path, demo,
                           num_loci=2000, mut_rate=mu).sfs

    lik1 = momi.likelihood._composite_log_likelihood(sfs, demo)
    lik2 = momi.likelihood._make_likelihood_fun(sfs.sampled_pops, sfs.configs.value, sfs.sampled_n, sfs._total_freqs, demo_func=demo_func)(x0)

    assert np.allclose(lik1, lik2)

def test_multiprocess_likelihood():
    x0 = np.random.normal(size=30)
    demo_func = lambda *x: simple_five_pop_demo(x=np.array(x), n_lins=(10,10,10,10,10)).rescaled()
    demo = demo_func(*x0)

    mu = .05
    sfs = momi.simulate_ms(scrm_path, demo,
                           num_loci=2000, mut_rate=mu).sfs

    lik1 = momi.likelihood._composite_log_likelihood(sfs, demo, mut_rate=mu)

    surface = momi.SfsLikelihoodSurface(sfs, demo_func=demo_func, batch_size=5, processes=2, mut_rate=mu)
    lik2 = surface.log_lik(x0)
    assert np.allclose(lik1, lik2)

def test_multiprocess_grad():
    x0 = np.random.normal(size=30)
    demo_func = lambda *x: simple_five_pop_demo(x=np.array(x), n_lins=(10,10,10,10,10)).rescaled()
    demo = demo_func(*x0)

    mu = .05
    sfs = momi.simulate_ms(scrm_path, demo,
                           num_loci=2000, mut_rate=mu).sfs

    lik1 = lambda x: momi.likelihood._composite_log_likelihood(sfs, demo_func(*x), mut_rate=mu)
    lik1 = ag.grad(lik1)(x0)

    momi.util.parsum.reset_grad_count()
    assert not momi.util.parsum.num_grad_calls()

    surface = momi.SfsLikelihoodSurface(sfs, demo_func=demo_func, batch_size=5, processes=2, mut_rate=mu)
    lik2 = ag.grad(surface.log_lik)(x0)

    assert np.allclose(lik1, lik2)
    assert momi.util.parsum.num_grad_calls()
    assert not momi.util.parsum.num_hess_calls()

def test_multiprocess_hess():
    x0 = np.random.normal(size=30)
    y = np.random.normal(size=30)
    demo_func = lambda *x: simple_five_pop_demo(x=np.array(x), n_lins=(10,10,10,10,10)).rescaled()
    demo = demo_func(*x0)

    mu = .05
    sfs = momi.simulate_ms(scrm_path, demo,
                           num_loci=2000, mut_rate=mu).sfs

    lik1 = lambda x: momi.likelihood._composite_log_likelihood(sfs, demo_func(*x), mut_rate=mu)
    lik1 = ag.hessian_vector_product(lik1)(x0, y)

    momi.util.parsum.reset_grad_count()
    assert not momi.util.parsum.num_hess_calls()

    surface = momi.SfsLikelihoodSurface(sfs, demo_func=demo_func, batch_size=-1, processes=2, mut_rate=mu)
    lik2 = lambda x: surface.log_lik(x)
    lik2 = ag.hessian_vector_product(lik2)(x0, y)

    assert np.allclose(lik1, lik2)
    assert momi.util.parsum.num_grad_calls()
    assert momi.util.parsum.num_hess_calls()
