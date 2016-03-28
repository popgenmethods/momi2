from __future__ import division
import pytest
import momi
from momi import expected_sfs
import momi.likelihood
from demo_utils import simple_admixture_demo
import autograd.numpy as np
import itertools, random
from collections import Counter

from test_ms import ms_path, scrm_path

### test subsampling of SNPs
@pytest.mark.parametrize("fold,normalized",
                         ((random.choice((True,False)),random.choice((True,False))),))
def test_subconfigs(fold, normalized):
    demo = simple_admixture_demo().rescaled()
   
    configs = momi.simulate_ms(scrm_path, demo,
                              num_loci=1000, mut_rate=1.).sfs.configs

    if fold:
        configs = configs.copy(fold=True)
    
    sub_idxs = np.array(random.sample(range(len(configs)), int(len(configs)/2)+1))
    assert len(sub_idxs) > 0 and len(sub_idxs) < configs

    val1 = momi.expected_sfs(demo, configs, normalized=normalized)[sub_idxs]

    sub_configs = momi.likelihood._SubConfigs(configs, sub_idxs)
    val2 = momi.expected_sfs(demo, sub_configs, normalized=normalized)

    assert np.allclose(val1,val2)

@pytest.mark.parametrize("fold,use_mut",
                         ((random.choice((True,False)),random.choice((True,False))),))
def test_subsfs(fold, use_mut):
    demo = simple_admixture_demo().rescaled()

    n_loci = 10
    mut_rate = 100.
    sfs = momi.simulate_ms(scrm_path, demo,
                           num_loci=n_loci, mut_rate=mut_rate).sfs

    if fold:
        sfs = sfs.copy(fold=True)

    locus = random.choice(range(n_loci))
    subsfs = momi.likelihood._SubSfs(sfs.configs, sfs._counts_ij[locus,:])

    if not use_mut:
        mut_rate = None
    
    val1 = momi.composite_log_likelihood(sfs, demo, mut_rate=mut_rate, vector=True)[locus]
    val2 = momi.composite_log_likelihood(subsfs, demo, mut_rate=mut_rate)

    assert np.isclose(val1, val2)

@pytest.mark.parametrize("fold",
                         (random.choice((True,False)),))
def test_subsfs2(fold):
    demo = simple_admixture_demo().rescaled()

    n_loci = 10
    mut_rate = 100.
    sfs = momi.simulate_ms(scrm_path, demo,
                           num_loci=n_loci, mut_rate=mut_rate).sfs

    if fold:
        sfs = sfs.copy(fold=True)

    subsfs_list = momi.likelihood._subsfs_list(sfs, 10, np.random)
    total = [([sfs.configs[i] for i in subsfs.configs.sub_idxs],
              np.squeeze(subsfs._counts_ij))
             for subsfs in subsfs_list]
    total = [Counter(dict(zip(cnfs,cnts))) for cnfs,cnts in total]
    total = dict(sum(total, Counter()))

    assert total == sfs.total

    
    
@pytest.mark.parametrize("fold",
                         (random.choice((True,False)),))
def test_subliks(fold):
    demo_func = lambda *x: simple_admixture_demo(x).rescaled()
    rnd = np.random.RandomState()    
    x0 = rnd.normal(size=7)

    demo = demo_func(*x0)

    n_loci = 100
    mut_rate = 10.
    sfs = momi.simulate_ms(scrm_path, demo,
                           num_loci=n_loci, mut_rate=mut_rate).sfs

    if fold:
        sfs = sfs.copy(fold=True)

    n_chunks = 10

    lik = lambda X,params,**kwargs: momi.composite_log_likelihood(X, demo_func(*params), comb=False, **kwargs)
    
    lik_funs = momi.likelihood._sgd_liks(lik, sfs, n_chunks, rnd, None, True)

       
    val0 = [f(x0) for f in lik_funs]
    val1 = lik(sfs,x0, mut_rate=None)
    val2 = np.sum(lik(sfs,x0,mut_rate=None, vector=True))

    assert np.isclose(np.sum(val0),val1) and np.isclose(val1,val2)

def test_stochastic_inference(method='adam',folded=False):
    num_runs = 1000
    mu=1.0
    def get_demo(t0, t1):
        return momi.Demography([('-ej', t0, 1, 2), ('-ej', t0 + t1, 2, 3)],
                               (1,2,3), (5,5,5))
    true_x = np.array([.5,.2])
    true_demo = get_demo(*true_x)

    sfs = momi.simulate_ms(ms_path, true_demo,
                           num_loci=num_runs, mut_rate=mu).sfs
    if folded:
        sfs = sfs.copy(fold=True)
    
    optimize_res = momi.composite_mle_search(sfs, get_demo, np.array([.1,.9]), mu, bounds=[(1e-100,None),(1e-100,None)], method=method, maxiter=10, options={'n_chunks':10}, output_progress =10)
    print optimize_res
    
    inferred_x = optimize_res.x
    error = (true_x - inferred_x) / true_x
    print "# Truth:\n", true_x
    print "# Inferred:\n", inferred_x
    print "# Max Relative Error: %f" % max(abs(error))
    print "# Relative Error:","\n", error

    assert max(abs(error)) < .1
    #assert False
