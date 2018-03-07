
import pytest
import autograd.numpy as np
from autograd import grad
import networkx as nx
import random
from demo_utils import *
import momi
from momi import expected_sfs, expected_total_branch_len
from numdifftools import Gradient, Hessian


def check_gradient(f, x):
    print(x, "\n", f(x))

    print("# grad2")
    grad2 = Gradient(f)(x)
    print("# building grad1")
    g = grad(f)
    print("# computing grad1")
    grad1 = g(x)

    print("gradient1\n", grad1, "\ngradient2\n", grad2)
    np.allclose(grad1, grad2)

    # check Hessian vector product
    y = np.random.normal(size=x.shape)
    gdot = lambda u: np.dot(g(u), y)
    hess1, hess2 = grad(gdot)(x), Gradient(gdot)(x)
    print("hess1\n", hess1, "\nhess2\n", hess2)
    np.allclose(hess1, hess2)


def test_simple():
    def f(x):
        return np.sum(np.outer(x, x))
    check_gradient(f, np.random.normal(size=10))


def test_simple_2():
    def f(x):
        return np.sum(1.0 / np.outer(x, x))
    check_gradient(f, np.random.normal(size=10))


def test_simple_3():
    def f(x):
        return np.sum(x) / np.sum(x)
    check_gradient(f, np.random.normal(size=10))


def sfs_func(demo_func, n_lins, normalized=True, states=None):
    # get random sfs entry
    if states is None:
        n = np.sum([x for x in n_lins])
        total_der = 0
        while total_der == 0 or total_der == n:
            total_der = 0
            #states = {}
            states = []
            for pop, n_pop in enumerate(n_lins):
                n_der = random.randint(0, n_pop)
                assert n_der >= 0 and n_der <= n_pop
                total_der += n_der
                #states[pop] = {'ancestral' : n_pop - n_der, 'derived' : n_der}
                states.append(n_der)

    def f(x):
        demo = demo_func(x)
        demo.set_mut_rate(1.0)
        configs = momi.build_config_list(demo.leafs, tuple(states), n_lins)
        #print(configs)
        # print demo.graph['cmd']
        #sfs, branch_len = expected_sfs(demo.demo_hist, configs), expected_total_branch_len(
        #    demo.demo_hist, sampled_pops=demo.pops, sampled_n=demo.n)
        sfs, branch_len = demo.expected_sfs(configs, length=1, return_dict=False), demo.expected_branchlen(
            dict(zip(configs.sampled_pops, configs.sampled_n)))
        if normalized:
            return np.squeeze(sfs / branch_len)
        return np.squeeze(sfs)
    return f


def test_admixture():
    n_lins = (2, 2)
    #f = sfs_func(simple_admixture_demo, n_lins, normalized=True)
    f = sfs_func(simple_admixture_demo, n_lins, normalized=False)
    x = np.random.normal(size=7)
    check_gradient(f, x)


@pytest.mark.parametrize("log_tau,growth_rate,end_growth_rate",
                         ((-random.expovariate(1), g, random.gauss(0, 1))
                          for g in (random.uniform(-10, 10),
                                    random.uniform(-.01, .01), 0.0)))
def test_exp_growth(log_tau, growth_rate, end_growth_rate):
    n_lins = (10,)

    f = sfs_func(exp_growth_model, n_lins, normalized=True)
    x = np.array([log_tau, growth_rate, end_growth_rate])

    check_gradient(f, x)
    #assert False


@pytest.mark.parametrize("n,epochs,normalized",
                         ((random.randint(2, 10), random.randint(1, 10), norm) for norm in (False, True)))
def test_piecewise_constant(n, epochs, normalized):
    n_lins = (n,)

    x = np.random.normal(size=2 * epochs - 1)
    f = sfs_func(piecewise_constant_demo, n_lins, normalized=normalized)

    check_gradient(f, x)


@pytest.mark.parametrize("n1,n2,normalized",
                         ((random.randint(1, 10), random.randint(1, 10), norm) for norm in (False, True)))
def test_simple_two_pop(n1, n2, normalized):
    n_lins = (n1, n2)

    x = np.random.normal(size=4)
    f = sfs_func(simple_two_pop_demo, n_lins, normalized=normalized)

    check_gradient(f, x)


## TODO reenable the test, it got really slow at some point...
#@pytest.mark.parametrize("n1,n2,n3,n4,n5,normalized",
#                         ((random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), random.randint(1, 5), norm) for norm in (False, True)))
#def test_simple_five_pop(n1, n2, n3, n4, n5, normalized):
#    n_lins = (n1, n2, n3, n4, n5)
#
#    x = np.random.normal(size=30)
#
#    def demo_func(y, n_lins):
#        return simple_five_pop_demo(y, n_lins)
#    f = sfs_func(demo_func, n_lins, normalized=normalized,
#                 states=[0,0,1,0,0])
#
#    check_gradient(f, x)
