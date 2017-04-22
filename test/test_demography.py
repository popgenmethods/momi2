
import pytest

import numpy as np

import momi
from momi import expected_sfs_tensor_prod, expected_total_branch_len, make_demography
from demo_utils import simple_admixture_demo
from momi.math_functions import hypergeom_quasi_inverse

import autograd


def test_constructor():
    pre_demo = simple_admixture_demo()
    demo = pre_demo.demo_hist._get_multipop_moran(pre_demo.pops, pre_demo.n)
    demo2 = momi.demography.Demography(
        demo._get_graph_structure(), *demo._get_differentiable_part())

    assert np.allclose(expected_total_branch_len(demo),
                       expected_total_branch_len(demo2))

    # make sure it fails if we don't pass in the array values
    demo3 = momi.demography.Demography(demo._get_graph_structure())
    try:
        expected_total_branch_len(demo3)
    except:
        return
    assert False


def test_constructor_grad():
    def fun1(x):
        pre_demo = simple_admixture_demo(x)
        return expected_total_branch_len(pre_demo.demo_hist, sampled_n=pre_demo.n, sampled_pops=pre_demo.pops)

    fun2_helper = lambda diff_vals, diff_keys, G: expected_total_branch_len(
        momi.demography.Demography(G, diff_keys, diff_vals))

    helper_grad = momi.util.count_calls(autograd.grad(fun2_helper))

    fun2_helper = autograd.primitive(fun2_helper)
    #fun2_helper.defgrad(lambda ans, diff_vals, diff_keys, G: lambda g: tuple(g*y for y in helper_grad(diff_vals.value, diff_keys, G)))
    fun2_helper.defgrad(lambda ans, diff_vals, diff_keys, G: lambda g: tuple(
        g * y for y in helper_grad(diff_vals, diff_keys, G)))

    def fun2(x):
        pre_demo = simple_admixture_demo(x)
        demo = pre_demo.demo_hist._get_multipop_moran(
            pre_demo.pops, pre_demo.n)
        return fun2_helper(*reversed([demo._get_graph_structure()] + list(demo._get_differentiable_part())))

    x_val = np.random.normal(size=7)

    assert not helper_grad.num_calls()
    assert np.allclose(autograd.grad(fun1)(x_val), autograd.grad(fun2)(x_val))
    assert helper_grad.num_calls()


class TestDemography(momi.demography.Demography):

    def __init__(self, demo):
        super(TestDemography, self).__init__(demo._G)

    def _n_at_node(self, node):
        if node[0] in self.sampled_pops and node[1] == 0:
            return self._G.node[node]['lineages']
        return np.sum(self._n_at_node(l) for l in self._G[node])


def test_pseudoinverse():
    demo = simple_admixture_demo()
    demo = demo.demo_hist._get_multipop_moran(demo.pops, demo.n)

    # construct from same event_list so that nodes have same labels
    demo0 = make_demography(demo.events, demo.sampled_pops,
                            demo.sampled_n, demo.sampled_t, demo.default_N)
    demo1 = TestDemography(demo)

    p = 20
    vecs = [np.random.normal(size=(p, n + 1)) for n in demo0.sampled_n]

    vals0, vals1 = [expected_sfs_tensor_prod(vecs, d)
                    for d in (demo0, demo1)]

    assert np.allclose(vals0, vals1)

    n_lins_diff = np.array(
        [demo0._n_at_node(v) - demo1._n_at_node(v) for v in demo0._G])
    assert all(n_lins_diff <= 0)
    assert any(n_lins_diff < 0)


def test_hypergeom_pinv_eye():
    i = np.random.randint(2, 50)
    assert np.allclose(hypergeom_quasi_inverse(i, i),
                       np.eye(i + 1, i + 1))


def test_copy():
    demo = make_demography([("-ej", 1., "a", "b")], ["a", "b"], (3, 2))
    demo.copy(sampled_n=(5, 6))


def test_P():
    t1 = np.random.exponential(.25)
    t2 = np.random.exponential(.25) + t1
    t3 = np.random.exponential(.5) + t2
    p1 = np.random.uniform(0, 1)
    p2 = np.random.uniform(0, 1)

    i = np.random.choice([0, 1])
    j = 1 - i

    root_event = ('-ej', t3, 0, 1)
    pulse_events0 = [('-ep', t1, 0, 1, p1),
                     ('-ep', t2, i, j, p2)]
    pulse_events1 = [('-ep', t1, 0, 'x', p1), ('-ej', t1, 'x', 1),
                     ('-ep', t2, i, 'y', p2), ('-ej', t2, 'y', j)]

    demo0 = make_demography(pulse_events0 + [root_event],
                            (0, 1), (5, 6))
    demo1 = make_demography(pulse_events1 + [root_event],
                            (0, 1), (5, 6))

    p = 20
    vecs = [np.random.normal(size=(p, n + 1)) for n in demo0.sampled_n]

    vals0, vals1 = [expected_sfs_tensor_prod(vecs, d)
                    for d in (demo0, demo1)]

    assert np.allclose(vals0, vals1)


def test_events_before_sample():
    n_events = 4
    t = [0.0]
    for i in range(n_events):
        t += [np.random.exponential(1. / float(n_events)) + t[-1]]
    t = t[1:]

    events = [('-ep', t[0], 'a', 'b', np.random.uniform(0, 1))]

    demo0 = make_demography(events + [('-en', 0.0, 'c', 10.0), ('-eg', 0.0, 'c', 1.0),
                                      ('-ej', t[1], 'a', 'c'),
                                      ('-ej', t[2], 'c', 'b')],
                            sampled_pops=('a', 'b'), sampled_n=(7, 5),
                            sampled_t=(0., t[3]))

    demo1 = make_demography(events + [('-en', t[1], 'a', 10.0 * np.exp(-t[1])), ('-eg', t[1], 'a', 1.0),
                                      ('-ej', t[2], 'a', 'b')],
                            sampled_pops=('a', 'b'), sampled_n=(7, 5),
                            sampled_t=(0., t[3]))

    vecs = [np.random.normal(size=(10, n + 1)) for n in demo0.sampled_n]
    val0, val1 = [expected_sfs_tensor_prod(vecs, d) for d in (demo0, demo1)]

    assert np.allclose(val0, val1)


def test_time_scale():
    n_events = 3
    t = [0.0]
    for i in range(n_events):
        t += [np.random.exponential(1. / float(n_events)) + t[-1]]
    t = t[1:]

    demo0 = make_demography([('-en', t[0], 'a', .3),
                             ('-eg', t[0], 'a', 0.5),
                             ('-ej', t[2], 'a', 'b')],
                            sampled_pops=('a', 'b'), sampled_n=(7, 5),
                            sampled_t=(0., t[1]),
                            time_scale='ms')

    demo1 = make_demography([('-en', 4.0 * t[0], 'a', .3),
                             ('-eg', 4.0 * t[0], 'a', 0.5 / 4.0),
                             ('-ej', 4.0 * t[2], 'a', 'b')],
                            sampled_pops=('a', 'b'), sampled_n=(7, 5),
                            sampled_t=(0., 4.0 * t[1]),
                            time_scale='standard')

    vecs = [np.random.normal(size=(10, n + 1)) for n in demo0.sampled_n]
    val0, val1 = [expected_sfs_tensor_prod(vecs, d) for d in (demo0, demo1)]

    assert np.allclose(val0, val1 / 4)
