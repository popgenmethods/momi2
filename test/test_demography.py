
import pytest

import numpy as np

import momi
from momi import expected_sfs_tensor_prod, expected_total_branch_len
from demo_utils import simple_admixture_demo
from momi.math_functions import hypergeom_quasi_inverse

import autograd


def test_constructor():
    pre_demo = simple_admixture_demo()
    demo = pre_demo._get_demo({"b":2,"a":3})
    demo2 = momi.demography.Demography(
        demo._get_graph_structure(), demo._get_differentiable_part())

    assert np.allclose(expected_total_branch_len(demo),
                       expected_total_branch_len(demo2))

    # make sure it fails if we don't pass in the array values
    demo3 = momi.demography.Demography(demo._get_graph_structure())
    try:
        expected_total_branch_len(demo3)
    except:
        return
    assert False


class NoLookdownDemography(momi.demography.Demography):
    def __init__(self, demo):
        super(NoLookdownDemography, self).__init__(demo._G)

    def _n_at_node(self, node):
        if node[0] in self.sampled_pops and node[1] == 0:
            return self._G.node[node]['lineages']
        return np.sum(self._n_at_node(l) for l in self._G[node])


def test_pseudoinverse():
    demo0 = simple_admixture_demo()._get_demo({"b":2,"a":3})
    demo1 = NoLookdownDemography(demo0)

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


def test_P():
    t1 = np.random.exponential(.25)
    t2 = np.random.exponential(.25) + t1
    t3 = np.random.exponential(.5) + t2
    p1 = np.random.uniform(0, 1)
    p2 = np.random.uniform(0, 1)

    i = np.random.choice([0, 1])
    j = 1 - i

    demo0 = momi.DemographicModel(1.0, .25)
    demo1 = momi.DemographicModel(1.0, .25)

    for d in (demo0, demo1):
        d.add_leaf(0)
        d.add_leaf(1)
        d.move_lineages(0, 1, t3)
    demo0.move_lineages(0, 1, t=t1, p=p1)
    demo0.move_lineages(i, j, t=t2, p=p2)

    demo1.move_lineages(0, 'x', t=t1, p=p1)
    demo1.move_lineages('x', 1, t=t1)
    demo1.move_lineages(i, 'y', t=t2, p=p2)
    demo1.move_lineages('y', j, t=t2)

    demo0 = demo0._get_demo({0:5,1:6})
    demo1 = demo1._get_demo({0:5,1:6})

    #root_event = ('-ej', t3, 0, 1)
    #pulse_events0 = [('-ep', t1, 0, 1, p1),
    #                 ('-ep', t2, i, j, p2)]
    #pulse_events1 = [('-ep', t1, 0, 'x', p1), ('-ej', t1, 'x', 1),
    #                 ('-ep', t2, i, 'y', p2), ('-ej', t2, 'y', j)]

    #demo0 = make_demography(pulse_events0 + [root_event],
    #                        (0, 1), (5, 6))
    #demo1 = make_demography(pulse_events1 + [root_event],
    #                        (0, 1), (5, 6))

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


    demo0 = momi.DemographicModel(1.0, .25)
    demo1 = momi.DemographicModel(1.0, .25)
    p = np.random.uniform(0, 1)
    for d in (demo0, demo1):
        d.add_leaf("a")
        d.add_leaf("b", t=t[3])
        d.move_lineages("a", "b", t=t[0], p=p)

    demo0.set_size("c", t=0, N=10, g=1)
    demo0.move_lineages("a", "c", t=t[1])
    demo0.move_lineages("c", "b", t=t[2])

    demo1.set_size("a", t=t[1], N=10*np.exp(-t[1]), g=1.0)
    demo1.move_lineages("a", "b", t=t[2])

    demo0, demo1 = [d._get_demo({"a":7,"b":5}) for d in (demo0, demo1)]

    #events = [('-ep', t[0], 'a', 'b', np.random.uniform(0, 1))]
    #demo0 = make_demography(events + [('-en', 0.0, 'c', 10.0), ('-eg', 0.0, 'c', 1.0),
    #                                  ('-ej', t[1], 'a', 'c'),
    #                                  ('-ej', t[2], 'c', 'b')],
    #                        sampled_pops=('a', 'b'), sampled_n=(7, 5),
    #                        sampled_t=(0., t[3]))

    #demo1 = make_demography(events + [('-en', t[1], 'a', 10.0 * np.exp(-t[1])), ('-eg', t[1], 'a', 1.0),
    #                                  ('-ej', t[2], 'a', 'b')],
    #                        sampled_pops=('a', 'b'), sampled_n=(7, 5),
    #                        sampled_t=(0., t[3]))

    vecs = [np.random.normal(size=(10, n + 1)) for n in demo0.sampled_n]
    val0, val1 = [expected_sfs_tensor_prod(vecs, d) for d in (demo0, demo1)]

    assert np.allclose(val0, val1)
