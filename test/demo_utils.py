#from momi import make_demography
import momi
from momi import demographic_history
import autograd.numpy as np
import random


def make_demo_hist(events, sampled_pops, sampled_n, *args, **kwargs):
    return DemoHistSample(sampled_pops, sampled_n,
                          demographic_history(events, *args, **kwargs))


class DemoHistSample(object):

    def __init__(self, pops, n, demo_hist):
        self.demo_hist = demo_hist
        self.pops = pops
        self.n = n


def simple_admixture_demo(x=np.random.normal(size=7), n_lins=(2, 3)):
    t = np.cumsum(np.exp(x[:5])) * 1e4
    p = 1.0 / (1.0 + np.exp(x[5:]))
    events = [("-ep", t[1], 'a', 2, 1. - p[1]), ('-ep', t[0], 'a', 3, 1. - p[0]),
              ('-ej', t[2], 2, 3), ('-ej', t[3], 3, 'b'), ('-ej', t[4], 'a', 'b')]
    # return make_demography(events, sampled_pops = ('b','a'), sampled_n =
    # n_lins, default_N = 1e4)
    return make_demo_hist(events, sampled_pops=('b', 'a'), sampled_n=n_lins, default_N=1e4)


def simple_admixture_3pop(x=None, n_lins=(4, 4, 4)):
    if x is None:
        x = np.random.normal(size=7)
    t = np.cumsum(np.exp(x[:5]))
    p = 1.0 / (1.0 + np.exp(x[5:]))
    events = [("-ep", t[1], 'a', 'c', 1. - p[1]), ('-ep', t[0], 'a', 'd', 1. - p[0]),
              ('-ej', t[2], 'c', 'd'), ('-ej', t[3], 'd', 'b'), ('-ej', t[4], 'a', 'b')]
    # return make_demography(events, sampled_pops = ('b','a','c'), sampled_n =
    # n_lins)
    return make_demo_hist(events, sampled_pops=('b', 'a', 'c'), sampled_n=n_lins)


def simple_two_pop_demo(x=np.random.normal(size=4), n_lins=(5, 6)):
    x = [1e4 * np.exp(xi) for xi in x]
    events = [('-en', 0., 1, x[1]), ('-en', 0., 0, x[2]),
              ('-ej', x[0], 0, 1), ('-en', x[0], 1, x[3])]
    # return make_demography(events, sampled_pops = (1,0), sampled_n = n_lins,
    # default_N = 1e4)
    return make_demo_hist(events, sampled_pops=(1, 0), sampled_n=n_lins, default_N=1e4)


def simple_three_pop_demo(t0, t1):
    events = [('-ej', t0, 1, 2), ('-ej', t0 + t1, 2, 3)]
    # return make_demography(events, (1,2,3), (1,1,1))
    return make_demo_hist(events, (1, 2, 3), (1, 1, 1))


def simple_nea_admixture_demo(N_chb_bottom, N_chb_top, pulse_t, pulse_p, ej_chb, ej_yri, sampled_n=(14, 10)):
    ej_chb = pulse_t + ej_chb
    ej_yri = ej_chb + ej_yri

    G_chb = -np.log(N_chb_top / N_chb_bottom) / ej_chb

    events = [('-en', 0., 'chb', N_chb_bottom),
              ('-eg', 0, 'chb', G_chb),
              ('-ep', pulse_t, 'chb', 'nea', pulse_p),
              ('-ej', ej_chb, 'chb', 'yri'),
              ('-ej', ej_yri, 'yri', 'nea'),
              ]

    return make_demo_hist(events, ('yri', 'chb'), sampled_n)
    # return make_demography(events, ('yri','chb'), sampled_n)
simple_nea_admixture_demo.bounds = [(.01, 100.),
                                    (.01, 100.),
                                    (.01, 5.),
                                    (.001, .25),
                                    (.01, 5.),
                                    (.01, 5.)]
simple_nea_admixture_demo.true_params = [10., .1, .25, .03, .25, 1.]


def piecewise_constant_demo(x=np.random.normal(size=15), n_lins=(10,)):
    assert x.shape[0] % 2 == 1
    n, = n_lins

    events_list = [('-en', 0., 0, 1e4 * np.exp(x[0]))]
    prev_time = 0.0
    for i in range(int((x.shape[0] - 1) / 2)):
        prev_time = np.exp(x[2 * i + 1]) + prev_time
        N = np.exp(x[2 * i + 2])
        events_list += [('-en', 1e4 * prev_time, 0, 1e4 * N)]
    return make_demo_hist(events_list, sampled_pops=(0,), sampled_n=n_lins, default_N=1e4)
    # return make_demography(events_list, sampled_pops = (0,), sampled_n =
    # n_lins, default_N = 1e4)


def exp_growth_model(x=np.random.normal(size=3), n_lins=(10,)):
    t, g, g2 = x
    t, g2 = np.exp(t), np.exp(g2)
    events = [('-eg', 0., 0, g / 1e4), ('-eg', t * 1e4,
                                        0, g2 / 1e4), ('-eg', 3 * t * 1e4, 0, 0.)]
    return make_demo_hist(events, sampled_pops=(0,), sampled_n=n_lins, default_N=1e4)
    # return make_demography(events, sampled_pops = (0,), sampled_n = n_lins,
    # default_N = 1e4)


def exp_growth_0_model(x, n_lins):
    x0 = np.array([x[0], 0.0, x[1]])
    return exp_growth_model(x0, n_lins)


def simple_five_pop_demo(x=np.random.normal(size=30), n_lins=(1, 2, 3, 4, 5)):
    assert len(x) == 30
    # make all params positive
    x = np.exp(x)

    # # allow negative growth rates
    # for i in range(15,20):
    #     x[i] = np.log(x[i])
    # # make times increasing
    # for i in range(1,15):
    #     x[i] = x[i] + x[i-1]

    t = np.cumsum(x[:15])
    # allow negative growth rates
    g = np.log(x[15:20])

    # number of edges is 2n-1
    events_list = [('-eg', t[0], 5, g[0]),
                   ('-eg', t[1], 4, g[1]),
                   ('-eg', t[2], 3, g[2]),
                   ('-eg', t[3], 2, g[3]),
                   ('-eg', t[4], 1, g[4]),
                   ('-ej', t[5], 5, 4), ('-en', t[5], 4, x[20]),
                   ('-en', t[6], 3, x[21]),
                   ('-en', t[7], 2, x[22]),
                   ('-en', t[8], 1, x[23]),
                   ('-ej', t[9], 4, 3), ('-en', t[9], 3, x[24]),
                   ('-en', t[10], 2, x[25]),
                   ('-en', t[11], 1, x[26]),
                   ('-ej', t[12], 3, 2), ('-en', t[12], 2, x[27]),
                   ('-en', t[13], 1, x[28]),
                   ('-ej', t[14], 2, 1), ('-en', t[14], 1, x[29])]
    demo = make_demo_hist(events_list, sampled_pops=list(
        range(1, len(n_lins) + 1)), sampled_n=n_lins)
    #demo = make_demography(events_list, sampled_pops = list(range(1,len(n_lins)+1)), sampled_n = n_lins)
    #demo = demo.rescaled(1e4)
    return demo


def random_tree_demo(num_leaf_pops, lins_per_pop):
    events_list = []
    sampled_pops = list(range(1, num_leaf_pops + 1))
    roots = list(sampled_pops)
    for i in roots:
        events_list += [('-en', 0.0, i, random.expovariate(1.0))]
    t = 0.0
    while len(roots) > 1:
        i, j = random.sample(roots, 2)
        t += random.expovariate(1.0)
        events_list += [('-ej', t, i, j),
                        ('-en', t, j, random.expovariate(1.0))]
        roots.remove(i)
    return make_demo_hist(events_list, sampled_pops, [lins_per_pop] * num_leaf_pops)
    # return make_demography(events_list, sampled_pops, [lins_per_pop] *
    # num_leaf_pops)
