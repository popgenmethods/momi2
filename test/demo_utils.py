import momi
import autograd.numpy as np
import random


def simple_admixture_demo(x=np.random.normal(size=7), n_lins=(2, 3)):
    t = np.cumsum(np.exp(x[:5]))
    p = 1.0 / (1.0 + np.exp(x[5:]))

    ret = momi.DemographicModel(1., .25)
    ret.add_leaf("b")
    ret.add_leaf("a")
    ret.move_lineages("a", 2, t[1], p=1.-p[1])
    ret.move_lineages("a", 3, t[0], p=1.-p[0])
    ret.move_lineages(2, 3, t[2])
    ret.move_lineages(3, "b", t[3])
    ret.move_lineages("a", "b", t[4])
    return ret


def simple_admixture_3pop(x=None, n_lins=(4, 4, 4)):
    if x is None:
        x = np.random.normal(size=7)
    t = np.cumsum(np.exp(x[:5]))
    p = 1.0 / (1.0 + np.exp(x[5:]))

    model = momi.DemographicModel(1., .25)
    model.add_leaf("b")
    model.add_leaf("a")
    model.add_leaf("c")
    model.move_lineages("a", "c", t[1], p=1.-p[1])
    model.move_lineages("a", "d", t[0], p=1.-p[0])
    model.move_lineages("c", "d", t[2])
    model.move_lineages("d", "b", t[3])
    model.move_lineages("a", "b", t[4])
    return model


def simple_two_pop_demo(x=np.random.normal(size=4), n_lins=(5, 6)):
    x = [np.exp(xi) for xi in x]
    model = momi.DemographicModel(1., .25)
    model.add_leaf(1)
    model.add_leaf(0)
    model.set_size(1, t=0.0, N=x[1])
    model.set_size(0, t=0.0, N=x[2])
    model.move_lineages(0, 1, t=x[0])
    model.set_size(1, t=x[0], N=x[3])
    return model


def simple_three_pop_demo(t0, t1, n_lins=(1,1,1)):
    model = momi.DemographicModel(1., .25)
    model.add_leaf(1)
    model.add_leaf(2)
    model.add_leaf(3)
    model.move_lineages(1, 2, t0)
    model.move_lineages(2, 3, t0+t1)
    return model


def simple_nea_admixture_demo(N_chb_bottom, N_chb_top, pulse_t, pulse_p, ej_chb, ej_yri, sampled_n=(14, 10)):
    ej_chb = pulse_t + ej_chb
    ej_yri = ej_chb + ej_yri

    G_chb = -np.log(N_chb_top / N_chb_bottom) / ej_chb

    model = momi.DemographicModel(1., .25)
    model.add_leaf("yri")
    model.add_leaf("chb")
    model.set_size("chb", 0., N=N_chb_bottom, g=G_chb)
    model.move_lineages("chb", "nea", t=pulse_t, p=pulse_p)
    model.move_lineages("chb", "yri", t=ej_chb)
    model.move_lineages("yri", "nea", t=ej_yri)
    return model
    #events = [('-en', 0., 'chb', N_chb_bottom),
    #          ('-eg', 0, 'chb', G_chb),
    #          ('-ep', pulse_t, 'chb', 'nea', pulse_p),
    #          ('-ej', ej_chb, 'chb', 'yri'),
    #          ('-ej', ej_yri, 'yri', 'nea'),
    #          ]

    #return make_demo_hist(events, ('yri', 'chb'), sampled_n)
    ## return make_demography(events, ('yri','chb'), sampled_n)
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

    model = momi.DemographicModel(1.0, .25)
    model.add_leaf(0, N=np.exp(x[0]))
    #events_list = [('-en', 0., 0, 1e4 * np.exp(x[0]))]
    prev_time = 0.0
    for i in range(int((x.shape[0] - 1) / 2)):
        prev_time = np.exp(x[2 * i + 1]) + prev_time
        N = np.exp(x[2 * i + 2])
        model.set_size(0, t=prev_time, N=N)
    return model
    #    events_list += [('-en', 1e4 * prev_time, 0, 1e4 * N)]
    #return make_demo_hist(events_list, sampled_pops=(0,), sampled_n=n_lins, default_N=1e4)
    # return make_demography(events_list, sampled_pops = (0,), sampled_n =
    # n_lins, default_N = 1e4)


def exp_growth_model(x=np.random.normal(size=3), n_lins=(10,)):
    t, g, g2 = x
    t, g2 = np.exp(t), np.exp(g2)
    model = momi.DemographicModel(1.0, .25)
    model.add_leaf(0, g=g)
    model.set_size(0, t=t, g=g2)
    model.set_size(0, t=3*t, g=0)
    return model
    #events = [('-eg', 0., 0, g / 1e4),
    #          ('-eg', t * 1e4, 0, g2 / 1e4),
    #          ('-eg', 3 * t * 1e4, 0, 0.)]
    #return make_demo_hist(events, sampled_pops=(0,), sampled_n=n_lins, default_N=1e4)
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

    model = momi.DemographicModel(1.0, .25)
    for pop in range(1, 6):
        model.add_leaf(pop)
    model.set_size(5, t[0], g=g[0])
    model.set_size(4, t[1], g=g[1])
    model.set_size(3, t[2], g=g[2])
    model.set_size(2, t[3], g=g[3])
    model.set_size(1, t[4], g=g[4])
    model.move_lineages(5, 4, t=t[5], N=x[20])
    model.set_size(3, t=t[6], N=x[21])
    model.set_size(2, t=t[7], N=x[22])
    model.set_size(1, t[8], N=x[23])
    model.move_lineages(4, 3, t[9], N=x[24])
    model.set_size(2, t[10], N=x[25])
    model.set_size(1, t[11], N=x[26])
    model.move_lineages(3, 2, t[12], N=x[27])
    model.set_size(1, t[13], N=x[28])
    model.move_lineages(2, 1, t[14], N=x[29])
    return model
    ## number of edges is 2n-1
    #events_list = [('-eg', t[0], 5, g[0]),
    #               ('-eg', t[1], 4, g[1]),
    #               ('-eg', t[2], 3, g[2]),
    #               ('-eg', t[3], 2, g[3]),
    #               ('-eg', t[4], 1, g[4]),
    #               ('-ej', t[5], 5, 4), ('-en', t[5], 4, x[20]),
    #               ('-en', t[6], 3, x[21]),
    #               ('-en', t[7], 2, x[22]),
    #               ('-en', t[8], 1, x[23]),
    #               ('-ej', t[9], 4, 3), ('-en', t[9], 3, x[24]),
    #               ('-en', t[10], 2, x[25]),
    #               ('-en', t[11], 1, x[26]),
    #               ('-ej', t[12], 3, 2), ('-en', t[12], 2, x[27]),
    #               ('-en', t[13], 1, x[28]),
    #               ('-ej', t[14], 2, 1), ('-en', t[14], 1, x[29])]
    #demo = make_demo_hist(events_list, sampled_pops=list(
    #    range(1, len(n_lins) + 1)), sampled_n=n_lins)
    ##demo = make_demography(events_list, sampled_pops = list(range(1,len(n_lins)+1)), sampled_n = n_lins)
    ##demo = demo.rescaled(1e4)
    #return demo


def random_tree_demo(num_leaf_pops, lins_per_pop):
    #events_list = []
    sampled_pops = list(range(1, num_leaf_pops + 1))
    model = momi.DemographicModel(1.0, .25)
    for p in sampled_pops:
        model.add_leaf(p, N=random.expovariate(1.0))
    roots = list(sampled_pops)
    #for i in roots:
    #    events_list += [('-en', 0.0, i, random.expovariate(1.0))]
    t = 0.0
    while len(roots) > 1:
        i, j = random.sample(roots, 2)
        t += random.expovariate(1.0)
        #events_list += [('-ej', t, i, j),
        #                ('-en', t, j, random.expovariate(1.0))]
        model.move_lineages(i, j, t, N=random.expovariate(1.0))
        roots.remove(i)
    #return make_demo_hist(events_list, sampled_pops, [lins_per_pop] * num_leaf_pops)
    return model
    # return make_demography(events_list, sampled_pops, [lins_per_pop] *
    # num_leaf_pops)
