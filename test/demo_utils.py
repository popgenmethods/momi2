from momi import Demography
import autograd.numpy as np
import random

def simple_admixture_demo(x=np.random.normal(size=7), n_lins=(2,3)):
    t = np.cumsum(np.exp(x[:5])) * 1e4
    p = 1.0 / (1.0 + np.exp(x[5:]))
    return Demography([("-ep", t[1], 'a', 2, 1.-p[1]),('-ep',t[0],'a',3, 1.-p[0]),
                       ('-ej',t[2],2,3),('-ej',t[3],3,'b'),('-ej',t[4],'a','b')],
                      sampled_pops = ('b','a'), sampled_n = n_lins, default_N = 1e4)

def simple_two_pop_demo(x=np.random.normal(size=4), n_lins=(5,6)):
    x = [1e4*np.exp(xi) for xi in x]
    return Demography([('-en',0.,1,x[1]), ('-en',0.,0,x[2]), ('-ej',x[0],0,1), ('-en',x[0],1,x[3])],
                      sampled_pops = (1,0), sampled_n = n_lins, default_N = 1e4)


def piecewise_constant_demo(x = np.random.normal(size=15), n_lins = (10,)):
    assert x.shape[0] % 2 == 1
    n, = n_lins

    events_list = [('-en',0.,0,1e4*np.exp(x[0]))]
    prev_time = 0.0
    for i in range(int((x.shape[0]-1)/2)):
        prev_time = np.exp(x[2*i+1]) + prev_time
        N = np.exp(x[2*i+2])
        events_list += [('-en',1e4*prev_time,0,1e4*N)]
    return Demography(events_list,
                      sampled_pops = (0,), sampled_n = n_lins, default_N = 1e4)


def exp_growth_model(x = np.random.normal(size=3), n_lins = (10,)):
    t,g,g2 = x
    t,g2 = np.exp(t), np.exp(g2)
    return Demography([('-eg',0.,0,g/1e4),('-eg',t*1e4,0,g2/1e4),('-eg',3*t*1e4,0,0.)],
                      sampled_pops = (0,), sampled_n = n_lins, default_N = 1e4)

def exp_growth_0_model(x, n_lins):
    x0 = np.array([x[0], 0.0, x[1]])
    return exp_growth_model(x0, n_lins)

def simple_five_pop_demo(x = np.random.normal(size=30), n_lins=(1,2,3,4,5)):
    assert len(x) == 30
    # make all params positive
    x = np.exp(x)
    # allow negative growth rates
    for i in range(15,20):
        x[i] = np.log(x[i])
    # make times increasing
    for i in range(1,15):
        x[i] = x[i] + x[i-1]
    
    # number of edges is 2n-1
    events_list = [('-eg',x[0],5,x[15]),
                   ('-eg',x[1],4,x[16]),
                   ('-eg',x[2],3,x[17]),
                   ('-eg',x[3],2,x[18]),
                   ('-eg',x[4],1,x[19]),
                   ('-ej',x[5],5,4),('-en',x[5],4,x[20]),
                   ('-en',x[6],3,x[21]),
                   ('-en',x[7],2,x[22]),
                   ('-en',x[8],1,x[23]),
                   ('-ej',x[9],4,3),('-en',x[9],3,x[24]),
                   ('-en',x[10],2,x[25]),
                   ('-en',x[11],1,x[26]),
                   ('-ej',x[12],3,2),('-en',x[12],2,x[27]),
                   ('-en',x[13],1,x[28]),
                   ('-ej',x[14],2,1),('-en',x[14],1,x[29])]
    demo = Demography(events_list,
                      sampled_pops = range(1,len(n_lins)+1), sampled_n = n_lins)
    demo = demo.rescaled(1e4)
    return demo

def random_tree_demo(num_leaf_pops, lins_per_pop):
    events_list = []
    sampled_pops = range(1,num_leaf_pops+1)
    roots = list(sampled_pops)
    for i in roots:
        events_list += [('-en', 0.0, i, random.expovariate(1.0))]
    t = 0.0
    while len(roots) > 1:
        i,j = random.sample(roots, 2)
        t += random.expovariate(1.0)
        events_list += [('-ej', t, i ,j),
                        ('-en', t, j, random.expovariate(1.0))]
        roots.remove(i)
    return Demography(events_list, sampled_pops, [lins_per_pop] * num_leaf_pops)
