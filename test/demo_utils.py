from momi import Demography
import autograd.numpy as np
import random

def simple_admixture_demo(x=np.random.normal(size=7), n_lins=(2,3)):
    t = np.cumsum(np.exp(x[:5])) * 1e4
    p = 1.0 / (1.0 + np.exp(x[5:]))
    return Demography("-d 1e4 -n %d %d -S $2 1 $3 -S $0 1 $1 -J $4 2 3 -J $t3 3 0 -J $t4 1 0" % n_lins, 
                      t[0], p[0], t[1], p[1], t[2], t3=t[3], t4=t[4])


def simple_two_pop_demo(x=np.random.normal(size=4), n_lins=(5,6)):
    #assert len(x) == 4
    return Demography("-d 1e4 -n %s -N 0 0 $1 -N 0 1 $2 -J $0 1 0 -N $0 * $3" % " ".join(map(str, n_lins)), *[1e4*np.exp(xi) for xi in x])


def piecewise_constant_demo(x = np.random.normal(size=15), n_lins = (10,)):
    assert x.shape[0] % 2 == 1
    n, = n_lins

    cmd = "-d 1e4 -n %d -N 0 0 $0" % n
    args = [1e4 * np.exp(x[0])]
    prev_time = 0.0
    var = 1
    for i in range(int((x.shape[0]-1)/2)):
        cmd += " -N $%d 0 $%d" % (var, var+1)
        var += 2
        prev_time = np.exp(x[2*i+1]) + prev_time
        N = np.exp(x[2*i+2])
        args += [1e4 * prev_time, 1e4 * N]
    return Demography(cmd, *args)


def exp_growth_model(x = np.random.normal(size=3), n_lins = (10,)):
    t,g,g2 = x
    t,g2 = np.exp(t), np.exp(g2)
    return Demography("-d 1e4 -n %d -G 0 0 $0 -G $1 0 $2 -G $3 0 0.0" % n_lins[0],
                           g/1e4, t*1e4 , g2/1e4, 3*t*1e4)

def exp_growth_0_model(x, n_lins):
    x0 = np.array([x[0], 0.0, x[1]])
    return exp_growth_model(x0, n_lins)

def simple_five_pop_demo(x = np.random.normal(size=30), n_lins=(1,2,3,4,5)):
    # number of edges is 2n-1   
    cmd = ["-I 5 %s" % (" ".join(map(str, n_lins))),
           "-eg $0 5 $15",
           "-eg $1 4 $16",
           "-eg $2 3 $17",
           "-eg $3 2 $18",
           "-eg $4 1 $19",
           "-ej $5 5 4 -en $5 4 $20",
           "-en $6 3 $21",
           "-en $7 2 $22",
           "-en $8 1 $23",
           "-ej $9 4 3 -en $9 3 $24",
           "-en $10 2 $25",
           "-en $11 1 $26",
           "-ej $12 3 2 -en $12 2 $27",
           "-en $13 1 $28",
           "-ej $14 2 1 -eN $14 $29"]

    cmd = " ".join(cmd)

    assert len(x) == 30
    # make all params positive
    args = map(np.exp, x)
    # allow negative growth rates
    for i in range(15,20):
        args[i] = np.log(args[i])
    # make times increasing
    for i in range(1,15):
        args[i] = args[i] + args[i-1]

    demo = Demography.from_ms(1e4, cmd, *args)
    return demo


def random_tree_demo(num_leaf_pops, lins_per_pop):
    cmd = "-I %d %s" % (num_leaf_pops, " ".join([str(lins_per_pop)] * num_leaf_pops))
    for i in range(num_leaf_pops):
        cmd += " -n %d %f" % (i+1, random.expovariate(1.0))
    roots = set([i+1 for i in range(num_leaf_pops)])
    t = 0.0
    while len(roots) > 1:
        i,j = random.sample(roots, 2)
        t += random.expovariate(1.0)
        cmd += " -ej %f %d %d" % (t, i, j)
        roots.remove(i)
        cmd += " -en %f %d %f" % (t, j, random.expovariate(1.0))
    return Demography.from_ms(1e4, cmd)
