from __future__ import division
import pytest
import autograd.numpy as np
from autograd.numpy import outer, sum, exp, log
from autograd import grad

import networkx as nx
import random

from numdifftools import Gradient, Hessian
from test_sims import simple_admixture_demo

from momi import make_demography, compute_sfs

EPS=1e-8

def num_grad(f,x,eps=EPS):
    ret = []
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = eps
        ret.append((f(x+dx) - f(x)) / eps)
    return np.array(ret)

# this is numerically unstable
def num_hess(f,x,eps=EPS):
    def g(y):
        return num_grad(f,y,eps)
    return num_grad(g,x,np.sqrt(eps))
    #return num_grad(g,x,eps)

def log_within(x, y, eps=1e-2, trunc=1e-10):
    approx_zero = np.logical_or(np.abs(x) < trunc, np.abs(y) < trunc)
    not_zero = np.logical_not(approx_zero)
    if np.any(not_zero):
        assert max(np.abs(np.log(x[not_zero] / y[not_zero]))) < eps
    if np.any(approx_zero):
        assert np.max(np.abs(x[approx_zero])) < trunc and np.max(np.abs(y[approx_zero])) < trunc

def check_gradient(f, x):
    print x, "\n", f(x)

    print "# grad2"
    grad2 = Gradient(f)(x)
    print "# building grad1"
    g = grad(f)
    print "# computing grad1"
    grad1 = g(x)

    print "gradient1\n", grad1, "\ngradient2\n", grad2
    log_within(grad1,grad2)

    ## check Hessian vector product
    ## comented out cuz slow
    y = np.random.normal(size=x.shape)
    gdot = lambda u : np.dot(g(u), y)
    hess1, hess2 = grad(gdot)(x), Gradient(gdot)(x)
    print "hess1\n",hess1,"\nhess2\n",hess2
    log_within(hess1,hess2, eps=1e-1, trunc=1e-5)


def test_simple():
    def f(x):
        #x = array(x)
        return sum(outer(x,x))
    check_gradient(f, np.random.normal(size=10))

def test_simple_2():
    def f(x):
        #x = array(x)
        return sum(np.divide(1.0 , outer(x,x)))
    check_gradient(f, np.random.normal(size=10))

def test_simple_3():
    def f(x):
        #return sum(x/x)
        return np.divide(sum(x) , sum(x))
    check_gradient(f, np.random.normal(size=10))

def simple_two_pop_demo(x, n_lins):
    #assert len(x) == 4
    leafs = sorted(n_lins.keys())
    counts = [n_lins[l] for l in leafs]
    return make_demography("-I %d %s -n 1 $1 -n 2 $2 -ej $0 2 1 -eN $0 $3" % (len(n_lins), " ".join(map(str, counts))), *(map(exp, x)))

def piecewise_constant_demo(x, n_lins):
    assert x.shape[0] % 2 == 1
    assert n_lins.keys() == ['a']
    n = n_lins['a']

    cmd = "-I 1 %d -en 0.0 1 $0" % n
    args = [exp(x[0])]
    prev_time = 0.0
    var = 1
    for i in range(int((x.shape[0]-1)/2)):
        cmd += " -eN $%d $%d" % (var, var+1)
        var += 2
        prev_time = exp(x[2*i+1]) + prev_time
        N = exp(x[2*i+2])
        args += [prev_time, N]
    return make_demography(cmd, *args)


def sfs_func(demo_func, n_lins, normalized=True):
    # get random sfs entry
    n = sum([x for _,x in n_lins.items()])
    total_der = 0
    while total_der == 0 or total_der == n:
        total_der = 0
        #states = {}
        states = []
        for pop,n_pop in sorted(n_lins.items()):
            n_der = random.randint(0,n_pop)
            assert n_der >= 0 and n_der <= n_pop
            total_der += n_der
            #states[pop] = {'ancestral' : n_pop - n_der, 'derived' : n_der}
            states.append(n_der)
    
    print states
    def f(x):
        demo = demo_func(x, n_lins)
        #print demo.graph['cmd']
        sfs, branch_len = compute_sfs(demo,states)
        if normalized:
            return sfs/branch_len
        return sfs
    return f

def test_admixture():
    n_lins = {'1':2,'2':2}
    f = sfs_func(simple_admixture_demo, n_lins, normalized=True)
    #f = sfs_func(simple_admixture_demo, n_lins, normalized=False)
    x = np.random.normal(size=7)
    check_gradient(f,x)

def exp_growth_model(x, n_lins):
    t,g,g2 = x
    t,g2 = exp(t), exp(g2)
    return make_demography("-I 1 %d -G $0 -eG $1 $2 -eG $3 0.0" % n_lins['1'],
                           g,t,g2,3*t)

@pytest.mark.parametrize("log_tau,growth_rate,end_growth_rate", 
                         ((-random.expovariate(1),g,random.gauss(0,1))
                          for g in (random.uniform(-10,10), 
                                    random.uniform(-.01,.01), 0.0)))
def test_exp_growth(log_tau, growth_rate, end_growth_rate):
    n_lins = {'1':10}

    f = sfs_func(exp_growth_model, n_lins, normalized=True)
    x=np.array([log_tau,growth_rate,end_growth_rate])

    check_gradient(f,x)
    #assert False

@pytest.mark.parametrize("n,epochs,normalized", 
                         ((random.randint(2,10),random.randint(1,10),norm) for norm in (False,True)))
def test_piecewise_constant_p(n, epochs, normalized):
    n_lins = {'a' : n}

    x = np.random.normal(size=2*epochs - 1)
    f = sfs_func(piecewise_constant_demo, n_lins, normalized=normalized)

    check_gradient(f,x)

@pytest.mark.parametrize("n1,n2,normalized", 
                         ((random.randint(1,10),random.randint(1,10),norm) for norm in (False,True)))
def test_simple_two_pop(n1,n2,normalized):
    n_lins = {'a' : n1, 'b' : n2}

    x = np.random.normal(size=4)
    f = sfs_func(simple_two_pop_demo, n_lins, normalized=normalized)
    
    check_gradient(f,x)


def simple_five_pop_demo(x, n_lins):
    # number of edges is 2n-1
    leafs = sorted(n_lins.keys())
    counts = [n_lins[l] for l in leafs]
    
    cmd = ["-I 5 %s" % (" ".join(map(str, counts))),
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
    args = map(exp, x)
    # allow negative growth rates
    for i in range(15,20):
        args[i] = log(args[i])
    # make times increasing
    for i in range(1,15):
        args[i] = args[i] + args[i-1]

    demo = make_demography(cmd, *args)
    return demo


@pytest.mark.parametrize("n1,n2,n3,n4,n5,normalized", 
                         ((random.randint(1,5),random.randint(1,5),random.randint(1,5),random.randint(1,5),random.randint(1,5),norm) for norm in (False,True)))
def test_simple_five_pop(n1,n2,n3,n4,n5,normalized):
    n_lins = {'a' : n1, 'b' : n2, 'c' : n3, 'd' : n4, 'e' : n5}

    x = np.random.normal(size=30)
    def demo_func(y,n_lins):
        return simple_five_pop_demo(x,n_lins)
    f = sfs_func(demo_func, n_lins, normalized=normalized)
    
    check_gradient(f,x)
