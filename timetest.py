from demography import Demography
from sum_product import SumProduct
import time
#import numpy
#import autograd.numpy as np
from autograd import grad
from autograd.numpy import dot, array

import sys
sys.stdout = open('timetest.txt','w')

def check_time(demofunc, data, *x):
    # do data
    data = data[:50]
    def func(y):
        # precompute
        start = time.time()
        demo = demofunc(*y)
        ret = -demo.totalSfsSum
        end = time.time()
        print "Precomputation: %f seconds" % (end - start)
       
        start = time.time()
        for states in data:
            demo.update_state(states)
            ret += SumProduct(demo).p()
        end = time.time()
        print "%f seconds per site (%d sites in %f seconds)" % ( (end-start) / float(len(data)), len(data), end-start)
        return ret

    print "Computing SFS"
    x = array(x)
    start = time.time()
    func(x)
    end=time.time()
    print "%f total seconds" % (end-start)

    print "\nComputing gradient"
    g = grad(func)
    start = time.time()
    g(x)
    end=time.time()
    print "%f total seconds" % (end-start)

    print "\nComputing Hessian-vector product"
    gdot = lambda x,y: dot(y,g(x))
    hp = grad(gdot)
    start = time.time()
    hp(x,x)
    end=time.time()
    print "%f total seconds" % (end-start)

def check_demo_time(demofunc, *x):
    demo = demofunc(*x)
    sumFreqs, sumSqFreqs, nonzeroFreqs = demo.simulate_sfs(num_sims=100, theta=1.0)
    
    data = []
    for key in sumFreqs:
        state = {}
        for i,pop in enumerate(sorted(demo.leaves)):
            state[pop] = {'derived' : key[i], 'ancestral' : demo.n_lineages_subtended_by[pop] - key[i]}
        data.append(state)

    print "\n=======================\nTiming demography %s\n" % demo.graph['cmd']
    print "%d variables" % len(x)
    check_time(demofunc, data, *x)


def two_pop_demo(x):
    return Demography.from_ms("-I 2 10 10 -ej $0 2 1", x)
check_demo_time(two_pop_demo, 1.0)

def four_pop_demo(t0,t1,t2):
    return Demography.from_ms("-I 4 10 10 10 10 -ej $0 4 3 -ej $1 3 2 -ej $2 2 1",
                              t0,t1,t2)
check_demo_time(four_pop_demo, 0.4, 1.0, 2.0)

def ten_pop_demo(*t):
    cmd10 = "-I 10 %s" % (" ".join(["10"]*10))
    for i in range(9):
        cmd10 += " -ej $%d %d %d" % (i, i+1, i+2)
    return Demography.from_ms(cmd10, *t)
check_demo_time(ten_pop_demo, *[.1 * i for i in range(1,10)])
