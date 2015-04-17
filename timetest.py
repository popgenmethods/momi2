from __future__ import division
from demography import Demography
from sum_product import SumProduct
import time
import numpy

import sys
sys.stdout = open('timetest.txt','w')

def check_time(cmd, data):
    # precompute
    start = time.time()
    demo = Demography.from_ms(cmd)
    demo.totalSfsSum
    end = time.time()
    print "Precomputation: %f seconds" % (end - start)

    # do data
    data = data[:50]
    start = time.time()
    for states in data:
        demo.update_state(states)
        p = SumProduct(demo).p()
    end = time.time()
    print "%f seconds per site (%d sites in %f seconds)" % ( (end-start) / len(data), len(data), end-start)

def check_demo_time(cmd):
    demo = Demography.from_ms(cmd)
    sumFreqs, sumSqFreqs, nonzeroFreqs = demo.simulate_sfs(num_sims=100, theta=1.0)
    
    data = []
    for key in sumFreqs:
        state = {}
        for i,pop in enumerate(sorted(demo.leaves)):
            state[pop] = {'derived' : key[i], 'ancestral' : demo.n_lineages_subtended_by[pop] - key[i]}
        data.append(state)

    print "Timing demography %s" % cmd
    check_time(cmd, data)

check_demo_time("-I 2 10 10 -ej 1.0 2 1")
check_demo_time("-I 4 10 10 10 10 -ej .4 4 3 -ej 1.0 3 2 -ej 2.0 2 1")

cmd10 = "-I 10 %s" % (" ".join(["10"]*10))
t = .1
for i in reversed(range(9)):
    cmd10 += " -ej %f %d %d" % (t, i+2, i+1)
    t += .1
check_demo_time(cmd10)
