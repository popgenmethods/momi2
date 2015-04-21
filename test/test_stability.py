from __future__ import division
import time
import moran_model
import numpy as np
from numpy import exp

def check_expm(t,v):
    print "\n========================\nTesting t=%g" % t
    P,d,Pinv = moran_model.moran_eigensystem(len(v)-1)
    print "exp eigenvalues = %s" % exp(t*d)

    # precompute
    moran_model.moran_action(t,v)
    moran_model._old_moran_action(t,v)

    numtimes=10

    print "\nTesting eigenvalue expm"
    start = time.time()
    for i in range(numtimes):
        x = moran_model.moran_action(t,v)
    end = time.time()
    print "%f seconds" % ((end - start)/numtimes)
    print "%d entries < 0" % sum(x < 0)

    print "\nTesting al-mohy/higham expm"
    start = time.time()
    for i in range(numtimes):
        x = moran_model._old_moran_action(t,v)
    end = time.time()
    print "%f seconds" % ((end - start)/numtimes)
    print "%d entries < 0" % sum(x < 0)

#v = np.array([1.0] + [0.0]*100)
#check_expm(1.0,v)
#check_expm(.5,v)
#check_expm(.1,v)
#check_expm(1e-2,v)
#check_expm(1e-3,v)
#check_expm(1e-6,v)
#check_expm(1e-10,v)
#check_expm(1e-16,v)
#check_expm(0.0,v)

v = np.array([1.0] + [0.0]*1000)
#check_expm(0.0,v)
#check_expm(1e-6,v)
#check_expm(1e-3,v)
check_expm(1e-1,v)
