from __future__ import division
import time
import moran_model
import numpy as np
from numpy import exp

def check_expm(t,v):
    print "\n========================\nTesting t=%g" % t
    #P,d,Pinv = moran_model.moran_eigensystem(len(v)-1)
    #print "exp eigenvalues = %s" % exp(t*d)

    action0 = moran_model.moran_action_eigen
    action1 = moran_model.moran_al_mohy_higham

    # precompute
    action0(t,v)
    action1(t,v)

    numtimes=10

    print "\nTesting eigenvalue expm"
    start = time.time()
    for i in range(numtimes):
        x = action0(t,v)
    end = time.time()
    print "%f seconds" % ((end - start)/numtimes)
    print "%d entries < 0" % np.sum(x < 0)

    #print x

    print "\nTesting al-mohy/higham expm"
    start = time.time()
    for i in range(numtimes):
        x = action1(t,v)
    end = time.time()
    print "%f seconds" % ((end - start)/numtimes)
    print "%d entries < 0" % np.sum(x < 0)

    #print x

if __name__ == "__main__":
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

    #v = np.array([1.0] + [0.0]*1000)
    #v = np.array([[[1.0] + [0.0]*10]*10]*10)
    #v = np.array([[[0.0] * 2 + [1.0] + [0.0]*2]*5]*10)
    #print v.shape
    #v = np.zeros((10,10,10))
    #v = np.zeros(1000)
    #v = np.zeros((100,5,2))
    v = np.zeros((100,5,2))
    v[5,...] = 1.0
    #check_expm(0.0,v)
    check_expm(1e-6,v)
    #check_expm(1e-3,v)
    #check_expm(1e-1,v)
