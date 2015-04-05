from __future__ import division
import pytest
import scipy.optimize
from ad import gh, adnumber
from ad.admath import log, exp
from size_history import ConstantTruncatedSizeHistory, PiecewiseHistory
import numpy as np
from numpy import array as arr
import random

# returns xhat = argmin |f(x) - f(xhat)|_2^2
# using gradient descent starting from x0
def minimize_l2_err(f, x0, x, upper_bound=None, lower_bound=1e-12):
    fx = f(x)
    def pre_objective(xhat):
        xhat = xhat
        ret = sum((f(xhat) - fx)**2)
        #print ret
        return ret
    start_err = pre_objective(x0)
    def objective(xhat):
        return pre_objective(xhat) / start_err

    grad, hess = gh(objective)
    bds = ((lower_bound,upper_bound),) * len(x0)

    print "x_start", x0, "loss(x_start)", start_err, "x_true", x
    if len(x) == 1:
        #res = scipy.optimize.minimize(objective, x0, method='trust-ncg', jac=grad, hess=hess)
        res = scipy.optimize.minimize(objective, x0, method='newton-cg', jac=grad, hess=hess)
        assert np.abs(x - res.x) < 1e-3 and res.fun < 1.0
    else:
        res = scipy.optimize.minimize(objective, x0, method='newton-cg', jac=grad, hess=hess)
        #res = scipy.optimize.minimize(objective, x0, method='trust-ncg', jac=grad, hess=hess)
        #res2 = scipy.optimize.minimize(objective, x0, method='newton-cg', jac=grad)
        res2 = scipy.optimize.minimize(objective, x0, method='tnc')
        #res2 = scipy.optimize.minimize(objective, x0, method='nelder-mead')
        #res2 = scipy.optimize.minimize(objective, x0)
        print "withHessian:\n",res
        print "withoutGradient:\n",res2
        assert res.nfev <= res2.nfev and res.fun < 1.0 and res2.fun < 1.0
#         assert res.fun < 1e-1
#         assert res2.fun < 1e-1
#         assert res.fun < f_start * 1e-1 or f_start < 1e-10
#         assert res2.fun < f_start * 1e-1 or f_start < 1e-10
    #assert np.all(res.x > lower_bound)

#def test_moran_action(epochs,params):

@pytest.mark.parametrize("N,n,t", ((random.uniform(.1,10),random.randint(1,20), random.uniform(.01,100)),))
def test_truncconst_sfs(N,n,t):
    def statistic(xhat):
        t_hat = exp(xhat[0])
        print N, t_hat
        est_hist = ConstantTruncatedSizeHistory(n,t_hat,N)
        ret = np.array([est_hist.freq(der,n) for der in range(1,n+1)])
        #ret = est_hist.etjj
        #print ret
        return ret
    x = np.array((t,))
    x0 = np.random.uniform(0.1,10,len(x)) * x

    x, x0 = log(x), log(x0)
    minimize_l2_err(statistic, x0, x)

def get_demo_from_epochs(x, n, addInfEpoch=True):
    #print x
    x = list(x)
    assert len(x) % 2 == 0
    numTimes = int(len(x) / 2)
    sizes = x[:numTimes]
    times = x[numTimes:]
    for i in range(1,len(times)):
        times[i] += times[i-1]

    if addInfEpoch:
        sizes = [1.0] + sizes
        #sizes = sizes + [1.0]
        times = times + [float('inf')]
    
    pieces = [ConstantTruncatedSizeHistory(n, t, N) for t,N in zip(times, sizes)]
    return PiecewiseHistory(pieces)

@pytest.mark.parametrize("epochs,params", ((e,p) for e in (1,9) 
                                           for p in (1,2,4) 
                                           #for p in (1,)
                                           if p <= e * 2))
def test_sfs(epochs,params):
    n = 50

    N_bds, t_bds = (.1,10), (.1/float(epochs),10/float(epochs))
    N = np.random.uniform(*N_bds,size=epochs)
    N0 = np.random.uniform(*N_bds,size=epochs)
    t = np.random.uniform(*t_bds,size=epochs)
    t0 = np.random.uniform(*t_bds,size=epochs)

    truth = np.concatenate((N,t))
    x0 = np.concatenate((N0,t0))

    params = np.random.choice(np.arange(len(truth)), size=params, replace=False)
    print params
    print truth
    x0 = log(x0[params])
    x = log(truth[params])

    def statistic(xhat):
        #print xhat

        demo_prms = list(truth)
        for i in range(len(xhat)):
            demo_prms[params[i]] = exp(xhat[i])
        
        est_hist = get_demo_from_epochs(demo_prms, n)
        ret = np.array([est_hist.freq(der,n) for der in range(1,n)])
        #ret = est_hist.etjj
        #print ret
        return ret
    #x0 = np.random.uniform(0.1,10,len(x)) * x

    minimize_l2_err(statistic, x0, x)
