from __future__ import division
import pytest
import scipy.optimize
from ad import gh, adnumber
from ad.admath import log
from size_history import ConstantTruncatedSizeHistory
import numpy as np
import random

# returns xhat = argmin |f(x) - f(xhat)|_2^2
# using gradient descent starting from x0
def minimize_l2_err(f, x0, x, upper_bound=None, lower_bound=1e-12):
    fx = f(x)
    def objective(xhat):
        return sum((f(xhat)-fx)**2)
    grad, hess = gh(objective)
    bds = ((lower_bound,upper_bound),) * len(x0)

    x0d = adnumber(x0)
    o0d = objective(x0d)
    print "starting_x", x0d, "starting_obj", o0d, "starting_deriv", o0d.d(x0d[0]), "all_derivs" , o0d.d()

    res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', jac=grad, bounds=bds)
    assert sum((np.asarray(log(x)) - np.asarray(log(res.x)))**2) < 1e-6
    assert np.all(res.x > lower_bound)


@pytest.mark.parametrize("N,n,t", ((random.uniform(.1,10),random.randint(1,100), t) for t in (random.uniform(.01,100), float('inf'))))
def test_constant_size_freqs(N,n,t):
    def statistic(xhat):
        N_hat = xhat[0]
        print N_hat, t
        est_hist = ConstantTruncatedSizeHistory(n,t,N_hat)
        #ret = np.array([est_hist.freq(der,n) for der in range(1,n)])
        ret = est_hist.etjj
        print ret
        return ret
    x = np.array((N,))
    x0 = np.random.uniform(0.1,10,len(x)) * x

    minimize_l2_err(statistic, x0, x)
    
