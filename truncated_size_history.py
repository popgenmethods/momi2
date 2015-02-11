from __future__ import division
from util import EPSILON, memoize
from math import exp, fsum, log, expm1
from cached_property import cached_property
import numpy as np

@memoize
def W(n, b, j):
    if j == 2:
        return 6.0 / float(n + 1)
    elif j == 3:
        return 30.0 * (n - 2 * b) / float(n + 1) / float(n + 2)
    else:
        jj = j - 2
        ret = W(n, b, jj) * -(1 + jj) * (3 + 2 * jj) * (n - jj) / jj / (2 * jj - 1) / (n + jj + 1)
        ret += W(n, b, jj + 1) * (3 + 2 * jj) * (n - 2 * b) / jj / (n + jj + 1)
        return ret
    return ret

Wvec = np.vectorize(W, excluded=['n'])

class TruncatedSizeHistory(object):
    def __init__(self, n_max, tau):
        self.n_max = n_max
        self.tau = tau
        # Compute first coalescent times on startup

    @cached_property
    def before_tmrca(self):
        b = np.arange(1, self.n_max)[:, None]
        j = np.arange(2, self.n_max + 1)
        # FIXME: this can be more efficiently computed via the V term
        # in the same Polanski Kimmel paper
        ww = Wvec(n, b, j) * b / self.n_max * self.etjj
        ret = self.tau - fsum(ww.flat)
        # TODO: add assertion back in
        return ret

    @cached_property
    def sfs(self):
        ret = {}
        n_max = self.n_max
        # compute the SFS for n_max via Polanski and Kimmel
        ww = Wvec(n, np.arange(1, n_max), np.arange(2, n_max + 1)[:, None])
        bv = (ww * self.first_coalescent_time[:, None]).sum(axis=0)
        ret = dict(enumerate(bv, 1))

        # compute entry for monomorphic site
        ret[(n_max, n_max)] = self.before_tmrca

        # use recurrence to compute SFS for n < maxSampleSize
        for n in range(n_max, 0, -1):
            for k in range(1, n + 1):
                ret[(k, n)] = (self.sfs[(k + 1, n + 1)] * (k + 1) / (n + 1) + 
                        ret[(k, n + 1)] * (n + 1 - k) / (n + 1))

        # check accuracy
        assert self.tau == ret[(1, 1)] or abs(log(self.tau / ret[(1, 1)])) < EPSILON, \
                "%.16f, %.16f"  % (self.tau, ret[(1, 1)])
        return ret

    def freq(self, n_derived, n):
        if n_derived == 0:
            return 0.0
        return self.sfs[(n_derived, n)]

    def transition(self, v):
        return moran_model.moran_action(self.scaled_time, v)

class ConstantTruncatedSizeHistory(TruncatedSizeHistory):
    '''Constant size population truncated to time tau.'''

    def __init__(self, N, n_max, tau):
        '''
        N = 1 / coalescent rate
        '''
        super(ConstantTruncatedSizeHistory, self).__init__(n_max, tau)
        self.N = 1. * N
        
    @cached_property
    def etjj(self):
        j = np.arange(2, self.n_max + 1)
        denom = j * (j - 1) / 2 / self.N
        scaled_time = denom * self.tau
        num = -expm1(-scaled_time) # equals 1 - exp(-scaledTime)
        assert all([num >= 0.0, num <= 1.0, num <= scaled_time]), "numerator=%g, scaledTime=%g" % (num, scaled_time)
        return num / denom
    
    @cached_property
    def scaled_time(self):
        '''
        integral of 1/haploidN(t) from 0 to tau.
        used for Moran model transitions
        '''
        return self.tau / self.N

    def __str__(self):
        return "(ConstantPopSize: N=%f, tau=%f)" % (self.N, self.tau)
