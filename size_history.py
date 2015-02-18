from __future__ import division
from util import EPSILON, memoize
from math import exp, fsum, log, expm1
from cached_property import cached_property
import numpy as np

import moran_model

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

class TruncatedSizeHistory(object):
    def __init__(self, N, n_max, tau):
        self.N = N
        self.n_max = n_max
        self.tau = tau

    def freq(self, n_derived, n):
        if n_derived == 0:
            return 0.0
        return self.sfs[(n_derived, n)]

    @cached_property
    def sfs(self):
        if self.n_max == 1:
            return {(1, 1): self.tau}
        ret = {}
        # compute the SFS for n_max via Polanski and Kimmel
        ww = np.zeros([self.n_max - 1, self.n_max - 1])
        for i in range(2, self.n_max + 1):
            for j in range(1, self.n_max):
                ww[i - 2, j - 1] = W(self.n_max, j, i)
        bv = (ww * self.etjj[:, None]).sum(axis=0)
        ret = dict(((b, self.n_max), v) for b, v in enumerate(bv, 1))
        # compute entry for monomorphic site
        ret[(self.n_max, self.n_max)] = self._before_tmrca(ret)

        # use recurrence to compute SFS for n < maxSampleSize
        for n in range(self.n_max - 1, 0, -1):
            for k in range(1, n + 1):
                ret[(k, n)] = (ret[(k + 1, n + 1)] * (k + 1) / (n + 1) + 
                        ret[(k, n + 1)] * (n + 1 - k) / (n + 1))

        # check accuracy
        assert self.tau == ret[(1, 1)] or abs(log(self.tau / ret[(1, 1)])) < EPSILON, \
                "%.16f, %.16f"  % (self.tau, ret[(1, 1)])
        return ret

    def _before_tmrca(self, partial_sfs):
        ret = self.tau - fsum([partial_sfs[(b, self.n_max)] * b / self.n_max 
                               for b in range(1, self.n_max)])
        # TODO: add assertion back in
        return ret

    def transition_prob(self, v):
        return moran_model.moran_action(self.scaled_time, v)


class ConstantTruncatedSizeHistory(TruncatedSizeHistory):
    '''Constant size population truncated to time tau.'''
    @cached_property
    def etjj(self):
        j = np.arange(2, self.n_max + 1)
        denom = j * (j - 1) / 2 / self.N
        scaled_time = denom * self.tau
        num = -np.expm1(-scaled_time) # equals 1 - exp(-scaledTime)
        assert np.all([num >= 0.0, num <= 1.0, num <= scaled_time]), "numerator=%g, scaledTime=%g" % (num, scaled_time)
        return num / denom
    
    @property
    def scaled_time(self):
        '''
        integral of 1/haploidN(t) from 0 to tau.
        used for Moran model transitions
        '''
        return self.tau / self.N

    def __str__(self):
        return "(ConstantPopSize: N=%f, tau=%f)" % (self.N, self.tau)
