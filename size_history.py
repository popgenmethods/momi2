from __future__ import division
from util import EPSILON, memoize
from math import exp, fsum, log, expm1
from util import cached_property
import numpy as np
import scipy.integrate
from scipy.special import comb as binom

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
    def __init__(self, n_max, tau):
        self.n_max = n_max
        self.tau = tau
        # precompute
        self.sfs

    def freq(self, n_derived, n):
        if n_derived == 0:
            return 0.0
        if (n_derived, n) not in self.sfs and n < self.n_max and n_derived > 0:
            self._sfs_recurrence()
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

        return ret

    def _sfs_recurrence(self):
        # use recurrence to compute SFS for n < maxSampleSize
        for n in range(self.n_max - 1, 0, -1):
            for k in range(1, n + 1):
                self.sfs[(k, n)] = (self.sfs[(k + 1, n + 1)] * (k + 1) / (n + 1) +
                        self.sfs[(k, n + 1)] * (n + 1 - k) / (n + 1))

        # check accuracy
        assert self.tau == self.sfs[(1, 1)] or abs(log(self.tau / self.sfs[(1, 1)])) < EPSILON, \
                "%.16f, %.16f"  % (self.tau, self.sfs[(1, 1)])


    def _before_tmrca(self, partial_sfs):
        ret = self.tau - fsum([partial_sfs[(b, self.n_max)] * b / self.n_max
                               for b in range(1, self.n_max)])
        # TODO: add assertion back in
        return ret

    def transition_prob(self, v):
        return moran_model.moran_action(self.scaled_time, v)


class ConstantTruncatedSizeHistory(TruncatedSizeHistory):
    '''Constant size population truncated to time tau.'''
    def __init__(self, n_max, tau, N):
        self.N = N
        super(ConstantTruncatedSizeHistory, self).__init__(n_max, tau)

    @cached_property
    def etjj(self):
        j = np.arange(2, self.n_max + 1)
        denom = binom(j, 2) / self.N
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


class FunctionalTruncatedSizeHistory(TruncatedSizeHistory):
    '''Size history parameterized by an arbitrary function f.'''

    def __init__(self, n_max, tau, f):
        '''Initialize the model. For t > 0, f(t) >= is the instantaneous
        rate of coalescence (i.e., the inverse of the population size).
        f should accept and return vectors of times.
        '''
        self._f = f
        super(FunctionalTruncatedSizeHistory, self).__init__(n_max, tau)

    def _R(self, t):
        return scipy.integrate.quad(self._f, 0, t)[0]

    @cached_property
    def etjj(self):
        ret = []
        # TODO: if this is too slow for large n_max, it could be sped up
        # by using vectorized integration (a la scipy.integrate.simps)
        for j in range(2, self.n_max + 1):
            j2 = binom(j, 2)
            # tau * P(Tjj > tau)
            r1 = self.tau * np.exp(-j2 * self.scaled_time)
            def _int(t):
                return t * self._f(t) * np.exp(-j2 * self._R(t))
            r2 = scipy.integrate.quad(_int, 0, self.tau)[0]
            ret.append(r1 + j2 * r2)
        return np.array(ret)

    @cached_property
    def scaled_time(self):
        return self._R(self.tau)


class ExponentialTruncatedSizeHistory(TruncatedSizeHistory):
    def __init__(self, n_max, tau, N_top, N_bottom):
        if tau == float('inf'):
            raise Exception("Exponential growth only permitted for finite time intervals")
        self.N_top, self.N_bottom = N_top, N_bottom
        # N_bottom = N_top * exp(tau * growth_rate)
        self.growth_rate = log(N_bottom / N_top) / tau
        super(ExponentialTruncatedSizeHistory, self).__init__(n_max, tau)

    @cached_property
    def etjj(self):
        j = np.arange(2, self.n_max+1)
        ret = scipy.special.expi(- scipy.misc.comb(j,2) / self.N_bottom * np.exp(self.growth_rate * self.tau) / self.growth_rate)
        ret = ret - scipy.special.expi(- scipy.misc.comb(j,2) / self.N_bottom / self.growth_rate)
        ret = ret * (1.0 / self.growth_rate * np.exp(1.0 / self.growth_rate * scipy.misc.comb(j,2) / self.N_bottom))

        assert np.all(np.ediff1d(ret) <= 0) # ret should be decreasing
        assert np.all(ret >= 0)
        assert np.all(ret <= self.tau)

        return ret

    @cached_property
    def scaled_time(self):
        '''
        integral of 1/haploidN(t) from 0 to tau.
        used for Moran model transitions
        '''
        return 1.0 / self.growth_rate / self.N_bottom * np.expm1(self.growth_rate * self.tau)


class PiecewiseHistory(TruncatedSizeHistory):
    def __init__(self, pieces):
        n_max = pieces[0].n_max
        assert all([p.n_max == n_max for p in pieces])
        tau = sum([p.tau for p in pieces])
        self.pieces = pieces
        super(PiecewiseHistory, self).__init__(n_max, tau)

    @cached_property
    def etjj(self):
        j = np.arange(2, self.n_max+1)
        jChoose2 = scipy.misc.comb(j,2)

        ret = np.zeros(len(j))
        noCoalProb = np.ones(len(j))
        for pop in self.pieces:
            ret = ret + noCoalProb * pop.etjj
            noCoalProb = noCoalProb * np.exp(- pop.scaled_time * jChoose2)
        return ret

    @cached_property
    def scaled_time(self):
        return sum([pop.scaled_time for pop in self.pieces])
