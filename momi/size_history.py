
from .util import memoize
import autograd.numpy as np
from autograd.numpy import sum, exp, log, expm1
from .math_functions import transformed_expi, expm1d
import scipy.integrate
from scipy.special import comb as binom
from . import moran_model

class SizeHistory(object):
    def __init__(self, tau, scaled_time):
        '''
        tau = length of time interval (in ms units)
        scaled_time = probability of 2 lineages coalescing within interval
                    = \int 2/N(t) dt, where N(t) = pop size in ms units
        '''
        self.tau = tau
        self.scaled_time = scaled_time

    def sfs(self, n):
        if n == 0:
            return np.array([0.])
        Et_jj = self.etjj(n)
        #assert np.all(Et_jj[:-1] - Et_jj[1:] >= 0.0) and np.all(Et_jj >= 0.0) and np.all(Et_jj <= self.tau)

        ret = np.sum(Et_jj[:, None] * Wmatrix(n), axis=0)

        before_tmrca = self.tau - np.sum(ret * np.arange(1, n) / n)
        # ignore branch length above untruncated TMRCA
        if self.tau == float('inf'):
            before_tmrca = 0.0
        
        ret = np.concatenate((np.array([0.0]), ret, np.array([before_tmrca])))
        return ret

    def transition_prob(self, v, axis=0):
        return moran_model.moran_action(self.scaled_time, v, axis=axis)

class ConstantHistory(SizeHistory):
    '''Constant size population truncated to time tau.'''
    def __init__(self, tau, N):
        if N <= 0.0:
            raise Exception("N must be positive")
        self.N = N
        super(ConstantHistory, self).__init__(tau, 2.0 * tau / N)

    def etjj(self, n):
        j = np.arange(2, n + 1)

        denom = binom(j, 2) / self.N * 2.0
        if self.tau == float('inf'):
            return 1.0 / denom

        scaled_time = denom * self.tau
        num = -expm1(-scaled_time) # equals 1 - exp(-scaled_time)
        ret = num / denom
        return ret

    def ms_cmd(self, pop_id, start_time, rescale=1.0):
        return "-en %f %d %f" % (start_time / rescale, pop_id, self.N / rescale)
    
class ExponentialHistory(SizeHistory):
    def __init__(self, tau, growth_rate, N_bottom):
        assert tau != float('inf')
        # if tau == float('inf'):
        #     ## TODO: make tau=inf work with automatic differentiation
        #     raise Exception("Exponential growth in final epoch not implemented. Try setting final growth rate to the constant 0.0, i.e. -eG t 0.0. Setting final growth rate to a variable (-eG t $0) will break, even if the variable equals 0.0.")
        self.N_bottom, self.growth_rate = N_bottom, growth_rate
        self.total_growth = tau * growth_rate
        self.N_top = N_bottom * exp(-self.total_growth)

        super(ExponentialHistory, self).__init__(tau, expm1d(self.total_growth) * tau / self.N_bottom * 2.0)

    def etjj(self, n):
        j = np.arange(2, n+1)
        jChoose2 = binom(j,2)

        pow0, pow1 = self.N_bottom/jChoose2/2.0 , self.total_growth
        ret = -transformed_expi(pow0 * self.growth_rate / exp(pow1))
        ret = ret * exp(-expm1d(pow1) * self.tau / pow0 - pow1)
        ret = ret + transformed_expi(pow0 * self.growth_rate)
        ret = ret * pow0

        return ret

    def ms_cmd(self, pop_id, start_time, rescale=1.0):
        return "-en %f %d %f -eg %f %d %f" % (start_time / rescale, pop_id, self.N_bottom / rescale, start_time / rescale, pop_id, self.growth_rate * rescale)

class PiecewiseHistory(SizeHistory):
    def __init__(self, pieces):
        tau = sum([p.tau for p in pieces])
        self.pieces = pieces
        super(PiecewiseHistory, self).__init__(tau, sum([pop.scaled_time for pop in self.pieces]))

    def etjj(self, n):
        j = np.arange(2, n+1)
        jChoose2 = binom(j,2)

        ret = np.zeros(len(j))
        noCoalProb = np.ones(len(j))
        for pop in self.pieces:
            ret = ret + noCoalProb * pop.etjj(n)
            if pop.scaled_time != float('inf'):
                noCoalProb = noCoalProb * exp(- pop.scaled_time * jChoose2)
            else:
                assert pop is self.pieces[-1]
        return ret

    def ms_cmd(self, pop_id, start_time, rescale=1.0):
        curr = start_time
        ret = []
        for pop in self.pieces:
            ret += [pop.ms_cmd(pop_id, curr, rescale=rescale)]
            curr += pop.tau
        return " ".join(ret)
    
@memoize
def W(n, b, j):
    if j == 2:
        return 6.0 / (n + 1)
    elif j == 3:
        return 30.0 * (n - 2 * b) / (n + 1) / (n + 2)
    else:
        jj = j - 2
        ret = W(n, b, jj) * -(1 + jj) * (3 + 2 * jj) * (n - jj) / jj / (2 * jj - 1) / (n + jj + 1)
        ret += W(n, b, jj + 1) * (3 + 2 * jj) * (n - 2 * b) / jj / (n + jj + 1)
        return ret

@memoize
def Wmatrix(n):
    ww = np.zeros([n - 1, n - 1])
    for i in range(2, n + 1):
        for j in range(1, n):
            ww[i - 2, j - 1] = W(n, j, i)
    return ww

# given vector [sfs{n,1},...,sfs{n,n}],
# returns (n-1)x(n-1) matrix whose (ij) entry is sfs{i,j}
def sfs_recurrence(sfs, tau):
    n_max = len(sfs)
    ret = np.zeros((n_max+1,n_max+1))
    ret[1:,n_max] = sfs

    for n in range(n_max - 1, 0, -1):
        for k in range(1, n + 1):
            ret[(k, n)] = (ret[(k + 1, n + 1)] * (k + 1) / (n + 1) + 
                    ret[(k, n + 1)] * (n + 1 - k) / (n + 1))

    # check accuracy
    assert tau == ret[(1, 1)] or abs(log(tau / ret[(1, 1)])) < 1e-14
    return ret[1:,1:]

## TODO: not currently used, and not yet compatible with automatic differentiation.
class FunctionalHistory(SizeHistory):
    '''Size history parameterized by an arbitrary function f.'''
    def __init__(self, tau, f):
        '''Initialize the model. For t > 0, f(t) >= is the instantaneous
        rate of coalescence (i.e., the inverse of the population size).
        f should accept and return vectors of times.
        '''
        self._f = f
        super(FunctionalHistory, self).__init__(tau, self._R(tau))

    def _R(self, t):
        return scipy.integrate.quad(self._f, 0, t)[0]

    def etjj(self, n):
        ret = []
        # TODO: if this is too slow for large n, it could be sped up
        # by using vectorized integration (a la scipy.integrate.simps)
        for j in range(2, n + 1):
            j2 = binom(j, 2)
            # tau * P(Tjj > tau)
            r1 = self.tau * exp(-j2 * self.scaled_time)
            def _int(t):
                return t * self._f(t) * exp(-j2 * self._R(t))
            r2 = scipy.integrate.quad(_int, 0, self.tau)[0]
            ret.append(r1 + j2 * r2)
        return ret

# class _TrivialHistory(object):
#     def sfs(self, n):
#         return np.zeros(n+1)

#     def transition_prob(self, v, axis=0):
#         return v + 0.0 # return copy of v
