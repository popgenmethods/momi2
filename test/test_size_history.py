import size_history
import util
import numpy as np
from scipy.misc import comb as binom
import random
from adarray import array

def q(n, b):
    return sum([k / binom(k, 2) * binom(n - b - 1, k - 2) / binom(n - 1, k - 1) for k in range(2, n + 1)])

def test_constant_sfs():
    n_max = 10
    hist = size_history.ConstantTruncatedSizeHistory(n_max, float("inf"), 1.0)
    for b in range(1, n_max):
        assert abs(hist.sfs[(b, n_max)] - q(n_max, b)) < 1e-8
#     for n in range(2, n_max + 1):
#         for b in range(1, n):
#             assert abs(hist.sfs[(b, n)] - q(n, b)) < 1e-8

def test_functional_sfs_matches_constant():
    tau = random.expovariate(1.)
    N = 1000. * random.random()
    n_max = 100
    hist1 = size_history.ConstantTruncatedSizeHistory(n_max, tau, N)
    hist2 = size_history.FunctionalTruncatedSizeHistory(n_max, tau, lambda t: 1. / N)
    assert np.allclose(hist1.etjj.x, array(hist2.etjj).x)
    assert np.allclose(hist1.scaled_time.x, hist2.scaled_time)
