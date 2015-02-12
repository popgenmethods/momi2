import size_history
import util
import numpy as np
from scipy.misc import comb as binom

def q(n, b):
    return sum([k / binom(k, 2) * binom(n - b - 1, k - 2) / binom(n - 1, k - 1) for k in range(2, n + 1)])

def test_constant_sfs():
    n_max = 10
    hist = size_history.ConstantTruncatedSizeHistory(1.0, n_max, float("inf"))
    for n in range(2, n_max + 1):
        for b in range(1, n):
            assert abs(hist.sfs[(b, n)] - q(n, b)) < 1e-8
