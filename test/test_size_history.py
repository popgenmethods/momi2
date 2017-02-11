import momi.size_history as size_history
import momi.util as util
import numpy as np
from scipy.misc import comb as binom
import random


def q(n, b):
    return sum([k / binom(k, 2) * binom(n - b - 1, k - 2) / binom(n - 1, k - 1) / 2.0 for k in range(2, n + 1)])


def test_constant_sfs():
    n_max = 10
    hist = size_history.ConstantHistory(float("inf"), 1.0)
    for b in range(1, n_max):
        assert abs(hist.sfs(n_max)[b] - q(n_max, b)) < 1e-8


def test_functional_sfs_matches_constant():
    tau = random.expovariate(1.)
    N = 1000. * random.random()
    n_max = 100
    hist1 = size_history.ConstantHistory(tau, N)
    hist2 = size_history.FunctionalHistory(tau, lambda t: 2. / N)
    assert np.allclose(hist1.etjj(n_max), hist2.etjj(n_max))
    assert np.allclose(hist1.scaled_time, hist2.scaled_time)
