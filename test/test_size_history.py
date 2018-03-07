import momi.size_history as size_history
import momi.util as util
import numpy as np
from scipy.special import comb as binom
import random


def q(n, b):
    return sum([k / binom(k, 2) * binom(n - b - 1, k - 2) / binom(n - 1, k - 1) / 2.0 for k in range(2, n + 1)])


def test_constant_sfs():
    n_max = 10
    hist = size_history.ConstantHistory(float("inf"), 1.0)
    for b in range(1, n_max):
        assert abs(hist.sfs(n_max)[b] - q(n_max, b)) < 1e-8

