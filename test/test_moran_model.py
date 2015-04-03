from __future__ import division
import numpy as np
import moran_model
import pytest

def test_rate_matrix():
    n = 4
    M = moran_model.rate_matrix(n).todense()
    assert np.allclose(M, 
            [[0, 0, 0, 0, 0],
             [(n - 1) / 2, -(n - 1), (n - 1) / 2, 0, 0],
             [0, 2 * (n - 2) / 2, -2 * (n - 2), 2 * (n - 2) / 2, 0],
             [0, 0, 3 * (n - 3) / 2, -3 * (n - 3), 3 * (n - 3) / 2],
             [0, 0, 0, 0, 0]])

@pytest.mark.parametrize("n,t", 
        ((n, t) for n in (5, 10, 50, 100, 250) 
            for t in (0.01, 0.1, 1.0, 10.0, 100.0) if n * t < 100))
def test_eig_vs_expm(n, t):
    print(n, t)
    v = np.random.random(n + 1)
    w1 = moran_model.moran_action(t, v)
    w2 = moran_model._old_moran_action(t, v)
    assert np.allclose(w1, w2)
