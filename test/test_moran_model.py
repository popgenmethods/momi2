from __future__ import division
import numpy as np
import moran_model

def test_rate_matrix():
    n = 4
    M = moran_model.rate_matrix(n).todense()
    assert np.allclose(M, 
            [[0, 0, 0, 0, 0],
             [(n - 1) / 2, -(n - 1), (n - 1) / 2, 0, 0],
             [0, 2 * (n - 2) / 2, -2 * (n - 2), 2 * (n - 2) / 2, 0],
             [0, 0, 3 * (n - 3) / 2, -3 * (n - 3), 3 * (n - 3) / 2],
             [0, 0, 0, 0, 0]])
