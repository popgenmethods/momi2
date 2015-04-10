from __future__ import division
import numpy as np
import moran_model
import pytest
from adarray import adnumber

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
def test_eig_vs_expm(n, t, ad_order2):
    print(n, t)
    
    v = np.random.random(n + 1)

    x,y,z = adnumber([1,1,1])

    t = x**2 * y**3 * z**4 * t

    v = x**3 * y**2 * z**3 * v

    w1 = moran_model.moran_action(t, v)
    w2 = moran_model._old_moran_action(t, v)
    assert np.allclose(w1.x, w2.x)

    assert np.max(np.log(np.array(w1.gradient([x,y,z])) / np.array(w2.gradient([x,y,z])))) < 1e-8

    assert np.max(np.log(np.array(w1.hessian([x,y,z])) / np.array(w2.hessian([x,y,z])))) < 1e-6
