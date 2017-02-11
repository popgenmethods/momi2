
import numpy as np
import momi.moran_model as moran_model
import pytest
from autograd import grad
from autograd.numpy import dot
import numdifftools as nd


def test_rate_matrix():
    n = 4
    M = moran_model.rate_matrix(n).todense()
    assert np.allclose(M,
                       [[0, 0, 0, 0, 0],
                        [(n - 1) / 2, -(n - 1), (n - 1) / 2, 0, 0],
                           [0, 2 * (n - 2) / 2, -2 * (n - 2),
                            2 * (n - 2) / 2, 0],
                           [0, 0, 3 * (n - 3) / 2, -3 * (n - 3),
                            3 * (n - 3) / 2],
                           [0, 0, 0, 0, 0]])

# @pytest.mark.parametrize("n,t",
#         ((n, t) for n in (5, 10, 50, 100, 250)
#             for t in (0.01, 0.1, 1.0, 10.0, 100.0) if n * t < 100))
# def test_eig_vs_expm(n, t):
#     print(n, t)

#     v = np.random.random(n + 1)

#     x,y,z = 1.0,2.0,3.0

#     t = x**2 * y**3 * z**4 * t

#     v = x**3 * y**2 * z**3 * v

#     w1 = moran_model.moran_action_eigen(t, v)
#     w2 = moran_model.moran_al_mohy_higham(t,v)

#     assert np.allclose(w1, w2)

#     def mma(y):
#         return dot(moran_model.moran_action(t,y), y)
#     mmagrad = grad(mma)
#     mmaegrad = nd.Jacobian(mma)
#     assert np.max(np.log(np.array(mmagrad(v)) / np.array(mmaegrad(v)))) < 1e-8

#     ## hessian vector product
#     mmahess_dot_y = grad(lambda x: dot(mmagrad(x),x))
#     mmahess_dot_y_e = nd.Jacobian(lambda x: dot(mmagrad(x),x))

#     #mmagrad_dot_y(v)
#     assert np.max(np.log(mmahess_dot_y(v) / mmahess_dot_y_e(v))) < 1e-6
