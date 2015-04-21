#import numpy as np
from util import memoize
import scipy.sparse
from scipy.sparse.linalg import expm_multiply
import autograd.numpy as np
from autograd.numpy import dot, diag, exp
#from adarray.ad.admath import exp


@memoize
def rate_matrix(n, sparse_format="csr"):
    i = np.arange(n + 1)
    diag = i * (n - i) / 2.
    diags = [diag[:-1], -2 * diag, diag[1:]]
    M = scipy.sparse.diags(diags, [1, 0, -1], (n + 1, n + 1), format=sparse_format)
    return M

@memoize
def moran_eigensystem(n):
    M = np.asarray(rate_matrix(n).todense())
    d, P = np.linalg.eig(M)
    return P, d, np.linalg.inv(P)

def moran_action(t, v):
    n = v.shape[0] - 1
    P, d, Pinv = moran_eigensystem(n)
    D = diag(exp(t * d))
    # TODO: this can be optimized using np.einsum()
    ret = dot(P, dot(D, dot(Pinv,v)))
    return ret

    ##TODO: add this checks back in
    #assert np.all(v >= 0.0)
#     if np.all(ret >= 0.0):
#         return ret
#     else:
#         ## slower but more accurate
#         return _old_moran_action(t,v)

def _old_moran_action(t, v):
    n = len(v) - 1
    return expm_multiply(rate_matrix(n)*t,v)
#     M = rate_matrix(n)
#     return expm_multiply(t * M, v)

