import numpy as np
from util import memoize
import scipy.sparse
from scipy.sparse.linalg import expm_multiply

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
    n = len(v) - 1
    P, d, Pinv = moran_eigensystem(n)
    D = np.diag(np.exp(t * d))
    # TODO: this can be optimized using np.einsum()
    return P.dot(D).dot(Pinv).dot(v)

def _old_moran_action(t, v):
    n = len(v) - 1
    M = rate_matrix(n)
    return expm_multiply(t * M, v)
