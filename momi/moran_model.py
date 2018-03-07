
from .util import memoize, check_probs_matrix
from .math_functions import par_einsum
import scipy.sparse
import autograd.numpy as np
from autograd.numpy import dot, diag, exp


def moran_transition(t, n):
    assert t >= 0.0
    P, d, Pinv = moran_eigensystem(n)
    D = diag(exp(t * d))
    return check_probs_matrix(dot(P, dot(D, Pinv)))

def moran_action(t, v, axis=0):
    if v.shape[axis] == 1:
        return v

    n = v.shape[axis] - 1
    PDPinv = moran_transition(t, n)

    vlen, output_dim = len(v.shape), list(range(len(v.shape)))
    output_dim[axis] = vlen
    ret = par_einsum(v, list(range(vlen)), PDPinv, [vlen, axis], output_dim)
    assert ret.shape == v.shape
    return ret


@memoize
def rate_matrix(n, sparse_format="csr"):
    i = np.arange(n + 1)
    diag = i * (n - i) / 2.
    diags = [diag[:-1], -2 * diag, diag[1:]]
    M = scipy.sparse.diags(
        diags, [1, 0, -1], (n + 1, n + 1), format=sparse_format)
    return M


@memoize
def moran_eigensystem(n):
    M = np.asarray(rate_matrix(n).todense())
    d, P = np.linalg.eig(M)
    return P, d, np.linalg.inv(P)
