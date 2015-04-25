from util import memoize, my_einsum
import scipy.sparse
from scipy.sparse.linalg import expm_multiply
import autograd.numpy as np
from autograd.numpy import dot, diag, exp, tensordot
from autograd.core import primitive

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

def moran_action(t, v, axis=0):
    n = v.shape[axis] - 1
    P, d, Pinv = moran_eigensystem(n)
    D = diag(exp(t * d))
    ## use einsum
    vlen,output_dim = len(v.shape), range(len(v.shape))
    output_dim[axis] = vlen
    ## seems more efficient to have PDPinv be a single arg, rather than 3 args
    ret = my_einsum(v, range(vlen), dot(P,dot(D,Pinv)), [vlen,axis], output_dim)
    assert ret.shape == v.shape
    return ret

## TODO: need to implement for autograd
def _old_moran_action(t, v):
    n = len(v) - 1
    return expm_multiply(rate_matrix(n)*t,v)
