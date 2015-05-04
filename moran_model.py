from __future__ import division
from util import memoize, truncate0
from math_functions import einsum2, swapaxes
import scipy.sparse
from scipy.sparse.linalg import expm_multiply
import autograd.numpy as np
from autograd.numpy import dot, diag, exp
from autograd.core import primitive

@primitive
def moran_action(t,v, axis=0):
    assert np.all(v >= 0.0) and t >= 0.0
    ## fast, but has cancellation errors for small t
    ret = moran_action_eigen(t,v,axis)
    # deal with small negative numbers from cancellation
    ret = truncate0(ret, axis=axis)
#     if np.any(ret < 0.0):
#         ## more accurate, and theoretically lower complexity, but slower in practice
#         ret = moran_al_mohy_higham(t,v,axis)
#         assert np.all(ret >= 0.0)
    return ret

def moran_action_eigen(t, v, axis=0, transpose=False):
    n = v.shape[axis] - 1
    P, d, Pinv = moran_eigensystem(n)
    D = diag(exp(t * d))

    PDPinv = dot(P,dot(D,Pinv))
    if transpose:
        PDPinv = np.transpose(PDPinv)

    vlen,output_dim = len(v.shape), range(len(v.shape))
    output_dim[axis] = vlen
    ret = einsum2(v, range(vlen), PDPinv, [vlen,axis], output_dim)
    assert ret.shape == v.shape
    return ret

def moran_al_mohy_higham(t, v, axis=0, transpose=False):
    return moran_apply(lambda M,x: expm_multiply(M*t,x), v, axis, transpose)

'''
Derivatives of action of matrix exponential of Moran rate matrix
'''
## derivative with respect to t
moran_action.defgrad(lambda ans,t,v,axis=0:
                         lambda g: einsum2(g, range(ans.ndim),
                                           moran_dot(ans,axis), range(ans.ndim), []))
## derivative with respect to v
moran_action.defgrad(lambda ans,t,v,axis=0:
                         lambda g: moran_action_eigen(t, g, axis, transpose=True), argnum=1)


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

@primitive
def moran_dot(v, axis, transpose=False):
    '''compute matrix product of Moran rate matrix (transpose) with v'''
    return moran_apply(lambda M,x: M.dot(x), v, axis, transpose)
moran_dot.defgrad(lambda ans,v,axis, transpose=False: lambda g: moran_dot(g,axis, not transpose))

def moran_apply(f,v,axis, transpose=False):
    '''
    get Moran rate matrix (transpose), make v have appropriate shape, and call f(M,v)
    '''
    n = v.shape[axis]-1
    v = swapaxes(v, 0, axis)
    old_shape = v.shape    

    v = np.reshape(v, [v.shape[0],-1])
    M = rate_matrix(n)
    if transpose:
        M = M.transpose()

    ret = f(M, v)
    ret = np.reshape(ret, old_shape)
    ret = swapaxes(ret, 0, axis)

    return ret

