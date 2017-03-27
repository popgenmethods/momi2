cimport numpy as np
import numpy as np
import cython
from cython.parallel import prange, parallel

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _par_matmul(np.ndarray[np.double_t, ndim=3] A,
                np.ndarray[np.double_t, ndim=3] B):
    if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
        raise ValueError("Invalid dimensions for matmul")

    cdef np.ndarray[np.double_t, ndim=3] C = np.zeros((A.shape[0],
                                                       A.shape[1],
                                                       B.shape[2]))

    cdef int I,J,K,L,JL,IJL
    I,J,K,L = A.shape[0], A.shape[1], A.shape[2], B.shape[2]
    JL = J*L
    IJL = I*JL

    cdef int i,j,k,l,jl,ijl
    for ijl in prange(IJL, schedule='guided', nogil=True):
        i = ijl // JL
        jl = ijl % JL
        j = jl // L
        l = jl % L
        for k in range(K):
            C[i,j,l] += A[i,j,k] * B[i,k,l]

    return C
