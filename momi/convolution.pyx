cimport numpy as np
import numpy as np
import cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sum_trailing_antidiagonals(np.ndarray[np.double_t, ndim=3] A):
    '''
    Sums the antidiagonals of the last two axes of A
    '''
    cdef int i,j,k,I,J,K
    I,J,K = A.shape[0],A.shape[1],A.shape[2]

    cdef np.ndarray[np.double_t, ndim=2] B = np.zeros((I, J + K - 1))

    for i in prange(I, schedule='guided', nogil=True):
        for j in range(J):
            for k in range(K):
                    B[i,j+k] += A[i,j,k]
    return B

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def add_trailing_axis(np.ndarray[np.double_t, ndim=2] B, int trailing_dim):
    '''
    Takes the last dimension of B, and expands it into two dimensions,
    so that the entries of the expanded matrix are constant along the antidiagonals
    of the last two dimensions.

    If sum_trailing_antidiagonals is viewed as multiplication by a tensor,
    this is equivalent to multiplying by the "transpose" of that tensor
    '''
    assert trailing_dim <= B.shape[1]

    cdef np.ndarray[np.double_t, ndim=3] A = np.zeros((B.shape[0],
                                                       B.shape[1] - trailing_dim + 1,
                                                       trailing_dim))

    cdef int i,j,k,I,J,K
    I,J,K = A.shape[0],A.shape[1],A.shape[2]

    cdef int ijk,jk,IJK,JK
    JK = J*K
    IJK = I*JK
    for ijk in prange(IJK, schedule='guided', nogil=True):
        i = ijk // JK
        jk = ijk % JK
        j = jk // K
        k = jk % K
        A[i,j,k] = B[i,j+k]
    return A

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convolve_sum_axes(np.ndarray[np.double_t, ndim=4] A,
                      np.ndarray[np.double_t, ndim=4] B):
    # first dimension is for data points (shared by A and B)
    assert A.shape[0] == B.shape[0]
    # last dimension is for matrix multiply (reduction)
    assert A.shape[3] == B.shape[3]

    cdef np.ndarray[np.double_t, ndim=4] C = np.zeros((A.shape[0],
                                                       A.shape[1],
                                                       B.shape[1],
                                                       A.shape[2] + B.shape[2] - 1))

    cdef int i,j,k,l,m,n,I,J,K,L,M,N
    I,J,L = A.shape[0],A.shape[1],A.shape[2]
    K,M = B.shape[1],B.shape[2]
    N = A.shape[3]

    cdef int ijk, IJK, jk, JK
    JK = J*K
    IJK = I*JK
    for ijk in prange(IJK, schedule='guided', nogil=True):
        i = ijk // JK
        jk = ijk % JK
        j = jk // K
        k = jk % K
        for l in range(L):
            for m in range(M):
                for n in range(N):
                    C[i,j,k,l+m] += A[i,j,l,n] * B[i,k,m,n]
    return C

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def transposed_convolve_sum_axes(np.ndarray[np.double_t, ndim=4] C,
                                 np.ndarray[np.double_t, ndim=4] B):
    '''
    If convolve_trailing_axes is viewed as multiplying A and B by a certain tensor,
    this is equal to multiplying C and B by that tensor, but with the tensor transposed along
    the A/C directions.
    '''
    Ashape = (C.shape[0], C.shape[1], C.shape[3] + 1 - B.shape[2], B.shape[3])

    cdef np.ndarray[np.double_t, ndim=4] A = np.zeros(Ashape)

    cdef int i,j,k,l,m,n,I,J,K,L,M,N
    I,J,L = A.shape[0],A.shape[1],A.shape[2]
    K,M = B.shape[1],B.shape[2]
    N = A.shape[3]

    cdef int ijln, IJLN, ijl, IJL, jl, JL
    JL = J*L
    IJL = I*JL
    IJLN = IJL*N
    for ijln in prange(IJLN, schedule='guided', nogil=True):
        n = ijln // IJL
        ijl = ijln % IJL
        i = ijl // JL
        jl = ijl % JL
        j = jl // L
        l = jl % L
        for k in range(K):
            for m in range(M):
                A[i,j,l,n] += C[i,j,k,l+m] * B[i,k,m,n]
    return A


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def roll_trailing_axes(np.ndarray[np.double_t, ndim=3] A):
    '''
    Returns array B[i,j,j+k] = A[i,j,k]
    '''
    cdef int i,j,k,I,J,K
    I,J,K = A.shape[0],A.shape[1],A.shape[2]

    cdef np.ndarray[np.double_t, ndim=3] B = np.zeros((I, J, J + K - 1))

    cdef int ijk,IJK,jk,JK
    JK = J*K
    IJK = I*JK
    for ijk in prange(IJK, schedule='guided', nogil=True):
        i = ijk // JK
        jk = ijk % JK
        j = jk // K
        k = jk % K
        B[i,j,j+k] = A[i,j,k]
    return B

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def unroll_trailing_axes(np.ndarray[np.double_t, ndim=3] B):
    '''
    Inverse of roll trailing axes
    '''
    cdef int i,j,k,I,J,K
    I,J = B.shape[0],B.shape[1]
    K = B.shape[2]-J+1

    cdef np.ndarray[np.double_t, ndim=3] A = np.zeros((I, J, K))

    cdef int ijk,IJK,jk,JK
    JK = J*K
    IJK = I*JK
    for ijk in prange(IJK, schedule='guided', nogil=True):
        i = ijk // JK
        jk = ijk % JK
        j = jk // K
        k = jk % K
        A[i,j,k] = B[i,j,j+k]
    return A
