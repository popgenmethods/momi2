cimport numpy as np
import numpy as np
import cython

@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def sum_trailing_antidiagonals(np.ndarray A):
    '''
    Sums the antidiagonals of the last two axes of A
    '''
    A = np.ascontiguousarray(A)
    assert A.ndim == 3

    cdef unsigned int i,j,k,I,J,K
    I,J,K = A.shape[0],A.shape[1],A.shape[2]

    cdef np.ndarray B = np.zeros((I, J + K - 1))

    cdef double[:,:,::1] Abuf = A
    cdef double[:,::1] Bbuf = B
    
    for i in range(I):
        for j in range(J):
            for k in range(K):
                    Bbuf[i,j+k] += Abuf[i,j,k]
    return B

@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def add_trailing_axis(np.ndarray B, unsigned int trailing_dim):
    '''
    Takes the last dimension of B, and expands it into two dimensions,
    so that the entries of the expanded matrix are constant along the antidiagonals
    of the last two dimensions.

    If sum_trailing_antidiagonals is viewed as multiplication by a tensor,
    this is equivalent to multiplying by the "transpose" of that tensor
    '''
    B = np.ascontiguousarray(B)
    assert B.ndim == 2
    assert trailing_dim < B.shape[1]

    cdef np.ndarray A = np.zeros((B.shape[0], B.shape[1] - trailing_dim + 1, trailing_dim))

    cdef unsigned int i,j,k,I,J,K
    I,J,K = A.shape[0],A.shape[1],A.shape[2]

    cdef double[:,:,::1] Abuf = A
    cdef double[:,::1] Bbuf = B

    for i in range(I):
        for j in range(J):
            for k in range(K):
                Abuf[i,j,k] = Bbuf[i,j+k]
    return A

@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def convolve_trailing_axes(np.ndarray A, np.ndarray B):
    A,B = (np.ascontiguousarray(x) for x in (A,B))
    assert A.ndim == 3 and B.ndim == 3
    # first dimension is for data points (shared by A and B)
    assert A.shape[0] == B.shape[0]
    
    cdef np.ndarray C = np.zeros((A.shape[0], A.shape[1], B.shape[1], A.shape[2] + B.shape[2] - 1))

    cdef unsigned int i,j,k,l,m,I,J,K,L,M
    I,J,L = A.shape[0],A.shape[1],A.shape[2]
    K,M = B.shape[1],B.shape[2]

    cdef double[:,:,::1] Abuf = A
    cdef double[:,:,::1] Bbuf = B
    cdef double[:,:,:,::1] Cbuf = C

    for i in range(I):
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    for m in range(M):
                        Cbuf[i,j,k,l+m] += Abuf[i,j,l] * Bbuf[i,k,m]
    return C

@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.cdivision(True)
def transposed_convolve_trailing_axes(np.ndarray C, np.ndarray B, tuple Ashape):
    '''
    If convolve_trailing_axes is viewed as multiplying A and B by a certain tensor,
    this is equal to multiplying C and B by that tensor, but with the tensor transposed along
    the A/C directions.
    '''
    B,C = (np.ascontiguousarray(x) for x in (B,C))    
    assert C.ndim == 4 and B.ndim == 3 and len(Ashape) == 3
    assert C.shape[0] == B.shape[0] and C.shape[0] == Ashape[0]
    assert Ashape[2] + B.shape[2] - 1 == C.shape[3]

    cdef np.ndarray A = np.zeros(Ashape)

    cdef unsigned int i,j,k,l,m,I,J,K,L,M
    I,J,L = A.shape[0],A.shape[1],A.shape[2]
    K,M = B.shape[1],B.shape[2]

    cdef double[:,:,::1] Abuf = A
    cdef double[:,:,::1] Bbuf = B
    cdef double[:,:,:,::1] Cbuf = C

    for i in range(I):
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    for m in range(M):
                        Abuf[i,j,l] += Cbuf[i,j,k,l+m] * Bbuf[i,k,m]
    return A
