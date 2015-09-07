cimport numpy as np
import numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sum_trailing_antidiagonals(np.ndarray[np.double_t, ndim=3] A):
    '''
    Sums the antidiagonals of the last two axes of A
    '''
    A = np.array(A)
    cdef unsigned int i,j,k,I,J,K
    I,J,K = A.shape[0],A.shape[1],A.shape[2]

    cdef np.ndarray[np.double_t, ndim=2] B = np.zeros((I, J + K - 1))
    
    for i in range(I):
        for j in range(J):
            for k in range(K):
                    B[i,j+k] += A[i,j,k]
    return B

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def add_trailing_axis(np.ndarray[np.double_t, ndim=2] B, unsigned int trailing_dim):
    '''
    Takes the last dimension of B, and expands it into two dimensions,
    so that the entries of the expanded matrix are constant along the antidiagonals
    of the last two dimensions.

    If sum_trailing_antidiagonals is viewed as multiplication by a tensor,
    this is equivalent to multiplying by the "transpose" of that tensor
    '''
    assert trailing_dim < B.shape[1]

    cdef np.ndarray[np.double_t, ndim=3] A = np.zeros((B.shape[0],
						       B.shape[1] - trailing_dim + 1,
                                                       trailing_dim))

    cdef unsigned int i,j,k,I,J,K
    I,J,K = A.shape[0],A.shape[1],A.shape[2]

    for i in range(I):
        for j in range(J):
            for k in range(K):
                A[i,j,k] = B[i,j+k]
    return A

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def convolve_trailing_axes(np.ndarray[np.double_t, ndim=3] A,
                           np.ndarray[np.double_t, ndim=3] B):
    # first dimension is for data points (shared by A and B)
    assert A.shape[0] == B.shape[0]
    
    cdef np.ndarray[np.double_t, ndim=4] C = np.zeros((A.shape[0],
                                                       A.shape[1],
                                                       B.shape[1],
                                                       A.shape[2] + B.shape[2] - 1))

    cdef unsigned int i,j,k,l,m,I,J,K,L,M
    I,J,L = A.shape[0],A.shape[1],A.shape[2]
    K,M = B.shape[1],B.shape[2]

    for i in range(I):
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    for m in range(M):
                        C[i,j,k,l+m] += A[i,j,l] * B[i,k,m]
    return C

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def transposed_convolve_trailing_axes(np.ndarray[np.double_t, ndim=4] C,
                                      np.ndarray[np.double_t, ndim=3] B,
                                      tuple Ashape):
    '''
    If convolve_trailing_axes is viewed as multiplying A and B by a certain tensor,
    this is equal to multiplying C and B by that tensor, but with the tensor transposed along
    the A/C directions.
    '''
    assert len(Ashape) == 3
    assert C.shape[0] == B.shape[0] and C.shape[0] == Ashape[0]
    assert Ashape[2] + B.shape[2] - 1 == C.shape[3]

    cdef np.ndarray[np.double_t, ndim=3] A = np.zeros(Ashape)

    cdef unsigned int i,j,k,l,m,I,J,K,L,M
    I,J,L = A.shape[0],A.shape[1],A.shape[2]
    K,M = B.shape[1],B.shape[2]

    for i in range(I):
        for j in range(J):
            for k in range(K):
                for l in range(L):
                    for m in range(M):
                        A[i,j,l] += C[i,j,k,l+m] * B[i,k,m]
    return A
