import numpy as np
from cython.parallel cimport prange
from functools import lru_cache

@lru_cache(None)
def Wmatrix(int n):
    assert n >= 1
    W = np.zeros([n - 1, n - 1], dtype=float)
    cdef double[:, :] vW = W
    cdef int b, j
    if n > 1:
        bb = np.arange(1, n)
        W[:, 0] = 6. / (n + 1)
        if n > 2:
            W[:, 1] = 30. * (n - 2 * bb) / (n + 1) / (n + 2)
            with nogil:
                for b in prange(1, n):
                    for j in range(2, n - 1):
                        vW[b - 1, j] = vW[b - 1, j - 2] * -(1 + j) * (3 + 2 * j) * (n - j) / j / (2 * j - 1) / (n + j + 1)
                        vW[b - 1, j] += vW[b - 1, j - 1] * (3 + 2 * j) * (n - 2 * b) / j / (n + j + 1)
    return W.T
