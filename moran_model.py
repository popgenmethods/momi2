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

def moran_action(t, v):
    return expm_multiply(rate_matrix(len(v) - 1) * t, v)

