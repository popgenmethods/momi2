from __future__ import division
import autograd.numpy as np
from autograd.core import primitive
import scipy
from util import memoize

def einsum2(*args):
    '''
    like numpy.einsum, using format
    einsum(op0, sublist0, op1, sublist1, ..., sublistout).
    however, sublist can have arbitrary labels (not just integers)

    format einsum(subscripts, *operands) NOT supported
    '''
    assert len(args) % 2 == 1

    args, enum_args = list(args), list(enumerate(args))
    # convert the index arguments to have integer type
    idx_argnum,idx_lists = zip(*(enum_args[1::2] + [enum_args[-1]]))
    idx_lists = map(list, idx_lists)
    idx_to_int = {idx: i for i,idx in enumerate(set(sum(idx_lists, [])))}

    for argnum,idxs in zip(idx_argnum,idx_lists):
        args[argnum] = [idx_to_int[i] for i in idxs]
    
    ## TODO: use np.tensordot instead (faster, easier to parallelize)
    return np.einsum(*args)

def sum_antidiagonals(arr, labels, axis0, axis1, new_axis):
    assert axis0 != axis1
    idx0,idx1 = labels.index(axis0), labels.index(axis1)

    ret = np.swapaxes(np.swapaxes(arr, idx0, 0), idx1, 1)[::-1,...]
    ret = np.array([np.trace(ret,offset=k) 
                    for k in range(-ret.shape[0]+1,ret.shape[1])])    

    # swap labels
    labels = list(labels)
    for i,idx in list(enumerate((idx0,idx1))):
        labels[i],labels[idx] = labels[idx],labels[i]
    labels = [new_axis] + labels[2:]
   
    return ret,labels

'''
Returns
-expi(-1/x) * exp(1/x) / x
for x s.t. abs(x) is decreasing
'''
def transformed_expi(x):
    abs_x = np.abs(x)
    ser = abs_x < 1./45.
    nser = np.logical_not(ser)

#     ret = np.zeros(x.shape)
#     ret[ser], ret[nser] = transformed_expi_series(x[ser]), transformed_expi_naive(x[nser])))
#     return ret

    ## We use np.concatenate to combine.
    ## would be better to use ret[ser] and ret[nser] as commented out above
    ## but array assignment not yet supported by autograd
    assert np.all(abs_x[:-1] >= abs_x[1:])
    return np.concatenate((transformed_expi_naive(x[nser]), transformed_expi_series(x[ser])))

def transformed_expi_series(x):
    c_n, ret = 1., 1.
    for n in range(1,11):
        c_n = -c_n * x * n
        ret = ret + c_n
    return ret

def transformed_expi_naive(x):
    return -expi(-1.0/x) * np.exp(1.0/x) / x

@primitive
def expi(x):
    return scipy.special.expi(x)
expi.defgrad(lambda ans,x: lambda g: g * np.exp(x) / x)

'''
returns (e^x-1)/x, for scalar x. works for x=0.
Taylor series is 1 + x/2! + x^2/3! + ...
'''
def expm1d(x):
    if x == 0.0:
        return expm1d_taylor(x)
    elif x == float('inf'):
        return float('inf')
    return np.expm1(x)/x
## used for higher order derivatives at x=0 and x=inf
def expm1d_taylor(x):
    c_n, ret = 1.,1.
    for n in range(2,11):
        c_n = c_n * x / float(n)
        ret = ret + c_n
    return ret

log_factorial = lambda n: scipy.special.gammaln(n+1)
log_binom = lambda n,k: log_factorial(n) - log_factorial(k) - log_factorial(n-k)
def hypergeom_mat(N,n):
    K = np.outer(np.arange(N+1), np.ones(n+1))
    k = np.outer(np.ones(N+1), np.arange(n+1))
    ret = log_binom(K,k)
    ret = ret + ret[::-1,::-1]
    ret = ret - log_binom(N,n)
    return np.exp(ret)

@memoize
def hypergeom_quasi_inverse(N,n):
    u,s,v = np.linalg.svd(hypergeom_mat(N,n), full_matrices=False)
    return np.dot(u, np.dot(np.diag(1/s), v))

# like einsum, but for labels in fft_labels, does multiplication in fourier domain
# (i.e. does convolution instead of multiplication for fft_labels)
def fft_einsum(in1, labels1, in2, labels2, out_labels, fft_labels):
    assert all([l in labels1 and l in labels2 for l in fft_labels])

    labels = out_labels,labels1,labels2
    fft_idx = []
    for lab in labels:
        fft_idx.append(np.array([lab.index(l) for l in fft_labels]))
    
    fft_shapes = np.array(in1.shape)[fft_idx[1]] + np.array(in2.shape)[fft_idx[2]] - 1
    fshape = np.array([_next_regular(int(d)) for d in fft_shapes])

    out_slice = np.array([slice(None)] * len(out_labels))
    out_slice[fft_idx[0]] = np.array([slice(s) for s in fft_shapes])
    
    ret = einsum2(np.fft.fftn(in1, fshape, fft_idx[1]), labels1,
                  np.fft.fftn(in2, fshape, fft_idx[2]), labels2,
                  out_labels)
    return np.real(np.fft.ifftn(ret, axes=fft_idx[0])[list(out_slice)])
                    
def _next_regular(target):
    """
    COPIED FROM SCIPY.SIGNAL
    -----------------
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match
