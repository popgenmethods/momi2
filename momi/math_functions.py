from __future__ import division
import autograd.numpy as np
from autograd.core import primitive
import scipy
from util import memoize, truncate0, set0

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

    # gradient of einsum doesn't work with repeated axes
    # get rid of repeated axes, and replace by appropriate multiplication with Identity
    additional_args=[]
    next_idx = len(idx_to_int)
    for argnum in idx_argnum[:-1]:
        arr,idxs = args[argnum-1], args[argnum]
        old_idxs,idxs = idxs,[]
        for i,old_idx in enumerate(old_idxs):
            if old_idx not in idxs:
                idxs += [old_idx]
            else:
                prev_i = idxs.index(old_idx)
                idxs += [next_idx]
                additional_args += [np.eye(arr.shape[prev_i], arr.shape[i]), [old_idx, next_idx]]
                next_idx += 1
        args[argnum] = idxs
                
    return np.einsum(*(args[:-1] + additional_args + [args[-1]]))

def convolve_axes(arr0, arr1, labs, axes, out_axis):
    lab0,lab1 = labs = [list(l) for l in labs]
    axis0,axis1 = axes
    idx0,idx1 = [l.index(a) for l,a in zip(labs,axes)]
    
    out_labs = set(lab0 + lab1)
    for old_axis in axis0,axis1:
        out_labs.remove(old_axis)
    out_labs = [out_axis] + sorted(list(out_labs))

    lab0[idx0] = lab1[idx1] = out_axis
    ret = fft_einsum(out_axis, arr0, lab0, arr1, lab1, out_labs)
    
    return ret, out_labs

def sum_antidiagonals(arr, labels, axis0, axis1, out_axis):
    out_labs = sorted(list(labels))
    for old_axis in axis0,axis1:
        out_labs.remove(old_axis)
    out_labs = [out_axis] + out_labs

    labels = list(labels)
    for i,l in enumerate(labels):
        if l in (axis0,axis1):
            labels[i] = out_axis
    ret = fft_einsum(out_axis, arr, labels, out_labs)
    
    return ret, out_labs

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




# like einsum2, but for fft_label, does multiplication in fourier domain
# (i.e. does convolution instead of multiplication for fft_label)
def fft_einsum(fft_label, *args, **kwargs):
    args, out_labels = list(args[:-1]), args[-1]
    assert len(args) % 2 == 0

    for i in range(int(len(args) / 2)):
        arr, labels = args[2*i], args[2*i+1]
        assert np.all(arr >= 0.0)

    ## for numerical stability, split up fft into low-precision and high-precision parts
    ## (a bit hacky, need to find a better solution)
    tol = kwargs.get('tol', 1e-13)
    sqrt_tol = np.sqrt(tol)
    for i in range(int(len(args) / 2)):
        arr, labels = args[2*i], args[2*i+1]
        for j,l in enumerate(labels):
            if l == fft_label:
                # the maximum along axis
                maxes = np.amax(arr, axis=j)[[slice(None)] * j + [np.newaxis]
                                             + [slice(None)] * (arr.ndim -j-1) ]
                if np.any(np.logical_and(arr < sqrt_tol * maxes, arr > 0)):
                    args1, args2 = [list(args) + [out_labels] for _ in range(2)]
                    args1[2*i] = set0(arr, arr < sqrt_tol * maxes)
                    args2[2*i] = set0(arr, arr >= sqrt_tol * maxes)
                    return fft_einsum(fft_label, *args1, **kwargs) + fft_einsum(fft_label, *args2, **kwargs)
                    
    fft_shapes = []
    for i in range(int(len(args) / 2)):
        arr, labels = args[2*i], args[2*i+1]
        for j,l in enumerate(labels):
            if l == fft_label:
                fft_shapes += [arr.shape[j]-1]
        
    fft_shape = sum(fft_shapes) + 1
    fshape = _next_regular(fft_shape)
    
    for i in range(int(len(args) / 2)):
        arr, labels = args[2*i], args[2*i+1]
        for j,l in enumerate(labels):
            if l == fft_label:
                arr = args[2*i] = np.fft.fftn(arr, [fshape], [j])

    out_fft_idx, = [j for j,l in enumerate(out_labels) if l == fft_label]
                
    out_slice = [slice(None)] * len(out_labels)
    out_slice[out_fft_idx] = slice(fft_shape)
    
    ret = einsum2(*(args + [out_labels]))
    ret = np.real(np.fft.ifftn(ret, axes=[out_fft_idx])[out_slice])

    ret = truncate0(ret, axis=out_fft_idx, tol=tol, strict=True)
    return ret


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
