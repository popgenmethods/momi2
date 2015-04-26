import functools
import autograd.numpy as np
from autograd.numpy.fft import fftn, ifftn
from autograd.core import primitive
from functools import partial
import itertools
import scipy

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

EPSILON = 1e-8

def H(n):
    return (1. / np.arange(1, n + 1)).sum()

def memoize(obj):
    cache = obj.cache = {}
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        # return obj(*args, **kwargs)
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer


class memoize_instance(object):
    """cache the return value of a method
    
    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.
    
    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    
    recipe from http://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res

@primitive
def swapaxes(a, axis1, axis2):
    return np.swapaxes(a, axis1, axis2)
swapaxes.defgrad(lambda ans,a,axis1,axis2:
                     lambda g: swapaxes(g, axis1,axis2))

@primitive
def my_trace(a, offset):
    return np.trace(a, offset)
my_trace.defgrad(lambda ans,a,offset:
                     lambda g: my_einsum(np.eye(a.shape[0], a.shape[1], k=offset),
                                         [0,1],
                                         g, range(2, len(a.shape)),
                                         range(len(a.shape))))

@primitive
def my_einsum(*args):
    '''
    assumes format einsum(op0, sublist0, op1, sublist1, ..., sublistout),
    NOT format einsum(subscripts, *operands)
    '''
    assert len(args) % 2 == 1

    args, enum_args = list(args), list(enumerate(args))
    # convert the index arguments to have integer type
    idx_argnum,idx_lists = zip(*(enum_args[1::2] + [enum_args[-1]]))
    idx_lists = map(list, idx_lists)
    idx_to_int = {idx: i for i,idx in enumerate(set(sum(idx_lists, [])))}

    for argnum,idxs in zip(idx_argnum,idx_lists):
        args[argnum] = [idx_to_int[i] for i in idxs]
    
    return np.einsum(*args)
def make_einsum_grad(argnum, ans, *args):
    if argnum % 2 == 1:
        raise Exception()
    grad_args = list(args)
    grad_args[-1] = args[argnum+1]
    grad_args[argnum+1] = args[-1]
    def grad(g):
       curr_args = list(grad_args)
       curr_args[argnum] = g
       return my_einsum(*curr_args)
    return grad
my_einsum.gradmaker = make_einsum_grad

def sum_antidiagonals(arr, labels, axis0, axis1, new_axis):
    assert axis0 != axis1
    idx0,idx1 = labels.index(axis0), labels.index(axis1)

    ret = swapaxes(swapaxes(arr, idx0, 0), idx1, 1)[::-1,...]
    ret = np.array([my_trace(ret,offset=k) 
                    for k in range(-ret.shape[0]+1,ret.shape[1])])    

    # swap labels
    labels = list(labels)
    for i,idx in list(enumerate((idx0,idx1))):
        labels[i],labels[idx] = labels[idx],labels[i]
    labels = [new_axis] + labels[2:]
   
    return ret,labels


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
    
    ret = my_einsum(my_fftn(in1, fshape, fft_idx[1]), labels1,
                    my_fftn(in2, fshape, fft_idx[2]), labels2,
                    out_labels)
    return np.real(ifftn(ret, axes=fft_idx[0])[list(out_slice)])
                    
@primitive
def my_fftn(x, s, axes):
    '''
    autograd fftn currently broken for arguments s,axes
    '''
    return fftn(x,s,axes)
def fftngrad(ans,x,s,axes):
    gslice = tuple(slice(0,int(sz)) for sz in x.shape)
    g_s = tuple(np.array(map(max, zip(x.shape, ans.shape)))[axes])
    return lambda g: my_fftn(g,g_s,axes)[gslice]
my_fftn.defgrad(fftngrad)


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
