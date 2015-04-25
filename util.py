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

def fftconvolve(in1, in2, axes=None):
    """
    Mostly copied from scipy.signal, but with axes argument added
    """
    assert len(in1.shape) == len(in2.shape)
    if axes is None:
        axes = np.arange(len(in1.shape))

    ## get output shape along axes to be convolved
    in_shapes = [np.array(arr.shape) for arr in in1,in2]
    shape = in_shapes[0][axes] + in_shapes[1][axes] -1
    for s in in_shapes:
        s[axes] = shape
    assert np.all(in_shapes[0] == in_shapes[1])
    
    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [_next_regular(int(d)) for d in shape]
    # slices for output array
    fslice = tuple([slice(0, int(sz)) for sz in in_shapes[0]])
    
    ret = ifftn(my_fftn(in1, fshape, axes) * my_fftn(in2, fshape, axes), axes=axes)[fslice]
    ret = np.real(ret)
    return ret

@primitive
def my_fftn(x, s, axes):
    '''
    autograd fftn currently broken for arguments s,axes
    '''
    return fftn(x,s,axes)
def fftngrad(ans,x,s,axes):
    gslice = tuple(slice(0,int(sz)) for sz in x.shape)
    g_s = tuple(map(max, zip(x.shape, ans.shape)))
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
