
from future.utils import raise_with_traceback
import autograd.numpy as np
import numdifftools
from functools import partial, wraps
from autograd.core import primitive
from autograd import hessian, grad, hessian_vector_product, jacobian
import autograd
import itertools
from collections import Counter
import sys, warnings, collections, logging, gc

logger = logging.getLogger(".".join(__name__.split(".")[:-1]))

def count_calls(fun):
    call_counter=[0]
    @wraps(fun)
    def new_fun(*args, **kwargs):
        call_counter[0] += 1
        return fun(*args, **kwargs)

    new_fun.num_calls = lambda : call_counter[0]
    def reset_count(): call_counter[0] = 0
    new_fun.reset_count = reset_count

    return new_fun
        
def check_symmetric(X):
    Xt = np.transpose(X)
    assert np.allclose(X, Xt)
    return 0.5 * (X + Xt)

def check_psd(X):
    X = check_symmetric(X)
    d,U = np.linalg.eigh(X)
    d = truncate0(d)
    ret = np.dot(U, np.dot(np.diag(d), np.transpose(U)))
    assert np.allclose(ret, X)
    return np.array(ret, ndmin=2)

def truncate0(x, axis=None, strict=False, tol=1e-13):
    '''make sure everything in x is non-negative'''
    # the maximum along axis
    maxes = np.maximum(np.amax(x, axis=axis), 1e-300)
    # the negative part of minimum along axis
    mins = np.maximum(-np.amin(x,axis=axis), 0.0)

    # assert the negative numbers are small (relative to maxes)
    assert np.all(mins <= tol * maxes)

    if axis is not None:
        idx = [slice(None)] * x.ndim
        idx[axis] = np.newaxis
        mins = mins[idx]
        maxes = maxes[idx]

    if strict:
        # set everything below the tolerance to 0
        return set0(x, x < tol * maxes)
    else:
        # set everything of same magnitude as most negative number, to 0
        return set0(x, x < 2*mins)

def check_probs_matrix(x):
    x = truncate0(x)
    rowsums = np.sum(x, axis=1)
    assert np.allclose(rowsums,1.0)
    return np.einsum('ij,i->ij',x,1.0/rowsums)

@primitive
def set0(x, indices):
    y = np.array(x)
    y[indices] = 0
    return y
set0.defgrad(lambda ans,x,indices: lambda g: set0(g,indices))

def closeleq(x,y):
    return np.logical_or(np.isclose(x,y), x <= y)
def closegeq(x,y):
    return np.logical_or(np.isclose(x,y), x >= y)


@primitive
def make_constant(x):
    return x
make_constant.defgrad_is_zero()

def memoize(obj):
    cache = obj.cache = {}
    @wraps(obj)
    def memoizer(*args, **kwargs):
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
        key = (self.func, args[1:], frozenset(list(kw.items())))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res

def force_primitive(subroutine):
    """
    Wraps subroutine(x_tuple,...) to be autograd.primitive.
    This improves memory usage, when computing first order derivatives
    of functions that call subroutine().
    (Note, this does not improve memory usage of second order derivatives, and
    may actually increase it!)
    """
    ## TODO: make faster, by storing results of the forward-pass whenever wrapped_subroutine is called    
    @autograd.primitive
    @wraps(subroutine)    
    def wrapped_subroutine(x_tuple, *args, **kwargs):
        return subroutine(x_tuple, *args, **kwargs)
    
    @count_calls # decorator used by unit tests making sure the new gradient is being called
    def subroutine_grad(ans, x_tuple, *args, **kwargs):
        return lambda g: tuple(g*y for y in autograd.grad(subroutine)(x_tuple, *args, **kwargs))
    
    wrapped_subroutine.defgrad(subroutine_grad)

    ## for unit tests associated with @count_calls
    wrapped_subroutine.num_grad_calls = subroutine_grad.num_calls
    wrapped_subroutine.reset_grad_count = subroutine_grad.reset_count
   
    return wrapped_subroutine
