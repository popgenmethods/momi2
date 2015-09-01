from __future__ import division
import functools
import autograd.numpy as np
from functools import partial
from autograd.core import primitive
import os
import itertools
from collections import Counter

def default_ms_path():
    return os.environ["MS_PATH"]

def sum_sfs_list(sfs_list):
    """
    Combines a list of SFS's into a single SFS by summing their entries.

    Parameters
    ----------
    sfs_list : list of dict
         A list where each entry is a dict mapping configs (tuples) to
         frequencies (floats or ints).

    Returns
    -------
    combined_sfs : dict
         The combined SFS, represented as a dict mapping configs (tuples)
         to frequencies (floats or ints).
    """
    return dict(sum([Counter(sfs) for sfs in sfs_list], Counter()))

def polymorphic_configs(demo):
    n = sum([demo.n_lineages(l) for l in demo.leaves])
    ranges = [range(demo.n_lineages(l)) for l in sorted(demo.leaves)]

    config_list = []
    for config in itertools.product(*ranges):
        if sum(config) == n or sum(config) == 0:
            continue
        config_list.append(config)
    return config_list

def check_symmetric(X):
    Xt = np.transpose(X)
    assert np.allclose(X, Xt)
    return 0.5 * (X + Xt)

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


@primitive
def check_probs_matrix(x):
    x = truncate0(x)
    rowsums = np.einsum('ij->i',x)
    assert np.allclose(rowsums,1.0)
    return np.einsum('ij,i->ij',x,1.0/rowsums)
check_probs_matrix.defgrad(lambda ans,x: lambda g: g)

@primitive
def set0(x, indices):
    y = np.array(x)
    y[indices] = 0
    return y
set0.defgrad(lambda ans,x,indices: lambda g: set0(g,indices))
    

@primitive
def make_constant(x):
    return x
make_constant.defgrad_is_zero()

def memoize(obj):
    cache = obj.cache = {}
    @functools.wraps(obj)
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
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res

def smooth_pos_map(x):
    # a smooth (up to second derivative) map from R to R+
    f1 = np.exp(x)
    f2 = -(x**3) / 6. + (x**2) / 2. + x + 1.0
    f3 = 1.5 * x + 5. / 6.
    
    return (x < 0) * f1 + np.logical_and(x >= 0, x < 1) * f2 + (x >= 1) * f3

def make_function(f):
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except TypeError:
            return f
    return func
