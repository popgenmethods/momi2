
from future.utils import raise_with_traceback
import autograd.numpy as np
import numdifftools
from functools import partial, wraps
from autograd.core import primitive, Node
from autograd import hessian, grad, hessian_vector_product, jacobian, value_and_grad, vector_jacobian_product
import autograd
import itertools
from collections import Counter
import sys, warnings, collections, logging, gc
import multiprocessing as mp

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


class rearrange_gradients(object):
    """
    rearrange_gradients()(fun) returns a function that uses a
    wrapped version of fun, except it is primitive, so that we
    don't store its gradient computations in memory.

    fun is assumed to return a scalar, and the signature of fun
    is assumed to be
           fun(x, *args, **kwargs)
    where x is a numpy.array, and derivative is ONLY taken with
    respect to x (not wrt any other args).

    to avoid doing the forward-pass twice, we precompute
    gradient and cache it on the forward-pass, thus saving
    us from having to do it again on the backward pass.

    second-order gradients remain well-defined, but use an
    extra forward-pass.
    """
    def __init__(self, get_value_and_grad=value_and_grad):
        self.get_value_and_grad = get_value_and_grad
    def __call__(self, fun):
        # Notation:
        # f(x) = fun(x, *args, **kwargs)
        # F is the mapping, F: f(x) -> final_result
        # G = scalar function of dF/dx
        # use y,z as dummy variables for x when needed

        get_dF_dx = count_calls(self.get_dF_dx)
        get_dG_dx = count_calls(self.get_dG_dx)

        @wraps(fun)
        def fun_wrapped(x, *args, **kwargs):
            for a in list(args) + list(kwargs.values()):
                if isinstance(a, Node):
                    raise NotImplementedError

            if isinstance(x, Node):
                # use y as a dummy variable for x
                f = lambda y: fun(y, *args, **kwargs)
                # a primitive version of f, to avoid storing its gradient in memory
                @primitive
                def f_primitive(y, df_dy_container):
                    # use value_and_grad to precompute df_dy
                    fy, df_dy = self.get_value_and_grad(fun)(y, *args, **kwargs)
                    # store df_dy for the backward pass
                    df_dy_container.append(df_dy)
                    return fy
                def make_dF_dy_chainrule(fy, y, df_dy_container):
                    df_dy, = df_dy_container
                    # needs to be primitive, so we can properly obtain the gradient of df_dy
                    @primitive
                    # use z as dummy variable for y
                    def dF_dz_primitive(z, dF_df):
                        return get_dF_dx(dF_df, df_dy)
                    make_dG_dz_chainrule = lambda dF_dz, z, dF_df: lambda dGdz_d2F: get_dG_dx(f, z, dF_df, dGdz_d2F)
                    dF_dz_primitive.defgrad(make_dG_dz_chainrule)

                    dF_dy_chainrule = lambda dF_df: dF_dz_primitive(y, dF_df)
                    return dF_dy_chainrule
                f_primitive.defgrad(make_dF_dy_chainrule)
                return f_primitive(x, [])
            else:
                return fun(x, *args, **kwargs)
        ## for unit tests associated with @count_calls
        def reset_grad_count():
            get_dF_dx.reset_count()
            get_dG_dx.reset_count()
        fun_wrapped.reset_grad_count = reset_grad_count
        fun_wrapped.num_grad_calls = get_dF_dx.num_calls
        fun_wrapped.num_hess_calls = get_dG_dx.num_calls
        return fun_wrapped

    def get_dF_dx(self, dF_df, df_dx):
        return dF_df * df_dx

    def get_dG_dx(self, f, x, dF_df, dGdx_d2F):
        dF_dx_fun = vec_jac_prod_fun(f, dF_df)
        return vec_jac_prod_fun(dF_dx_fun, dGdx_d2F)(x)

class rearrange_tuple_gradients(rearrange_gradients):
    """
    similar to rearrange_gradients, except fun is assumed to have
    signature
    fun(x_tuple, *args, **kwargs)
    where x_tuple is a tuple of numpy arrays.

    also, second-order gradients are disabled.
    """
    def get_dF_dx(self, dF_df, df_dx_tuple):
        return tuple(dF_df*df_dx for df_dx in df_dx_tuple)

    def get_dG_dx(self, *args, **kwargs):
        raise HessianDisabledError("Autograd hessians disabled to allow for computational improvements to memory usage. To disable these memory savings and allow Hessian usage, use SfsLikelihoodSurface(..., batch_size=-1).")

class HessianDisabledError(NotImplementedError):
    pass

## use parsum() to compute parallel sums of arbitrary order gradients

def _parsum_val_and_grad(x, autograd_process_list):
    for agproc in autograd_process_list:
        agproc.put(x, [value_and_grad])
    return tuple(map(sum, zip(*[agproc.get() for agproc in autograd_process_list])))

@rearrange_gradients(get_value_and_grad = lambda fun: _parsum_val_and_grad)
def parsum(x, autograd_process_list):
    return _parsum(x, autograd_process_list, [])

@primitive
def _parsum(x, autograd_process_list, g_list=[]):
    for agproc in autograd_process_list:
        agproc.put(x, g_list)
    return sum([agproc.get() for agproc in autograd_process_list])

_parsum.defgrad(lambda ans, x, queue_list, g_list: lambda g: _parsum(x, queue_list, [g] + list(g_list)))

class AutogradProcess(object):
    def __init__(self, funmaker, *args, **kwargs):
        self.inqueue = mp.SimpleQueue()
        self.outqueue = mp.SimpleQueue()
        self.proc = mp.Process(target=vec_jac_prod_worker, args=tuple([self.inqueue, self.outqueue, funmaker] + list(args)), kwargs=kwargs)
        self.proc.start()

    def put(self, x, g_list):
        self.inqueue.put([x] + list(g_list))

    def get(self):
        return self.outqueue.get()

    def join(self):
        self.inqueue.put(None)
        self.proc.join()

    def __del__(self):
        self.join()

def vec_jac_prod_worker(in_queue, out_queue, basefun_maker, *args, **kwargs):
    basefun = basefun_maker(*args, **kwargs)
    while True:
        nxt_item = in_queue.get()
        if nxt_item is None:
            break
        x, g_list = nxt_item[0], nxt_item[1:]
        if list(g_list) == [value_and_grad]:
            out_queue.put(value_and_grad(basefun)(x))
        else:
            fun = basefun
            for g in g_list[::-1]:
                fun = vec_jac_prod_fun(fun, g)
            out_queue.put(fun(x))

def vec_jac_prod_fun(fun, vec):
    return lambda x: vector_jacobian_product(fun)(x,vec)
