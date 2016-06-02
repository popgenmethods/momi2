
from future.utils import raise_with_traceback
import functools
import autograd.numpy as np
import numdifftools
from functools import partial, update_wrapper, wraps
from autograd.core import primitive
from autograd import hessian, grad, hessian_vector_product, jacobian
import autograd
import itertools
from collections import Counter
import scipy, scipy.optimize
import sys, warnings, collections, logging

logger = logging.getLogger(".".join(__name__.split(".")[:-1]))

def mypartial(fun, *args, **kwargs):
    ret = functools.partial(fun, *args, **kwargs)
    functools.update_wrapper(ret, fun)
    return ret

def count_calls(fun):
    call_counter=[0]
    def new_fun(*args, **kwargs):
        call_counter[0] += 1
        return fun(*args, **kwargs)
    update_wrapper(new_fun, fun)

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
        key = (self.func, args[1:], frozenset(list(kw.items())))
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

# def make_function(f):
#     def func(*args, **kwargs):
#         try:
#             return f(*args, **kwargs)
#         except TypeError:
#             return f
#     return func

def wrap_minimizer(minimizer):
    def wrapped(f, start_params, maxiter, bounds = None, **kwargs):
        fixed_params = []    
        if bounds is not None:
            for i,b in enumerate(bounds):
                if b is not None:
                    try:
                        if b[0] == b[1]:
                            fixed_params += [(i,b[0])]
                    except (TypeError,IndexError) as e:
                        fixed_params += [(i,b)]
            if any(start_params[i] != b for i,b in fixed_params):
                raise ValueError("start_params does not agree with fixed parameters in bounds")

        if fixed_params:
            fixed_idxs, fixed_offset = list(map(np.array, list(zip(*fixed_params))))

            fixed_idxs = np.array([(i in fixed_idxs) for i in range(len(start_params))])
            proj0 = np.eye(len(fixed_idxs))[:,fixed_idxs]
            proj1 = np.eye(len(fixed_idxs))[:,~fixed_idxs]

            fixed_offset = np.dot(proj0, fixed_offset)

            get_x = lambda x0: np.dot(proj1, x0) + fixed_offset

            def subfun(fun):
                if fun is None: return None
                new_fun = lambda x0, *fargs, **fkwargs: fun(get_x(x0), *fargs, **fkwargs)
                functools.update_wrapper(new_fun, fun)
                return new_fun
            f0 = subfun(f)
            if "f_validation" in kwargs:
                kwargs["f_validation"] = subfun(kwargs["f_validation"])
            
            start0 = np.array([s for (fxd,s) in zip(fixed_idxs,start_params) if not fxd])
            bounds0 = [b for (fxd,b) in zip(fixed_idxs, bounds) if not fxd]
            ret = minimizer(f0, start0, maxiter, bounds0, **kwargs)
            ret.x = np.dot(proj1,ret.x) + fixed_offset
        else:
            ret = minimizer(f, start_params, maxiter, bounds, **kwargs)
        return ret
    functools.update_wrapper(wrapped, minimizer)
    return wrapped

@wrap_minimizer
def _minimize(f, start_params, maxiter, bounds,
              jac = True, method = 'tnc', tol = None, options = {},
              f_name="objective", f_validation=None, callback=None):
    options = dict(options)
    if maxiter is not None:
        options['maxiter'] = maxiter
   
    if jac:
        f = autograd.value_and_grad(f)
    
    f = wrap_objective(f, f_name, jac)

    hist = lambda : None
    hist.itr = 0
    hist.f_vals = []
    hist.validations = []
    hist.result = None

    user_callback = callback   
    def callback(x):
        for y,fy,gy in reversed(f.hist.recent):
            if np.allclose(y,x):
                fx,gx = fy,gy
                break
        assert np.allclose(y,x)

        if user_callback is not None:
            user_callback(x,fx)
        
        hist.f_vals += [fx]
        
        while f.hist.recent:
            f.hist.recent.pop()

        logger.info("iter = {i} ; x = {x} ; {f} =  {fx}".format(i=hist.itr, x=list(x), f=f_name, fx=fx))
        
        hist.itr += 1
        if f_validation is not None:
            hist.validations += [f_validation(x)]
            assert len(hist.validations) == len(hist.f_vals)

            if len(hist.f_vals) >= 2 and hist.f_vals[-1] < hist.f_vals[-2] and hist.validations[-1] > hist.validations[-2]:
                # validation function has failed to improve
                hist.result = {'status':1, 'success':True, 'fun':hist.f_vals[-1], 'x':x, 'message':"Validation function stopped improving", 'nit':hist.itr}
                if jac:
                    hist.result["jac"] = gx
                    hist.result["nfev"] = f.hist.nfev #nfev is not correct if using finite difference approximation to gradient
                raise Exception()

    try:
        if method=='gradient_descent':
            #assert jac is True
            ret = gradient_descent(start_params, f, bounds=bounds, callback=callback, **options)
        else:
            ret = scipy.optimize.minimize(f, start_params, jac=jac, method=method, bounds=bounds, tol=tol, options=options, callback=callback)
        #assert ret.nfev == f.hist.nfev-1 or not jac
    except:
        if hist.result:
            ret = scipy.optimize.OptimizeResult(hist.result)
        else:
            raise
    return ret

def gradient_descent(x0, f_and_g, bounds=None, maxiter=1000, eps=None, callback=None):
    if eps is not None:
        fun = f_and_g
        gr = numdifftools.Gradient(fun, step=eps)
        f_and_g = lambda x: (fun(x),gr(x))

    if bounds is None:
        bounds = [(None,None)]*len(x0)
    old_bounds, bounds = bounds, []
    for lower, upper in old_bounds:
        if lower is None: lower = -float('inf')
        if upper is None: upper = float('inf')
        bounds.append((lower,upper))
    lower, upper = list(map(np.array, zip(*bounds)))
    def truncate(x): return np.maximum(np.minimum(x, upper), lower)

    old_f_and_g = f_and_g
    @wraps(old_f_and_g)
    def f_and_g(x):
        f_and_g.nfev+=1
        return old_f_and_g(x)
    f_and_g.nfev=0

    def backtrack_linesearch(y,stepsize,reverse=False):
        ## if reverse==True, do a "forward" line search for maximum stepsize satisfying condition
        fy,gy = f_and_g(y)
        while True:
            x = truncate(y - gy*stepsize)
            fx,gx = f_and_g(x)
            condition = (fx <= fy + 0.5 * np.dot(gy, x-y))
            if reverse: condition = not condition
            if condition: break
            else:
                if reverse: stepsize = 2.0*stepsize
                else: stepsize = 0.5*stepsize
        return (x,fx,gx),(y,fy,gy),stepsize
    
    x=np.array(x0,dtype=float)
    y=np.array(x)

    ## get initial stepsize
    _,(_,fy,gy),stepsize = backtrack_linesearch(y,1.0,reverse=True)
    x,fx,gx = y,fy,gy
    # initial callback
    callback(x)
    
    nesterov_it=0
    for nit in range(1,maxiter+1):
        prev_x,prev_fx,prev_gx = x,fx,gx
        prev_y,prev_fy,prev_gy = y,fy,gy

        ## update x with backtracking linesearch
        (x,fx,gx),(y,fy,gy),stepsize = backtrack_linesearch(y, stepsize)
        assert np.all(y == prev_y) # y shouldn't change here
        callback(x)

        ## check convergence
        if np.allclose(y,x):
            success=True
            message="jac(x[k])=~0"
            break        
        if np.allclose(x,prev_x):
            success=True
            message="|x[k]-x[k-1]|=~0"
            break
        # if np.isclose(prev_fx,fx):
        #     success=True
        #     message="|f(x[k])-f(x[k-1])|=~0"
        #     break

        ## check if restart nesterov
        if np.dot(gy, x - prev_x) > 0:
            nesterov_it=0
            prev_x = x

        ## add nesterov momentum
        y = truncate(x + (nesterov_it)/float(nesterov_it+2)*(x-prev_x))
        nesterov_it+=1

        ## allow stepsize to grow
        stepsize = stepsize * 2.0
    try: success,message
    except NameError:
        success=False
        message="Maximum number of iterations reached"
    return scipy.optimize.OptimizeResult({'success':success, 'message':message,
                                          'nfev':f_and_g.nfev, 'nit':nit,
                                          'x':x, 'fun':fx, 'jac':gy})

def wrap_objective(fun,name,jac):
    hist = lambda : None
    
    hist.recent = []
    hist.nfev = 0
    
    @wraps(fun)
    def new_fun(x):
        hist.nfev += 1
        
        try:
            ret = fun(x)
            
            if jac: fx, gx = ret
            else: fx,gx = ret,0

            for (y,fun_name) in ((fx,name), (gx,"jac")):
                if np.any(np.isnan(y)) or not np.all(np.isfinite(y)):
                    raise OptimizationError("%s ( %s ) == %s. Try setting stricter bounds (e.g. lower bound of 1e-100 instead of 0) or higher truncate_probs" % (fun_name,str(x),str(y)))
                
            hist.recent.append((x,fx,gx))
            
            return ret
        except Exception as e:
            raise_with_traceback(OptimizationError("at %s( %s ):\n%s: %s" % (name, str(x), type(e).__name__, str(e))))

    new_fun.hist = hist
            
    return new_fun

class OptimizationError(Exception):
    pass

def _npstr(x):
    return np.array_str(x, max_line_width=sys.maxsize)

def _get_stochastic_optimizer(method):
    if method == "adam":
        return adam
    if method == "adadelta":
        return adadelta
    raise ValueError("Unrecognized method")

def wrap_sgd(optimizer):
    def wrapped_optimizer(meta_fun, start_params, maxiter, bounds, *args, **kwargs):
        if bounds is None:
            bounds = [None] * len(start_params)
        bounds = [b if b is not None else (None,None)
                  for b in bounds]
        upper_bounds = [b if b is not None else float('inf')
                        for _,b in bounds]
        lower_bounds = [b if b is not None else -float('inf')
                        for b,_ in bounds]
        bounds = list(zip(lower_bounds, upper_bounds))

        n_minibatches = meta_fun.n_minibatches

        meta_grad = grad(meta_fun)
        meta_fun_and_grad = lambda x,minibatch: (meta_fun(x,minibatch=minibatch), meta_grad(x,minibatch=minibatch))

        fun_and_grad_list = [mypartial(meta_fun_and_grad, minibatch=i) for i in range(n_minibatches)]
        fun_and_grad_list = [wrap_objective(fg, 'objective',jac=True) for fg in fun_and_grad_list]
        
        return optimizer(fun_and_grad_list, start_params, maxiter, bounds, *args, **kwargs)
    functools.update_wrapper(wrapped_optimizer, optimizer)
    return wrapped_optimizer
        
## based on code from autograd/examples
@wrap_minimizer
@wrap_sgd
def adam(fun_and_jac_list, start_params, maxiter, bounds,
         tol=None,
         random_generator=np.random, step_size=1., b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    lower_bounds, upper_bounds = list(zip(*bounds))
    if tol is not None:
        raise NotImplementedError("tol not yet implemented")
    
    x = start_params    
    m = np.zeros(len(x))
    v = np.zeros(len(x))

    step_size = step_size / float(len(fun_and_jac_list))
    
    history = OptimizeHistory()
    for curr_pass in range(maxiter):
        history.new_batch()
        for i in random_generator.permutation(len(fun_and_jac_list)):
            fun_and_jac = fun_and_jac_list[i]
            
            f,g = fun_and_jac(x)
            history.update(x,f,g)

            m = (1 - b1) * g      + b1 * m  # First  moment estimate.
            v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
            mhat = m / float(1 - b1**(i + 1))    # Bias correction.
            vhat = v / float(1 - b2**(i + 1))
            x = x-step_size*mhat/(np.sqrt(vhat) + eps)
            x = np.maximum(np.minimum(x, upper_bounds), lower_bounds)
    return scipy.optimize.OptimizeResult({'x':x, 'fun':f, 'jac':g, 'history':history})

@wrap_minimizer
@wrap_sgd
def adadelta(fun_and_jac_list, start_params, maxiter, bounds,
             tol=None,
             random_generator=np.random, rho=.95, eps=1e-6):
    lower_bounds, upper_bounds = list(zip(*bounds))
    if tol is not None:
        raise NotImplementedError("tol not yet implemented")
    
    x = start_params    
    EG2 = 0.
    EDelX2 = 0.

    history = OptimizeHistory()
    for curr_pass in range(maxiter):
        history.new_batch()
        for i in random_generator.permutation(len(fun_and_jac_list)):
            fun,jac = fun_and_jac_list[i]
            
            f = fun(x)
            g = jac(x)
            history.update(x,f,g)

            EG2 = rho * EG2 + (1.-rho) * (g**2)
            stepsize = - np.sqrt(EDelX2 + eps) / np.sqrt(EG2 + eps) * g

            EDelX2 = rho * EDelX2 + (1.-rho) * (stepsize**2)
            
            x = x-stepsize
            x = np.maximum(np.minimum(x, upper_bounds), lower_bounds)
    return scipy.optimize.OptimizeResult({'x':x, 'fun':f, 'jac':g, 'history':history})

class OptimizeHistory(object):
    def __init__(self):
        self.x = []
        self.f = []
        self.g = []

    def new_batch(self):
        self.x += [[]]
        self.f += [[]]
        self.g += [[]]

    def update(self, x,f,g):
        self.x[-1] += [x]
        self.f[-1] += [f]
        self.g[-1] += [g]
