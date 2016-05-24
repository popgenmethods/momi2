
from future.utils import raise_with_traceback
import functools
import autograd.numpy as np
from functools import partial, update_wrapper, wraps
from autograd.core import primitive
from autograd import hessian, grad, hessian_vector_product, jacobian
import autograd
import itertools
from collections import Counter
import scipy, scipy.optimize
import sys, warnings, collections


# class mypartial(functools.partial):
#     def __init__(self, *args, **kwargs):
#         super().__init__(self, *args, **kwargs)
#         functools.update_wrapper(self, self.func)


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
              output_progress = False, f_name="objective", f_validation=None, callback=None):
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

        if output_progress and hist.itr % int(output_progress) == 0:
            print(("iter %d: %s(%s) == %f" % (hist.itr, f_name, str(x), fx)))
        
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
        ret = scipy.optimize.minimize(f, start_params, jac=jac, method=method, bounds=bounds, tol=tol, options=options, callback=callback)
        #assert ret.nfev == f.hist.nfev-1 or not jac
    except:
        if hist.result:
            ret = scipy.optimize.OptimizeResult(hist.result)
        else:
            raise
    return ret

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

# class InterruptOptimization(Exception):
#     def __init__(self, **kwargs):
#         self.result = dict(kwargs)

# def _verbosify(func, before = None, after = None, print_freq = 1):
#     i = [0] # can't do i=0 because of unboundlocalerror
#     def new_func(*args, **kwargs):
#         ii = i[0]
#         if ii % print_freq == 0 and before is not None:
#             print(before(ii,*args, **kwargs))
#         ret = func(*args, **kwargs)
#         if ii % print_freq == 0 and after is not None:
#             print(after(ii,ret, *args, **kwargs))
#         i[0] += 1
#         return ret
#     new_func.__name__ = "_verbose" + func.__name__
#     return new_func

def _npstr(x):
    return np.array_str(x, max_line_width=sys.maxsize)

def _get_stochastic_optimizer(method):
    if method == "adam":
        return adam
    if method == "adadelta":
        return adadelta
    raise ValueError("Unrecognized method")

def wrap_sgd(optimizer):
    def wrapped_optimizer(meta_fun, start_params, maxiter, bounds, output_progress, *args, **kwargs):
        if bounds is None:
            bounds = [None] * len(start_params)
        bounds = [b if b is not None else (None,None)
                  for b in bounds]
        upper_bounds = [b if b is not None else float('inf')
                        for _,b in bounds]
        lower_bounds = [b if b is not None else -float('inf')
                        for b,_ in bounds]
        bounds = list(zip(lower_bounds, upper_bounds))

        #meta_grad = wrap_objective(grad(meta_fun), 'jac')        
        #meta_fun = wrap_objective(meta_fun, 'objective', check_inf=False, output_progress=output_progress)

        n_minibatches = meta_fun.n_minibatches
        #fun_list = [mypartial(meta_fun, minibatch=i) for i in range(n_minibatches)]
        #grad_list = [mypartial(meta_grad, minibatch=i) for i in range(n_minibatches)]

        meta_grad = grad(meta_fun)
        meta_fun_and_grad = lambda x,minibatch: (meta_fun(x,minibatch=minibatch), meta_grad(x,minibatch=minibatch))

        fun_and_grad_list = [mypartial(meta_fun_and_grad, minibatch=i) for i in range(n_minibatches)]
        fun_and_grad_list = [wrap_objective(fg, 'objective',jac=True) for fg in fun_and_grad_list]
        
        #return optimizer(zip(fun_list, grad_list), start_params, maxiter, bounds, *args, **kwargs)
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



## TODO: uncomment and rewrite simulate_inference
# def simulate_inference(ms_path, num_loci, mu, additional_ms_params, true_ms_params, init_opt_params, demo_factory, n_iter=10, transform_params=lambda x:x, verbosity=0, method='trust-ncg', surface_type='kl', n_sfs_dirs=0, tensor_method='greedy-hosvd', conf_intervals=False):
#     '''
#     Simulate a SFS, then estimate the demography via maximum composite
#     likelihood, using first and second-order derivatives to search 
#     over log-likelihood surface.

#     num_loci: number of unlinked loci to simulate
#     true_ms_params: dictionary of true parameters, in ms parameter space
#     init_opt_params: array of initial parameters, in optimization parameter space
#     mu: mutation rate per locus.
#     demo_str: a string given the demography in ms-format
#     n_iter: number of iterations to use in basinhopping
#     transform_params: a function transforming the parameters in optimization space,
#                       to the values expected by make_demography
#     verbosity: 0=no output, 1=medium output, 2=high output
#     '''
#     start = time.clock()
    
#     def myprint(*args,**kwargs):
#         level = kwargs.get('level',1)
#         if level <= verbosity:
#             print(*args)
           
#     true_ms_params = pd.Series(true_ms_params)
#     old_transform_params = transform_params
#     transform_params = lambda x: pd.Series(old_transform_params(x))

#     def demo_func_ms(**params):
#         if callable(demo_factory):
#             return demo_factory(**params)
#         else:
#             return Demography.from_ms(1.0,demo_factory, **params)
    
#     true_demo = demo_func_ms(**true_ms_params)
#     myprint("# True demography:")
#     myprint(true_demo)
    
#     myprint("# Simulating %d unlinked loci" % num_loci)
#     ## ms_output = file object containing the ms output
#     ms_output = simulate_ms(ms_path, true_demo, num_loci=num_loci, mu_per_locus=mu, additional_ms_params = additional_ms_params)

#     ## sfs_list = list of dictionaries
#     ## sfs_list[i][config] = count of config at simulated locus i
#     sfs_list = sfs_list_from_ms(ms_output)
#     ms_output.close()

#     total_snps = sum([x for sfs in sfs_list for _,x in sfs.iteritems()])
#     myprint("# Total %d SNPs observed" % total_snps)

#     uniq_snps = len({x for sfs in sfs_list for x in sfs.keys()})
#     myprint("# %d unique SNPs observed" % uniq_snps)
    
#     # m estimator surface
#     idx = true_ms_params.index
#     ## TODO: make calling demo_func less convoluted
#     f_surface, f_cov = get_likelihood_surface(true_demo, sfs_list, mu,
#                                       lambda x: demo_func_ms(**pd.Series(x,index=idx)),
#                                       surface_type,
#                                       tensor_method, n_sfs_dirs)

#     # construct the function to minimize, and its derivatives
#     def f(params):
#         return f_surface(pd.Series(transform_params(params), index=idx).values)        
#         # try:
#         #     return surface.evaluate(pd.Series(transform_params(params), index=idx).values)
#         # except MemoryError:
#         #     raise
#         # ## TODO: define a specific exception type for out-of-bounds or overflow errors
#         # except Exception:
#         #    # in case parameters are out-of-bounds or so extreme they cause overflow/stability issues. just return a very large number. note the gradient will be 0 in this case and the gradient descent may stop.            
#         #     return 1e100

#     def results_df(est_params, opt_space=True):
#         if opt_space:
#             est_params = transform_params(est_params)
#         return pd.DataFrame({'True': true_ms_params,
#                              'Est' : est_params,
#                              'Est/True': est_params / true_ms_params},
#                             columns=['True','Est','Est/True'])
    
#     g, hp = grad(f), hessian_vector_product(f)
#     def f_verbose(params):
#         # for verbose output during the gradient descent
#         myprint("Evaluating objective. Current position:",level=2)
#         myprint(results_df(params),level=2)
#         return f(params)
#     def g_verbose(params):
#         myprint("Evaluating gradient",level=2)
#         return g(params)
#     def hp_verbose(params, v):
#         myprint("Evaluating hessian-vector product",level=2)
#         return hp(params, v)

#     myprint("# Start demography:")
#     myprint(demo_func_ms(**transform_params(init_opt_params)))
#     myprint("# Performing optimization.")

#     def print_basinhopping(x,f,accepted):
#         myprint("\n***BASINHOPPING***")
#         myprint("at local minima %f" % f)
#         myprint(results_df(x))
#         if accepted:
#             myprint("Accepted")
#         else:
#             myprint("Rejected")
    
#     #optimize_res = scipy.optimize.minimize(f_verbose, init_opt_params, jac=g_verbose, hessp=hp_verbose, method='newton-cg')
#     optimize_res = scipy.optimize.basinhopping(f_verbose, init_opt_params,
#                                                niter=n_iter, interval=1,
#                                                T=float(total_snps),
#                                                minimizer_kwargs={'method':method,
#                                                                  'jac':g_verbose,
#                                                                  'hessp':hp_verbose},
#                                                callback=print_basinhopping)

#     opt_end = time.clock()
    
#     myprint("\n\n# Global minimum: %f" % optimize_res.fun, level=0)
#     myprint(results_df(optimize_res.x),level=0)
    
#     inferred_ms_params = transform_params(optimize_res.x)

#     ret = {'truth': true_ms_params,
#            'est': inferred_ms_params,
#            'init': transform_params(init_opt_params),
#            'opt_res': optimize_res,
#            'time': {'opt': opt_end - start},
#            'num_snps': {'total' : total_snps, 'unique': uniq_snps},
#            }

#     if conf_intervals:
#         ## estimate sigma hat at plugin
#         #sigma = surface.max_covariance(inferred_ms_params.values)
#         sigma = f_cov(inferred_ms_params.values)

#         # recommend to call check_symmetric on matrix inverse,
#         # as linear algebra routines may not perfectly preserve symmetry due to numerical errors
#         ## TODO: what if sigma not full rank?
#         ## TODO: use eigh to compute inverse? (also in likelihood)
#         sigma_inv = check_symmetric(np.linalg.inv(sigma))

#         ## marginal p values
#         sd = np.sqrt(np.diag(sigma))
#         z = (inferred_ms_params - true_ms_params) / sd
#         z_p = pd.Series((1.0 - scipy.stats.norm.cdf(np.abs(z))) * 2.0 , index=idx)

#         coord_results = results_df(inferred_ms_params, opt_space=False)
#         coord_results['p value'] = z_p
#         myprint(coord_results)

#         ## global p value
#         resids = inferred_ms_params - true_ms_params
#         eps_norm = np.dot(resids, np.dot(sigma_inv, resids))
#         wald_p = 1.0 - scipy.stats.chi2.cdf(eps_norm, df=len(resids))

#         myprint("# Chi2 test for params=true_params")
#         myprint("# X, 1-Chi2_cdf(X,df=%d)" % len(resids))    
#         myprint(eps_norm, wald_p)

#         conf_end = time.clock()

#         ret.update({'sigma': sigma,
#                     'sigma_inv': sigma_inv,
#                     'p_vals': {'z': z_p, 'wald': wald_p},
#                     })
#         ret['time']['conf'] = conf_end - opt_end

#     return ret

# def get_likelihood_surface(true_demo, sfs_list, mu, demo_func, surface_type, tensor_method, n_sfs_dirs):
#     if surface_type == 'kl' and n_sfs_dirs <= 0:
#         sfs = sum_sfs_list(sfs_list)
#         mu = make_function(mu)
#         f = lambda params: -unlinked_log_likelihood(sfs, demo_func(params), mu(params) * len(sfs_list), adjust_probs = 1e-80)
#         f_cov = lambda params: unlinked_mle_approx_cov(params, sfs_list, demo_func, mu)
#         return f, f_cov       

#     if surface_type == 'kl' or n_sfs_dirs <= 0:
#         raise Exception("Either must use KL divergence, or must specify number of directions")
    
#     leaves = sorted(true_demo.leaves)
#     if tensor_method=='random':
#         sfs_dirs = []
#         for leaf in leaves:
#             sfs_dirs += [np.random.normal(size=(n_sfs_dirs, true_demo.n_lineages(leaf)+1))]
#     elif tensor_method=='greedy-hosvd':
#         sfs_dirs = zip(*greedy_hosvd(get_sfs_tensor(sum_sfs_list(sfs_list),
#                                                     [true_demo.n_lineages(l) for l in leaves]),
#                                      n_sfs_dirs, verbose=True))
#         sfs_dirs = [np.array(x) for x in sfs_dirs]
#         #sfs_dirs = dict(zip(leaves, sfs_dirs))
#     else:
#         raise Exception("Unrecognized tensor_method")

#     if surface_type == 'pgs-diag':
#         ret = PGSurface_Diag(sfs_list, sfs_dirs, mu, demo_func).evaluate
#     elif surface_type == 'pgs-exact':
#         ret = PGSurface_Exact(sfs_list, sfs_dirs, mu, demo_func).evaluate
#     elif surface_type == 'pgs-emp':
#         ret = PGSurface_Empirical(sfs_list, sfs_dirs, mu, demo_func).evaluate
#     elif surface_type == 'pws':
#         ret = PoissonWishartSurface(sfs_list, sfs_dirs, mu, demo_func).evaluate
#     else:
#         Exception("Unrecognized surface type")
#     return ret,None
