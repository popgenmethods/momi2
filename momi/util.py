
import functools
import autograd.numpy as np
from functools import partial
from autograd.core import primitive
from autograd import hessian, grad, hessian_vector_product, jacobian
import itertools
from collections import Counter
import scipy, scipy.optimize
import sys
import collections
import warnings

class mylist(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args)
        for k,v in kwargs.iteritems():
            setattr(self,k,v)

# def sum_sfs_list(sfs_list):
#     """
#     Combines a list of SFS's into a single SFS by summing their entries.

#     Parameters
#     ----------
#     sfs_list : list of dict
#          A list where each entry is a dict mapping configs (tuples) to
#          frequencies (floats or ints).

#     Returns
#     -------
#     combined_sfs : dict
#          The combined SFS, represented as a dict mapping configs (tuples)
#          to frequencies (floats or ints).
#     """
#     return dict(sum([Counter(sfs) for sfs in sfs_list], Counter()))

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

def optimize(f, start_params,
             jac = True, hess = False, hessp = False,
             method = 'tnc', maxiter = 100, bounds = None, tol = None, options = {},
             output_progress = False, **kwargs):
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
        fixed_idxs, fixed_offset = map(np.array, zip(*fixed_params))
        
        fixed_idxs = np.array([(i in fixed_idxs) for i in range(len(start_params))])
        proj0 = np.eye(len(fixed_idxs))[:,fixed_idxs]
        proj1 = np.eye(len(fixed_idxs))[:,~fixed_idxs]

        fixed_offset = np.dot(proj0, fixed_offset)

        f0 = lambda x: f(np.dot(proj1, x) + fixed_offset)
        start0 = np.array([s for (fxd,s) in zip(fixed_idxs,start_params) if not fxd])
        bounds0 = [b for (fxd,b) in zip(fixed_idxs, bounds) if not fxd]
        ret = optimize(f0, start0, jac=jac, hess=hess, hessp=hessp,
                       method=method, maxiter=maxiter, bounds=bounds0,
                       tol=tol, options=options, output_progress=output_progress,
                       **kwargs)
        ret.x = np.dot(proj1,ret.x) + fixed_offset
        return ret
        
    if (hess or hessp) and not isinstance(method, collections.Callable) and method.lower() not in ('newton-cg','trust-ncg','dogleg'):
        raise ValueError("Only methods newton-cg, trust-ncg, and dogleg use hessian")
    if bounds is not None and not isinstance(method, collections.Callable) and method.lower() not in ('l-bfgs-b', 'tnc', 'slsqp'):
        raise ValueError("Only methods l-bfgs-b, tnc, slsqp use bounds")

    if maxiter is None:
        raise ValueError("maxiter must be a finite positive integer")
    if 'maxiter' in options:
        raise ValueError("Please specify maxiter thru function argument 'maxiter', rather than 'options'")
    
    options = dict(options)
    options['maxiter'] = maxiter

    def safe_fun(fun,name, check_inf=True):
        def new_fun(*a):
            try:
                ret = fun(*a)
                if np.any(np.isnan(ret)) or (check_inf and not np.all(np.isfinite(ret))):
                    raise OptimizationError("%s ( %s ) == %s. Consider setting stricter bounds? (e.g. set a lower bound of 1e-100 instead of 0)" % (name,str(*a),str(ret)))
                return ret
            except Exception, e:
                raise OptimizationError("Exception at %s( %s )!\n\nCaused by:\n%s" % (name, str(*a), str(e)))
        return new_fun
    
    kwargs = dict(kwargs)
    kwargs.update({kw : safe_fun(d(f),kw)
                   for kw, b, d in [('jac', jac, grad), ('hessp', hessp, hessian_vector_product), ('hess', hess, hessian)]
                   if b})

    f = safe_fun(f, "objective", check_inf=False)
    
    if output_progress:
        f = _verbosify(f,
                       before = lambda i,x: "evaluation %d" % i,
                       after = lambda i,ret,x: "objective ( %s ) == %g" % (_npstr(x), ret),
                       print_freq = output_progress)

    return scipy.optimize.minimize(f, start_params, method=method, bounds=bounds, tol=tol, options=options, **kwargs)

class OptimizationError(Exception):
    pass

def _verbosify(func, before = None, after = None, print_freq = 1):
    i = [0] # can't do i=0 because of unboundlocalerror
    def new_func(*args, **kwargs):
        ii = i[0]
        if ii % print_freq == 0 and before is not None:
            print(before(ii,*args, **kwargs))
        ret = func(*args, **kwargs)
        if ii % print_freq == 0 and after is not None:
            print(after(ii,ret, *args, **kwargs))
        i[0] += 1
        return ret
    new_func.__name__ = "_verbose" + func.__name__
    return new_func

def _npstr(x):
    return np.array_str(x, max_line_width=sys.maxsize)

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
