from __future__ import division, print_function

from likelihood_surface import NegativeLogLikelihood, L2ErrorSurface
from parse_ms import make_demography, simulate_ms, sfs_list_from_ms
from util import check_symmetric

import scipy
import scipy.stats

import pandas as pd

## Functions for computing derivatives
import autograd.numpy as np
from autograd import grad, hessian_vector_product

import time

def simulate_inference(ms_path, num_loci, theta, additional_ms_params, true_ms_params, init_opt_params, demo_factory, n_iter=10, transform_params=lambda x:x, verbosity=0, method='trust-ncg', n_sfs_dirs=0):
    '''
    Simulate a SFS, then estimate the demography via maximum composite
    likelihood, using first and second-order derivatives to search 
    over log-likelihood surface.

    num_loci: number of unlinked loci to simulate
    true_ms_params: dictionary of true parameters, in ms parameter space
    init_opt_params: array of initial parameters, in optimization parameter space
    theta: mutation rate per locus.
    demo_str: a string given the demography in ms-format
    n_iter: number of iterations to use in basinhopping
    transform_params: a function transforming the parameters in optimization space,
                      to the values expected by make_demography
    verbosity: 0=no output, 1=medium output, 2=high output
    '''
    start = time.clock()
    
    def myprint(*args,**kwargs):
        level = kwargs.get('level',1)
        if level <= verbosity:
            print(*args)
           
    true_ms_params = pd.Series(true_ms_params)
    old_transform_params = transform_params
    transform_params = lambda x: pd.Series(old_transform_params(x))

    def demo_func_ms(**params):
        if callable(demo_factory):
            return demo_factory(**params)
        else:
            return make_demography(demo_factory, **params)
    
    def demo_func_opt(params):
        return demo_func_ms(**transform_params(params))
    
    true_demo = demo_func_ms(**true_ms_params)
    myprint("# True demography:")
    myprint(true_demo.ms_cmd)
    
    myprint("# Simulating %d unlinked loci" % num_loci)
    ## ms_output = file object containing the ms output
    ms_output = simulate_ms(true_demo, num_sims=num_loci, theta=theta, ms_path=ms_path, additional_ms_params = additional_ms_params)

    ## sfs_list = list of dictionaries
    ## sfs_list[i][config] = count of config at simulated locus i
    sfs_list = sfs_list_from_ms(ms_output,
                                true_demo.n_at_leaves # tuple with n at each leaf deme
                                )
    ms_output.close()

    total_snps = sum([x for sfs in sfs_list for _,x in sfs.iteritems()])
    myprint("# Total %d SNPs observed" % total_snps)

    uniq_snps = len({x for sfs in sfs_list for x in sfs.keys()})
    myprint("# %d unique SNPs observed" % uniq_snps)
    
    # log-likelihood surface
    surface = get_likelihood_surface(true_demo, sfs_list, theta, demo_func_opt, n_sfs_dirs)

    # construct the function to minimize, and its derivatives
    def f(params):
        try:
            return surface.evaluate(params)
        except MemoryError:
            raise
        ## TODO: define a specific exception type for out-of-bounds or overflow errors
        except Exception:
           # in case parameters are out-of-bounds or so extreme they cause overflow/stability issues. just return a very large number. note the gradient will be 0 in this case and the gradient descent may stop.            
            return 1e100

    def results_df(est_params, opt_space=True):
        if opt_space:
            est_params = transform_params(est_params)
        return pd.DataFrame({'True': true_ms_params,
                             'Est' : est_params,
                             'Est/True': est_params / true_ms_params},
                            columns=['True','Est','Est/True'])
    
    g, hp = grad(f), hessian_vector_product(f)
    def f_verbose(params):
        # for verbose output during the gradient descent
        myprint("Evaluating objective. Current position:",level=2)
        myprint(results_df(params),level=2)
        return f(params)
    def g_verbose(params):
        myprint("Evaluating gradient",level=2)
        return g(params)
    def hp_verbose(params, v):
        myprint("Evaluating hessian-vector product",level=2)
        return hp(params, v)

    myprint("# Start demography:")
    myprint(demo_func_opt(init_opt_params).ms_cmd)
    myprint("# Performing optimization.")

    def print_basinhopping(x,f,accepted):
        myprint("\n***BASINHOPPING***")
        myprint("at local minima %f" % f)
        myprint(results_df(x))
        if accepted:
            myprint("Accepted")
        else:
            myprint("Rejected")
    
    #optimize_res = scipy.optimize.minimize(f_verbose, init_opt_params, jac=g_verbose, hessp=hp_verbose, method='newton-cg')
    optimize_res = scipy.optimize.basinhopping(f_verbose, init_opt_params,
                                               niter=n_iter, interval=1,
                                               T=float(total_snps),
                                               minimizer_kwargs={'method':method,
                                                                 'jac':g_verbose,
                                                                 'hessp':hp_verbose},
                                               callback=print_basinhopping)

    opt_end = time.clock()
    
    myprint("\n\n# Global minimum: %f" % optimize_res.fun)
    
    inferred_ms_params = transform_params(optimize_res.x)

    ## reparametrize surface by ms params
    idx = true_ms_params.index
    surface = get_likelihood_surface(true_demo, sfs_list, theta,
                                     lambda x: demo_func_ms(**pd.Series(x,index=idx)),
                                     n_sfs_dirs)
    ## estimate sigma hat at plugin
    sigma = surface.max_covariance(inferred_ms_params.values)

    # recommend to call check_symmetric on matrix inverse,
    # as linear algebra routines may not perfectly preserve symmetry due to numerical errors
    sigma_inv = check_symmetric(np.linalg.inv(sigma))
   
    ## marginal p values
    sd = np.sqrt(np.diag(sigma))
    z = (inferred_ms_params - true_ms_params) / sd
    z_p = pd.Series((1.0 - scipy.stats.norm.cdf(np.abs(z))) * 2.0 , index=idx)

    coord_results = results_df(inferred_ms_params, opt_space=False)
    coord_results['p value'] = z_p
    myprint(coord_results)
    
    ## global p value
    resids = inferred_ms_params - true_ms_params
    eps_norm = np.dot(resids, np.dot(sigma_inv, resids))
    wald_p = 1.0 - scipy.stats.chi2.cdf(eps_norm, df=len(resids))
    
    myprint("# Chi2 test for params=true_params")
    myprint("# X, 1-Chi2_cdf(X,df=%d)" % len(resids))    
    myprint(eps_norm, wald_p)

    conf_end = time.clock()
    
    return {'truth': true_ms_params,
            'est': inferred_ms_params,
            'init': transform_params(init_opt_params),
            'sigma': sigma,
            'sigma_inv': sigma_inv,
            'p_vals': {'z': z_p, 'wald': wald_p},
            'opt_res': optimize_res,
            'time': {'opt': opt_end - start, 'conf': conf_end - opt_end,
                     'total' : conf_end - start},
            'num_snps': {'total' : total_snps, 'unique': uniq_snps},
            }


def get_likelihood_surface(true_demo, sfs_list, theta, demo_func, n_sfs_dirs=0):
    if n_sfs_dirs <= 0:
        return NegativeLogLikelihood(sfs_list, theta=theta, demo_func=demo_func)
    else:
        sfs_dirs = {}
        for leaf in true_demo.leaves:
            sfs_dirs[leaf] = np.random.normal(size=(n_sfs_dirs, true_demo.n_lineages(leaf)+1))
        return L2ErrorSurface(sfs_list, sfs_dirs, theta, demo_func)
