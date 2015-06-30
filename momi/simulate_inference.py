from __future__ import division, print_function

from likelihood_surface import CompositeLogLikelihood
from parse_ms import make_demography, simulate_ms, sfs_list_from_ms
from util import check_symmetric

import scipy
import scipy.stats

import pandas as pd

## Functions for computing derivatives
import autograd.numpy as np
from autograd import grad, hessian_vector_product

def simulate_inference(ms_path, num_loci, theta, additional_ms_params, true_ms_params, init_opt_params, demo_factory, n_iter=10, transform_params=lambda x:x, verbosity=0):
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
    def myprint(*args,**kwargs):
        level = kwargs.get('level',1)
        if level <= verbosity:
            print(*args)
           
    true_ms_params = pd.Series(true_ms_params)
    old_transform_params = transform_params
    transform_params = lambda x: pd.Series(old_transform_params(x))

    def demo_func_ms(**params):
        try:
            return demo_factory(**params)
        except TypeError:
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

    myprint("# %d unique SNPs observed" % len({x for sfs in sfs_list for x in sfs.keys()}))
    
    # log-likelihood surface
    surface = CompositeLogLikelihood(sfs_list, theta=theta, demo_func=demo_func_opt)

    # construct the function to minimize, and its derivatives
    def f(params):
        try:
            return -surface.log_likelihood(params)
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
                                               minimizer_kwargs={'method':'newton-cg',
                                                                 'jac':g_verbose,
                                                                 'hessp':hp_verbose},
                                               callback=print_basinhopping)
    
    myprint("\n\n# Global minimum: %f" % optimize_res.fun)
    
    inferred_ms_params = transform_params(optimize_res.x)

    ## reparametrize surface by ms params
    idx = true_ms_params.index
    surface = CompositeLogLikelihood(sfs_list, theta=theta,
                                     demo_func=lambda x: demo_func_ms(**pd.Series(x,index=idx)))
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

    return {'truth': true_ms_params,
            'est': inferred_ms_params,
            'init': transform_params(init_opt_params),
            'sigma': sigma,
            'sigma_inv': sigma_inv,
            'p_vals': {'z': z_p, 'wald': wald_p},
            'opt_res': optimize_res}
