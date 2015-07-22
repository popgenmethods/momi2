from __future__ import division
from momi import make_demography, NegativeLogLikelihood
import scipy as sp
import scipy.optimize as spopt
import scipy.stats
## Thinly-wrapped numpy that supports automatic differentiation
import autograd.numpy as anp
## Functions for computing derivatives
from autograd import grad, hessian_vector_product

'''
This example fits a piecewise exponential size history
to a single population.

It provides an example of using Python I/O to read in
a file of SFS counts.

It also provides a different approach to optimization then
example_inference.py, which uses second-order Newton algorithm
on an unconstrained parameter space. By contrast, in this example:
1) We use a bounded parameter space (population sizes, waiting times must be positive)
2) We only provide first-order gradient. scipy defaults to using L-BFGS or BFGS algorithm in this case.
3) We do the gradient descent within a larger global optimization algorithm, basinhopping, that is more robust to local minima.
'''

def main(sfs_file, n_epochs):
    '''
    Given a file with SFS counts and number of epochs, infer population size history.
    '''
    sfs_list = read_data(sfs_file)
    theta_list = [avg_pairwise_differences(sfs) for sfs in sfs_list]
    x, fx = infer_params(sfs_list, theta_list, n_epochs)
    print fx
    print piecewise_exponential_demo(x).ms_cmd

def read_data(filename):
    '''
    Read in the SFS in file name, which is assumed to have the same format
    as used in fastNeutrino.
    '''
    f = open(filename)
    lines = f.readlines()
    f.close()

    first_line, lines = lines[0], lines[1:]

    global n
    n,L = [int(x) for x in first_line.strip().split()]

    sfs_list = [{} for locus in range(L)]
    for n_ancestral,line in enumerate(lines):
        n_derived = n - n_ancestral
        line = line.strip()
        if line == "":
            continue

        counts = [int(x) for x in line.split()]
        assert len(counts) == L

        for locus,count in enumerate(counts):
            # config is a 1-tuple with the derived count, because there is only 1 population
            config = (n_derived,)
            sfs_list[locus][config] = count

    assert all([len(locus) == n+1 for locus in sfs_list])

    # remove the monomorphic sites
    monomorphic_counts = [locus.pop((0,)) + locus.pop((n,)) for locus in sfs_list]
    assert all([len(locus) == n-1 for locus in sfs_list])

    return sfs_list

def avg_pairwise_differences(sfs):
    '''
    Gets the average pairwise differences, used to estimate theta.
    '''
    total_pairwise_diffs = 0
    for n_der, in sfs:
        total_pairwise_diffs += n_der * (n - n_der) * sfs[(n_der,)]
    return total_pairwise_diffs / (n * (n-1) / 2)

def piecewise_exponential_demo(params):
    '''
    Function that returns a piecewise exponential demo from a parameter vector.
    params is assumed to have length 3*(n_epochs-1)+1, where:
    (1) first n_epochs parameters are the start size at the bottom of each epoch.
    (2) next n_epochs-1 parameters are the time lengths of the epochs
    (3) final n_epochs-1 parameters are the growth rates within each epochs
    '''
    assert (len(params) -1) % 3 == 0
    n_epochs = int((len(params) - 1) / 3)+1
    # TODO: function doesn't work for n_epochs == 1 (constant size history)
    assert n_epochs >= 2

    # the first n_epochs parameters are the epoch start sizes
    start_sizes = params[:n_epochs]
    params = params[n_epochs:]
    # the next n_epoch-1 parameters are epoch lengths, and then growth rates
    epoch_lens, growth_rates = params[:n_epochs-1], params[n_epochs-1:]

    assert anp.all(start_sizes > 0) and anp.all(epoch_lens > 0)

    # the start times of the epochs
    start_times = anp.cumsum(epoch_lens)

    # turn the parameter vectors into lists
    start_sizes, start_times, growth_rates = (list(x) for x in (start_sizes, start_times, growth_rates))
    # handle the first and last epochs separately
    final_size, final_time = start_sizes.pop(), start_times.pop()
    init_size, init_growth = start_sizes.pop(0), growth_rates.pop(0)

    # now construct the ms command line
    demo_string = ["-I 1 %d" % n]
    args = []
    def add_args(args, *arg_vals):
        # add the arguments, and return their position
        args += arg_vals
        return range(len(args) - len(arg_vals), len(args))

    demo_string += ["-eN 0 $%d -eG 0 $%d" % tuple(add_args(args, init_size, init_growth))]    
    for size, growth, time in zip(start_sizes, growth_rates, start_times):
        size_idx, growth_idx, time_idx = add_args(args, size, growth, time)
        demo_string += ["-eN $%d $%d -eG $%d $%d" % (time_idx, size_idx, time_idx, growth_idx)]
    demo_string += ["-eN $%d $%d" % tuple(add_args(args, final_time, final_size))]

    demo_string = " ".join(demo_string)
    return make_demography(demo_string, *args)

def infer_params(sfs_list, theta_list, n_epochs):
    '''
    Infers a piecewise-exponential size history,
    with n_epochs finite intervals.
    (or n_epochs+1 total intervals, including the final epoch of infinite length)

    Uses L-BFGS within basinhopping algorithm to infer the parameters.
    '''
    surface = NegativeLogLikelihood(sfs_list, theta=theta_list,
                                     demo_func=piecewise_exponential_demo,
                                     )

    # construct the function to minimize, and its derivatives
    def f(params):
       try:
           return surface.evaluate(params)
       except Exception:
           # in case parameters are out-of-bounds or so extreme they cause overflow/stability issues. just return a very large number. note the gradient will be 0 in this case and the gradient descent may stop.
           return 1e100
    g, hp = grad(f), hessian_vector_product(f)

    # bounds for the L-BFGS. constrained so that population sizes, waiting times are positive. growth rates (the last n_epochs-1 parameters) are unconstrained
    bounds = [(1e-16,None)] * (2*(n_epochs-1)+1) + [(None,None)] * (n_epochs-1)

    # note that it is recommended to set the lower bound to 1e-16
    # instead of 0. otherwise L-BFGS will sometimes try values of
    # 0-1e-16, which is within numerical precision of 0 but still
    # an invalid parameter value since it is negative.
    # In particular this will cause the L-BFGS to stop, because
    # it gets a gradient of 0 at that point.

    # random initial parameters
    init_sizes = anp.random.exponential(size=n_epochs)
    init_lens = anp.random.exponential(size=n_epochs-1) / n_epochs
    init_rates = anp.random.normal(size=n_epochs-1)

    init_params = anp.concatenate([init_sizes, init_lens, init_rates])

    # print after every basinhopping iteration
    def print_fun(x, f, accepted):
        print("at minima %.4f accepted %d" % (f, int(accepted)))

    # now do the inference
    optimize_res = spopt.basinhopping(f, init_params,
                                      niter=100,
                                      interval=1,
                                      T=1000,
                                      minimizer_kwargs={'jac':g,'bounds':bounds},
                                      callback=print_fun)

    return optimize_res.x, optimize_res.fun

if __name__=="__main__":
    main("JXY_sfs.txt", 3)
