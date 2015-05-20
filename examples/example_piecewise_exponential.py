from __future__ import division
from momi import make_demography, CompositeLogLikelihood
import scipy as sp
import scipy.optimize as spopt
import scipy.stats
## Thinly-wrapped numpy that supports automatic differentiation
import autograd.numpy as anp
## Functions for computing derivatives
from autograd import grad, hessian_vector_product

def main(sfs_file, n_epochs):
    sfs_list = read_data(sfs_file)
    theta_list = [avg_pairwise_differences(sfs) for sfs in sfs_list]
    x, fx = infer_params(sfs_list, theta_list, n_epochs)
    print fx
    print piecewise_exponential_demo(x).ms_cmd

def infer_params(sfs_list, theta_list, n_epochs):
    surface = CompositeLogLikelihood(sfs_list, theta=theta_list,
                                     demo_func=piecewise_exponential_demo,
                                     )

    # construct the function to minimize, and its derivatives
    def f(params):
       try:
           return -surface.log_likelihood(params)
       except:
           # in case the basinhopping proposes parameters that are out-of-bounds or so extreme they cause overflow/stability issues
           return 1e100
    g, hp = grad(f), hessian_vector_product(f)

    # random initial parameters
    init_lens = anp.random.exponential(size=n_epochs) / n_epochs
    init_sizes = anp.random.exponential(size=n_epochs+1)
    init_rates = anp.random.normal(size=n_epochs)

    bounds = [(1e-16,None)] * (2*n_epochs+1) + [(None,None)] * n_epochs

    init_params = anp.concatenate([init_lens, init_sizes, init_rates])

    def print_fun(x, f, accepted):
        print("at minima %.4f accepted %d" % (f, int(accepted)))

    optimize_res = spopt.basinhopping(f, init_params,
                                      niter=100,
                                      interval=1,
                                      minimizer_kwargs={'jac':g,'bounds':bounds},
                                      callback=print_fun)

    return optimize_res.x, optimize_res.fun
                                      
def avg_pairwise_differences(sfs):
    total_pairwise_diffs = 0
    for n_der, in sfs:
        total_pairwise_diffs += n_der * (n - n_der) * sfs[(n_der,)]
    return total_pairwise_diffs / (n * (n-1) / 2)

def read_data(filename):
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

def piecewise_exponential_demo(params):
    assert (len(params) -1) % 3 == 0
    n_epochs = int((len(params) - 1) / 3)
    # TODO: function doesn't work for n_epochs == 0 (constant size history)
    assert n_epochs >= 1 

    # the first n_epochs+1 parameters are the epoch start sizes
    start_sizes = params[:n_epochs+1]
    params = params[n_epochs+1:]
    # the next n_epoch parameters are epoch lengths, and then growth rates
    epoch_lens, growth_rates = params[:n_epochs], params[n_epochs:]

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

if __name__=="__main__":
    main("JXY_sfs.txt", 1)
