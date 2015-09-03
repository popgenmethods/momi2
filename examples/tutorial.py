import momi
from numpy import allclose
import pandas

"""
This tutorial gives an overview of the momi package.
Use the help() function to view documentation and more information.
"""
## enter help screen of the module. press 'q' to exit
help(momi)

## TODO: write help(momi)!!

print "Section 0: Creating Demographies"
print "--------------------------------"
"""
Let's start by creating a demographic history. 
momi uses a command line format loosely based on the program ms.

Users who prefer the original ms command line can also use that format, but be warned:
the ms command line follows different conventions from the rest of momi,
in particular when it comes to population labels and the scaling of parameters.
"""
# a demography constructed with the momi format
demo = momi.Demography(
    # default diploid pop size = 1e4    
    "-d 1e4 " +
    # Sample 10 alleles from deme 0 and 12 alleles from deme 1
    "-n 10 12 " +
    # at present (t=0), deme 0 exponentially growing at rate .00005 per generation
    "-G 0 0 5e-5 " +
    # at present (t=0), deme 1 has diploid pop size of 5000
    "-N 0 1 5000 " +
    # at t=3500 generations ago, 75% of lineages in deme 0 migrate to a new deme 2
    "-S 3500 0 .25 " +
    # deme 2 joins onto deme 1 (so effect is a pulse from deme 1 to deme 0, forward in time)
    "-J 3500 2 1 " +
    # at t=15000 generations ago, deme 1 joins onto deme 0
    "-J 15000 1 0 " +
    # at t=15000 generations ago, deme 0 has growth rate set to 0
    "-G 15000 0 0"
)
print demo

# same demography, using special variables $
demo = momi.Demography("-d 1e4 -n 10 12 -G 0 0 $growth0 -N 0 1 $size1 -S $pulse_time 0 $prob -J $pulse_time 2 1 -J $join_time 1 0 -G $join_time 0 0",
                       growth0 = 5e-5, size1 = 5000, pulse_time = 3500, prob = .25, join_time = 15000)
print demo


# same demography, using ms command line
demo = momi.Demography.from_ms(1e4,
                               "-I 2 10 12 -eg 0 1 .5 -en 0 2 .5 -es .35 1 .25 -ej .35 3 2 -ej 1.5 2 1 -eg 1.5 1 0")
print demo

# ms command line can also use special variables
demo = momi.Demography.from_ms(1e4,
                               "-I 2 10 12 -eg 0 1 $growth0 -en 0 2 $size1 -es $pulse_time 1 $prob -ej $pulse_time 3 2 -ej $join_time 2 1 -eg $join_time 1 0",
                               growth0 = 5e-5 * 1e4, size1 = 5000 / 1e4, pulse_time = 3500 / 1e4, prob = .25, join_time = 15000 / 1e4)
print demo

"""
Notice how the flags -G,-J,-N,-S correspond to -eg,-ej-en,-es, respectively.

This example illustrates two important differences between
the momi and ms formats:
1) In momi, parameters are in per-generation units,
   whereas in ms, parameters are rescaled by the reference size (1e4 in this example):
   sizes and times are divided by 1e4, and growth rate is multiplied by 1e4.
2) In momi, populations are indexed starting at 0, 
   whereas in ms populations are indexed starting at 1.
   (So population 0 in momi corresponds to population 1 in ms)

There is a third major difference, not illustrated by this example:
3) In momi, new populations created by -S are labeled in order of position of -S flag in command line.
   In ms, new populations created by -es are labeled in order of time of -es event.

   You'll have to worry about this if there is more than one -S event.
   But in this example, there is only one -S event, so we refer the reader to
   help(momi.Demography) for more details (in particular, see Note ##TODO).
"""

## TODO: write help(momi.Demography)! make it no longer a subclass of nx.Digraph

print "\n"
print "Section 1: Coalescent Statistics"
print "--------------------------------"
"""
Let's examine some statistics of the above demography.
"""

eTmrca = momi.expected_tmrca(demo)
print "Expected TMRCA: ", eTmrca, "generations"

eTmrca_0 = momi.expected_deme_tmrca(demo, 0)
print "Expected TMRCA of samples in deme 0: ", eTmrca_0, "generations"

eL = momi.expected_total_branch_len(demo)
print "Expected total branch length: ", eL, "generations"

"""
See help(momi.expected_tmrca), etc. for details.
These functions are all wrappers for momi.expected_sfs_tensor_prod(),
which can efficiently compute a variety of summary statistics for the SFS and the coalescent.
"""


print "\n"
print "Section 2: Expected Sample Frequency Spectrum (SFS)"
print "---------------------------------------------------"
"""
The expected SFS for configuration (i_0,i_1,...) is the expected length of branches with
     i_0 descendants in pop 0, i_1 descendants in pop 1, ...

Equivalently, it is the expected number of mutations with
     i_0 derived alleles in pop 0, i_1 derived alleles in pop 1, ...
when the mutation rate mu == 1.
To get the expected number of mutations in general, multiply the expected SFS by mu.

To get the probability that a random mutation has
     i_0 descendants in pop 0, i_1 descendants in pop 1, ...
take the expected SFS, and divide by expected_total_branch_len().
"""

# a list of configs
config_list = [(1,0), (0,1), (1,3), (10,0), (0,12), (2,2)]
# an array of the SFS entries corresponding to each config in config_list
eSFS = momi.expected_sfs(demo, config_list)
# the SFS renormalized to be probabilities
eSFS_normalized = momi.expected_sfs(demo, config_list, normalized=True)

# use numpy.allclose to check equality of vectors
assert allclose(eSFS_normalized, eSFS / eL)

print pandas.DataFrame({"Config": config_list, "E[SFS]": eSFS, "Prob": eSFS_normalized})

"""
momi.expected_sfs also includes options for dealing with sampling error and ascertainment bias.
See help(momi.expected_sfs) for more details.
"""

print "\n"
print "Section 3: Observed SFS"
print "-----------------------"
"""
The observed SFS gives the number of observed SNPs for each configuration.
Its expected value is the total mutation rate, times the expected SFS (previous section).

momi represents the observed SFS as a dictionary, mapping configs (tuples) to counts (ints).

Here, we use ms to simulate a dataset, read in the output,
and then construct the observed SFS using momi.sfs_list_from_ms().
We then print out the SFS and some statistics for illustration.
"""

print "Reading dataset from ms...\n"

n_loci, mu_per_locus = 1000, 1e-3

# file of output from ms
ms_output = momi.simulate_ms(demo, num_sims=n_loci, mu=mu_per_locus, additional_ms_params="-r %f 10000" % mu_per_locus)

# get a list with the observed SFS at each locus
sfs_list = momi.sfs_list_from_ms(ms_output, demo.n_at_leaves)

# aggregate into a single SFS for the whole dataset
combined_sfs = momi.sum_sfs_list(sfs_list)

# The observed SFS is represented as a dictionary, mapping configs (tuples) to their counts
print "Observed SFS for locus 0:\n", sfs_list[0], "\n"
print "Observed SFS for all loci:\n", combined_sfs, "\n"

print "Number of singleton mutations:\n", combined_sfs[(1,0)] + combined_sfs[(0,1)], "\n"
print "Total number of mutations:\n", sum(combined_sfs.values())


## TODO rescale simulate_ms? it's a bit confusing with the -r parameter (make sure to update mu_per_locus below)
## TODO save an ms file in repository and read that in
## TODO: change API of sfs_list_from_ms, simulate_ms to use list of lines, instead of file object?
## TODO: change sfs_list_from_ms so it doesn't need demo.n_at_leaves


print "\n"
print "Section 4: Composite likelihood"
print "-------------------------------"
"""
We construct a composite likelihood by using a Poisson random field (PRF) approximation.

This assumes that the number of observed SNPs for each configuration 
are independent Poisson with rate lambda == mutation_rate * expected_sfs.
"""

# the mutation rate for the whole dataset
combined_mu = n_loci * mu_per_locus

# compute the composite likelihood
composite_log_lik = momi.unlinked_log_likelihood(combined_sfs, demo, mu=combined_mu)

print "Composite log likelihood:", composite_log_lik


print "\n"
print "Section 5: Inference"
print "--------------------"
"""
The Maximum Composite Likelihood Estimator (MCLE) is the demography that maximizes unlinked_log_likelihood().
Under certain conditions, this demography converges to the true demography.

To find the MCLE, we first define a function mapping from parameters to valid demographies.

For optimization, it's a good idea to choose parameters so that they are of roughly the same
order of magnitude (this is related to 'preconditioning').

Here, we use scaled_growth = growth * 1e4, and scaled_pop_size,scaled_time = pop_size,time / 1e4 (similar to ms).
We also parametrize the pulse prob by its logit, for the sake of illustration.
"""
import autograd.numpy as np ## thinly wrapped version of numpy; see comment below
def demo_func(params):
    scaled_growth0, scaled_size1, scaled_pulse_time, logit_prob, scaled_wait_time = params
    
    return momi.Demography("-d 1e4 -n 10 12 -G 0 0 $growth0 -N 0 1 $size1 -S $pulse_time 0 $prob -J $pulse_time 2 1 -J $join_time 1 0 -G $join_time 0 0",
                           growth0 = scaled_growth0 / 1e4,
                           size1 = scaled_size1 * 1e4,
                           pulse_time = scaled_pulse_time * 1e4,
                           prob = 1.0 / (1.0 + np.exp(-logit_prob)),
                           join_time = (scaled_pulse_time + scaled_wait_time) * 1e4,
    )

true_params = np.array([5e-5 * 1e4, 5000 / 1e4, 3500 / 1e4, np.log(.25 / (1 - .25)), (15000 - 3500) / 1e4])
demo = demo_func(true_params)
print demo

"""
Notice how we use autograd.numpy for math functions. This allows us to automatically take derivatives:
"""
def lik_func(params):
    return momi.unlinked_log_likelihood(combined_sfs, demo_func(params), mu=combined_mu)

from autograd import grad
lik_grad = grad(lik_func)

true_lik = lik_func(true_params)
true_lik_grad = lik_grad(true_params)

print ""
print "Composite log likelihood at truth:", true_lik
print "Gradient at truth:", true_lik_grad

"""
See the comment at the end of this section for more details on autograd and automatic differentiation.

Next we define (lower,upper) bounds on the parameter space, pick a random start value, and then search for the MCLE.
"""
# (lower,upper) bounds on the parameter space
bounds = [(-.001 * 1e4, .001 * 1e4), # scaled growth rate
          (1e2 / 1e4, 1e6 / 1e4), # scaled pop size
          (1 / 1e4, 1e5 / 1e4), # scaled pulse time
          (-10,10), # logit(pulse prob)
          (1 / 1e4, 1e5 / 1e4), # scaled wait time
          ]

# random starting values from the triangular distribution
import random
start_params = np.array([random.triangular(bounds[i][0], bounds[i][1], mode)
                      for i,mode in enumerate([0, 1, 1, 0, 1])])

print ""
print "True parameters:", true_params
print "Start parameters:", start_params
print "Searching for MCLE..."

mcle_search_result = momi.unlinked_mle_search(combined_sfs, demo_func, combined_mu, start_params, verbose = 20, bounds = bounds, maxiter = 500)
est_params = mcle_search_result.x

# search for the MCLE using gradient information
#mcle_search_result = momi.unlinked_mle_search0(combined_sfs, demo_func, combined_mu, start_params, bounds = bounds, verbose=True)
#mcle_search_result = momi.unlinked_mle_search1(combined_sfs, demo_func, combined_mu, start_params, bounds = bounds, maxiter=200, verbose=True)
#mcle_search_result = momi.unlinked_mle_search1(combined_sfs, demo_func, combined_mu, start_params, bounds = bounds, verbose=True)
#est_params = mcle_search_result.x

# transform_params2 = lambda params: np.array([params[0], np.exp(params[1]), np.exp(params[2]), params[3], np.exp(params[4])])
# demo_func2 = lambda params: demo_func(transform_params2(params))
# start_params2 = np.array([start_params[0], np.log(start_params[1]), np.log(start_params[2]), start_params[3], np.log(start_params[4])])
# mcle_search_result = momi.unlinked_mle_search2(combined_sfs, demo_func2, combined_mu, start_params2, verbose=True)
# est_params = transform_params2(mcle_search_result.x)

print mcle_search_result

print "Log-likelihood at truth:", true_lik
print "Log-likelihood at estimated:", -mcle_search_result.fun


## TODO: write documentation for unlinked_mle_search()
## TODO: clean up printing in this section

print "Est/Truth:",  est_params / true_params
print "Estimated Demography:\n", demo_func(est_params)

"""
Note can use unlinked_mle_search0 if unsure about autograd
or unlinked_mle_search2 if want to use hessian
"""

"""
Now let's create a function mapping parameters to Demography.
We also want this function to work with automatic differentiation,
so we can use hill-climbing later.

momi uses the autograd package to automatically compute derivatives.
In principle, autograd can compute the derivative of arbitrary functions written with python code.
In practice, here are a few rules to keep in mind:

0) Arithmetic operations +,-,*,/,** all work automatically

1) For more complicated mathematical operations, use autograd.numpy and autograd.scipy,
   thinly wrapped versions of numpy/scipy that support auto-differentiation.

   For most users, numpy contains all the mathematical operations that are needed:
   exp, log, trigonemetric functions, matrix operations, fourier transform, etc.

   If needed, it is also possible to use autograd to define derivatives of your own
   mathematical operations.

2) Do NOT convert floats to strings, as this will break with autograd.
   Instead, use positional or keyword arguments to pass parameters to Demography() constructor.

3) Other do's and don'ts: (copy and pasted from autograd tutorial, with 'np' shorthand for 'autograd.numpy')

    Do use

    Most of numpy's functions
    Most numpy.ndarray methods
    Some scipy functions
    Indexing and slicing of arrays x = A[3, :, 2:4]
    Explicit array creation from lists A = np.array([x, y])

    Don't use

    Assignment to arrays A[0,0] = x
    Implicit casting of lists to arrays A = np.sum([x, y]), use A = np.sum(np.array([x, y])) instead.
    A.dot(B) notation (use np.dot(A, B) instead)
    In-place operations (such as a += b, use a = a + b instead)

Documentation for autograd can be found at https://github.com/HIPS/autograd/
"""

    
"""
We note that special arguments can be used in the same way with Demography.from_ms()
"""








# ## now inference
# ## start by making a function that maps parameters into demographies

# ## later we will want to make sure this function also works with automatic differentiation,
# ## but let's worry about that later
## to make the function non-differentiable, insert parameters as string
# def demo_func():
#     x += 3 ## NOTE for later this line is not differentiable!


# res = unlinked_mle_search0(..., verbose=True)


# ## automatic differentiation

# # start by defining function that maps parameters to log likelihoods
# def log_lik_func(params):
#     return unlinked_log_likelihood(demo_func(params),...)

# log_lik_func(truth)

# # define gradient
# g = grad(log_lik_func)
# try:
#     print "trying to take gradient of log likelihood function"
#     g(truth)
# except Exception,e:
#     print "gradient failed!"
#     print e

# ## lets rewrite the function so its differentiable
# ## write long comment about how to make differentiable functions
# def demo_func2(...):
#     ## make pulse probs be in logistic scale, to illustrate autograd.numpy
#     pass

# def log_lik_func(params):
#     return unlinked_log_likelihood(demo_func(params),...)
# g = grad(log_lik_func)
# print g(params) ## it worked!

# ## now try to do inference using the first derivative
# res = unlinked_mle_search1(...)

# ## finally, confidence intervals

# cov = unlinked_mle_approx_cov(...)
# # compute marginal confidence intervals, Wald p-values
