import momi
from numpy import allclose
import pandas

"""
This tutorial gives an overview of the momi package.

Recommended usage: copy/paste blocks of code into ipython session

To rerun results with new random dataset, run as
     python tutorial.py /path/to/ms [--save (optional)]
(--save will overwrite tutorial_data.txt)

Use the help() function to view documentation.
"""
## enter help screen of the module. press 'q' to exit
help(momi)



print "Section 0: Creating Demographies"
print "--------------------------------"
"""
Let's start by creating a demographic history. 
momi uses a command line format loosely based on the program ms by Richard Hudson.

Users who prefer the ms command line can also use that format, but be warned:
the ms command line follows different conventions from the rest of momi,
in particular when it comes to population labels and the scaling of parameters.
"""

# a demography from the ms command line format
demo = momi.Demography.from_ms(1e4, # the reference diploid population size
                               # ms command line
                               "-I 2 10 12 -eg 0 1 .5 -en 0 2 .5 -es .35 1 .25 -ej .35 3 2 -ej 1.5 2 1 -eg 1.5 1 0")
print demo


# the same demography constructed with the momi format
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

# Demography() can use special variables with $
demo = momi.Demography("-d 1e4 -n 10 12 -G 0 0 $growth0 -N 0 1 $size1 -S $pulse_time 0 $prob -J $pulse_time 2 1 -J $join_time 1 0 -G $join_time 0 0",
                       growth0 = 5e-5, size1 = 5000, pulse_time = 3500, prob = .25, join_time = 15000)
print demo

# Demography.from_ms() can also use special variables
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
   But in this example, there is only one -S event, so let's not worry about it now.

See Also:
---------
help(momi.Demography), help(momi.Demography.__init__), help(momi.Demography.from_ms)
"""



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

print pandas.DataFrame({"Config": config_list, "E[SFS]": eSFS, "Prob": eSFS_normalized}).set_index('Config')

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

n_loci, mu_per_locus, recom_per_locus = 1000, 1e-3, 1e-3
data_file = 'tutorial_data.txt'

import sys
if len(sys.argv) < 2:
    # use saved dataset
    print "Reading dataset from %s" % data_file  
    ms_output = file(data_file,'r')
else:
    # to generate new dataset, run script as
    # python tutorial.py /path/to/ms [--save (optional)]
    print "Generating new dataset..."
    ms_path = sys.argv[1]
    ms_output = momi.simulate_ms(ms_path, demo, n_loci, mu_per_locus, additional_ms_params="-r %f 10000" % (1e4 * recom_per_locus))

    if '--save' in sys.argv[2:]:
        print "Saving generated dataset in %s" % data_file
        
        data_file = file(data_file,'w')
        for line in ms_output:
            data_file.write(line)
        data_file.close()

        ms_output.seek(0)

print "Processing dataset from ms..."
        
# get a list with the observed SFS at each locus
sfs_list = momi.sfs_list_from_ms(ms_output)

# aggregate into a single SFS for the whole dataset
combined_sfs = momi.sum_sfs_list(sfs_list)

# The observed SFS is represented as a dictionary, mapping configs (tuples) to their counts
print "Observed SFS for locus 0:\n", sfs_list[0], "\n"
print "Observed SFS for all loci:\n", combined_sfs, "\n"

print "Number of singleton mutations:\n", combined_sfs[(1,0)] + combined_sfs[(0,1)], "\n"
print "Total number of mutations:\n", sum(combined_sfs.values())



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
print "True params:", true_params
print "Composite log likelihood at truth:", true_lik
print "Gradient at truth:", true_lik_grad

"""
See the FOOTNOTE at the end of this section for more details on autograd and automatic differentiation.

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
lower_bounds, upper_bounds = [l for l,u in bounds], [u for l,u in bounds]
start_params = np.array([random.triangular(lower, upper, mode)
                         for lower, upper, mode in zip(lower_bounds, upper_bounds, [0, 1, 1, 0, 1])])

print ""
print "Searching for MCLE..."

mcle_search_result = momi.unlinked_mle_search(combined_sfs, demo_func, combined_mu, start_params,
                                              bounds = bounds, maxiter = 500, output_progress = 25)

print ""
print "Search results:"
# print info such as whether search succeeded, number function/gradient evaluations, etc
print mcle_search_result
# note the printed function & gradient values are for -1*log_likelihood

print ""
print "Log-likelihood at truth:", true_lik
print "Log-likelihood at estimated:", -mcle_search_result.fun

est_params = mcle_search_result.x
print ""
print "Est params:", est_params
print "Est/Truth:",  est_params / true_params
print "Estimated Demography:\n", demo_func(est_params)

"""
FOOTNOTE: using autograd for automatic differentiation

autograd uses the magic of 'operator overloading' to compute derivatives automatically.

Here are a few rules to keep in mind, to make sure autograd works correctly:

0) If you don't want to worry about autograd at all, you can tell unlinked_mle_search not to use the gradient (jacobian) with:
        momi.unlinked_mle_search(..., jac=False, ...)

   Conversely, if you'd like to use more derivatives, i.e. the hessian or hessian-vector-products, you can do:
        momi.unlinked_mle_search(..., hess=True, ...)
        momi.unlinked_mle_search(..., hessp=True, ...)

1) Arithmetic operations +,-,*,/,** all work with autograd

2) For more complicated mathematical operations, use autograd.numpy and autograd.scipy,
   thinly wrapped versions of numpy/scipy that support auto-differentiation.

   For most users, numpy contains all the mathematical operations that are needed:
   exp, log, trigonemetric functions, matrix operations, fourier transform, etc.

   If needed, it is also possible to use autograd to define derivatives of your own
   mathematical operations.

3) Do NOT convert floats to strings when creating Demography.
   Instead, use special $ arguments to pass floats to Demography() or Demography.from_ms()

4) Other do's and don'ts: (copy and pasted from autograd tutorial, with 'np' shorthand for 'autograd.numpy')

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



print "\n"
print "Section 6: Confidence intervals"
print "-------------------------------"

"""
As the number of independent loci goes to infinity,
the MCLE is asymptotically Gaussian, with mean at the truth,
and covariance given by the inverse 'Godambe information'.
(assuming certain regularity conditions, e.g. that the parameter space is identifiable).

This can be used to construct approximate confidence intervals,
which have the correct coverage properties in the limit.
"""

print ""
print "Computing approximate covariance of MCLE..."

## the approximate covariance matrix of the MCLE
mcle_cov = momi.unlinked_mle_approx_cov(est_params, sfs_list, demo_func, mu_per_locus)
print mcle_cov

# marginal confidence intervals
print ""
print "Approximate 95% confidence intervals for parameters:"

import scipy.stats
conf_lower, conf_upper = scipy.stats.norm.interval(.95, loc = est_params, scale = np.sqrt(np.diag(mcle_cov)))
print pandas.DataFrame({"Truth" : true_params, "Lower" : conf_lower, "Upper" : conf_upper}, columns = ["Lower","Upper","Truth"], index=['scaled_growth0','scaled_size1','scaled_pulse_time','logit_prob','scaled_wait_time'])


# higher dimensional confidence regions, using wald test
print ""
print "Smallest alpha, so that level-alpha confidence region contains Truth:"
print "(alpha = 0 is a single point, alpha = 1 is whole parameter space)"

# wald test: residual * cov^{-1} * residual should be Chi-squared with n_params degrees of freedom

inv_cov = np.linalg.inv(mcle_cov)
# make sure the numerical inverse is still symmetric
assert allclose(inv_cov, inv_cov.T)
inv_cov = (inv_cov + inv_cov.T) / 2.0

resids = est_params - true_params
wald_stat = np.dot(resids, np.dot(inv_cov, resids))
print "alpha = ", scipy.stats.chi2.cdf(wald_stat, df=len(resids))

"""
Two important caveats for approximate confidence intervals

0) The approximate confidence intervals are correct only in the limit as n_loci -> infinity.
   There is no guarantee of correctness for any finite value of n_loci.

1) unlinked_mle_approx_cov() assumes that the loci are i.i.d.
   Mild violations of this assumption may be acceptable: see Notes of help(momi.unlinked_mle_approx_cov)
"""



## TODO: Section 7, inference with summary statistics and expected_sfs_tensor_prod
