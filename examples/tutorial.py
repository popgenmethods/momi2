import momi
from numpy import isclose

"""
This tutorial gives an overview of the momi package.
Use the help() function to view documentation and more information.
"""
## enter help screen of the module. press 'q' to exit
help(momi)

## TODO: write help(momi)!!

"""
Creating Demographies
---------------------
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
    # Sample 10 alleles from deme 0 and 7 alleles from deme 1
    "-n 10 7 " +
    # at present (t=0), deme 0 exponentially growing at rate .00005 per generation
    "-G 0 0 5e-5 " +
    # at present (t=0), deme 1 has diploid pop size of 5000
    "-N 0 1 5000 " +
    # at t=3500 generations, 75% of lineages in deme 0 migrate to a new deme 2
    "-S 3500 0 .25 " +
    # deme 2 joins onto deme 1 (so effect is a pulse from deme 1 to deme 0)
    "-J 3500 2 1 " +
    # at t=15000 generations, deme 1 joins onto deme 0
    "-J 15000 1 0 " +
    # at t=15000 generations, deme 0 has growth rate set to 0
    "-G 15000 0 0"
)

# equivalent demography, specified from the ms command line
demo2 = momi.Demography.from_ms(1e4,
                                "-I 2 10 7 " +
                                "-eg 0 1 .5 " +
                                "-en 0 2 .5 " +
                                "-es .35 1 .25 " +
                                "-ej .35 3 2 " +
                                "-ej 1.5 2 1 " +
                                "-eg 1.5 1 0"
                                )

# print out demo2 in momi format; check it is indeed equivalent to demo
print demo2

"""
Notice how the flags -G,-J,-N,-S correspond to -eg,-ej-en,-es, respectively.

This example also illustrates two important differences between
the momi and ms formats:
1) In momi, parameters are in per-generation units,
   whereas in ms, parameters are rescaled by the reference size (1e4):
   sizes and times are divided by 1e4, and growth rate is multiplied by 1e4.
2) In momi, populations are indexed starting at 0, 
   whereas in ms populations are indexed starting at 1.
   (So population 0 in momi corresponds to population 1 in ms)

There is a third major difference, not illustrated here,
that we will revisit in more detail later (in the Inference section):
3) In momi, new populations created by -S are labeled according to their position in the command line.
   In ms, new populations created by -es are labeled according to the time of the admixture event.

   In this example, there is only one admixture event, 
   so the ordering of populations is the same in both commands.

See also:
help(momi.Demography.__init__) for info on the momi format
help(momi.Demography.from_ms) for info on the ms format
help(momi.Demography) for info on the Demography object
"""

## TODO: write help(momi.Demography)! make it no longer a subclass of nx.Digraph

"""
Coalescent Statistics
---------------------
We illustrate some functions for computing statistics of the multipopulation coalescent.
"""

eTmrca = momi.expected_tmrca(demo)
print "Expected TMRCA: ", eTmrca, "generations"

eTmrca_0 = momi.expected_deme_tmrca(demo, 0)
print "Expected TMRCA of samples in deme 0: ", eTmrca_0, "generations"

eL = momi.expected_total_branch_len(demo)
print "Expected total branch length: ", eL, "generations"

# check that demo2 has the same values for these statistics
# use numpy.isclose to check equality of floating point numbers
assert isclose(eTmrca, momi.expected_tmrca(demo2))
assert isclose(eTmrca_0, momi.expected_deme_tmrca(demo2, 0))
assert isclose(eL, momi.expected_total_branch_len(demo2))

"""
See help(momi.expected_tmrca), etc. for details.

The workhorse behind these statistics, and momi.expected_sfs (next Section),
is momi.expected_sfs_tensor_prod(), which can efficiently compute a variety
of summary statistics of the expected SFS and the coalescent.
"""

"""
Expected Site Frequency Spectrum (SFS)
--------------------------------------
"""

# ## sfs

# config_list = ...
# print expected_sfs(config_list, demo)


# ## simulate/read in data from ms

# ms_file = file(...)
# ## UNCOMMENT next 2 lines to generate a new dataset. note user must make sure ms_path is pointing to ms or scrm!
# # ms_path = momi.default_ms_path() ##
# # ms_file = momi.simulate_ms(ms_path = ms_path...)

# sfs_list = momi.sfs_list_from_ms(ms_file)
# print sfs_list

# # combine into a single sfs
# combined_sfs = momi.sum_sfs_list(sfs_list)
# print combined_sfs


# ## compute log likelihood of combined sfs

# print unlinked_log_likelihood(...)


# ## now inference
# ## start by making a function that maps parameters into demographies

# ## later we will want to make sure this function also works with automatic differentiation,
# ## but let's worry about that later
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
