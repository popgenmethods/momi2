
## can call help(momi) for an overview of the useful functions in momi

## create demographies

demo = make_demography(...)
demo2 = demo_from_ms(...)

## consult help functions help(make_demography) and help(demo_from_ms) for more details
help(make_demography)
help(demo_from_ms)

# should be the same
print demo
print demo2

## compute some statistics of the multipopulation coalescent

eTmrca = momi.expected_tmrca(demo)
eL = momi.expected_total_branch_len(demo)

print eTmrca, eL

# we can check that demo2 has the same values for E[Tmrca] and E[L]
# use numpy.isclose to check equality of floating point numbers
assert isclose(eTmrca, expected_tmrca(demo2))
assert isclose(eL, expected_total_branch_len(demo2))

## sfs

config_list = ...
print expected_sfs(config_list, demo)


## simulate/read in data from ms

ms_file = file(...)
## UNCOMMENT next 2 lines to generate a new dataset. note user must make sure ms_path is pointing to ms or scrm!
# ms_path = momi.default_ms_path() ##
# ms_file = momi.simulate_ms(ms_path = ms_path...)

sfs_list = momi.sfs_list_from_ms(ms_file)
print sfs_list

# combine into a single sfs
combined_sfs = momi.sum_sfs_list(sfs_list)
print combined_sfs


## compute log likelihood of combined sfs

print unlinked_log_likelihood(...)


## now inference
## start by making a function that maps parameters into demographies

## later we will want to make sure this function also works with automatic differentiation,
## but let's worry about that later
def demo_func():
    x += 3 ## NOTE for later this line is not differentiable!


res = unlinked_mle_search0(..., verbose=True)


## automatic differentiation

# start by defining function that maps parameters to log likelihoods
def log_lik_func(params):
    return unlinked_log_likelihood(demo_func(params),...)

log_lik_func(truth)

# define gradient
g = grad(log_lik_func)
try:
    print "trying to take gradient of log likelihood function"
    g(truth)
except Exception,e:
    print "gradient failed!"
    print e

## lets rewrite the function so its differentiable
## write long comment about how to make differentiable functions
def demo_func2(...):
    ## make pulse probs be in logistic scale, to illustrate autograd.numpy
    pass

def log_lik_func(params):
    return unlinked_log_likelihood(demo_func(params),...)
g = grad(log_lik_func)
print g(params) ## it worked!

## now try to do inference using the first derivative
res = unlinked_mle_search1(...)

## finally, confidence intervals

cov = unlinked_mle_approx_cov(...)
# compute marginal confidence intervals, Wald p-values
