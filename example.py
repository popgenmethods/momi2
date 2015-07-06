## to run this file, you must either be in the momi directory, or put the momi directory in your PYTHONPATH,
## e.g. to run this file (or another one using momi code), you can do
## PYTHONPATH=$momi_dir python filename

from size_history import PiecewiseHistory, ConstantTruncatedSizeHistory, ExponentialTruncatedSizeHistory
from demography import Demography
from sum_product import SumProduct


## First, let's compute the SFS for a single population with a three-epoch history

n = 20 ## sample size

## specify the epochs backwards in time from the present
epochs = [ExponentialTruncatedSizeHistory(n_max=n, tau=0.25, N_top=1.0, N_bottom=10.0), # an exponential growth period for 0.25 units of time. "top"=past, "bottom"=present
          ConstantTruncatedSizeHistory(n_max = n, tau=0.1, N=0.1), # a bottleneck for 0.1 units of time
          ConstantTruncatedSizeHistory(n_max = n, tau=float('inf'), N=1.0), # ancestral population size = 1
          ]
## turn the list of epochs into a single population history
demo1 = PiecewiseHistory(epochs)

## print the SFS entries
print "Printing SFS entries for three epoch history"
print [demo1.freq(i, n) for i in range(1,n)]


## Next, print out the "truncated SFS" for just the two recent epochs
## i.e., the frequency spectrum for mutations that occur within the two recent epochs
epochs_trunc = epochs[:2]
demo1_trunc = PiecewiseHistory(epochs_trunc)

print "\nPrinting truncated SFS for two recent epochs"
print [demo1_trunc.freq(i,n) for i in range(1,n)]


## Finally, let's compute SFS entries for a multipopulation demography
# For our demography, we'll have three leaf populations: 'a','b','c', with 10,5,8 sampled alleles respectively

## specify demography via a newick string
## to specify additional population parameters, follow branch length with [&&momi:...]
## where ... contains:
##    lineages= # alleles (if population is leaf)
##    model= population history (default=constant)
##    N, N_top, N_bottom= parameters for constant/exponential size history
##    model_i= model for i-th epoch of piecewise history (either constant or exponential)
##    N_i, N_top_i, N_bottom_i, tau_i= parameters for i-th epoch of piecewise history
newick_str = """
((
a:.25[&&momi:lineages=10:model=constant:N=10.0],
b:.3[&&momi:lineages=5:model=exponential:N_top=1.0:N_bottom=10.0]
):.1[&&momi:model=constant:N=1.5],
c:.3[&&momi:lineages=8:model=piecewise:model_0=exponential:tau_0=.2:N_top_0=.1:N_bottom_0=1.0:model_1=constant:tau_1=.1:N_1=.3]
)[&&momi:N=3.0]
"""
demo3 = Demography.from_newick(newick_str)

demo3.update_state({'a': {'derived' : 1, 'ancestral': 9},
                    'b': {'derived' : 3, 'ancestral' : 2},
                    'c': {'derived' : 0, 'ancestral' : 8}})
sp = SumProduct(demo3)

print "\nSFS entry for (1,3,0) for 3-population demography"
print sp.p()
