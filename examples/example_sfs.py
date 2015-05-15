from momi import make_demography, compute_sfs
import numpy as np

'''
Start by constructing demography.
Use the same format as the program ms,
http://home.uchicago.edu/~rhudson1/source/mksamples.html
In particular see msdoc.pdf.

Main differences with ms:
0) only the demographic parameters are specified (no -t,-T,-r)
1) the flag -I must always be specified
2) flags for continuous migration not implemented
3) an additional flag for ancient DNA:
   -a i t: leaf population i starts at time t, instead of time 0
'''
demo = make_demography("-I 2 5 5"
                       + " -a 2 .01"
                       + " -g 1 .1"
                       + " -es .5 1 .1 -ej .5 4 2"
                       + " -es .1 1 .05 -ej .1 3 2"
                       + " -ej 1.0 2 1 -eg 1.0 1 0")

'''
Variables can be defined in the command-line string
with the "$" sign, and provided as arguments to
make_demography.
'''
demo2 = make_demography("-I 2 5 5"
                        + " -a 2 $t_a"
                        + " -g 1 $g0"
                        + " -es $t1 1 $p1 -ej $t1 4 2"
                        + " -es $t2 1 $p2 -ej $t2 3 2"
                        + " -ej $t3 2 1 -eg $t3 1 0",
                        t_a = .01, g0=.1, t1=.5, p1=.1, t2=.1, p2=.05, t3=1.0)



'''
In ms, the label of a population created by -es
depends on the temporal order in which it was created.
This is inconvenient for parameter estimation because we do not
necessarily know the temporal orders a priori.

To refer to population by its order within the command line,
use #. So in the previous example, replacing "3" by "#4" and 
"4" by "#3" yields the same demography, because $t1 > $t2.
'''
demo3 = make_demography("-I 2 5 5"
                       + " -a 2 $t_a"
                       + " -g 1 $g0"
                       + " -es $t1 1 $p1 -ej $t1 #3 2"
                       + " -es $t2 1 $p2 -ej $t2 #4 2"
                       + " -ej $t3 2 1 -eg $t3 1 0",
                       t_a = .01, g0=.1, t1=.5, p1=.1, t2=.1, p2=.05, t3=1.0)

                      
'''
Now create some configs (samples) to compute the SFS entries for.
Each config is represented by a tuple (d_1,d_2,...), where d_i is
the number of derived alleles in population i.
'''
config_list = [(1,0), (0,1) ,(1,3), (5,0), (0,5), (2,2)]

'''
Now compute the SFS.

sfs = expected branch length with (d_1,d_2) leafs in populations 1,2.
normalizing_constant = expected total branch length

All branch lengths are in ms-scaled units, which is off by a factor of 2
from the more common scaling convention in population genetics.
'''
sfs, normalizing_constant = compute_sfs(demo, config_list)

'''
Check that demo2, demo3 indeed give the same SFS
'''
for other_demo in (demo2,demo3):
    other_sfs, other_normalizing_constant = compute_sfs(other_demo, config_list)
    assert np.all(sfs == other_sfs) and normalizing_constant == other_normalizing_constant

## print the SFS now
print "\nPrinting SFS without error model\n"
def print_sfs(config_list, sfs, normalizing_constant):
    print "\n"
    print "\t".join(["Config", "ExpectedCount", "ConditionalProbability"])
    for c,s in zip(config_list,sfs):
        print "\t".join([str(c), str(s), str(s / normalizing_constant)])
    print "Normalizing constant: ", normalizing_constant
print_sfs(config_list, sfs, normalizing_constant)


'''
Errors & Ascertainment bias

It is often important to account for errors in observations
Either because of poor coverage, or because rare mutations are missed entirely

compute_sfs takes in two arguments relating to errors & sampling bias:
(1) error_matrices: assume independent errors in each leaf population
(2) min_freqs: ignore SNPs with very low frequencies in all subpopulations.

Arguments (1),(2) adjust the sfs entries and the normalizing constant to account
for these errors.

More complex models of error/sampling bias can be handled by using the function
raw_compute_sfs directly, instead of the wrapper function compute_sfs.
'''
print "\n\nPrinting SFS with simple linear error model and minimum allele frequency\n"

# error_matrix_list[leaf_pop][i,j] = Prob(observing i derived in leaf_pop | actually j derived in leaf_pop)
error_matrix_list = []
for leaf in sorted(demo.leaves):
    n_lins = demo.n_lineages(leaf)

    error_mat = np.zeros((n_lins+1, n_lins+1))
    for true_derived in range(n_lins+1):
        # a similar error model to Gravel et al 2010
        # probability of missing a SNP decays exponentially with derived count
        # if SNP isn't missed, then its frequency is gotten exactly
        unobserved_prob = np.exp(-true_derived)
        error_mat[0,true_derived] += unobserved_prob
        error_mat[true_derived, true_derived] += 1.0 - unobserved_prob
    error_matrix_list.append(error_mat)

# to be considered, each allele of SNP must attain frequency >= 3 in leaf pop 1, or frequency >= 1 in leaf pop 2
min_freqs = [3,1]

sfs, normalizing_constant = compute_sfs(demo, config_list, error_matrices = error_matrix_list, min_freqs = min_freqs)

# note a warning is raised because there is a config that does not attain minimum frequency.
# its entry is set to 0

print_sfs(config_list, sfs, normalizing_constant)



