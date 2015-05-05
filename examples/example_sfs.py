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
'''
demo = make_demography("-I 2 5 5"
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
                        + " -g 1 $g0"
                        + " -es $t1 1 $p1 -ej $t1 4 2"
                        + " -es $t2 1 $p2 -ej $t2 3 2"
                        + " -ej $t3 2 1 -eg $t3 1 0",
                        g0=.1, t1=.5, p1=.1, t2=.1, p2=.05, t3=1.0)



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
                       + " -g 1 $g0"
                       + " -es $t1 1 $p1 -ej $t1 #3 2"
                       + " -es $t2 1 $p2 -ej $t2 #4 2"
                       + " -ej $t3 2 1 -eg $t3 1 0",
                       g0=.1, t1=.5, p1=.1, t2=.1, p2=.05, t3=1.0)

                      
'''
Now create some configs (samples) to compute the SFS entries for.
Each config is represented by a tuple (d_1,d_2,...), where d_i is
the number of derived alleles in population i.
'''
config_list = [(1,3), (5,0), (0,5), (2,2)]

'''
Now compute the SFS.

sfs = expected length of branches with (d_1,d_2) leafs in populations 1,2.
branch_len = expected total branch length below TMRCA.

All branch lengths are in ms-scaled units, which is off by a factor of 2
from the more common scaling convention in population genetics.
'''
sfs, branch_len = compute_sfs(demo, config_list)
for other_demo in (demo2,demo3):
    other_sfs, other_branch_len = compute_sfs(other_demo, config_list)
    assert np.all(sfs == other_sfs) and branch_len == other_branch_len

print "SFS entries"
for c,s in zip(config_list,sfs):
    print str(c) + "\t" + str(s)
print "Total branch length", branch_len
