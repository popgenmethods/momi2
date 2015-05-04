from demography import make_demography
from sum_product import compute_sfs
import numpy as np

'''
An example demography with 2 populations,
with exponential growth at population 1,
and two pulse events from 2 to 1.

Note in ms, the label of a population created by -es
depends on the temporal order in which it was created.
To refer to population by its order within the command line,
use #. So in this example, replacing "#3" by "4" and "#4" by "3",
would yield the same demography, because $t1 > $t2.
'''
demo = make_demography("-I 2 5 5"
                       + " -g 1 $g0"
                       + " -es $t1 1 $p1 -ej $t1 #3 2"
                       + " -es $t2 1 $p2 -ej $t2 #4 2"
                       + " -ej $t3 2 1 -eg $t3 1 0",
                       g0=.1, t1=.5, p1=.1, t2=.1, p2=.05, t3=1.0)
                       
# the same demography, but using the standard ms labeling conventions
demo2 = make_demography("-I 2 5 5"
                        + " -g 1 $g0"
                        + " -es $t1 1 $p1 -ej $t1 4 2"
                        + " -es $t2 1 $p2 -ej $t2 3 2"
                        + " -ej $t3 2 1 -eg $t3 1 0",
                        g0=.1, t1=.5, p1=.1, t2=.1, p2=.05, t3=1.0)


'''
Now create some configs (samples) to compute the SFS entries for.
Each config is represented by a tuple (d_1,d_2), where d_i is
the number of derived alleles in population i.
'''
config_list = [(1,3), (5,0), (0,5), (2,2)]

'''
sfs = expected length of branches with (d_1,d_2) leafs in populations 1,2.
branch_len = expected total branch length below TMRCA.

All branch lengths are in ms-scaled units, which is off by a factor of 2
from the "standard" scaling convention in population genetics.
'''
sfs, branch_len = compute_sfs(demo, config_list)
sfs2, branch_len2 = compute_sfs(demo2, config_list)

assert np.all(sfs == sfs2) and branch_len == branch_len2

print "SFS entries"
for c,s in zip(config_list,sfs):
    print str(c) + "\t" + str(s)
print "Total branch length", branch_len
