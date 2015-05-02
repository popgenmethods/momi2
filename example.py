from __future__ import division
from demography import make_demography
#from sum_product import mle_estimated_variance
from maximum_likelihood import fit_log_likelihood_example
from scipy.optimize import minimize
import autograd.numpy as np
from autograd.numpy import log,exp,dot
from autograd import grad, hessian_vector_product
import scipy

from util import memoize, aggregate_sfs
from scipy.stats import chi2

def example_admixture_demo(x):
    '''
    An example demography with a pulse migration.

    Function is somewhat involved because population labels
    depend on time of split, and because we choose a parametrization
    s.t. the parameter space is all of \mathbb{R}^8

    Input parameter is
    x = np.array([g2,t3,p4,t5,p6,t7,t8,t9])
    where,
    growth rate $2 = g2
    times $3,$5 = exp(t3-1),exp(t5-1)
    probs $4,$6 = 1/(1+p) for p=p4,p6
    times $7,$8 = $3+exp(t7-1), $5+exp(t8-1)
    time $9 = max($7,$8) + exp(t9-1)
    '''
    ms_cmd = ["-I 2 10 10",
              "-g 1 $2", # pop 1 starts with growth rate $2
              "-es $3 1 $4",  # pop 1 pulses to pop $0 at t=$3, w.p. $4
              "-es $5 2 $6", # pop 2 pulses to pop $1 at t=$5, w.p. $6
              "-ej $7 $0 1", # pops 1 and $0 coalesce at t=$7
              "-ej $8 $1 2", # pops 2 and $1 coalesce at t=$8
              "-ej $9 1 2"] # pops 1 and 2 coalesce at t=$9
    ms_cmd = " ".join(ms_cmd)
    
    g2,t3,p4,t5,p6,t7,t8,t9 = x
    
    t3,t5 = map(exp, (t3-1,t5-1))
    p4,p6 = map(lambda p: 1/(1+exp(p)), (p4,p6))
    t7,t8 = t3 + exp(t7-1), t5 + exp(t8-1)
    t9 = np.maximum(t7,t8) + exp(t9-1)

    pop0,pop1 = 3,4
    if t3 > t5:
        # ms population labels are swapped in this case
        pop0,pop1 = pop1,pop0

    demo =  make_demography(ms_cmd,
                            pop0,pop1,g2,t3,p4,t5,p6,t7,t8,t9)
    return demo, 1.0

true_x = np.random.normal(size=8)
num_sims = 10000
fit_log_likelihood_example(example_admixture_demo, num_sims, true_x, np.random.normal(size=8))
#fit_log_likelihood_example(example_admixture_demo, num_sims, true_x, true_x)
