from demography import make_demography
#from maximum_likelihood import LogLikelihoodPRF, fit_log_likelihood_example
from likelihood_surface import fit_log_likelihood_example
from scipy.optimize import basinhopping, minimize
import autograd.numpy as np
from autograd.numpy import log,exp,dot
from autograd import grad, hessian_vector_product

from util import memoize, aggregate_sfs

def simple_human_demo(n,
                      #t_bottleneck_to_africa_split,
                      t_africa_split_to_eurasia_split,
                      t_eurasia_split_to_present,
                      #ancestral_size,
                      africa_size, eurasia_size,
                      eur_present_size, asia_present_size):
    demo_cmd = " ".join(["-I 3 %s" % (" ".join(map(str,n))),
                         "-n 1 $0", # present pop size of africa
                         "-n 2 $1", # present pop size of europe
                         "-n 3 $2", # present pop size of asia
                         "-ej $3 3 2 -en $3 2 $4", # eurasia merge and bottleneck
                         "-ej $5 2 1", # eurasia,africa merge
                         "-en $6 1 $7", # ancestral pop size
                         ])
    ancestral_size=africa_size
    t_bottleneck_to_africa_split=0.0

    eurasia_split = exp(t_eurasia_split_to_present)
    africa_split = eurasia_split + exp(t_africa_split_to_eurasia_split)
    bottleneck = africa_split + exp(t_bottleneck_to_africa_split)

    demo = make_demography(demo_cmd,
                           exp(africa_size),
                           exp(eur_present_size),
                           exp(asia_present_size),
                           eurasia_split, exp(eurasia_size),
                           africa_split,
                           bottleneck, exp(ancestral_size))
    return demo

def check_simple_human_demo():
    #n = [10] * 3
    n = [5] * 3
    #theta = 1.0
    num_sims = 10000
    #true_params = np.exp(np.random.normal(size=6))
    true_x = np.random.normal(size=6)   
    init_x = np.random.normal(size=len(true_x))
    demo_func = lambda x: simple_human_demo(n, *x)

    fit_log_likelihood_example(demo_func, num_sims, true_x, init_x)

if __name__ == "__main__":
    #set_order(2)
    check_simple_human_demo()
