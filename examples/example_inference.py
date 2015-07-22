from __future__ import division, print_function
import sys

from momi import simulate_inference, make_demography

import cPickle as pickle

## Thinly-wrapped numpy that supports automatic differentiation
import autograd.numpy as anp

def main():
    if len(sys.argv) > 2:
        raise Exception("Too many command-line arguments.")

    if len(sys.argv) == 2:
        ms_path = sys.argv[1]
    else:
        ## If ms_path is None, momi uses system variable $MS_PATH
        ms_path = None

    pulse_demo_str = " ".join(["-I 2 10 10 -g 1 $growth",
                               "-es $split_t 1 $split_p -ej $split_t 3 2",
                               "-ej $join_t 2 1 -eG $join_t 0.0"])

    #def demo_factory(**x):
    #    return make_demography(pulse_demo_str, **x)

    
    res = simulate_inference(ms_path=ms_path,
                             num_loci=1000,
                             theta=10.0,
                             additional_ms_params='-r 10.0 10000',
                             true_ms_params = transform_pulse_params(anp.array([1.0, -1.0, -1.0, 1.0])),
                             init_opt_params = anp.random.normal(size=4),
                             #demo_factory = demo_factory,
                             demo_factory = pulse_demo_str,
                             transform_params = transform_pulse_params,
                             n_iter = 1,
                             verbosity = 2,
                             n_sfs_dirs = 100)
    with open('example_inference.pickle','wb') as f:
        pickle.dump(res, f)
        
def transform_pulse_params(params):
    '''
    Transforms parameter space that is all of \mathbb{R}^4,
    to the parameter values expected by ms.

    This is useful because it allows us to do the optimization
    on an unconstrained parameter space.

    Note also that only math functions from autograd.numpy used, to allow for
    automatic differentiation.
    
    autograd.numpy supports nearly all functions from numpy, and also 
    makes it easy to define your own differentiable functions as necessary.
    But there are some restrictions:
    1) avoid A.dot(B) notation. Instead use anp.dot(A,B)
    2) avoid in place operations, x += y. Instead use x = x + y
    3) avoid assignment to arrays, e.g. A[0,0] = x breaks if A or x are differentiable
       indexing and slicing are OK though, e.g. x = A[0,0] is fine
    See the autograd tutorial (on its github page) for more details
    '''
    growth, logit_pulse_prob, log_pulse_time, log_join_wait_time = params

    ret = {'growth' : growth,
           'split_t' : anp.exp(log_pulse_time),
           'split_p' : anp.exp(logit_pulse_prob) / (anp.exp(logit_pulse_prob)+1)}
    ret['join_t'] = ret['split_t'] + anp.exp(log_join_wait_time)

    return ret
    
if __name__=="__main__":
    main()
