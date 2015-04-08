from __future__ import division
import math
import cmath
from ad import __author__, ADF, to_auto_diff, _apply_chain_rule
import numpy as np

e = math.e

'''math functions. copied from admath, except without the @_vectorize decorator'''
def exp(x):
    """
    Return the exponential value of x
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = exp(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [exp(x)]
        qc_wrt_args = [exp(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [exp(xi) for xi in x]
#        except TypeError:
#         if x.imag:
#             return cmath.exp(x)
#         else:
#             return math.exp(x.real)
        return np.exp(x)


def expm1(x):
    """
    Return e**x - 1. For small floats x, the subtraction in exp(x) - 1 can 
    result in a significant loss of precision; the expm1() function provides 
    a way to compute this quantity to full precision::
        >>> exp(1e-5) - 1  # gives result accurate to 11 places
        1.0000050000069649e-05
        >>> expm1(1e-5)    # result accurate to full precision
        1.0000050000166668e-05
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = expm1(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [exp(x)]
        qc_wrt_args = [exp(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [expm1(xi) for xi in x]
#        except TypeError:
        return np.expm1(x) 

''' edited to not have a base argument (similar to np.log)'''
def log(x):
    if isinstance(x,ADF):
        
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = log(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1./x]
        qc_wrt_args = [-1./x**2]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [log(xi) for xi in x]
#        except TypeError:
#         if x.imag:
#             return cmath.log(x, base)
#         else:
#             return math.log(x.real, base)
        return np.log(x)
