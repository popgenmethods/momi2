import autograd
import autograd.numpy as np
import numdifftools
from functools import wraps, partial
from .util import logger, count_calls, closeleq, closegeq
import scipy, scipy.optimize
import itertools

def _find_minimum(f, start_params, optimizer, bounds=None,
                  callback=None,
                  opt_kwargs={}, **kwargs):
    fixed_params = []
    if bounds:
        bounds = [(None,None) if b is None else b for b in bounds]
        for i,b in enumerate(bounds):
            try:
                if b[0] == b[1] and b[0] is not None:
                    fixed_params += [(i,b[0])]
            except (TypeError,IndexError) as e:
                fixed_params += [(i,b)]
                    
        if any(start_params[i] != b for i,b in fixed_params):
            raise ValueError("start_params does not agree with fixed parameters in bounds")
        
        if fixed_params:
            fixed_idxs, fixed_offset = list(map(np.array, list(zip(*fixed_params))))

            fixed_idxs = np.array([(i in fixed_idxs) for i in range(len(start_params))])
            proj0 = np.eye(len(fixed_idxs))[:,fixed_idxs]
            proj1 = np.eye(len(fixed_idxs))[:,~fixed_idxs]

            fixed_offset = np.dot(proj0, fixed_offset)
            get_x = lambda x0: np.dot(proj1, x0) + fixed_offset
            def restricted(fun):
                if fun is None: return None
                new_fun = lambda x0, *fargs, **fkwargs: fun(get_x(x0), *fargs, **fkwargs)
                return wraps(fun)(new_fun)
            f = restricted(f)
            callback = restricted(callback)
            
            start_params = np.array([s for (fxd,s) in zip(fixed_idxs,start_params) if not fxd])
            bounds = [b for (fxd,b) in zip(fixed_idxs, bounds) if not fxd]

    opt_kwargs = dict(opt_kwargs)
    assert all([k not in opt_kwargs for k in ['bounds','callback']])
    if callback: opt_kwargs['callback'] = callback
    if bounds and not all([l is None and h is None for (l,h) in bounds]):
        opt_kwargs['bounds'] = bounds

    ret = _find_minimum_helper(f, start_params, optimizer, opt_kwargs, **kwargs)
    if fixed_params:
        ret.x = get_x(ret.x)
    return ret

def _find_minimum_helper(f, start_params, optimizer,
                       opt_kwargs={}, gradmakers={}, replacefun=None):
    opt_kwargs = dict(opt_kwargs)
    for k,v in gradmakers.items():
        assert k not in opt_kwargs
        opt_kwargs[k] = v(f)
    if replacefun:
        f = replacefun(f)
    return optimizer(f, start_params, **opt_kwargs)
             
def nesterov(fun, x0, fun_and_jac, maxiter=1000, bounds=None, callback=None):
    if callback is None:
        callback = lambda *a,**kw:None
    
    if bounds is None:
        bounds = [(None,None)]*len(x0)
    old_bounds, bounds = bounds, []
    for lower, upper in old_bounds:
        if lower is None: lower = -float('inf')
        if upper is None: upper = float('inf')
        bounds.append((lower,upper))
    lower, upper = list(map(np.array, zip(*bounds)))
    def truncate(x): return np.maximum(np.minimum(x, upper), lower)

    fun = count_calls(fun)

    def backtrack_linesearch(y,stepsize,reverse=False):
        ## if reverse==True, do a "forward" line search for maximum stepsize satisfying condition
        fy,gy = fun_and_jac(y)
        while True:
            x = truncate(y - gy*stepsize)
            fx = fun(x)
            if any([not np.isfinite(fz) for fz in (fx,fy)]):
                raise ValueError("Non-finite value of objective function")
            condition = (fx <= fy + 0.5 * np.dot(gy, x-y))
            if reverse: condition = not condition
            if condition: break
            else:
                if reverse: stepsize = 2.0*stepsize
                else: stepsize = 0.5*stepsize
        return x,fx,gy,stepsize
    
    x=np.array(x0,dtype=float)
    y=np.array(x)
    callback(x,fun(x),0)

    ## get initial stepsize
    _,_,_,stepsize = backtrack_linesearch(y,1.0,reverse=True)   
    nesterov_it=0
    for nit in range(1,maxiter+1):
        prev_x = x

        ## update x with backtracking linesearch
        x,fx,gy,stepsize = backtrack_linesearch(y, stepsize)
        callback(x,fx,nit)
        
        ## check convergence
        if np.allclose(y,x):
            success=True
            message="jac(x[k])=~0"
            break        
        if np.allclose(x,prev_x):
            success=True
            message="|x[k]-x[k-1]|=~0"
            break
        
        ## check if restart nesterov
        if np.dot(gy, x - prev_x) > 0:
            nesterov_it=0
            prev_x = x

        ## add nesterov momentum
        y = truncate(x + (nesterov_it)/float(nesterov_it+2)*(x-prev_x))
        nesterov_it+=1

        ## allow stepsize to grow
        stepsize = stepsize * 2.0
    try: success,message
    except NameError:
        success=False
        message="Maximum number of iterations reached"
    return scipy.optimize.OptimizeResult({'success':success, 'message':message,
                                          'nfev':fun.num_calls, 'nit':nit,
                                          'x':x, 'fun':fx, 'jac':gy})


def svrg(fun, x0, fun_and_jac, pieces, iter_per_epoch, maxiter=1000, bounds=None, callback=None, rgen=np.random, stepsize=1./16., rescale_stepsize=.5, lbfgs=10, only_stop_on_epoch=True):
    logger.info("Doing SVRG with initial stepsize %f and %d L-BFGS components" % (stepsize, lbfgs))
    if rescale_stepsize < 0: rescale_stepsize = False
    assert iter_per_epoch <= pieces
    
    x0 = np.array(x0)
   
    if callback is None:
        callback = lambda *a,**kw:None
        
    if bounds is None:
        bounds = [(None,None) for _ in x0]
    lower,upper = zip(*bounds)
    lower = [-float('inf') if l is None else l
             for l in lower]
    upper = [float('inf') if u is None else u
             for u in upper]
   
    def truncate(x):
        return np.maximum(np.minimum(x, upper), lower)

    def get_r(s,y,B):
        sy = np.dot(s,y)
        Bs = np.dot(B,s)
        sBs = np.dot(s, Bs)
        if sy >= .2 * sBs:
            theta = 1.
        else:
            theta = .8 * sBs / (sBs - sy)
        r = theta*y + (1.-theta)*Bs
        return r, Bs, sBs

    sy_list = []
    def update_Hess(B, new_x, prev_x, new_g, prev_g):       
        s = new_x-prev_x
        y = new_g-prev_g

        sy_list.append((s,y))
        if len(sy_list) >= lbfgs:
            del sy_list[0]
        
        r,_,_ = get_r(s,y,B)
        B = np.dot(r,r) / np.dot(s,r) * I

        for (s,y) in sy_list:
            r,Bs,sBs = get_r(s,y,B)
            B = B - np.outer(Bs,Bs)/sBs + np.outer(r,r) / np.dot(s,r)        
        return B

    I = np.eye(len(x0))
       
    finished = False
    x = x0
    nit = 0
    for epoch in itertools.count():
        if finished: break
        if epoch > 0:
            prev_w = w
            prev_gbar = gbar
            prev_fbar = fbar
            
        w = x
        fbar, gbar = fun_and_jac(w, None)
       
        if lbfgs >= 0:
            if epoch > 0 and not np.allclose(w, prev_w):
                B = update_Hess(B, w, prev_w, gbar, prev_gbar)
            else:
                u = truncate(w - gbar)
                f_u, g_u = fun_and_jac(u,0)
                f_w, g_w = fun_and_jac(w,0)
                g = pieces*(g_u - g_w) + gbar
                s, y = (u-w), (g-gbar)
                r,_,_ = get_r(s,y,I)
                B = np.dot(r,r) / np.dot(s,r) * I
            H = scipy.linalg.pinvh(B)
        else:
            H = I
        
        if epoch > 0 and rescale_stepsize and fbar > prev_fbar:
            stepsize *= rescale_stepsize 
            logger.info("Detected divergence, decreasing stepsize to %f" % stepsize)
            x = truncate(x_avg)
            f_autoregressive = f_avg
        else:
            f_autoregressive = fbar
        
        start = x
        x_avg, f_avg = 0,0
        for k in range(iter_per_epoch):
            i = rgen.randint(pieces)
            
            f_w, g_w = fun_and_jac(w,i)
            f_x, g_x = fun_and_jac(x,i)

            f = pieces*(f_x - f_w) + fbar            
            g = pieces*(g_x - g_w) + gbar
                       
            xnext = truncate(x - stepsize*np.dot(H,g))
                
            ## print a running (autoregressive) average of f
            f_autoregressive = f * .1 + f_autoregressive * .9
            callback(x, f_autoregressive, nit)
            #logger.info("stepsize: %g" % stepsize)

            xprev = x
            x = xnext

            x_avg = (x_avg*k + x)/float(k+1)
            f_avg = (f_avg*k + f)/float(k+1)
            
            nit += 1
            if nit >= maxiter:
                finished = True
                break

            #if np.allclose(xprev,x) and (k == 0 or not only_stop_on_epoch):
            if np.allclose(xprev, truncate(xprev-g)) and (k == 0 or not only_stop_on_epoch):
                finished=True
                success=True
                message="|x[k]-x[k-1]|~=0"
                break
            

    try: success,message
    except NameError:
        success=False
        message="Maximum number of iterations reached"
    return scipy.optimize.OptimizeResult({'success':success, 'message':message,
                                          'nit':nit,
                                          'x':x, 'fun':fbar, 'jac':gbar})
    
    
custom_opts = {"nesterov": nesterov}
stochastic_opts = {"svrg": svrg}



# def stoch_avg_grad(fun, x0, fun_and_jac, pieces, maxiter=1000, bounds=None, callback=None, rgen=np.random):
#     if callback is None:
#         callback = lambda *a,**kw:None
        
#     if bounds is None:
#         bounds = [(None,None) for _ in x0]
#     lower,upper = zip(*bounds)
#     lower = [-float('inf') if l is None else l
#              for l in lower]
#     upper = [float('inf') if u is None else u
#              for u in upper]

#     curr_funs = np.zeros(pieces)
#     sumfun = 0.0
    
#     curr_grads = np.zeros((pieces, len(x0)))
#     sumgrad = np.zeros(len(x0))
    
#     def truncate(x, truncate_grads=False):
#         ret = np.maximum(np.minimum(x, upper), lower)
#         isntclose = np.logical_not(np.isclose(ret, x))
#         if truncate_grads and np.any(isntclose):
#             curr_grads[:,isntclose] = 0.0
#             sumgrad[isntclose] = 0.0
#         return ret

#     def backtrack_linesearch(y, i, stepsize, reverse=False):
#         ## if reverse==True, do a "forward" line search for maximum stepsize satisfying condition
#         fy,gy = fun_and_jac(y,i)
#         while True:
#             x = truncate(y - gy*stepsize)
#             fx = fun(x,i)
#             if any([not np.isfinite(fz) for fz in (fx,fy)]):
#                 raise ValueError("Non-finite value of objective function")            
#             condition = (fx <= fy + 0.5 * np.dot(gy, x-y))
#             if reverse: condition = not condition
#             if condition: break
#             else:
#                 if reverse: stepsize = 2.0*stepsize
#                 else: stepsize = 0.5*stepsize
#         return (fy,gy), stepsize

#     x = np.array(x0, dtype=float)
#     ## get the initial stepsize
#     _,stepsize = backtrack_linesearch(x, 0, 1.0, reverse=True)

#     ## the initial pass
#     for nit in range(pieces):
#         (f,g),stepsize = backtrack_linesearch(x, nit, stepsize)
#         curr_funs[nit] = f
#         sumfun += f
#         curr_grads[nit,:] = g
#         sumgrad += g
       
#         x = truncate(x - stepsize / float(nit+1) * sumgrad, truncate_grads=True)
#         callback(x, sumfun*float(pieces)/(nit+1.0), nit)        
#         stepsize *= 2.0**(1.0/pieces)

#     ## regular updates
#     finished = False
#     while not finished:
#         #for i in rgen.permutation(pieces):
#         for i in rgen.randint(pieces, size=pieces):
#             nit += 1
#             if nit >= maxiter:
#                 finished = True
#                 break

#             (f,g),stepsize = backtrack_linesearch(x, i, stepsize)

#             sumfun -= curr_funs[i]
#             curr_funs[i] = f
#             sumfun += f

#             ##step = sumgrad + pieces*(g - curr_grads[i,:]) # SAGA update
            
#             sumgrad -= curr_grads[i,:]
#             curr_grads[i,:] = g
#             sumgrad += g

#             prev_x = x
#             x = truncate(x - stepsize / float(pieces) * sumgrad, truncate_grads=True)
#             #x = truncate(x - stepsize * step, truncate_grads=True)
#             callback(x, sumfun, nit)
#             stepsize *= 2.0**(1.0/pieces)

#             if np.allclose(prev_x,x):
#                 finished=True
#                 success=True
#                 message="|x[k]-x[k-1]|~=0"
#                 break

#     try: success,message
#     except NameError:
#         success=False
#         message="Maximum number of iterations reached"
#     return scipy.optimize.OptimizeResult({'success':success, 'message':message,
#                                           'nit':nit,
#                                           'x':x, 'fun':sumfun, 'jac':sumgrad})        

# def adam(fun_and_jac_list, start_params, maxiter, bounds,
#          tol=None,
#          random_generator=np.random, step_size=1., b1=0.9, b2=0.999, eps=10**-8):
#     """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
#     It's basically RMSprop with momentum and some correction terms."""
#     lower_bounds, upper_bounds = list(zip(*bounds))
#     if tol is not None:
#         raise NotImplementedError("tol not yet implemented")
    
#     x = start_params    
#     m = np.zeros(len(x))
#     v = np.zeros(len(x))

#     step_size = step_size / float(len(fun_and_jac_list))
    
#     history = OptimizeHistory()
#     for curr_pass in range(maxiter):
#         history.new_batch()
#         for i in random_generator.permutation(len(fun_and_jac_list)):
#             fun_and_jac = fun_and_jac_list[i]
            
#             f,g = fun_and_jac(x)
#             history.update(x,f,g)

#             m = (1 - b1) * g      + b1 * m  # First  moment estimate.
#             v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
#             mhat = m / float(1 - b1**(i + 1))    # Bias correction.
#             vhat = v / float(1 - b2**(i + 1))
#             x = x-step_size*mhat/(np.sqrt(vhat) + eps)
#             x = np.maximum(np.minimum(x, upper_bounds), lower_bounds)
#     return scipy.optimize.OptimizeResult({'x':x, 'fun':f, 'jac':g, 'history':history})

# def adadelta(fun_and_jac_list, start_params, maxiter, bounds,
#              tol=None,
#              random_generator=np.random, rho=.95, eps=1e-6):
#     lower_bounds, upper_bounds = list(zip(*bounds))
#     if tol is not None:
#         raise NotImplementedError("tol not yet implemented")
    
#     x = start_params    
#     EG2 = 0.
#     EDelX2 = 0.

#     history = OptimizeHistory()
#     for curr_pass in range(maxiter):
#         history.new_batch()
#         for i in random_generator.permutation(len(fun_and_jac_list)):
#             fun,jac = fun_and_jac_list[i]
            
#             f = fun(x)
#             g = jac(x)
#             history.update(x,f,g)

#             EG2 = rho * EG2 + (1.-rho) * (g**2)
#             stepsize = - np.sqrt(EDelX2 + eps) / np.sqrt(EG2 + eps) * g

#             EDelX2 = rho * EDelX2 + (1.-rho) * (stepsize**2)
            
#             x = x-stepsize
#             x = np.maximum(np.minimum(x, upper_bounds), lower_bounds)
#     return scipy.optimize.OptimizeResult({'x':x, 'fun':f, 'jac':g, 'history':history})

# class OptimizeHistory(object):
#     def __init__(self):
#         self.x = []
#         self.f = []
#         self.g = []

#     def new_batch(self):
#         self.x += [[]]
#         self.f += [[]]
#         self.g += [[]]

#     def update(self, x,f,g):
#         self.x[-1] += [x]
#         self.f[-1] += [f]
#         self.g[-1] += [g]
