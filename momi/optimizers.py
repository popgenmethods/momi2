import json
import autograd
import autograd.numpy as np
from functools import wraps, partial
from .util import count_calls, closeleq, closegeq
import scipy
import scipy.optimize
import itertools
import logging

logger = logging.getLogger(__name__)

callback_logger = logging.getLogger(__package__).getChild("progress")


class LoggingCallback(object):

    def __init__(self, user_callback=None):
        self.user_callback = user_callback

    def callback(self, x, fx, i):
        # msg = "Optimization step, {{'it': {i}, 'fun': {fx}, 'x': {x}}}".format(
        #     i=i, x=list(x), fx=fx)
        # logger.info(msg)
        if self.user_callback:
            x = np.array(x)
            x = x.view(LoggingCallbackArray)
            x.iteration = i
            x.fun = fx
            self.user_callback(x)

# special array subclass for LoggingCallback


class LoggingCallbackArray(np.ndarray):
    pass


def _find_minimum(f, start_params, optimizer, bounds=None,
                  callback=None,
                  opt_kwargs={}, **kwargs):
    fixed_params = []
    if bounds is not None:
        bounds = [(None, None) if b is None else b for b in bounds]
        for i, b in enumerate(bounds):
            try:
                if b[0] == b[1] and b[0] is not None:
                    fixed_params += [(i, b[0])]
            except (TypeError, IndexError) as e:
                fixed_params += [(i, b)]

        if any(start_params[i] != b for i, b in fixed_params):
            raise ValueError(
                "start_params does not agree with fixed parameters in bounds")

        if fixed_params:
            fixed_idxs, fixed_offset = list(
                map(np.array, list(zip(*fixed_params))))

            fixed_idxs = np.array([(i in fixed_idxs)
                                   for i in range(len(start_params))])
            proj0 = np.eye(len(fixed_idxs))[:, fixed_idxs]
            proj1 = np.eye(len(fixed_idxs))[:, ~fixed_idxs]

            fixed_offset = np.dot(proj0, fixed_offset)
            get_x = lambda x0: np.dot(proj1, x0) + fixed_offset

            def restricted(fun):
                if fun is None:
                    return None
                new_fun = lambda x0, * \
                    fargs, **fkwargs: fun(get_x(x0), *fargs, **fkwargs)
                return wraps(fun)(new_fun)
            f = restricted(f)
            callback = restricted(callback)

            start_params = np.array(
                [s for (fxd, s) in zip(fixed_idxs, start_params) if not fxd])
            bounds = [b for (fxd, b) in zip(fixed_idxs, bounds) if not fxd]

    opt_kwargs = dict(opt_kwargs)
    assert all([k not in opt_kwargs for k in ['bounds', 'callback']])
    if callback:
        opt_kwargs['callback'] = callback
    if bounds is not None and not all([l is None and h is None for (l, h) in bounds]):
        opt_kwargs['bounds'] = bounds

    ret = _find_minimum_helper(
        f, start_params, optimizer, opt_kwargs, **kwargs)
    if fixed_params:
        ret.x = get_x(ret.x)
    return ret


def _find_minimum_helper(f, start_params, optimizer,
                         opt_kwargs={}, gradmakers={}, replacefun=None):
    opt_kwargs = dict(opt_kwargs)
    for k, v in gradmakers.items():
        assert k not in opt_kwargs
        opt_kwargs[k] = v(f)
    if replacefun:
        f = replacefun(f)
    return optimizer(f, start_params, **opt_kwargs)

stochastic_opts = {}


def is_stoch_opt(fun):
    stochastic_opts[fun.__name__] = fun
    return fun


@is_stoch_opt
def sgd(fun, x0, fun_and_jac, pieces, stepsize, num_iters, bounds=None, callback=None, iter_per_output=10, rgen=np.random):
    x0 = np.array(x0)

    if callback is None:
        callback = lambda *a, **kw: None

    if bounds is None:
        bounds = [(None, None) for _ in x0]
    lower, upper = zip(*bounds)
    lower = [-float('inf') if l is None else l
             for l in lower]
    upper = [float('inf') if u is None else u
             for u in upper]

    def truncate(x):
        return np.maximum(np.minimum(x, upper), lower)

    x = x0
    for nit in range(num_iters):
        i = rgen.randint(pieces)
        f_x, g_x = fun_and_jac(x, i)
        x = truncate(x - stepsize * g_x)
        if nit % iter_per_output == 0:
            callback(x, f_x, nit)

    return scipy.optimize.OptimizeResult({'x': x, 'fun': f_x, 'jac': g_x})


@is_stoch_opt
def adam(fun, x0, fun_and_jac, pieces, num_iters, stepsize=.1, b1=0.9, b2=0.999, eps=10**-8, svrg_epoch=-1, bounds=None, callback=None, rgen=np.random, xtol=1e-6, w=None, fbar=None, gbar=None, checkpoint_file=None, checkpoint_iter=10, start_iter=0, m=None, v=None):
    x0 = np.array(x0)

    if callback is None:
        callback = lambda *a, **kw: None

    if bounds is None:
        bounds = [(None, None) for _ in x0]
    lower, upper = zip(*bounds)
    lower = [-float('inf') if l is None else l
             for l in lower]
    upper = [float('inf') if u is None else u
             for u in upper]

    def truncate(x):
        return np.maximum(np.minimum(x, upper), lower)

    x = x0
    if m is None:
        m = np.zeros(len(x))
    else:
        m = np.array(m)
    if v is None:
        v = np.zeros(len(x))
    else:
        v = np.array(v)

    if w is not None:
        w = np.array(w)
    if gbar is not None:
        gbar = np.array(gbar)

    prev_close = False
    success = False
    for nit in range(start_iter, num_iters):
        i = rgen.randint(pieces)
        f_x, g_x = fun_and_jac(x, i)

        if svrg_epoch > 0 and nit // svrg_epoch and nit % svrg_epoch == 0:
            w = x
            fbar, gbar = fun_and_jac(w, None)
            #logger.info("SVRG pivot, {0}".format(
            #    {"w": list(w), "fbar": fbar, "gbar": list(gbar)}))
        if w is not None:
            f_w, g_w = fun_and_jac(w, i)
            f_x = f_x - f_w + fbar
            g_x = g_x - g_w + gbar

        callback(x, f_x, nit)

        m = (1 - b1) * g_x + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g_x**2) + b2 * v  # Second moment estimate.

        mhat = m / (1 - b1**(nit + 1))    # Bias correction.
        vhat = v / (1 - b2**(nit + 1))

        prev_x = x
        x = truncate(x - stepsize * mhat / (np.sqrt(vhat) + eps))
        #logger.info("Adam moment estimates, {0}".format(
        #    {"x": list(x), "m": list(m), "v": list(v)}))

        # require x to not change for 2 steps in a row before stopping
        if xtol < 0 or not np.allclose(x, prev_x, xtol, xtol):
            prev_close = False
        elif prev_close:
            success = True
            break
        else:
            prev_close = True

        if checkpoint_file is not None and nit % checkpoint_iter == 0:
            def to_list(y):
                if y is None:
                    return None
                else:
                    return list(y)

            with open(checkpoint_file, "w") as f:
                json.dump({
                    "start_iter": nit+1,
                    "fbar": fbar,
                    "gbar": to_list(gbar),
                    "w": to_list(w),
                    "m": to_list(m),
                    "v": to_list(v),
                    "x0": to_list(x)}, f)


    if success:
        message = "|x[k]-x[k-1]|~=0"
    else:
        message = "Maximum number of iterations reached"

    return scipy.optimize.OptimizeResult({'x': x, 'fun': f_x, 'jac': g_x, 'nit': nit, 'message': message,
                                          'success': success})


@is_stoch_opt
def svrg(fun, x0, fun_and_jac, pieces, stepsize, iter_per_epoch, max_epochs=100, bounds=None, callback=None, rgen=np.random, quasinewton=True, init_epoch_svrg=False, xtol=1e-6):
    x0 = np.array(x0)

    if quasinewton is not True and quasinewton is not False:
        init_H = quasinewton
        quasinewton = True
    else:
        init_H = None

    if callback is None:
        callback = lambda *a, **kw: None

    if bounds is None:
        bounds = [(None, None) for _ in x0]
    lower, upper = zip(*bounds)
    lower = [-float('inf') if l is None else l
             for l in lower]
    upper = [float('inf') if u is None else u
             for u in upper]

    def truncate(x):
        return np.maximum(np.minimum(x, upper), lower)

    def update_Hess(H, new_x, prev_x, new_g, prev_g):
        if np.allclose(new_x, prev_x):
            return H

        s = new_x - prev_x
        y = new_g - prev_g
        sy = np.dot(s, y)
        Bs = np.linalg.solve(H, s)

        y_Bs = y - Bs
        if np.abs(np.dot(s, y_Bs)) < 1e-8 * np.linalg.norm(s) * np.linalg.norm(y_Bs):
            # skip SR1 update
            return H

        Hy = np.dot(H, y)
        s_Hy = s - Hy
        H = H + np.outer(s_Hy, s_Hy) / np.dot(s_Hy, y)
        return H

    I = np.eye(len(x0))

    finished = False
    x = x0
    nit = 0
    history = {k: [] for k in ('x', 'f', 'jac')}
    for epoch in itertools.count():
        if epoch > 0:
            prev_w = w
            prev_gbar = gbar
        else:
            prev_w, prev_gbar = None, None

        w = x
        if epoch > 0 or init_epoch_svrg is True:
            fbar, gbar = fun_and_jac(w, None)
            logger.info("SVRG pivot, {0}".format(
                {"w": list(w), "fbar": fbar, "gbar": list(gbar)}))
            #callback(w, fbar, epoch)
            for k, v in (('x', w), ('f', fbar), ('jac', gbar)):
                history[k].append(v)
        elif init_epoch_svrg is False:
            gbar = None
        else:
            fbar, gbar = init_epoch_svrg

        if quasinewton:
            if prev_gbar is not None:
                H = update_Hess(H, w, prev_w, gbar, prev_gbar)
            else:
                assert epoch == 0 or (epoch == 1 and not init_epoch_svrg)
                if init_H is not None:
                    H = init_H
                else:
                    f_w, g_w = fun_and_jac(w, 0)
                    if epoch == 0:
                        u = truncate(w - g_w)
                    else:
                        u = prev_w
                    f_u, g_u = fun_and_jac(u, 0)
                    s, y = (u - w), (g_u - g_w)
                    H = np.abs(np.dot(s, y) / np.dot(y, y)) * I

            H_eigvals, H_eigvecs = scipy.linalg.eigh(H)
            Habs = np.einsum("k,ik,jk->ij",
                             np.abs(H_eigvals),
                             H_eigvecs, H_eigvecs)
            Babs = scipy.linalg.pinvh(Habs)

        if epoch > 0 and xtol >= 0 and np.allclose(w, prev_w, xtol, xtol):
            success = True
            message = "|x[k]-x[k-1]|~=0"
            break
        if epoch >= max_epochs:
            success = False
            message = "Maximum number of iterations reached"
            break

        for k in range(iter_per_epoch):
            i = rgen.randint(pieces)

            f_x, g_x = fun_and_jac(x, i)

            if gbar is not None:
                f_w, g_w = fun_and_jac(w, i)
                g = (g_x - g_w) + gbar
                f = (f_x - f_w) + fbar
            else:
                assert epoch == 0 and init_epoch_svrg is False
                g = g_x
                f = f_x
            callback(x, f, nit)

            if not quasinewton:
                xnext = truncate(x - stepsize * g)
            else:
                xnext = x - stepsize * np.dot(Habs, g)
                if not np.allclose(xnext, truncate(xnext)):
                    model_fun = lambda y: np.dot(
                        g, y - x) + .5 / stepsize * np.dot(y - x, np.dot(Babs, y - x))
                    model_grad = autograd.grad(model_fun)
                    xnext = scipy.optimize.minimize(
                        model_fun, x, jac=model_grad, bounds=bounds).x
                xnext = truncate(xnext)
            x = xnext
            nit += 1

    history = {k: np.array(v) for k, v in history.items()}
    res = scipy.optimize.OptimizeResult({'success': success, 'message': message,
                                         'nit': nit, 'nepoch': epoch, 'history': history,
                                         'x': x, 'fun': fbar, 'jac': gbar})
    if quasinewton:
        res['hess_inv'] = H

    return res
