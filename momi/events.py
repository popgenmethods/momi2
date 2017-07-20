import collections as co
import autograd.numpy as np
from .math_functions import hypergeom_quasi_inverse, binom_coeffs, _apply_error_matrices, convolve_trailing_axes, sum_trailing_antidiagonals


class ParamsDict(co.OrderedDict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(
            dict(self))

    def __dir__(self):
        return list(self.keys())


class Parameter(object):
    def __init__(self, name, x0, opt_trans, inv_opt_trans,
                 transform_x, x_bounds):
        self.name = name
        self.x = x0
        self.opt_trans = opt_trans
        self.inv_opt_trans = inv_opt_trans
        self.x_bounds = list(x_bounds)
        self.transform_x = transform_x

    def copy(self):
        return Parameter(
            name=self.name, x0=self.x,
            opt_trans=self.opt_trans,
            inv_opt_trans=self.inv_opt_trans,
            transform_x=self.transform_x,
            x_bounds=self.x_bounds)

    @property
    def opt_x_bounds(self):
        opt_x_bounds = []
        for bnd in self.x_bounds:
            if bnd is None:
                opt_x_bounds.append(None)
            else:
                opt_x_bounds.append(self.inv_opt_trans(bnd))
        return opt_x_bounds

    def update_params_dict(self, params_dict, x=None):
        if x is None:
            x = self.x
        params_dict[self.name] = self.transform_x(
            x, params_dict)


class LeafEvent(object):
    def __init__(self, t, pop, N_e, gen_time):
        self.t = TimeValue(t, N_e, gen_time)
        self.pop = pop

    def __repr__(self):
        return "LeafEvent(t={}, pop={})".format(
            self.t.x, self.pop)

    def add_to_plot(self, params_dict, demo_plot):
        demo_plot.add_leaf(self.pop, self.t(params_dict, scaled=False))


class SizeEvent(object):
    def __init__(self, t, N, pop, N_e, gen_time):
        self.t = TimeValue(t, N_e, gen_time)
        self.N = SizeValue(N, N_e)
        self.pop = pop

    def oldstyle_event(self, prm_dict):
        return [("-en", self.t(prm_dict), self.pop, self.N(prm_dict))]

    def __repr__(self):
        return "SizeEvent(t={}, pop={}, N={})".format(
            self.t.x, self.pop, self.N.x)

    def add_to_plot(self, params_dict, demo_plot):
        demo_plot.set_size(self.pop, self.t(params_dict, scaled=False),
                           self.N(params_dict, scaled=False))


class JoinEvent(object):
    def __init__(self, t, pop1, pop2, N_e, gen_time):
        self.t = TimeValue(t, N_e, gen_time)
        self.pop1 = pop1
        self.pop2 = pop2

    def __repr__(self):
        return "JoinEvent(t={}, pop1={}, pop2={})".format(
            self.t.x, self.pop1, self.pop2)

    def oldstyle_event(self, prm_dict):
        return [("-ej", self.t(prm_dict), self.pop1, self.pop2)]

    def add_to_plot(self, params_dict, demo_plot):
        demo_plot.move_lineages(self.pop1, self.pop2,
                                self.t(params_dict, scaled=False), 1)


class PulseEvent(object):
    def __init__(self, t, p, pop1, pop2, N_e, gen_time):
        self.t = TimeValue(t, N_e, gen_time)
        self.p = EventValue(p)
        self.pop1 = pop1
        self.pop2 = pop2

    def __repr__(self):
        s = "PulseEvent(t={}, pop1={}, pop2={}, p={})"
        return s.format(self.t.x, self.pop1, self.pop2,
                        self.p.x)

    def oldstyle_event(self, prm_dict):
        return [("-ep", self.t(prm_dict), self.pop1,
                 self.pop2, self.p(prm_dict))]

    def add_to_plot(self, params_dict, demo_plot):
        demo_plot.move_lineages(self.pop1, self.pop2,
                                self.t(params_dict, scaled=False),
                                self.p(params_dict, scaled=False))

class GrowthEvent(object):
    def __init__(self, t, g, pop, N_e, gen_time):
        self.t = TimeValue(t, N_e, gen_time)
        self.pop = pop
        self.g = RateValue(g, N_e, gen_time)

    def __repr__(self):
        return "GrowthEvent(t={}, pop={}, g={})".format(
            self.t.x, self.pop, self.g.x)

    def oldstyle_event(self, prm_dict):
        return [("-eg", self.t(prm_dict), self.pop,
                 self.g(prm_dict))]

    def add_to_plot(self, params_dict, demo_plot):
        demo_plot.set_growth(self.pop, self.t(params_dict, scaled=False),
                             self.g(params_dict, scaled=False))


class EventValue(object):
    def __init__(self, x):
        self.x = x
        self.scale = 1.0

    def __call__(self, params_dict, scaled=True):
        if isinstance(self.x, str):
            x = params_dict[self.x]
        elif callable(self.x):
            x = self.x(params_dict)
        else:
            x = self.x
        if scaled:
            return x / self.scale
        else:
            return x


class SizeValue(EventValue):
    def __init__(self, N, N_e):
        self.x = N
        self.scale = N_e


class TimeValue(EventValue):
    def __init__(self, t, N_e, gen_time):
        self.x = t
        self.scale = 4.0 * N_e * gen_time


class RateValue(EventValue):
    def __init__(self, r, N_e, gen_time):
        self.x = r
        self.scale = .25 / N_e / gen_time
