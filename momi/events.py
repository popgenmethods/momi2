import collections as co
import autograd.numpy as np
import networkx as nx
import msprime
from .math_functions import hypergeom_quasi_inverse, binom_coeffs, _apply_error_matrices, convolve_trailing_axes, sum_trailing_antidiagonals
from .size_history import ConstantHistory, ExponentialHistory, PiecewiseHistory


# FIXME: we always assume default_N=1.0 for now
# NOTE sample_sizes should be an OrderedDict?
def _build_demo_graph(events, sample_sizes, params_dict, default_N):
    _G = nx.DiGraph()
    #_G.graph['event_cmds'] = tuple(events)
    _G.graph['default_N'] = default_N
    _G.graph['events_as_edges'] = []
    # the nodes currently at the root of the graph, as we build it up from the
    # leafs
    _G.graph['roots'] = {}

    for e in events:
        e.add_to_graph(_G, sample_sizes, params_dict)

    assert _G.node
    _G.graph['roots'] = [r for _, r in list(
        _G.graph['roots'].items()) if r is not None]

    if len(_G.graph['roots']) != 1:
        raise DemographyError("Must have a single root population")

    node, = _G.graph['roots']
    _set_sizes(_G.node[node], float('inf'))

    _G.graph['sampled_pops'] = tuple(sample_sizes.keys())
    _G.graph["events"] = tuple(events)
    _G.graph["params"] = co.OrderedDict(params_dict)
    return _G

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


# TODO rename to ScaledParameter
class Parameter(object):
    def __init__(self, name, x0, transform_x,
                 inv_transform_x, x_bounds, rgen):
        self.name = name
        self.x = x0
        self.x_bounds = list(x_bounds)
        self.transform_x = transform_x
        self.inv_transform_x = inv_transform_x
        self.rgen = rgen

        # TODO some sort of test that inv_transform_x is actually the inverse of transform_x

    def copy(self):
        return Parameter(
            name=self.name, x0=self.x,
            transform_x=self.transform_x,
            inv_transform_x=self.inv_transform_x,
            x_bounds=self.x_bounds,
            rgen=self.rgen)

    def resample(self, params):
        self.x = self.inv_transform_x(self.rgen(params),
                                      params)

    def update_params_dict(self, params_dict, x=None):
        if x is None:
            x = self.x
        params_dict[self.name] = self.transform_x(
            x, params_dict)


def get_event_from_old(oldstyle_event):
    flag, t = oldstyle_event[:2]
    rest = oldstyle_event[2:]
    if flag=="-en":
        i,N = rest
        return SizeEvent(t,N,i,1.,.25)
    elif flag=="-eg":
        i,g = rest
        return GrowthEvent(t,g,i,1.,.25)
    elif flag=="-ej":
        i,j = rest
        return JoinEvent(t,i,j,1.,.25)
    elif flag=="-ep":
        i,j,p = rest
        return PulseEvent(t,p,i,j,1.,.25)
    elif flag=="-eSample":
        i,n = rest
        return LeafEvent(t,i,1.,.25)
    else:
        assert False

class LeafEvent(object):
    def __init__(self, t, pop, N_e, gen_time):
        self.t = TimeValue(t, N_e, gen_time)
        self.pop = pop

    def __repr__(self):
        return "LeafEvent(t={}, pop={})".format(
            self.t.x, self.pop)

    def add_to_plot(self, params_dict, demo_plot):
        demo_plot.add_leaf(self.pop, self.t(params_dict, scaled=False))

    def add_to_graph(self, G, sample_sizes, params_dict):
        t=self.t(params_dict)
        i=self.pop
        n=sample_sizes[self.pop]
        G.add_node((i, 0), lineages=n)

        if i in G.graph['roots']:
            if G.graph['roots'][i] is None:
                raise DemographyError(
                    "Invalid events: pop {0} removed by move_lineages before sampling time".format(i))

            #G.node[(i,0)]['model'] = _TrivialHistory()
            G.node[(i, 0)]['sizes'] = [
                {'t': t, 'N': G.graph['default_N'], 'growth_rate':None}]
            _set_sizes(G.node[(i, 0)], t)

            prev = G.graph['roots'][i]
            _set_sizes(G.node[prev], t)

            assert prev[0] == i and prev[1] != 0
            newpop = (i, prev[1] + 1)
            _ej_helper(G, t, (i, 0), prev, newpop)
        else:
            newpop = (i, 0)
            G.node[newpop]['sizes'] = [
                {'t': t, 'N': G.graph['default_N'], 'growth_rate':None}]
        G.graph['roots'][i] = newpop

    def get_msprime_event(self, params_dict, pop_ids_dict):
        return None

def _get_pop_id(pop, pop_ids_dict):
    if pop not in pop_ids_dict:
        pop_ids_dict[pop] = len(pop_ids_dict)
    return pop_ids_dict[pop]

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

    def add_to_graph(self, G, sample_sizes, params_dict):
        t=self.t(params_dict)
        i=self.pop
        N=self.N(params_dict)
        _check_en_eg_pops(G, '-en', t, i, N)
        G.node[G.graph['roots'][i]]['sizes'].append(
            {'t': t, 'N': N, 'growth_rate': None})

    def get_msprime_event(self, params_dict, pop_ids_dict):
        t = self.t(params_dict)
        i = _get_pop_id(self.pop, pop_ids_dict)
        N = self.N(params_dict)
        return msprime.PopulationParametersChange(
            t, initial_size=N / 4, growth_rate=0, population_id=i)


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

    def add_to_graph(self, G, sample_sizes, params_dict):
        t=self.t(params_dict)
        i=self.pop1
        j=self.pop2
        if i not in G.graph['roots']:
            G.graph['roots'][i] = None
            # don't need to do anything else
            return
        _check_ej_ep_pops(G, '-ej', t, i, j)

        i0, j0 = (G.graph['roots'][k] for k in (i, j))
        j1 = (j, j0[1] + 1)
        assert j1 not in G.nodes()

        for k in i0, j0:
            # sets the TruncatedSizeHistory, and N_top and growth_rate for all
            # epochs
            _set_sizes(G.node[k], t)
        _ej_helper(G, t, i0, j0, j1)

        G.graph['roots'][j] = j1
        G.graph['roots'][i] = None

    def get_msprime_event(self, params_dict, pop_ids_dict):
        t = self.t(params_dict)
        i = _get_pop_id(self.pop1, pop_ids_dict)
        j = _get_pop_id(self.pop2, pop_ids_dict)
        return msprime.MassMigration(t, i, j)

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
                                self.p(params_dict, scaled=False),
                                pulse_name=self.p.x)

    def add_to_graph(self, G, sample_sizes, params_dict):
        t=self.t(params_dict)
        i=self.pop1
        j=self.pop2
        pij=self.p(params_dict)
        if pij < 0. or pij > 1.:
            raise DemographyError("Invalid pulse {0} from {1} to {2} at {3}: pulse probability must be between 0,1".format(pij, j, i, t))

        if i not in G.graph['roots']:
            # don't need to do anything
            return

        _check_ej_ep_pops(G, '-ep', t, i, j, pij)

        children = {k: G.graph['roots'][k] for k in (i, j)}
        for v in list(children.values()):
            _set_sizes(G.node[v], t)

        parents = {k: (v[0], v[1] + 1) for k, v in list(children.items())}
        assert all([par not in G.node for par in list(parents.values())])

        prev_sizes = {k: G.node[c]['sizes'][-1] for k, c in list(children.items())}
        for k, s in list(prev_sizes.items()):
            G.add_node(parents[k], sizes=[
                    {'t': t, 'N': s['N_top'], 'growth_rate':s['growth_rate']}])

        G.add_edge(parents[i], children[i], prob=1. - pij)
        G.add_edge(parents[j], children[i], prob=pij)
        G.add_edge(parents[j], children[j])

        new_event = tuple((parents[u], children[v])
                        for u, v in ((i, i), (j, i), (j, j))
                        )
        G.graph['events_as_edges'] += [new_event]

        for k, v in list(parents.items()):
            G.graph['roots'][k] = v

    def get_msprime_event(self, params_dict, pop_ids_dict):
        t = self.t(params_dict)
        i = _get_pop_id(self.pop1, pop_ids_dict)
        j = _get_pop_id(self.pop2, pop_ids_dict)
        pij=self.p(params_dict)

        return msprime.MassMigration(
            t, i, j, proportion=pij)

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

    def add_to_graph(self, G, sample_sizes, params_dict):
        t=self.t(params_dict)
        i=self.pop
        growth_rate=self.g(params_dict)
        _check_en_eg_pops(G, '-eg', t, i, growth_rate)
        G.node[G.graph['roots'][i]]['sizes'].append(
            {'t': t, 'growth_rate': growth_rate})

    def get_msprime_event(self, params_dict, pop_ids_dict):
        t = self.t(params_dict)
        i = _get_pop_id(self.pop, pop_ids_dict)
        g = self.g(params_dict)
        return msprime.PopulationParametersChange(
            t, growth_rate=g, population_id=i)

## helper functions for building demo.
## TODO remove/rename these!!

def _ej_helper(G, t, i0, j0, j1):
    prev = G.node[j0]['sizes'][-1]
    G.add_node(j1, sizes=[{'t': t, 'N': prev['N_top'],
                           'growth_rate':prev['growth_rate']}])

    new_edges = ((j1, i0), (j1, j0))
    G.graph['events_as_edges'].append(new_edges)
    G.add_edges_from(new_edges)


def _check_en_eg_pops(G, *event):
    flag, t, i = event[:3]
    if i in G.graph['roots'] and G.graph['roots'][i] is None:
        raise DemographyError(
            "Invalid set_size event at time {0}: pop {1} was already removed by previous move_lineages".format(t, i))

    if i not in G.graph['roots']:
        G.graph['roots'][i] = (i, 1)
        G.add_node(G.graph['roots'][i],
                   sizes=[{'t': t, 'N': G.graph['default_N'], 'growth_rate':None}],
                   )


def _check_ej_ep_pops(G, *event):
    flag, t, i, j = event[:4]
    for k in (i, j):
        if k in G.graph['roots'] and G.graph['roots'][k] is None:
            raise DemographyError(
                "Invalid move_lineages event at time {0}: pop {1} was already removed by previous move_lineages".format(t, k))

    if j not in G.graph['roots']:
        G.graph['roots'][j] = (j, 1)
        G.add_node(G.graph['roots'][j],
                   sizes=[{'t': t, 'N': G.graph['default_N'], 'growth_rate':None}],
                   )


def _set_sizes(node_data, end_time):
    assert 'model' not in node_data

    # add 'model_func' to node_data, add information to node_data['sizes']
    sizes = node_data['sizes']
    # add a dummy epoch with the end time
    sizes.append({'t': end_time})

    # do some processing
    N, growth_rate = sizes[0]['N'], sizes[0]['growth_rate']
    pieces = []
    for i in range(len(sizes) - 1):
        sizes[i]['tau'] = tau = (sizes[i + 1]['t'] - sizes[i]['t'])

        if 'N' not in sizes[i]:
            sizes[i]['N'] = N
        if 'growth_rate' not in sizes[i]:
            sizes[i]['growth_rate'] = growth_rate
        growth_rate = sizes[i]['growth_rate']
        N = sizes[i]['N']

        if growth_rate is not None and tau != float('inf'):
            pieces.append(ExponentialHistory(
                tau=tau, growth_rate=growth_rate, N_bottom=N))
            N = pieces[-1].N_top
        else:
            if growth_rate != 0. and growth_rate is not None and tau == float('inf'):
                raise DemographyError("Final epoch must have 0 growth rate")
            pieces.append(ConstantHistory(tau=tau, N=N))

        sizes[i]['N_top'] = N

        if not all([sizes[i][x] >= 0.0 for x in ('tau', 'N', 'N_top')]):
            raise DemographyError("Negative time or population size in {sizes}".format(
                sizes=[{k: str(v) for k, v in s.items()} for s in sizes]))
    sizes.pop()  # remove the final dummy epoch

    assert len(pieces) > 0
    if len(pieces) == 0:
        node_data['model'] = pieces[0]
    else:
        node_data['model'] = PiecewiseHistory(pieces)


class DemographyError(Exception):
    pass


## represent values as float, str, or lambda

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

