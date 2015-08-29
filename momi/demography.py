from __future__ import division
import networkx as nx
from util import default_ms_path, memoize_instance, memoize
from math_functions import einsum2, sum_antidiagonals, convolve_axes
import scipy, scipy.misc
import autograd.numpy as np

from size_history import ConstantHistory, ExponentialHistory, PiecewiseHistory

import os, itertools
from operator import itemgetter

class Demography(nx.DiGraph):
    def __init__(self, demo, *args, **kwargs):
        if isinstance(demo, nx.DiGraph):
            ## make a copy of demography object
            if len(args) != 0 or len(kwargs) != 0:
                raise ValueError("Too many arguments for copying Demography object")
            super(Demography, self).__init__(demo)           
        else:
            ## construct demography from string
            demo = _demo_graph_from_str(demo, args, kwargs)
            super(Demography, self).__init__(demo)
        ## set leaves and event_tree
        self.leaves = set([k for k, v in self.out_degree().items() if v == 0])
        self.event_tree = _build_event_tree(self)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Demography('%s')" % self.graph['cmd']

    @memoize_instance
    def n_lineages(self, node):
        return np.sum(self.node[l]['lineages'] for l in self.leaves_subtended_by(node))

    @property
    def n_at_leaves(self):
        return tuple(self.n_lineages(l) for l in sorted(self.leaves))

    @memoize_instance
    def leaves_subtended_by(self, node):
        return self.leaves & set(nx.dfs_preorder_nodes(self, node))

    def truncated_sfs(self, node):
        '''The truncated SFS at node.'''
        return self.node[node]['model'].sfs(self.n_lineages(node))

    def apply_transition(self, node, array, axis):
        '''Apply Moran model transition at node to array along axis.'''
        return self.node[node]['model'].transition_prob(array, axis)
   
    @property
    def root(self):
        '''The root (ancestral) population.'''
        ret, = self.parent_pops(self.event_root)
        return ret

    @property
    def event_root(self):
        '''The root of the junction tree.'''
        return self.event_tree.root

    def event_type(self, event):
        if len(event) == 1:
            return 'leaf'
        elif len(self.parent_pops(event)) == 2:
            return 'admixture'
        elif len(self.event_tree[event]) == 2:
            return 'merge_clusters'
        else:
            return 'merge_subpops'

    def sub_pops(self, event):
        '''
        The group of subpopulations corresponding to this event in the junction tree.
        '''
        return self.event_tree.node[event]['subpops']

    def parent_pops(self, event):
        '''The populations arising due to this event, backwards in time.'''
        return self.event_tree.node[event]['parent_pops']

    def child_pops(self, event):
        '''
        Returns dict of 
        {child_pop : child_event},
        which gives populations arising from this event forward in time,
        and the corresponding child events in the junction tree.
        '''
        return self.event_tree.node[event]['child_pops']
   
    @memoize_instance
    def admixture_prob(self, admixture_node):
        '''
        Array with dim [n_admixture_node+1, n_parent1_node+1, n_parent2_node+1],
        giving probability of derived counts in child, given derived counts in parents
        '''
        n_node = self.n_lineages(admixture_node)

        # admixture node must have two parents
        edge1,edge2 = self.in_edges([admixture_node], data=True)
        nd = self.node[admixture_node]
        parent1,parent2 = edge1[0], edge2[0]
        prob1,prob2 = nd['splitprobs'][parent1], nd['splitprobs'][parent2]
        assert prob1 + prob2 == 1.0

        n_from_1 = np.arange(n_node+1)
        n_from_2 = n_node - n_from_1
        binom_coeffs = (prob1**n_from_1) * (prob2**n_from_2) * scipy.misc.comb(n_node, n_from_1)
        ret = einsum2(der_in_admixture_node(n_node), range(4),
                      binom_coeffs, [0],
                      [1,2,3])
        assert ret.shape == tuple([n_node+1] * 3)
        return ret, [admixture_node, parent1, parent2]

@memoize
def der_in_admixture_node(n_node):
    '''
    returns 4d-array, [n_from_parent1, der_in_child, der_in_parent1, der_in_parent2].
    Used by Demography.admixture_prob
    '''
    # axis0=n_from_parent, axis1=der_from_parent, axis2=der_in_parent
    der_in_parent = np.tile(np.arange(n_node+1), (n_node+1,n_node+1,1))
    n_from_parent = np.transpose(der_in_parent, [2,0,1])
    der_from_parent = np.transpose(der_in_parent, [0,2,1])
    
    anc_in_parent = n_node - der_in_parent
    anc_from_parent = n_from_parent - der_from_parent
    
    x = scipy.misc.comb(der_in_parent, der_from_parent) * scipy.misc.comb(anc_in_parent, anc_from_parent) / scipy.misc.comb(n_node, n_from_parent)

    ret,labels = convolve_axes(x, x[::-1,...], [[c for c in 'ijk'], [c for c in 'ilm']], ['j','l'], 'n')
    return np.einsum('%s->inkm' % ''.join(labels), ret[...,:(n_node+1)])


def _build_event_tree(demo):
    def node_time(v):
        return demo.node[v]['sizes'][0]['t']
    
    eventEdgeList = []
    currEvents = {l : (l,) for l in demo.leaves}
    eventDict = {e : {'subpops' : (l,), 'parent_pops' : (l,), 'child_pops' : {}, 't' : node_time(l)} for l,e in currEvents.iteritems()}

    for e in demo.graph['events']:
        # get the population edges forming the event
        parent_pops, child_pops = map(set, zip(*e))
        child_events = set([currEvents[c] for c in child_pops])
        assert len(e) == 2 and len(parent_pops) + len(child_pops) == 3 and len(child_events) in (1,2)

        sub_pops = set(itertools.chain(*[eventDict[c]['subpops'] for c in child_events]))
        sub_pops.difference_update(child_pops)
        sub_pops.update(parent_pops)

        t = node_time(list(parent_pops)[0])
        for p in parent_pops:
            ## TODO: remove try/except when autograd fixes assertion bug
            try:
                assert np.isclose(node_time(p), t)
            except TypeError:
                pass
        
        eventDict[e] = {'parent_pops' : tuple(parent_pops), 'subpops' : tuple(sub_pops), 'child_pops' : {c : currEvents[c] for c in child_pops}, 't' : t}
        currEvents.update({p : e for p in sub_pops})
        for p in child_pops:
            del currEvents[p]
        eventEdgeList += [(e, c) for c in child_events]
    ret = nx.DiGraph(eventEdgeList)
    for e in eventDict:
        ret.add_node(e, **(eventDict[e]))

    assert len(currEvents) == 1
    root, = [v for k,v in currEvents.iteritems()]
    ret.root = root

    return ret

### methods for constructing demography from string
## TODO: clean up this code!

def _get_cmd_list(cmd):
    cmd_list = []
    for arg in cmd.split():
        if arg[0] == '-' and arg[1].isalpha():
            curr_args = [arg[1:]]
            cmd_list.append(curr_args)
        else:
            curr_args.append(arg)       
    return cmd_list

def _demo_graph_from_str(demo_string, demo_args, demo_kwargs, **kwargs):
    parser = _DemographyStringParser(demo_args, demo_kwargs, **kwargs)
    
    cmd_list = _get_cmd_list(demo_string)

    if cmd_list[0][0] != "d" or cmd_list[1][0] != "n" or any([cmd[0] in "dn" for cmd in cmd_list[2:]]):
        raise ValueError("Demography string must begin with -d followed by -n")
    
    for i in range(len(cmd_list)):
        if cmd_list[i][0] == "a" and cmd_list[i-1][0].isupper():
            raise ValueError("-a flag must precede all other flags except for -d and -n")

    for event in cmd_list:
        parser.add_event(*event)
    return parser.to_nx()

class _DemographyStringParser(object):
    def __init__(self, demo_args, demo_kwargs, add_pop_idx=0, **kwargs):
        self.params = _ParamsMap(demo_args, demo_kwargs, **kwargs)

        self.add_pop_idx = add_pop_idx

        self.events,self.edges,self.nodes = [],[],{}
        # the nodes currently at the root of the graph, as we build it up from the leafs
        self.roots = {}

        self.cmd_list = []

    def get_event(self, event_flag):
        if event_flag in "d n a N G S J".split():
            return getattr(self, '_' + event_flag)
        raise Exception("Invalid flag -%s" % event_flag)
        
    def add_event(self, event_flag, *args):
        args = getattr(self, '_' + event_flag)(*args)
        self.cmd_list.append("-%s %s" % (event_flag, " ".join(map(str,args))))
        
    def get_pop(self, i):
        return int(i) + self.add_pop_idx

    def _apply_all_pops(self, func, t, x, xfunc):
        assert self.roots
        for i in self.roots:
            assert i != "*" # avoid infinite recursion
            if self.roots[i] is not None:
                func(t,str(i-self.add_pop_idx),x)
        return (self.params.time(t), "*", xfunc(x))

    def _S(self, t,i,p):
        t,p = self.params.time(t), self.params.pulse(p)
        i = self.get_pop(i)

        child = self.roots[i]
        self.set_sizes(self.nodes[child], t)

        parents = ((child,), len(self.roots)+1)
        assert all([par not in self.nodes for par in parents])

        self.nodes[child]['splitprobs'] = {par : prob for par,prob in zip(parents, [p,1-p])}

        prev = self.nodes[child]['sizes'][-1]
        self.nodes[parents[0]] = {'sizes':[{'t':t,'N':prev['N_top'], 'growth_rate':prev['growth_rate']}]}
        self.nodes[parents[1]] = {'sizes':[{'t':t,'N':self.default_N, 'growth_rate':None}]}

        new_edges = tuple([(par, child) for par in parents])
        self.events.append( new_edges )
        self.edges += list(new_edges)

        self.roots[i] = parents[0]
        self.roots[len(self.roots)] = parents[1]
        
        return t,i,p
    
    def _J(self, t,i,j):
        t = self.params.time(t)
        i,j = map(self.get_pop, [i,j])

        for k in i,j:
            # sets the TruncatedSizeHistory, and N_top and growth_rate for all epochs
            self.set_sizes(self.nodes[self.roots[k]], t)

        new_pop = (self.roots[i], self.roots[j])
        self.events.append( ((new_pop,self.roots[i]),
                        (new_pop,self.roots[j]))  )

        assert new_pop not in self.nodes
        prev = self.nodes[self.roots[j]]['sizes'][-1]
        self.nodes[new_pop] = {'sizes':[{'t':t,'N':prev['N_top'], 'growth_rate':prev['growth_rate']}]}

        self.edges += [(new_pop, self.roots[i]), (new_pop, self.roots[j])]

        self.roots[j] = new_pop
        #del self.roots[i]
        self.roots[i] = None

        return t,i,j
    
    def _N(self, t,i,N):
        if i == "*":
            return self._apply_all_pops(self._N, t, N, self.params.size)
        t,N = self.params.time(t), self.params.size(N)
        i = self.get_pop(i)
        self.nodes[self.roots[i]]['sizes'].append({'t':t,'N':N,'growth_rate':None})
        return t,i,N        

    def _G(self, t,i,growth_rate):
        if i=="*":
            return self._apply_all_pops(self._G, t, growth_rate, self.params.growth)
        
        if self.params.growth(growth_rate) == 0.0 and growth_rate[0] != "$":
            growth_rate = None
        else:
            growth_rate = self.params.growth(growth_rate)
            
        t,i = self.params.time(t), self.get_pop(i)
        self.nodes[self.roots[i]]['sizes'].append({'t':t,'growth_rate':growth_rate})

        if growth_rate is None:
            growth_rate=0.0
        return t,i,growth_rate

    def _a(self, i, t):
        ## flag designates leaf population i is archaic, starting at time t
        assert self.roots
        if self.events:
            raise ValueError("-a should be called before any demographic changes")
        assert not self.edges and len(self.nodes) == len(self.roots)

        i,t = self.get_pop(i), self.params.time(t)
        pop = self.roots[i]
        assert len(self.nodes[pop]['sizes']) == 1
        self.nodes[pop]['sizes'][0]['t'] = t

        return i,t
    
    def _n(self, *lins_per_pop):
        # -n should be called immediately after -d, so everything should be empty
        assert all([not x for x in self.roots,self.events,self.edges,self.nodes])
        assert hasattr(self, "default_N")
        
        npop = len(lins_per_pop)
        lins_per_pop = map(int, lins_per_pop)

        for i in range(npop):
            self.nodes[i] = {'sizes':[{'t':0.0,'N':self.default_N,'growth_rate':None}],'lineages':lins_per_pop[i]}
            self.roots[i] = i
        return lins_per_pop

    def _d(self, default_N):
        assert all([not x for x in self.roots,self.events,self.edges,self.nodes])
        
        self.default_N = self.params.size(default_N)
        return self.default_N,

    def to_nx(self):
        assert self.nodes
        self.roots = [r for _,r in self.roots.iteritems() if r is not None]

        if len(self.roots) != 1:
            raise ValueError("Must have a single root population")

        node, = self.roots
        self.set_sizes(self.nodes[node], float('inf'))

        cmd = " ".join(self.cmd_list)
        ret = nx.DiGraph(self.edges, cmd=cmd, events=self.events)
        for v in self.nodes:
            ret.add_node(v, **(self.nodes[v]))
        return ret
    
    def set_sizes(self, node_data, end_time):
        # add 'model_func' to node_data, add information to node_data['sizes']
        sizes = node_data['sizes']
        # add a dummy epoch with the end time
        sizes.append({'t': end_time})

        # do some processing
        N, growth_rate = sizes[0]['N'], sizes[0]['growth_rate']
        pieces = []
        for i in range(len(sizes) - 1):
            sizes[i]['tau'] = tau = (sizes[i+1]['t'] - sizes[i]['t'])

            if 'N' not in sizes[i]:
                sizes[i]['N'] = N
            if 'growth_rate' not in sizes[i]:
                sizes[i]['growth_rate'] = growth_rate
            growth_rate = sizes[i]['growth_rate']
            N = sizes[i]['N']

            if growth_rate is not None:
                pieces.append(ExponentialHistory(tau=tau,growth_rate=growth_rate,N_bottom=N))
                N = pieces[-1].N_top
            else:
                pieces.append(ConstantHistory(tau=tau, N=N))

            sizes[i]['N_top'] = N

            if not all([sizes[i][x] >= 0.0 for x in 'tau','N','N_top']):
                raise ValueError("Negative time or population size. (Were events specified in correct order?")
        sizes.pop() # remove the final dummy epoch

        assert len(pieces) > 0
        if len(pieces) == 0:
            node_data['model'] = pieces[0]
        else:
            node_data['model'] = PiecewiseHistory(pieces)

class _ParamsMap(dict):
    def __init__(self, demo_args, demo_kwargs, gens_per_time=1.0):
        super(_ParamsMap, self).__init__(**demo_kwargs)
        for i,x in enumerate(demo_args):
            self[str(i)] = x
        self.gens_per_time = gens_per_time

    def time(self, var):
        return self.gens_per_time * self._get(var)

    def growth(self,var):
        return self._get(var) / self.gens_per_time

    def pulse(self,var):
        return self._get(var)

    def size(self,var):
        return self._get(var) * self.gens_per_time
            
    def _get(self, var):
        if var[0] == "$":
            ret = self[var[1:]]
        else:
            ret = float(var)
        if np.isnan(ret):
            raise Exception("nan in params %s" % (str(self.params_dict)))
        return ret
