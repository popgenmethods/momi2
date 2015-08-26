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
    '''
    Use momi.make_demography to construct demography from ms command line
    '''
    def __init__(self, to_copy, *args, **kwargs):
        '''makes a copy of a demography object'''
        super(Demography, self).__init__(to_copy, *args, **kwargs)
        self.leaves = set([k for k, v in self.out_degree().items() if v == 0])
        self.event_tree = _build_event_tree(self)

    @property
    def ms_cmd(self):
        '''The ms command line equivalent to this demography'''
        #return self.graph['cmd']
        return _make_ms_cmd(self)

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

def _make_ms_cmd(demo):
    # the events and their data in depth-first ordering
    events = [(e,demo.event_tree.node[e]) for e in nx.dfs_postorder_nodes(demo.event_tree)]
    # add times, and sort according to times; preserve original ordering for same times
    time_event_list = sorted([(d['t'],e,d) for e,d in events],
                             key=itemgetter(0))

    # pre-checking of the leaf events
    def is_leaf_event(e):
        return len(demo.event_tree[e]) == 0
    leaf_events = [(t,e,d) for t,e,d in time_event_list if is_leaf_event(e)]
    # assert leaf events have no child pops
    assert all([len(d['child_pops']) == 0 for t,e,d in leaf_events])
    # make sure leafs all start at time 0
    if any([t != 0.0 for t,e,d in leaf_events]):
        raise Exception("ms command line doesn't allow for archaic leaf populations")
    # sort leaf events according to their pop labels
    leaf_events = sorted(leaf_events, key=itemgetter(1))
    
    # put leaf events at beginning of time_event_list
    non_leaf_events = [(t,e,d) for t,e,d in time_event_list if not is_leaf_event(e)]
    time_event_list = leaf_events + non_leaf_events
    
    # add sample sizes to the command line
    ret = ["-I %d" % len(demo.leaves)]
    for _,_,d in leaf_events:
        l, = d['parent_pops']
        ret += [str(demo.n_lineages(l))]
          
    pops = {}
    pop_times = {}
    next_pop = 1
        
    for t,e,d in time_event_list:
        parent_pops = d['parent_pops']
        child_pops = d['child_pops']

        if len(child_pops) == 0:
            p, = parent_pops
            pops[p] = next_pop
            next_pop += 1
        elif len(child_pops) == 1:
            assert len(parent_pops) == 2
            c, = child_pops
            p0,p1 = parent_pops
            ret += ["-es %f %d %f" % (t, pops[c], demo.node[c]['splitprobs'][p0])]
            pops[p0] = pops[c]
            pops[p1] = next_pop
            next_pop += 1
        elif len(child_pops) == 2:
            assert len(parent_pops) == 1
            c0,c1 = child_pops
            p, = parent_pops
            ret += ["-ej %f %d %d" % (t, pops[c1], pops[c0])]
            pops[p] = pops[c0]
        else:
            assert False

        for p in parent_pops:
            pop_times[p] = t
            ret += [demo.node[p]['model'].ms_cmd(pops[p], pop_times[p])]
            
    return " ".join(ret)

def _get_cmd_list(cmd):
    cmd_list = []
    for arg in cmd.split():
        if arg[0] == '-' and arg[1].isalpha():
            curr_args = [arg[1:]]
            cmd_list.append(curr_args)
        else:
            curr_args.append(arg)       
    return cmd_list

## TODO: make this constructor for Demography
def _make_demo(demo_string, demo_args, demo_kwargs, **kwargs):
    parser = _DemographyStringParser(demo_args, demo_kwargs, **kwargs)
    
    cmd_list = _get_cmd_list(demo_string)
    if cmd_list[0][0] != "d":
        cmd_list = [("d", "1.0")] + cmd_list

    if cmd_list[0][0] != "d" or cmd_list[1][0] != "n" or any([cmd[0] in "dn" for cmd in cmd_list[2:]]):
        raise IOError("Demography string must begin with -d (optional) followed by -n (required)")
    
    for i in range(len(cmd_list)):
        if cmd_list[i][0] == "a" and cmd_list[i-1][0].isupper():
            raise IOError("-a flag must precede all other flags except for -d and -n")

    for event in cmd_list:
        parser.add_event(*event)
    return Demography(parser.to_nx())

class _DemographyStringParser(object):
    def __init__(self, demo_args, demo_kwargs, scale_time=1.0, add_pop_idx=0):
        self.params = _ParamsMap(*demo_args, **demo_kwargs)

        self.scale_time = scale_time
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

    def _apply_all_pops(self, func, t, x):
        assert self.roots
        for i in self.roots:
            assert i != "*" # avoid infinite recursion
            if self.roots[i] is not None:
                func(t,str(i-self.add_pop_idx),x)
        return (self.params.get(t), "*", self.params.get(x))

    def _S(self, t,i,p):
        t,p = map(self.params.get, (t,p))
        i = self.get_pop(i)

        child = self.roots[i]
        self.set_sizes(self.nodes[child], t)

        parents = ((child,), len(self.roots)+1)
        assert all([par not in self.nodes for par in parents])

        self.nodes[child]['splitprobs'] = {par : prob for par,prob in zip(parents, [p,1-p])}

        prev = self.nodes[child]['sizes'][-1]
        self.nodes[parents[0]] = {'sizes':[{'t':t,'N':prev['N_top'], 'alpha':prev['alpha']}]}
        self.nodes[parents[1]] = {'sizes':[{'t':t,'N':self.default_N, 'alpha':None}]}

        new_edges = tuple([(par, child) for par in parents])
        self.events.append( new_edges )
        self.edges += list(new_edges)

        self.roots[i] = parents[0]
        self.roots[len(self.roots)] = parents[1]
        
        return t,i,p
    
    def _J(self, t,i,j):
        t = self.params.get(t)
        i,j = map(self.get_pop, [i,j])

        for k in i,j:
            # sets the TruncatedSizeHistory, and N_top and alpha for all epochs
            self.set_sizes(self.nodes[self.roots[k]], t)

        new_pop = (self.roots[i], self.roots[j])
        self.events.append( ((new_pop,self.roots[i]),
                        (new_pop,self.roots[j]))  )

        assert new_pop not in self.nodes
        prev = self.nodes[self.roots[j]]['sizes'][-1]
        self.nodes[new_pop] = {'sizes':[{'t':t,'N':prev['N_top'], 'alpha':prev['alpha']}]}

        self.edges += [(new_pop, self.roots[i]), (new_pop, self.roots[j])]

        self.roots[j] = new_pop
        #del self.roots[i]
        self.roots[i] = None

        return t,i,j
    
    def _N(self, t,i,N):
        if i == "*":
            return self._apply_all_pops(self._N, t, N)
        t,N = map(self.params.get, (t,N))
        i = self.get_pop(i)
        self.nodes[self.roots[i]]['sizes'].append({'t':t,'N':N,'alpha':None})
        return t,i,N        

    def _G(self, t,i,alpha):
        if i=="*":
            return self._apply_all_pops(self._G, t, alpha)
        
        if self.params.get(alpha) == 0.0 and alpha[0] != "$":
            alpha = None
        else:
            alpha = self.params.get(alpha)
            
        t,i = self.params.get(t), self.get_pop(i)
        self.nodes[self.roots[i]]['sizes'].append({'t':t,'alpha':alpha})

        if alpha is None:
            alpha=0.0
        return t,i,alpha

    def _a(self, i, t):
        ## flag designates leaf population i is archaic, starting at time t
        assert self.roots
        if self.events:
            raise IOError("-a should be called before any demographic changes")
        assert not self.edges and len(self.nodes) == len(self.roots)

        i,t = self.get_pop(i), self.params.get(t)
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
            self.nodes[i] = {'sizes':[{'t':0.0,'N':self.default_N,'alpha':None}],'lineages':lins_per_pop[i]}
            self.roots[i] = i
        return lins_per_pop

    def _d(self, default_N):
        assert all([not x for x in self.roots,self.events,self.edges,self.nodes])
        
        self.default_N = self.params.get(default_N)
        return self.default_N,

    def to_nx(self):
        assert self.nodes
        self.roots = [r for _,r in self.roots.iteritems() if r is not None]

        if len(self.roots) != 1:
            raise IOError("Must have a single root population")

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
        N, alpha = sizes[0]['N'], sizes[0]['alpha']
        pieces = []
        for i in range(len(sizes) - 1):
            sizes[i]['tau'] = tau = (sizes[i+1]['t'] - sizes[i]['t'])

            if 'N' not in sizes[i]:
                sizes[i]['N'] = N
            if 'alpha' not in sizes[i]:
                sizes[i]['alpha'] = alpha
            alpha = sizes[i]['alpha']
            N = sizes[i]['N']

            if alpha is not None:
                pieces.append(ExponentialHistory(tau=tau,growth_rate=alpha,N_bottom=N))
                N = pieces[-1].N_top
            else:
                pieces.append(ConstantHistory(tau=tau, N=N))

            sizes[i]['N_top'] = N

            if not all([sizes[i][x] >= 0.0 for x in 'tau','N','N_top']):
                raise IOError("Negative time or population size. (Were events specified in correct order?")
        sizes.pop() # remove the final dummy epoch

        assert len(pieces) > 0
        if len(pieces) == 0:
            node_data['model'] = pieces[0]
        else:
            node_data['model'] = PiecewiseHistory(pieces)

class _ParamsMap(dict):
    def __init__(self, *args, **kwargs):
        super(_ParamsMap, self).__init__(kwargs)
        for i,x in enumerate(args):
            self[str(i)] = x
            
    def get(self, var):
        if var[0] == "$":
            ret = self[var[1:]]
        else:
            ret = float(var)
        if np.isnan(ret):
            raise Exception("nan in params %s" % (str(self.params_dict)))
        return ret       
            
def demo_from_ms_cmd(ms_cmd, *args, **kwargs):
    params = _ParamsMap(*args, **kwargs)
    
    cmd_list = _get_cmd_list(ms_cmd)
    
    if cmd_list[0][0] != 'I':
        raise IOError("ms command must begin with -I to specify samples per population")
    n_pops = int(cmd_list[0][1])
    
    ## first replace the # sign convention
    pops_by_time = [(0.0, idx) for idx in range(1,n_pops+1)]
    for cmd in cmd_list:
        if cmd[0] == 'es':
            n_pops += 1
            pops_by_time += [(params.get(cmd[1]), n_pops)]
    pops_by_time = [p[1] for p in sorted(pops_by_time, key=itemgetter(0))]

    pops_map = dict(zip(pops_by_time, range(1, len(pops_by_time)+1)))
    for cmd in cmd_list:
        for i in range(len(cmd)):
            if cmd[i][0] == "#":
                # replace with pop idx according to time ordering
                cmd[i] = str(pops_map[int(cmd[i][1:])])
    
    ## next sort ms command according to time
    non_events = [cmd for cmd in cmd_list if cmd[0][0] != 'e']
    events = [cmd for cmd in cmd_list if cmd[0][0] == 'e']
    
    time_events = [(params.get(cmd[1]), cmd) for cmd in events]
    time_events = sorted(time_events, key=itemgetter(0))
    
    events = [cmd for t,cmd in time_events]

    cmd_list = non_events + events

    ## next replace flags with their alternative names:
    for cmd in cmd_list:
        if cmd[0] == "I":
            if len(cmd[2:]) != int(cmd[1]):
                raise IOError("Wrong number of arguments for -I (note continuous migration is not allowed)")
            del cmd[1]
            cmd[0] = "n"
        elif cmd[0] in ["es","ej","eg","en"]:
            cmd[0] = cmd[0][1].upper()
        elif cmd[0] in ["n", "g"]:
            cmd[0] = cmd[0].upper()
            cmd.insert(1, "0.0")
        elif cmd[0] in ["eN", "eG"]:
            cmd[0] = cmd[0][1]
            cmd.insert(2, "*")
        elif cmd[0] == "G":
            cmd.insert(1, "0.0")
            cmd.insert(2, "*")
        cmd[0] = "-" + cmd[0]
            
    return _make_demo(" ".join(sum(cmd_list, [])), args, kwargs, add_pop_idx=-1)

