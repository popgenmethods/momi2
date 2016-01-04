
import networkx as nx
from .util import memoize_instance, memoize
from .math_functions import einsum2, sum_antidiagonals, convolve_axes, binom_coeffs, roll_axes, hypergeom_quasi_inverse
import scipy, scipy.misc
import autograd.numpy as np

from .size_history import ConstantHistory, ExponentialHistory, PiecewiseHistory
from .parse_ms import _convert_ms_cmd

import os, itertools
from operator import itemgetter

class Demography(object):
    @classmethod
    def from_ms(cls, default_N_diploid, ms_cmd, *args, **kwargs):
        """ Construct demography using format of Richard Hudson's program ms

        See examples/tutorial.py for examples

        Paramters
        ---------
        default_N_diploid : float
            the number of diploids corresponding to ms scaled pop size = 1.0
        ms_cmd : str
            ms command line string, with some modifications:
            0) only demography flags: no -t,-T,-r
            1) no continuous migration
            2) -I must be at start of command line
            3) can pass in special variables with $
        *args, **kwargs : optional
            additional arguments corresponding to special variables $

        See Also:
        ---------
        ms program (http://home.uchicago.edu/rhudson1/source/mksamples.html)
        see msdoc.pdf from ms program for more documentation

        Demography.__init__ : default Demography constructor

        Additional modifications
        ------------------------
        These modifications may or may not be in future versions:
        4) -a flag can set populations to archaic, same as in Demography(),
           but with ms label and scaling conventions
        5) can use hashtag '#i' to specify the i-th population
           in the command line from left to right (with i starting at 1,
           as per ms convention)
        """
        return Demography("-msformat $%d %s" % (len(args), ms_cmd),
                          *(list(args) + [default_N_diploid]),
                          **kwargs)
    
    def __init__(self, demo_str, *args, **kwargs):
        """ Construct demography from command line

        See examples/tutorial.py for examples.
        ## TODO: additional examples with multiple -S events

        Parameters:
        -----------
        demo_str : str
             a string of the format
                  -d <default_N_diploid> -n <n_0> ... <n_(D-1)> ... <additional flags>
             required flags are:
                 -d <default_N_diploid>
                     must be first flag.
                     set reference diploid size (== half reference haploid size)
                     coalescence happens at rate 1 / (2*N_diploid)
                 -n <n_0> ... <n_(D-1)>
                     must be second flag
                     n_i == # alleles (haploids) sampled from deme i
             additional flags are:
                 -a <t_gens> <i>
                     set population i to be archaic, with samples from t_gens ago
                     must occur after -d,-n, but before any -G,-J,-S,-N events
                 -G <t_gens> <i> <growth_rate>
                     set growth rate for population i, so at t > t_gens ago
                          N(t) = N(t_gens) * exp( -growth_rate * (t - t_gens) )
                     the effect is cancelled by -N or -G events occurring on the same
                     population, backwards from the present
                     (if i == *, growth rate for all populations is set)
                 -J <t_gens> <i> <j>
                     t_gens generations ago, all lineages from <i> move into <j>
                     (i.e. <i> and <j> find a common ancestor)
                 -N <t_gens> <i> <N_diploid>
                     set diploid size of population <i> for t > t_gens ago
                     coalescence occurs at rate 1/(2*N_diploid)
                     (if i == *, size for all populations is set)
                 -S <t_gens> <i> <p>
                     t_gens ago, each lineage in pop <i> independently with
                     probability 1-p to a new pop.

                     if this is the k-th -S flag in the string, from left to right,
                     and there are D initial populations, then the new population
                     is labeled D+k-1
                 
                     (contrast to ms: k determined by order of -es flag, from present to the past)

        Other Parameters
        ----------------
            *args, **kwargs: optional
                pass in special variables, specified with $ in demo_str
                e.g.
                # using **kwargs
                > demo = Demography("-d $N_present -n 10 -N $bottleneck_t 0 $N_ancestral",
                                    bottleneck_t = 1e3, N_present = 1e5, N_ancestral = 1e2)
                # using *args
                > demo = Demography("-d $1 -n 10 -N $0 0 $2", 1e3, 1e5, 1e2)
        """
        cmd_list = _get_cmd_list(demo_str)
        params = _ParamsMap(args, kwargs)
        
        if cmd_list[0][0] == "msformat":           
            N_e = params.size(cmd_list[0][1])
           
            cmd_list = _convert_ms_cmd(cmd_list[1:], params)
            parser = _DemographyStringParser(args, kwargs, add_pop_idx=-1, gens_per_time=N_e)
        else:
            parser = _DemographyStringParser(args, kwargs)

        if cmd_list[0][0] != "d" or cmd_list[1][0] != "n" or any([cmd[0] in "dn" for cmd in cmd_list[2:]]):
            raise ValueError("Demography string must begin with -d followed by -n")

        for i in range(len(cmd_list)):
            if cmd_list[i][0] == "a" and cmd_list[i-1][0].isupper():
                raise ValueError("-a flag must precede all other flags except for -d and -n")

        non_events = [cmd for cmd in cmd_list if not cmd[0].isupper()]
        events = [cmd for cmd in cmd_list if cmd[0].isupper()]

        ## label new pops by their order in the cmd
        npops = len(cmd_list[1][1:])
        for e in events:
            if e[0] == "S":
                ## add extra param giving the pop label
                ## this is the TRUE momi label, as an int
                ## (not the ms label even if its from ms cmd line)
                ## TODO: this code is very opaque, clean it up!                
                e.append(npops)
                npops += 1

        ## sort events by time
        events = sorted(events, key=lambda x: params.time(x[1]))

        ## process all events
        cmd_list = non_events + list(events)
        for event in cmd_list:
            parser.add_event(*event)

        self.G = parser.to_nx()
        
        self.leaves = set(self.G.graph['sampled_pops'])
        self.event_tree = _build_event_tree(self)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Demography('%s')" % self.G.graph['cmd']

    @memoize_instance
    def n_lineages(self, node):
        if node in self.leaves:
            node = (node,0)
        return np.sum(self.G.node[(l,0)]['lineages'] for l in self.leaves_subtended_by(node))

    @property
    def default_N_diploid(self):
        return self.G.graph['default_N_diploid']
    
    @property
    def n_at_leaves(self):
        return tuple(self.n_lineages(l) for l in sorted(self.leaves))

    @memoize_instance
    def leaves_subtended_by(self, node):
        return self.leaves & set([x[0] for x in nx.dfs_preorder_nodes(self.G, node)])

    def truncated_sfs(self, node):
        '''The truncated SFS at node.'''
        return self.G.node[node]['model'].sfs(self.n_lineages(node))

    def apply_transition(self, node, array, axis):
        '''Apply Moran model transition at node to array along axis.'''
        assert array.shape[axis] == self.n_lineages(node)+1
        if array.shape[axis] == 1:
            return array
        return self.G.node[node]['model'].transition_prob(array, axis)
   
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
        elif len(event) == 3:
            return 'pulse'
        # elif len(self.parent_pops(event)) == 2:
        #     return 'admixture'
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
    def pulse_prob(self, event):
        ## returns 4-tensor
        ## running time is O(n^5), because of pseudo-inverse
        ## if pulse from ghost population, only costs O(n^4)
        recipient, non_recipient, donor, non_donor = self.pulse_nodes(event)
        
        admixture_prob, admixture_idxs = self._admixture_prob(recipient)

        pulse_idxs = admixture_idxs + [non_recipient]
        pulse_prob = einsum2(admixture_prob, admixture_idxs,
                             binom_coeffs(self.n_lineages(non_recipient)), [non_recipient],
                             pulse_idxs)
        pulse_prob = einsum2(pulse_prob, pulse_idxs,
                             binom_coeffs(self.n_lineages(recipient)), [donor],
                             pulse_idxs)
        pulse_prob = roll_axes(pulse_prob, pulse_idxs, non_recipient, donor)

        donor_idx = pulse_idxs.index(donor)
        pulse_prob = einsum2(pulse_prob, pulse_idxs,
                             1.0 / binom_coeffs(pulse_prob.shape[donor_idx]-1), [donor],
                             pulse_idxs)

        # reduce the number of lineages in donor to only the number necessary
        N,n = pulse_prob.shape[donor_idx]-1, self.n_lineages(donor)
        assert N >= n
        if N > n:
            assert -1 not in pulse_idxs        
            tmp_idxs = [-1 if x == donor else x for x in pulse_idxs]
            pulse_prob = einsum2(pulse_prob, tmp_idxs,
                                 hypergeom_quasi_inverse(N, n),
                                 [-1,donor], pulse_idxs)
        assert pulse_prob.shape[donor_idx] == n + 1

        return pulse_prob, pulse_idxs

    def pulse_nodes(self, event):
        parent_pops = self.parent_pops(event)    
        child_pops_events = self.child_pops(event)
        assert len(child_pops_events) == 2
        child_pops, child_events = zip(*child_pops_events.items())

        child_in = self.G.in_degree(child_pops)
        recipient, = [k for k,v in child_in.items() if v == 2]
        non_recipient, = [k for k,v in child_in.items() if v == 1]

        parent_out = self.G.out_degree(parent_pops)
        donor, = [k for k,v in parent_out.items() if v == 2]
        non_donor, = [k for k,v in parent_out.items() if v == 1]

        return recipient, non_recipient, donor, non_donor
    
    @memoize_instance
    def admixture_prob(self, admixture_node):
        return self._admixture_prob(admixture_node)
    
    def _admixture_prob(self, admixture_node):
        '''
        Array with dim [n_admixture_node+1, n_parent1_node+1, n_parent2_node+1],
        giving probability of derived counts in child, given derived counts in parents
        '''
        n_node = self.n_lineages(admixture_node)

        # admixture node must have two parents
        edge1,edge2 = self.G.in_edges([admixture_node], data=True)
        parent1,parent2 = [e[0] for e in (edge1,edge2)]
        prob1,prob2 = [e[2]['prob'] for e in (edge1,edge2)]        
        assert prob1 + prob2 == 1.0

        n_from_1 = np.arange(n_node+1)
        n_from_2 = n_node - n_from_1
        binom_coeffs = (prob1**n_from_1) * (prob2**n_from_2) * scipy.misc.comb(n_node, n_from_1)
        ret = einsum2(der_in_admixture_node(n_node), list(range(4)),
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
        return demo.G.node[v]['sizes'][0]['t']
    
    eventEdgeList = []
    currEvents = {k : (k,) for k,v in demo.G.out_degree().items() if v == 0}
    eventDict = {e : {'subpops' : (v,), 'parent_pops' : (v,), 'child_pops' : {}, 't' : node_time(v)} for v,e in currEvents.items()}

    for e in demo.G.graph['events']:
        # get the population edges forming the event
        parent_pops, child_pops = list(map(set, list(zip(*e))))
        child_events = set([currEvents[c] for c in child_pops])
        #assert len(e) == 2 and len(parent_pops) + len(child_pops) == 3 and len(child_events) in (1,2)

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
    root, = [v for k,v in currEvents.items()]
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

class _DemographyStringParser(object):
    def __init__(self, demo_args, demo_kwargs, add_pop_idx=0, **kwargs):
        self.params = _ParamsMap(demo_args, demo_kwargs, **kwargs)

        self.add_pop_idx = add_pop_idx

        #self.events,self.edges,self.nodes = [],[],{}
        self.events = []
        self.G = nx.DiGraph()
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
                _,_,xval = func(t,str(i-self.add_pop_idx),x)
        return (self.params.time(t), "*", xval)

    # def _S(self, t,i,p, new_label):
    #     t,p = self.params.time(t), self.params.pulse(p)
    #     i = self.get_pop(i)
        
    #     child = self.roots[i]
    #     self.set_sizes(self.G.node[child], t)

    #     parents = ((i,child[1]+1), (len(self.roots)+1,0))
    #     assert all([par not in self.G.node for par in parents])

    #     prev = self.G.node[child]['sizes'][-1]
    #     self.G.add_node(parents[0], sizes=[{'t':t,'N':prev['N_top'], 'growth_rate':prev['growth_rate']}])
    #     self.G.add_node(parents[1], sizes=[{'t':t,'N':self.default_N, 'growth_rate':None}])

    #     for par,prob in zip(parents, [p,1-p]):
    #         self.G.add_edge(par, child, prob=prob)
    #     self.events += [tuple((par,child) for par in parents)]

    #     self.roots[i] = parents[0]
        
    #     assert new_label not in self.roots
    #     self.roots[new_label] = parents[1]

    #     return t,i,p

    def _S(self, t,i,p, new_label):
        self.G.add_node((new_label,0),
                        sizes=[{'t':self.params.time(t),'N':self.default_N,'growth_rate':None}],
                        lineages=0,
                        )
        self.roots[new_label] = (new_label,0)
        self._P(t,i,new_label-self.add_pop_idx,p)
        
        return self.params.time(t),self.get_pop(i),self.params.pulse(p)
    
    def _J(self, t,i,j):
        t = self.params.time(t)
        i,j = list(map(self.get_pop, [i,j]))

        for k in i,j:
            # sets the TruncatedSizeHistory, and N_top and growth_rate for all epochs
            self.set_sizes(self.G.node[self.roots[k]], t)

        new_pop = (j, self.roots[j][1]+1)
        assert new_pop not in self.G.nodes()
        prev = self.G.node[self.roots[j]]['sizes'][-1]
        self.G.add_node(new_pop, sizes=[{'t':t,'N':prev['N_top'], 'growth_rate':prev['growth_rate']}])
        
        new_edges = ((new_pop,self.roots[i]), (new_pop,self.roots[j]))       
        self.events.append(new_edges)
        self.G.add_edges_from(new_edges)

        self.roots[j] = new_pop
        #del self.roots[i]
        self.roots[i] = None

        return t,i,j
    
    def _N(self, t,i,N):
        if i == "*":
            return self._apply_all_pops(self._N, t, N)
        t,N = self.params.time(t), self.params.size(N)
        i = self.get_pop(i)
        self.G.node[self.roots[i]]['sizes'].append({'t':t,'N':N,'growth_rate':None})
        return t,i,N        

    def _G(self, t,i,growth_rate):
        if i=="*":
            return self._apply_all_pops(self._G, t, growth_rate)
        
        if self.params.growth(growth_rate) == 0.0 and growth_rate[0] != "$":
            growth_rate = None
        else:
            growth_rate = self.params.growth(growth_rate)
            
        t,i = self.params.time(t), self.get_pop(i)
        self.G.node[self.roots[i]]['sizes'].append({'t':t,'growth_rate':growth_rate})

        if growth_rate is None:
            growth_rate=0.0
        return t,i,growth_rate

    # def _P(self, t, i, j, pij):
    #     cur_max = max(self.roots.keys())
    #     ii = cur_max+1
        
    #     self._S(t,i,pij,ii)
    #     self._J(t,ii,j)
        
    #     del self.roots[ii]

    #     return t,i,j,pij

    def _P(self, t, i, j, pii):
        t = self.params.time(t)
        i,j = list(map(self.get_pop, [i,j]))
        pii = self.params.pulse(pii)

        children = {k: self.roots[k] for k in (i,j)}
        for v in children.values():
            self.set_sizes(self.G.node[v], t)

        parents = {k: (v[0],v[1]+1) for k,v in children.items()}
        assert all([par not in self.G.node for par in parents.values()])

        prev_sizes = {k: self.G.node[c]['sizes'][-1] for k,c in children.items()}
        for k,s in prev_sizes.items():
            self.G.add_node(parents[k], sizes=[{'t':t,'N':s['N_top'],'growth_rate':s['growth_rate']}])

        self.G.add_edge(parents[i], children[i], prob=pii)
        self.G.add_edge(parents[j], children[i], prob=1.-pii)
        self.G.add_edge(parents[j], children[j])

        new_event = tuple((parents[u], children[v])
                          for u,v in ( (i,i),(j,i),(j,j) )
                          )
        self.events += [new_event]
       
        for k,v in parents.items():
            self.roots[k] = v

        return t,i,j,pii
    
    def _a(self, t, i):
        ## flag designates leaf population i is archaic, starting at time t
        assert self.roots
        if self.events:
            raise ValueError("-a should be called before any demographic changes")
        assert not self.G.edges() and len(self.G.nodes()) == len(self.roots)

        i,t = self.get_pop(i), self.params.time(t)
        pop = self.roots[i]
        assert len(self.G.node[pop]['sizes']) == 1
        self.G.node[pop]['sizes'][0]['t'] = t

        return i,t
    
    def _n(self, *lins_per_pop):
        # -n should be called immediately after -d, so everything should be empty
        assert all([not x for x in (self.roots,self.events,self.G.edge,self.G.node)])
        assert hasattr(self, "default_N")
        
        npop = len(lins_per_pop)
        lins_per_pop = list(map(int, lins_per_pop))

        self.G.graph['sampled_pops'] = range(npop)        
        for i in self.G.graph['sampled_pops']:
            self.G.add_node((i,0),
                            sizes=[{'t':0.0,'N':self.default_N,'growth_rate':None}],
                            lineages=lins_per_pop[i])
            self.roots[i] = (i,0)
        return lins_per_pop

    def _d(self, default_N):
        assert all([not x for x in (self.roots,self.events,self.G.edge,self.G.node)])
        
        self.default_N = self.params.size(default_N)
        return self.default_N,

    def to_nx(self):
        assert self.G.node
        self.roots = [r for _,r in self.roots.items() if r is not None]

        if len(self.roots) != 1:
            raise ValueError("Must have a single root population")

        node, = self.roots
        self.set_sizes(self.G.node[node], float('inf'))

        cmd = " ".join(self.cmd_list)
        self.G.graph.update({'cmd':cmd,'events':self.events})
        self.G.graph['default_N_diploid'] = self.default_N
        return self.G
    
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

            if not all([sizes[i][x] >= 0.0 for x in ('tau','N','N_top')]):
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
            raise Exception("nan in params %s" % (str(self)))
        return ret
