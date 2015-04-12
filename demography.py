import networkx as nx
from Bio import Phylo
from cStringIO import StringIO
from cached_property import cached_property
from size_history import ConstantTruncatedSizeHistory
from sum_product import SumProduct

from size_history import ExponentialTruncatedSizeHistory as ExpHist
from size_history import ConstantTruncatedSizeHistory as ConstHist
from size_history import PiecewiseHistory

from numpy import log, exp ## TODO: import from adarray.admath

class Demography(nx.DiGraph):
    @classmethod
    def from_ms(cls, ms_cmd):
        return cls(_nx_from_ms(ms_cmd))

    @classmethod
    def from_newick(cls, newick, default_lineages=None, default_N=1.0):
        t = cls(_newick_to_nx(newick, default_lineages))
        # add models to edges
        for v0, v1, d in t.edges(data=True):
            n_sub = t.n_lineages_subtended_by[v1]
            nd = t.node_data[v1]
            if 'model' not in nd or nd['model'] == "constant":
                nd['model'] = ConstantTruncatedSizeHistory(
                        N=nd.get('N', default_N),
                        tau=d['branch_length'], 
                        n_max=n_sub)
            else:
                raise Exception("Unsupported model type: %s" % nd['model'])
        nd = t.node_data[t.root]
        # FIXME: all possible size histories for root
        nd['model'] = ConstantTruncatedSizeHistory(
                N=nd.get('N', default_N),
                n_max=t.n_lineages_subtended_by[t.root], 
                tau=float("inf"))
        return t


    def __init__(self, *args, **kwargs):
        super(Demography, self).__init__(*args, **kwargs)
        nd = self.node_data
        if not all('lineages' in nd[k] for k in self.leaves):
            raise Exception("'lineages' attribute must be set for each leaf node.")
        for v in self:
            if 'model_func' in nd[v] and 'model' not in nd[v]:
                nd[v]['model'] = nd[v]['model_func'](self.n_lineages_subtended_by[v])
#         if not all('model' in nd[k] for k in self):
#             raise Exception("'model' attribute must be set for all nodes.")

    @cached_property
    def event_tree(self):
        ## TODO: turn self['events'] into a tree (via nx.DiGraph)
        return self['events']

    @cached_property
    def totalSfsSum(self):
        return normalizing_constant(self)

    @cached_property
    def root(self):
        nds = [node for node, deg in self.in_degree().items() if deg == 0]
        assert len(nds) == 1
        return nds[0]
    
    @cached_property
    def node_data(self):
        return dict(self.nodes(data=True))

    @cached_property
    def leaves(self):
        return set([k for k, v in self.out_degree().items() if v == 0])

    @cached_property
    def n_lineages_subtended_by(self):
        nd = self.node_data
        return {v: sum(nd[l]['lineages'] for l in self.leaves_subtended_by[v]) for v in self}

    @cached_property
    def n_derived_subtended_by(self):
        nd = self.node_data
        return {v: sum(nd[l]['derived'] for l in self.leaves_subtended_by[v]) for v in self}

    @cached_property
    def leaves_subtended_by(self):
        return {v: self.leaves & set(nx.dfs_preorder_nodes(self, v)) for v in self}

    def is_leaf(self, node):
        return node in self.leaves

    def update_state(self, state):
        nd = self.node_data
        for node in state:
            ndn = nd[node]
            ndn.update(state[node])
            if ndn['lineages'] != ndn['derived'] + ndn['ancestral']:
                raise Exception("derived + ancestral must add to lineages at node %s" % node)
        # Invalidate the caches which depend on node state
        # FIXME: breaks for version 1.0.0 of cached_property module!
        self.n_derived_subtended_by # make sure cache exists
        del self.n_derived_subtended_by #reset cache
        del self.node_data  #reset cache

    def to_newick(self):
        return _to_newick(self, self.root)


_field_factories = {
    "N": float, "lineages": int, "ancestral": int, 
    "derived": int, "model": str
    }
def _extract_momi_fields(comment):
    for field in comment.split("&&"):
        if field.startswith("momi:"):
            attrs = field.split(":")
            assert attrs[0] == "momi"
            attrs = [a.split("=") for a in attrs[1:]]
            attrdict = dict((a, _field_factories[a](b)) for a, b in attrs)
            return attrdict
    return {}

def _newick_to_nx(newick, default_lineages=None):
    newick = StringIO(newick)
    phy = Phylo.read(newick, "newick")
    phy.rooted = True
    edges = []
    nodes = []
    node_data = {}
    clades = [phy.root]
    phy.root.name = phy.root.name or "root"
    i = 0
    while clades:
        clade = clades.pop()
        nd = _extract_momi_fields(clade.comment or "")
        if 'lineages' not in nd and default_lineages is not None:
            nd['lineages'] = default_lineages
        nodes.append((clade.name, nd))
        for c_clade in clade.clades:
            clades += clade.clades
            if c_clade.name is None:
                c_clade.name = "node%d" % i
                i += 1
            ed = {'branch_length': c_clade.branch_length}
            edges.append((clade.name, (c_clade.name), ed))
    t = nx.DiGraph(data=edges)
    t.add_nodes_from(nodes)
    tn = dict(t.nodes(data=True))
    for node in node_data:
        tn[node].update(node_data[node])
    return t

def _to_newick(G, root):
    parent = list(G.predecessors(root))
    try:
        edge_length = str(G[parent[0]][root]['branch_length'])
    except IndexError:
        edge_length = None
    if not G[root]:
        assert edge_length is not None
        return root + ":" + edge_length
    else:
        children = list(G[root])
        assert len(children) == 2
        dta = [(_to_newick(G, child),) for child in children]
        ret = "(%s,%s)" % (dta[0] + dta[1])
        if edge_length:
            ret += ":" + edge_length
        return ret


def normalizing_constant(demography):
    # to_directed() makes a deep-copy of the nx.DiGraph
    demography = Demography(demography.to_directed())
    # set all alleles to be of ancestral type
    state = {}
    for v in demography.leaves:
        state[v] = {}
        state[v]['derived'] = 0
        state[v]['ancestral'] = demography.node_data[v]['lineages']
    demography.update_state(state)
    # now create the Sum-Product
    sp = SumProduct(demography)

    ret = 0.0
    for node in demography:
        # 1 - partial_likelihood_bottom is probability of at least one derived leaf lineage
        ret += ((1.0 - sp.partial_likelihood_bottom(node)) * sp.truncated_sfs(node)).sum()

    # subtract off the term for all alleles derived
    state = {}
    for v in demography.leaves:
        state[v] = {}
        state[v]['derived'] = demography.node_data[v]['lineages']
        state[v]['ancestral'] = 0
    demography.update_state(state)
    # now create the Sum-Product
    sp = SumProduct(demography)

    ret -= sp.p(normalized=False)
    return ret



'''Construct demography from ms command'''
def _nx_from_ms(ms_cmd):
    ms_cmd = ms_cmd.strip()
    if not ms_cmd.startswith("-I "):
        raise IOError(("Must start cmd with -I", ms_cmd))

    # if ms_cmd = "-I 2 1 2 -G -.3 -ej 1.0 2 1"
    # then make cmd_list = [[I,2,1,2],[G,-.3],[ej,1.0,2,1]]
    cmd_list = []
#     curr_args = []
#     cmd_list.append(curr_args)

    for arg in ms_cmd.split():
        if arg[0] == '-' and arg[1].isalpha():
            curr_args = [arg[1:]]
            cmd_list.append(curr_args)
        else:
            curr_args.append(arg)

    #n,_ = map(int, cmd_list[0])
    #cmd_list = cmd_list[1:]
    assert cmd_list[0][0] == 'I'

    # now parse the ms cmd, store results in kwargs
    kwargs = {'events':[],'edges':[],'nodes':{},'roots':{},'cmd':ms_cmd}
    ## TODO: convert roots to [None] (will be useful for adding pops with -es)
    for cmd in cmd_list:
        if cmd[0] not in valid_params:
            raise NotImplementedError("-%s not implemented" % cmd[0])
        cmdfunc, args = eval("_" + cmd[0]), cmd[1:]
        cmdfunc(*args, **kwargs)

    # return nx.DiGraph from parsed ms cmd
    return _nx_from_parsed_ms(**kwargs)

valid_params = set(["G","I","n","g","eG","eg","eN","en","ej"])


def _nx_from_parsed_ms(events, edges, nodes, roots, cmd, **kwargs):
    #print cmd
    assert nodes
    if len(roots) != 1:
        raise IOError(("Must have a single root population", cmd))
    
    root, = list(roots.iteritems())
    _,node = root

    if 'alpha' in nodes[node]['sizes'][-1] and nodes[node]['sizes'][-1]['alpha'] is not None:
        raise IOError(("Root ancestral population must not have growth parameter",cmd))
    set_model(nodes[node], float('inf'), cmd)

    ret = nx.DiGraph(edges, cmd=cmd, events=events)
    for v in nodes:
        ret.add_node(v, **(nodes[v]))
        #ret.node[v].update(nodes[v])
    return ret

def event_tree(events, demography):
    ## TODO: need to implement this for pulse migration
    pass

def _ej(t,i,j, events, nodes, roots, edges, cmd, **kwargs):
    t = float(t)

    for k in i,j:
        # sets the TruncatedSizeHistory, and N_top and alpha for all epochs
        set_model(nodes[roots[k]], t, cmd)

    new_pop = "(%s,%s)" % (roots[i],roots[j])
    events.append( [(new_pop,roots[i]),
                    (new_pop,roots[j])]  )

    assert new_pop not in nodes
    prev = nodes[roots[j]]['sizes'][-1]
    nodes[new_pop] = {'sizes':[{'t':t,'N':prev['N_top'], 'alpha':prev['alpha']}]}

    edges += [(new_pop, roots[i]), (new_pop, roots[j])]

    roots[j] = new_pop
    del roots[i]

def _en(t,i,N, nodes, roots, **kwargs):
    t,N = map(float, [t,N])
    nodes[roots[i]]['sizes'].append({'t':t,'N':N,'alpha':None})

def _eN(t,N, roots, **kwargs):
    assert roots
    for i in roots:
        _en(t, i, N, roots=roots, **kwargs)

def _eg(t,i,alpha, roots, nodes, **kwargs):
    t,alpha = map(float, [t,alpha])
    nodes[roots[i]]['sizes'].append({'t':t,'alpha':alpha})

def _eG(t,alpha, roots, **kwargs):
    assert roots
    for i in roots:
        _eg(t,i,alpha, roots=roots, **kwargs)

def _n(i,N, nodes, events, edges, roots, **kwargs):
    assert roots
    if events:
        raise IOError(("-n should be called before any demographic changes", kwargs['cmd']))
    assert not edges and len(nodes) == len(roots)

    assert len(nodes[i]['sizes']) == 1
    nodes[i]['sizes'][0]['N'] = float(N)

def _g(i,alpha, nodes, events, edges, roots, **kwargs):
    assert roots
    if events:
        raise IOError(("-g,-G should be called before any demographic changes", kwargs['cmd']))
    assert not edges and len(nodes) == len(roots)

    assert len(nodes[i]['sizes']) == 1
    nodes[i]['sizes'][0]['alpha'] = float(alpha)

def _G(rate, roots, nodes, **kwargs):
    assert roots
    for i in roots:
        _g(i, rate, roots=roots, nodes=nodes, **kwargs)

def _I(*args, **kwargs):
    # -I should be called first, so kwargs should be empty
    assert all([not kwargs[x] for x in 'roots','events','edges','nodes'])

    npop = int(args[0])
    lins_per_pop = map(int,args[1:])
    if len(lins_per_pop) != npop:
        raise IOError(("Bad args for -I. Note continuous migration is not implemented.", kwargs['cmd']))

    kwargs['nodes'].update({str(i+1) : {'sizes':[{'t':0.0,'N':1.0,'alpha':None}],'lineages':lins_per_pop[i]} for i in range(npop)})
    kwargs['roots'].update({k : k for k in kwargs['nodes']})


def set_model(node_data, end_time, cmd):
    # add 'model_func' to node_data, add information to node_data['sizes']
    sizes = node_data['sizes']
    # add a dummy epoch with the end time
    sizes.append({'t': end_time})

    N, alpha = sizes[0]['N'], sizes[0]['alpha']
    for i in range(len(sizes) - 1):
        # ms times are rescaled by 0.5
        sizes[i]['tau'] = tau = (sizes[i+1]['t'] - sizes[i]['t']) * 2.0

        if 'N' not in sizes[i]:
            sizes[i]['N'] = N
        if 'alpha' not in sizes[i]:
            sizes[i]['alpha'] = alpha
        alpha = sizes[i]['alpha']
        N = sizes[i]['N']

        if alpha is not None:
            N = N * exp(-alpha * tau / 2.0)
        sizes[i]['N_top'] = N

        if not all([sizes[i][x] >= 0.0 for x in 'tau','N','N_top']):
            raise IOError(("Negative time or population size. (Were events specified in correct order?", cmd))
    sizes.pop() # remove the final dummy epoch

    # construct function that returns size history from number of lineages
    def model_func(n_max):
        pieces = []
        for size in sizes:
            #print size
            if size['alpha'] is not None:
                pieces.append(ExpHist(n_max=n_max, tau=size['tau'], N_top=size['N_top'], N_bottom=size['N']))
            else:
                pieces.append(ConstHist(n_max=n_max, tau=size['tau'], N=size['N']))
        assert len(pieces) > 0
        if len(pieces) == 0:
            return pieces[0]
        return PiecewiseHistory(pieces)
    # now store the size history function
    node_data['model_func'] = model_func

