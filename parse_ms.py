import networkx as nx
from cStringIO import StringIO
from Bio import Phylo
from size_history import ConstantHistory, ExponentialHistory, PiecewiseHistory

from autograd.numpy import isnan, exp,min

import newick
import sh, os, random
import itertools
from collections import Counter

'''
Construct networkx DiGraph from ms command.
Use demography.make_demography instead of calling this directly.
'''
def _to_nx(ms_cmd, *params):
    def toFloat(var):
        if var[0] == "$":
            ret = params[int(var[1:])]
        else:
            ret = float(var)
        if isnan(ret):
            raise Exception("nan in %s" % (ms_cmd))
        return ret
                           

    ms_cmd = ms_cmd.strip()
    if not ms_cmd.startswith("-I "):
        raise IOError(("Must start cmd with -I", ms_cmd))

    cmd_list = []

    for arg in ms_cmd.split():
        if arg[0] == '-' and arg[1].isalpha():
            curr_args = [arg[1:]]
            cmd_list.append(curr_args)
        else:
            curr_args.append(arg)
    assert cmd_list[0][0] == 'I'

    # replace variables in the ms cmd string
    ### TODO: fix this try/except! it's needed in general, but breaks for autograd
    try:
        ## MUST do this in reversed order, otherwise $26 will turn into ($2)6
        for i in reversed(range(len(params))):
            ms_cmd = ms_cmd.replace("$" + str(i), str(float(params[i])))
    except:
        pass

    # now parse the ms cmd, store results in kwargs
    kwargs = {'events':[],'edges':[],'nodes':{},'roots':{},'cmd':ms_cmd,'toFloat':toFloat}
    for cmd in cmd_list:
        if cmd[0] not in valid_params:
            raise NotImplementedError("-%s not implemented" % cmd[0])
        cmdfunc, args = eval("_" + cmd[0]), cmd[1:]
        cmdfunc(*args, **kwargs)

    # return nx.DiGraph from parsed ms cmd
    return _nx_from_parsed_ms(**kwargs)

valid_params = set(["G","I","n","g","eG","eg","eN","en","ej","es"])


def _nx_from_parsed_ms(events, edges, nodes, roots, cmd, toFloat, **kwargs):
    assert nodes
    roots = [r for _,r in roots.iteritems() if r is not None]

    if len(roots) != 1:
        raise IOError(("Must have a single root population", cmd))
    
    node, = roots
    set_model(nodes[node], toFloat('inf'), cmd)

    ret = nx.DiGraph(edges, cmd=cmd, events=events)
    for v in nodes:
        ret.add_node(v, **(nodes[v]))
    return ret

def _es(t,i,p, events, nodes, roots, edges, cmd, toFloat, **kwargs):
    t,p = map(toFloat, (t,p))
    i = int(i)

    child = roots[i]
    set_model(nodes[child], t, cmd)

    parents = ((child,), len(roots)+1)
    assert all([par not in nodes for par in parents])

    nodes[child]['splitprobs'] = {par : prob for par,prob in zip(parents, [p,1-p])}

    prev = nodes[child]['sizes'][-1]
    nodes[parents[0]] = {'sizes':[{'t':t,'N':prev['N_top'], 'alpha':prev['alpha']}]}
    nodes[parents[1]] = {'sizes':[{'t':t,'N':1.0, 'alpha':None}]}

    new_edges = tuple([(par, child) for par in parents])
    events.append( new_edges )
    edges += list(new_edges)

    roots[i] = parents[0]
    roots[len(roots)+1] = parents[1]

def _ej(t,i,j, events, nodes, roots, edges, cmd, toFloat, **kwargs):
    t = toFloat(t)
    i,j = map(int, [i,j])

    for k in i,j:
        # sets the TruncatedSizeHistory, and N_top and alpha for all epochs
        set_model(nodes[roots[k]], t, cmd)

    new_pop = (roots[i], roots[j])
    events.append( ((new_pop,roots[i]),
                    (new_pop,roots[j]))  )

    assert new_pop not in nodes
    prev = nodes[roots[j]]['sizes'][-1]
    nodes[new_pop] = {'sizes':[{'t':t,'N':prev['N_top'], 'alpha':prev['alpha']}]}

    edges += [(new_pop, roots[i]), (new_pop, roots[j])]

    roots[j] = new_pop
    #del roots[i]
    roots[i] = None

def _en(t,i,N, nodes, roots, toFloat, **kwargs):
    t,N = map(toFloat, [t,N])
    i = int(i)
    nodes[roots[i]]['sizes'].append({'t':t,'N':N,'alpha':None})

def _eN(t,N, roots, **kwargs):
    assert roots
    for i in roots:
         if roots[i] is not None:
             _en(t, i, N, roots=roots, **kwargs)

def _eg(t,i,alpha, roots, nodes, toFloat, **kwargs):
    if toFloat(alpha) == 0.0 and alpha[0] != "$":
        alpha = None
    else:
        alpha = toFloat(alpha)
    t,i = toFloat(t), int(i)
    nodes[roots[i]]['sizes'].append({'t':t,'alpha':alpha})

def _eG(t,alpha, roots, **kwargs):
    assert roots
    for i in roots:
        if roots[i] is not None:
            _eg(t,i,alpha, roots=roots, **kwargs)

def _n(i,N, nodes, events, edges, roots, toFloat, **kwargs):
    assert roots
    if events:
        raise IOError(("-n should be called before any demographic changes", kwargs['cmd']))
    assert not edges and len(nodes) == len(roots)
    i = roots[int(i)]
    assert len(nodes[i]['sizes']) == 1
    nodes[i]['sizes'][0]['N'] = toFloat(N)

def _g(i,alpha, nodes, events, edges, roots, toFloat, **kwargs):
    assert roots
    if events:
        raise IOError(("-g,-G should be called before any demographic changes", kwargs['cmd']))
    assert not edges and len(nodes) == len(roots)
    i = roots[int(i)]
    assert len(nodes[i]['sizes']) == 1
    if toFloat(alpha) == 0.0 and alpha[0] != "$":
        alpha = None
    else:
        alpha = toFloat(alpha)
    nodes[i]['sizes'][0]['alpha'] = alpha

def _G(rate, roots, nodes, **kwargs):
    assert roots
    for i in roots:
        if roots[i] is not None:
            _g(i, rate, roots=roots, nodes=nodes, **kwargs)

def _I(*args, **kwargs):
    # -I should be called first, so kwargs should be empty
    assert all([not kwargs[x] for x in 'roots','events','edges','nodes'])

    npop = int(args[0])
    lins_per_pop = map(int,args[1:])
    if len(lins_per_pop) != npop:
        raise IOError(("Bad args for -I. Note continuous migration is not implemented.", kwargs['cmd']))

    for i in range(npop):
        leaf_name = i+1
        kwargs['nodes'][leaf_name] = {'sizes':[{'t':0.0,'N':1.0,'alpha':None}],'lineages':lins_per_pop[i]}
        kwargs['roots'][i+1] = leaf_name

def set_model(node_data, end_time, cmd):
    # add 'model_func' to node_data, add information to node_data['sizes']
    sizes = node_data['sizes']
    # add a dummy epoch with the end time
    sizes.append({'t': end_time})

    # do some processing
    N, alpha = sizes[0]['N'], sizes[0]['alpha']
    pieces = []
    for i in range(len(sizes) - 1):
        sizes[i]['tau'] = tau = (sizes[i+1]['t'] - sizes[i]['t']) * 2.0

        if 'N' not in sizes[i]:
            sizes[i]['N'] = N
        if 'alpha' not in sizes[i]:
            sizes[i]['alpha'] = alpha
        alpha = sizes[i]['alpha']
        N = sizes[i]['N']

        if alpha is not None:
            pieces.append(ExponentialHistory(tau=tau, growth_rate=alpha/2.0, N_bottom=N))
            N = pieces[-1].N_top
        else:
            pieces.append(ConstantHistory(tau=tau, N=N))

        sizes[i]['N_top'] = N

        if not all([sizes[i][x] >= 0.0 for x in 'tau','N','N_top']):
            raise IOError(("Negative time or population size. (Were events specified in correct order?", cmd))
    sizes.pop() # remove the final dummy epoch

    assert len(pieces) > 0
    if len(pieces) == 0:
        node_data['model'] = pieces[0]
    else:
        node_data['model'] = PiecewiseHistory(pieces)


'''Simulate SFS from Demography. Call from demography.simulate_sfs instead.'''
def simulate_sfs(demo, num_sims, theta=None, seed=None, additionalParams=""):
    if any([(x in additionalParams) for x in "-t","-T","seed"]):
        raise IOError("additionalParams should not contain -t,-T,-seed,-seeds")

    lins_per_pop = [demo.n_lineages(l) for l in sorted(demo.leaves)]
    n = sum(lins_per_pop)
    pops_by_lin = []
    for pop in range(len(lins_per_pop)):
        for i in range(lins_per_pop[pop]):
            pops_by_lin.append(pop)
    assert len(pops_by_lin) == int(n)

    scrm_args = demo.ms_cmd
    if additionalParams:
        scrm_args = "%s %s" % (scrm_args, additionalParams)

    if seed is None:
        seed = random.randint(0,999999999)
    scrm_args = "%s --seed %s" % (scrm_args, str(seed))

    assert scrm_args.startswith("-I ")
    if not theta:
        scrm_args = "-T %s" % scrm_args
    else:
        scrm_args = "-t %f %s" % (theta, scrm_args)
    scrm_args = "%d %d %s" % (n, num_sims, scrm_args)

    #lines = sh.Command(os.environ["MSPATH"])(*ms_cmd.split(),_ok_code=[0,16,17,18])
    lines = sh.Command(os.environ["SCRM_PATH"])(*scrm_args.split())

    def f(x):
        if x == "//":
            f.i += 1
        return f.i
    f.i = 0
    runs = itertools.groupby((line.strip() for line in lines), f)
    next(runs)
    sumCounts = Counter()
    sumSqCounts = Counter()
    nonzeroCounts = Counter()
    for i, lines in runs:
        lines = list(lines)
        if theta:
            currCounts = read_empirical_sfs(lines, len(lins_per_pop), pops_by_lin)
        else:
            currCounts = read_tree_lens(lines, len(lins_per_pop), pops_by_lin)

        for config in currCounts:
            sumCounts[config] += currCounts[config]
            sumSqCounts[config] += currCounts[config]**2
            nonzeroCounts[config] += 1
    return sumCounts,sumSqCounts,nonzeroCounts

def read_empirical_sfs(lines, num_pops, pops_by_lin):
    currCounts = Counter()
    n = len(pops_by_lin)

    assert lines[0] == "//"
    nss = int(lines[1].split(":")[1])
    if nss == 0:
        return currCounts
    # remove header
    lines = lines[3:]
    # remove trailing line if necessary
    if len(lines) == n+1:
        assert lines[n] == ''
        lines = lines[:-1]
    # number of lines == number of haplotypes
    assert len(lines) == n
    # columns are snps
    for column in range(len(lines[0])):
        dd = [0] * num_pops
        for i, line in enumerate(lines):
            dd[pops_by_lin[i]] += int(line[column])
        assert sum(dd) > 0
        currCounts[tuple(dd)] += 1
    return currCounts

def read_tree_lens(lines, num_pops, pops_by_lin):
    assert lines[0] == "//"
    return NewickSfs(lines[1], num_pops, pops_by_lin).sfs

class NewickSfs(newick.tree.TreeVisitor):
    def __init__(self, newick_str, num_pops, pops_by_lin):
        self.tree = newick.parse_tree(newick_str)
        self.sfs = Counter()
        self.num_pops = num_pops
        self.pops_by_lin = pops_by_lin

        self.tree.dfs_traverse(self)

    def pre_visit_edge(self,src,b,len,dst):
        dd = [0] * self.num_pops
        # get the # lineages in each pop below edge
        for leaf in dst.get_leaves_identifiers():
            dd[self.pops_by_lin[int(leaf)-1]] += 1
        # add length to the sfs entry. multiply by 2 cuz of ms format
        self.sfs[tuple(dd)] += len * 2.0


'''Construct ms cmd line from newick (for back compatibility)'''
## TODO: solve back compatibitily issues and remove newick format
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

def _from_newick(newick, default_lins=None, default_N = 1.0):
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
    return _newick_nx_to_cmd(t, default_lins, default_N)

def _newick_nx_to_cmd(newick_nx, default_lins, default_N):
    leafs = sorted([k for k, v in newick_nx.out_degree().items() if v == 0])
    
    nd = newick_nx.node

    for l in leafs:
        if 'lineages' not in newick_nx.node[l]:
            nd[l]['lineages'] = default_lins
        nd[l]['height'] = 0.0
    
    cmd = "-I %d" % len(leafs)
    cmd = "%s %s" % (cmd, " ".join([str(nd[l]['lineages']) for l in leafs]))

    roots = {l : i for i,l in enumerate(leafs,start=1)}
    
    # add heights
    for parent in nx.dfs_postorder_nodes(newick_nx):
        for child in newick_nx[parent]:
            ed = newick_nx[parent][child]
            h = nd[child]['height'] + ed['branch_length']
            if 'height' in nd[parent]:
                assert abs(nd[parent]['height'] - h) < 1e-12
            else:
                nd[parent]['height'] = h

    # add sizes
    for node in newick_nx:
        node = newick_nx.node[node]
        if "model" in node and node['model'] != 'constant':
            raise NotImplementedError("Unsupported model type: %s" % nd['model'])
        if 'N' not in node:
            node['N'] = default_N

    # add leaf sizes to cmd
    for i,l in enumerate(leafs,start=1):
        if nd[l]['N'] != 1.0:
            cmd = "%s -n %d %f" % (cmd, i, nd[l]['N'])

    # add other sizes and join-on events
    for node in nx.dfs_postorder_nodes(newick_nx):
        if node in leafs:
            continue
        
        c1,c2 = newick_nx[node]
        t = newick_nx.node[node]['height'] / 2.0
        cmd = "%s -ej %f %d %d" % (cmd, t, roots[c1], roots[c2])
        roots[node] = roots[c2]
        del roots[c1], roots[c2]

        N = newick_nx.node[node]['N']
        if N != newick_nx.node[c2]['N']:
            cmd = "%s -en %f %i %f" % (cmd, t, roots[node], N)
    return cmd

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
