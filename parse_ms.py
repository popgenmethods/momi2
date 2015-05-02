import networkx as nx
from cStringIO import StringIO
from Bio import Phylo
from size_history import ConstantHistory, ExponentialHistory, PiecewiseHistory

from autograd.numpy import isnan, exp,min

import newick
import sh, os, random
import itertools
from collections import Counter

def sort_ms_cmd(args_list, params):
    ret = {}
    for arg in args_list:
        if arg[0].startswith("e"):
            t = params.getfloat(arg[1])
        else:
            t = 0.0
        if t not in ret:
            ret[t] = []
        ret[t].append(arg)
    ret = sum([x for t,x in sorted(ret.iteritems())], [])
    return ret

'''
Construct networkx DiGraph from ms command.
Use demography.make_demography instead of calling this directly.
'''
def _to_nx(ms_cmd, *params):
    get_params = GetParams(params)

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

    cmd_list = sort_ms_cmd(cmd_list, get_params)
    ms_cmd = ["-" + " ".join([str(x) for x in arg]) for arg in cmd_list]
    ms_cmd = " ".join(ms_cmd)
    # replace variables in the ms cmd string
    ### TODO: fix this try/except! it's needed in general, but breaks for autograd
    try:
        ## MUST do this in reversed order, otherwise ($26) will turn into ($2)6
        for i in reversed(range(len(params))):
            ms_cmd = ms_cmd.replace("$" + str(i), str(params[i]))
    except:
        pass

    # now parse the ms cmd, store results in kwargs
    kwargs = {'events':[],'edges':[],'nodes':{},'roots':{},'cmd':ms_cmd,'params':get_params}
    for cmd in cmd_list:
        if cmd[0] not in valid_params:
            raise NotImplementedError("-%s not implemented" % cmd[0])
        cmdfunc, args = eval("_" + cmd[0]), cmd[1:]
        cmdfunc(*args, **kwargs)

    # return nx.DiGraph from parsed ms cmd
    return _nx_from_parsed_ms(**kwargs)

valid_params = set(["G","I","n","g","eG","eg","eN","en","ej","es"])

def _es(t,i,p, events, nodes, roots, edges, cmd, params, **kwargs):
    t,p = map(params.getfloat, (t,p))
    i = params.getint(i)

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

def _ej(t,i,j, events, nodes, roots, edges, cmd, params, **kwargs):
    t = params.getfloat(t)
    i,j = map(params.getint, [i,j])

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

def _en(t,i,N, nodes, roots, params, **kwargs):
    t,N = map(params.getfloat, [t,N])
    i = params.getint(i)
    nodes[roots[i]]['sizes'].append({'t':t,'N':N,'alpha':None})

def _eN(t,N, roots, **kwargs):
    assert roots
    for i in roots:
         if roots[i] is not None:
             _en(t, i, N, roots=roots, **kwargs)

def _eg(t,i,alpha, roots, nodes, params, **kwargs):
    if params.getfloat(alpha) == 0.0 and alpha[0] != "$":
        alpha = None
    else:
        alpha = params.getfloat(alpha)
    t,i = params.getfloat(t), params.getint(i)
    nodes[roots[i]]['sizes'].append({'t':t,'alpha':alpha})

def _eG(t,alpha, roots, **kwargs):
    assert roots
    for i in roots:
        if roots[i] is not None:
            _eg(t,i,alpha, roots=roots, **kwargs)

def _n(i,N, nodes, events, edges, roots, params, **kwargs):
    assert roots
    if events:
        raise IOError(("-n should be called before any demographic changes", kwargs['cmd']))
    assert not edges and len(nodes) == len(roots)
    i = roots[params.getint(i)]
    assert len(nodes[i]['sizes']) == 1
    nodes[i]['sizes'][0]['N'] = params.getfloat(N)

def _g(i,alpha, nodes, events, edges, roots, params, **kwargs):
    assert roots
    if events:
        raise IOError(("-g,-G should be called before any demographic changes", kwargs['cmd']))
    assert not edges and len(nodes) == len(roots)
    i = roots[params.getint(i)]
    assert len(nodes[i]['sizes']) == 1
    if params.getfloat(alpha) == 0.0 and alpha[0] != "$":
        alpha = None
    else:
        alpha = params.getfloat(alpha)
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

class GetParams(object):
    def __init__(self, params):
        self.params = list(params)

    def get(self, var, vartype):
        if isinstance(var,vartype):
            return var
        if var[0] == "$":
            ret = self.params[int(var[1:])]
        else:
            ret = vartype(var)
        if isnan(ret):
            raise Exception("nan in params %s" % (str(self.params)))
        return ret

    def getint(self, var):
        return self.get(var, int)

    def getfloat(self, var):
        return self.get(var, float)

def _nx_from_parsed_ms(events, edges, nodes, roots, cmd, **kwargs):
    assert nodes
    roots = [r for _,r in roots.iteritems() if r is not None]

    if len(roots) != 1:
        raise IOError(("Must have a single root population", cmd))
    
    node, = roots
    set_model(nodes[node], float('inf'), cmd)

    ret = nx.DiGraph(edges, cmd=cmd, events=events)
    for v in nodes:
        ret.add_node(v, **(nodes[v]))
    return ret

def set_model(node_data, end_time, cmd):
    # add 'model_func' to node_data, add information to node_data['sizes']
    sizes = node_data['sizes']
    # add a dummy epoch with the end time
    sizes.append({'t': end_time})

    # do some processing
    N, alpha = sizes[0]['N'], sizes[0]['alpha']
    pieces = []
    for i in range(len(sizes) - 1):
        #sizes[i]['tau'] = tau = (sizes[i+1]['t'] - sizes[i]['t']) * 2.0
        sizes[i]['tau'] = tau = (sizes[i+1]['t'] - sizes[i]['t'])

        if 'N' not in sizes[i]:
            sizes[i]['N'] = N
        if 'alpha' not in sizes[i]:
            sizes[i]['alpha'] = alpha
        alpha = sizes[i]['alpha']
        N = sizes[i]['N']

        if alpha is not None:
            #pieces.append(ExponentialHistory(tau=tau, growth_rate=alpha/2.0, N_bottom=N))
            pieces.append(ExponentialHistory(tau=tau, growth_rate=alpha, N_bottom=N))
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
    if theta:
        return [read_empirical_sfs(list(lines), len(lins_per_pop), pops_by_lin)
                for i,lines in runs]
    else:
        return [read_tree_lens(list(lines), len(lins_per_pop), pops_by_lin)
                for i, lines in runs]

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
        #self.sfs[tuple(dd)] += len * 2.0
        self.sfs[tuple(dd)] += len
