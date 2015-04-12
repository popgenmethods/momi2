import networkx as nx
from size_history import ExponentialTruncatedSizeHistory as ExpHist
from size_history import ConstantTruncatedSizeHistory as ConstHist
from size_history import PiecewiseHistory

from numpy import log, exp ## TODO: import from adarray.admath

import newick
import sh, os, random
import itertools
from collections import Counter

'''Construct networkx DiGraph from ms command'''
def to_nx(ms_cmd):
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


'''Simulate SFS from Demography'''
def simulate_sfs(demo, num_sims, theta=None, seed=None, additionalParams=""):
    if any([(x in additionalParams) for x in "-t","-T","seed"]):
        raise IOError("additionalParams should not contain -t,-T,-seed,-seeds")

    lins_per_pop = [demo.n_lineages_subtended_by[l] for l in sorted(demo.leaves)]
    n = sum(lins_per_pop)
    pops_by_lin = []
    for pop in range(len(lins_per_pop)):
        for i in range(lins_per_pop[pop]):
            pops_by_lin.append(pop)
    assert len(pops_by_lin) == int(n)

    scrm_args = demo.graph['cmd']
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
