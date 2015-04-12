import networkx as nx
from demography import Demography
from numpy import log, exp ## TODO: import from adarray.admath
from size_history import ExponentialTruncatedSizeHistory as ExpHist
from size_history import ConstantTruncatedSizeHistory as ConstHist
from size_history import PiecewiseHistory

def get_demo(ms_cmd):
    # if ms_cmd = "3 1 -t 1.0 -T -I 2 1 2"
    # then cmd_list = [[3,1], [t,1.0], [T], [I,2,1,2]]
    cmd_list = []
    curr_args = []
    cmd_list.append(curr_args)

    for arg in ms_cmd.split():
        if arg[0] == '-' and arg[1].isalpha():
            curr_args = [arg[1:]]
            cmd_list.append(curr_args)
        else:
            curr_args.append(arg)

    n,_ = map(int, cmd_list[0])
    cmd_list = cmd_list[1:]

    # remove all cmds in ignore_params
    cmd_list = [cmd for cmd in cmd_list if cmd[0] not in ignore_params]

    ## TODO: convert roots to [None] (will be useful for adding pops with -es)
    kwargs = {'events':[],'edges':[],'nodes':{},'roots':{},'n':n,'cmd':ms_cmd}
    for cmd in cmd_list:
        if cmd[0] not in valid_params:
            raise NotImplementedError("-%s not implemented" % cmd[0])
        cmdfunc, args = eval("_" + cmd[0]), cmd[1:]
        cmdfunc(*args, **kwargs)

    return demo(**kwargs)

valid_params = set(["G","I","n","g","eG","eg","eN","en","ej"])
ignore_params = set(["seeds","seed","t","s","T","L","p","r","c"])

def demo(events, edges, nodes, roots, n, cmd, **kwargs):
    print cmd
    if not nodes:
        assert not any([events, edges, roots])
        _I(1, n, edges=edges, nodes=nodes, roots=roots, n=n, cmd=cmd)
    if len(roots) != 1:
        raise IOError(("Must have a single root population", cmd))
    
    root, = list(roots.iteritems())
    _,node = root

    if 'alpha' in nodes[node]['sizes'][-1] and nodes[node]['sizes'][-1]['alpha'] is not None:
        raise IOError(("Root ancestral population must not have growth parameter",cmd))
    set_model(nodes[node], float('inf'))

    ret = nx.DiGraph(edges)
    for v in nodes:
        ret.add_node(v, **(nodes[v]))
        #ret.node[v].update(nodes[v])

    demography = Demography(ret)
    demography.graph['cmd'] = cmd
    demography.graph['events'] = event_tree(events, demography)
    return demography

def event_tree(events, demography):
    ## TODO: need to implement this for pulse migration
    pass

def _ej(t,i,j, events, nodes, roots, edges, **kwargs):
    t = float(t)

    for k in i,j:
        # sets the TruncatedSizeHistory, and N_top and alpha for all epochs
        set_model(nodes[roots[k]], t)

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
    for i in roots:
        _en(t, i, N, roots=roots, **kwargs)

def _eg(t,i,alpha, roots, nodes, **kwargs):
    t,alpha = map(float, [t,alpha])
    nodes[roots[i]]['sizes'].append({'t':t,'alpha':alpha})

def _eG(t,alpha, roots, **kwargs):
    for i in roots:
        _eg(t,i,alpha, roots=roots, **kwargs)

def _n(i,N, nodes, events, edges, roots, **kwargs):
    if not nodes or events:
        raise IOError(("-n should be called after -I and before any demographic changes", kwargs['cmd']))
    assert not edges and len(nodes) == len(roots)

    assert len(nodes[i]['sizes']) == 1
    nodes[i]['sizes'][0]['N'] = float(N)

def _g(i,alpha, nodes, events, edges, roots, **kwargs):
    if not nodes or events:
        raise IOError(("-g should be called after -I and before any demographic changes", kwargs['cmd']))
    assert not edges and len(nodes) == len(roots)

    assert len(nodes[i]['sizes']) == 1
    nodes[i]['sizes'][0]['alpha'] = float(alpha)

def _G(rate, roots, nodes, n, **kwargs):
    if not nodes:
        _I(1,n, n=n, roots=roots, nodes=nodes, **kwargs)  
    for i in roots:
        _g(i, rate, n=n, roots=roots, nodes=nodes, **kwargs)

def _I(*args, **kwargs):
    if kwargs['nodes']:
        raise IOError(("-I should be called earlier", kwargs['cmd']))
    assert all([not kwargs[x] for x in 'roots','events','edges'])
    npop = int(args[0])
    lins_per_pop = map(int,args[1:])
    if len(lins_per_pop) != npop or sum(lins_per_pop) != kwargs['n']:
        raise IOError(("Bad args for -I. Note continuous migration is not implemented.", kwargs['cmd']))

    kwargs['nodes'].update({str(i+1) : {'sizes':[{'t':0.0,'N':1.0,'alpha':None}],'lineages':lins_per_pop[i]} for i in range(npop)})
    kwargs['roots'].update({k : k for k in kwargs['nodes']})


def set_model(node_data, end_time):
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

        assert all([sizes[i][x] >= 0.0 for x in 'tau','N','N_top'])
    sizes.pop() # remove the final dummy epoch

    # construct function that returns size history from number of lineages
    def model_func(n_max):
        pieces = []
        for size in sizes:
            print size
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
