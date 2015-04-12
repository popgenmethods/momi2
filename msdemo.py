import networkx as nx
from demography import Demography
from numpy import log, exp ## TODO: import from adarray.admath
from size_history import ExponentialTruncatedSizeHistory as ExpHist
from size_history import ConstantTruncatedSizeHistory as ConstHist
from size_history import PiecewiseHistory

def get_demo(ms_cmd):
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

    # remove all cmds in ignore_params
    #cmd_list = [cmd for cmd in cmd_list if cmd[0] not in ignore_params]

    ## TODO: convert roots to [None] (will be useful for adding pops with -es)
    kwargs = {'events':[],'edges':[],'nodes':{},'roots':{},'cmd':ms_cmd}
    for cmd in cmd_list:
        if cmd[0] not in valid_params:
            raise NotImplementedError("-%s not implemented" % cmd[0])
        cmdfunc, args = eval("_" + cmd[0]), cmd[1:]
        cmdfunc(*args, **kwargs)

    return demo(**kwargs)

valid_params = set(["G","I","n","g","eG","eg","eN","en","ej"])
#ignore_params = set(["seeds","seed","t","s","T","L","p","r","c"])

def demo(events, edges, nodes, roots, cmd, **kwargs):
    #print cmd
    assert nodes
    if len(roots) != 1:
        raise IOError(("Must have a single root population", cmd))
    
    root, = list(roots.iteritems())
    _,node = root

    if 'alpha' in nodes[node]['sizes'][-1] and nodes[node]['sizes'][-1]['alpha'] is not None:
        raise IOError(("Root ancestral population must not have growth parameter",cmd))
    set_model(nodes[node], float('inf'), cmd)

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
