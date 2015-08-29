from __future__ import division
import bisect
import networkx as nx

from demography import Demography, _demo_graph_from_str, _get_cmd_list, _ParamsMap
from size_history import ConstantHistory, ExponentialHistory, PiecewiseHistory
from util import default_ms_path

from autograd.numpy import isnan, exp

import random
import subprocess
import itertools
from collections import Counter
from cStringIO import StringIO

from operator import itemgetter

def to_ms_cmd(demo):
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
    

def make_demography(ms_cmd, *args, **kwargs):
    '''
    Returns a demography from partial ms command line

    See examples/example_sfs.py for more details
    '''
    params = _ParamsMap(args, kwargs)
    
    cmd_list = _get_cmd_list(ms_cmd)
    
    if cmd_list[0][0] != 'I':
        raise IOError("ms command must begin with -I to specify samples per population")
    n_pops = int(cmd_list[0][1])
    
    ## first replace the # sign convention
    pops_by_time = [(0.0, idx) for idx in range(1,n_pops+1)]
    for cmd in cmd_list:
        if cmd[0] == 'es':
            n_pops += 1
            pops_by_time += [(params.time(cmd[1]), n_pops)]
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
    
    time_events = [(params.time(cmd[1]), cmd) for cmd in events]
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

    cmd_list = [['-d','1']] + cmd_list
    return Demography(_demo_graph_from_str(" ".join(sum(cmd_list, [])), args, kwargs, add_pop_idx=-1))


def sfs_list_from_ms(ms_file, n_at_leaves):
    '''
    ms_file = file object containing output from ms
    n_at_leaves[i] = n sampled in leaf deme i

    Returns a list of empirical SFS's, one for each ms simulation
    '''
    lines = ms_file.readlines()

    def f(x):
        if x == "//":
            f.i += 1
        return f.i
    f.i = 0
    runs = itertools.groupby((line.strip() for line in lines), f)
    next(runs)

    pops_by_lin = []
    for pop in range(len(n_at_leaves)):
        for i in range(n_at_leaves[pop]):
            pops_by_lin.append(pop)

    return [_sfs_from_1_ms_sim(list(lines), len(n_at_leaves), pops_by_lin)
            for i,lines in runs]

def simulate_ms(demo, num_sims, theta, ms_path=default_ms_path(), seeds=None, additional_ms_params=""):
    '''
    Given a demography, simulate from it using ms

    Returns a file-like object with the ms output
    '''
    #if ms_path is None:
    #    ms_path = default_ms_path()

    if any([(x in additional_ms_params) for x in "-t","-T","seed"]):
        raise IOError("additional_ms_params should not contain -t,-T,-seed,-seeds")

    lins_per_pop = [demo.n_lineages(l) for l in sorted(demo.leaves)]
    n = sum(lins_per_pop)

    ms_args = to_ms_cmd(demo)
    if additional_ms_params:
        ms_args = "%s %s" % (ms_args, additional_ms_params)

    if seeds is None:
        seeds = [random.randint(0,999999999) for _ in range(3)]
    seeds = " ".join([str(s) for s in seeds])
    ms_args = "%s -seed %s" % (ms_args, seeds)

    assert ms_args.startswith("-I ")
    ms_args = "-t %f %s" % (theta, ms_args)
    ms_args = "%d %d %s" % (n, num_sims, ms_args)

    return run_ms(ms_args)

def run_ms(ms_args, ms_path=default_ms_path()):
    try:
        lines = subprocess.check_output([ms_path] + ms_args.split(),
                                        stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        ## ms gives really weird error codes, so ignore them
        lines = e.output
    return StringIO(lines)

'''
Helper functions for simulating SFS with ms.
'''

def _sfs_from_1_ms_sim(lines, num_pops, pops_by_lin):
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
