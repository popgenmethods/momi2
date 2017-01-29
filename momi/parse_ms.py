
import bisect
import networkx as nx

from .size_history import ConstantHistory, ExponentialHistory, PiecewiseHistory
from .data_structure import seg_site_configs
from autograd.numpy import isnan, exp, array, ones

import random
import subprocess
import itertools
from collections import Counter, defaultdict
from io import StringIO

from operator import itemgetter

def seg_sites_from_ms(ms_file, sampled_pops):
    lines = ms_file.readlines()

    if " -T " in lines[0]:
        ## TODO: implement this
        raise NotImplementedError("Reading SFS from ms file with -T not yet implemented")
    
    firstline = lines[0].replace('\ ','\_').split() # don't split up escaped spaces in /ms/path

    # get out pops_by_lin
    pops_by_lin = []
    npop = 1
    for i,flag in enumerate(firstline):
        if flag == "-I":
            npop = int(firstline[i+1])
        elif flag == "-es":
            npop += 1

        if flag == "-I" or flag == "-eI":
            for j,nn in enumerate(firstline[(i+2):(i+2+npop)]):
                for _ in range(int(nn)):
                    pops_by_lin += [j]
    if pops_by_lin == []:
        pops_by_lin = [0]*int(firstline[1])
    
    def f(x):
        if x == "//":
            f.i += 1
        return f.i
    f.i = 0
    runs = itertools.groupby((line.strip() for line in lines), f)
    next(runs)

    configs = (_snp_sequence_from_1_ms_sim(list(lines), pops_by_lin)
               for i,lines in runs)
    return seg_site_configs(sampled_pops, configs)
    #if sampled_pops is not None:
    #    ret = mylist(ret, sampled_pops=tuple(sampled_pops))
    #return ret

def simulate_ms(ms_path, demo, num_loci, mut_rate, seeds=None, additional_ms_params="", cmd_format="ms", raw_output=False, sampled_pops=None, sampled_n=None):
    """
    Use ms to simulate from a Demography, and get an open file object containing the output.

    Parameters
    ----------
    ms_path : str
         path to ms or similar program (e.g. scrm)
    demo : Demography.
         must have demo.default_N == 1.0, otherwise throws exception
         (you can call demo.rescaled() to rescale the demography to correct units)
    num_loci : int
         number of independent loci
    mut_rate : float
         rate of mutations occurring per unit time, per locus.
    seeds : optional, iterable
         a list or tuple of 3 seeds
    additional_ms_params : optional, str
         additional commands to append to the ms command, e.g. recombination.
    cmd_format: optional, str
         Default is "ms". If "scrm", allows demography to have archaic populations;
         they will be specified in the scrm format. In this case, ms_path must be
         a path to scrm (or program that uses the same input format)
    raw_output : optional, bool
         If True, returns the raw ms output as a file object.
         If False (the default), returns a momi.SegSites object

    Returns
    -------
    If raw_output=False (the default), returns a momi.SegSites object.
    Otherwise, returns open readable file-like object (StringIO) containing the ms output

    See Also
    --------
    sfs_list_from_ms : convert ms file output to a list of SFS
    to_ms_cmd : converts a Demography to a partial ms command
    """   
    demo = demo._get_multipop_moran(sampled_pops, sampled_n)

    if any([(x in additional_ms_params) for x in ("-t","-T","seed")]):
        raise IOError("additional_ms_params should not contain -t,-T,-seed,-seeds")

    lins_per_pop = list(demo.sampled_n)
    n = sum(lins_per_pop)

    ms_args = to_ms_cmd(demo, cmd_format=cmd_format)
    if additional_ms_params:
        ms_args = "%s %s" % (ms_args, additional_ms_params)

    if seeds is None:
        seeds = [random.randint(0,999999999) for _ in range(3)]
    seeds = " ".join([str(s) for s in seeds])
    ms_args = "%s -seed %s" % (ms_args, seeds)

    assert ms_args.startswith("-I ")
    ms_args = "-t %g %s" % (mut_rate, ms_args)
    ms_args = "%d %d %s" % (n, num_loci, ms_args)
   
    ret = run_ms(ms_args, ms_path=ms_path)
    if not raw_output:
        ret = seg_sites_from_ms(ret, demo.sampled_pops)
    return ret

def run_ms(ms_args, ms_path):
    """
    Runs the command
    "%s %s" % (ms_path, ms_args)
    and returns the output as a file-like object
    """
    try:
        lines = subprocess.check_output([ms_path] + ms_args.split(),
                                        stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        ## ms gives really weird error codes, so ignore them
        lines = e.output
    return StringIO(lines.decode('unicode-escape'))

def to_ms_cmd(demo, cmd_format="ms"):
    """
    Converts a Demography to a partial ms command.

    Parameters
    ----------
    demo : Demography.
         must be in ms units (i.e., demo.default_N == 1.0), otherwise throws exception
         (you can call demo.rescaled() to rescale the demography to correct units)
    cmd_format: optional, str
         Default is "ms". If "scrm", allows demography to have archaic populations,
         specified in the scrm format.
    
    Returns
    -------
    ms_demo_str : str
         the Demography in ms syntax
    """
    if cmd_format not in ("ms","scrm"):
        raise Exception("Unrecognized cmd_format")
    
    if demo.default_N != 1.0:
        raise Exception("Please rescale demography to be in ms units (i.e. default_N=1.0). This can be done with Demography.rescaled().")

    sampled_t = demo.sampled_t
    if sampled_t is None:
        sampled_t = 0.0
    sampled_t = array(sampled_t) * ones(len(demo.sampled_pops))
    
    if cmd_format == "ms" and not all(sampled_t == 0.0):
        raise Exception("ms command line doesn't allow for archaic leaf populations. Try setting cmd_format='scrm'")

    pops = dict(list(zip(demo.sampled_pops, list(range(1,len(demo.sampled_pops)+1)))))
    sampled_n = demo.sampled_n
       
    npop = len(pops)

    sizes = defaultdict(list)
    def get_size_events(pop, time):
        cur_size, cur_growth, cur_t = 1.0, 0.0, 0.0
        for event in sizes[pop]:
            flag = event[0]
            if flag == '-en':
                flag,t,i,N = event
                cur_size, cur_growth, cur_t = N,0.,t
            else:
                assert flag == '-eg'
                flag,t,i,g = event
                cur_size = exp( (cur_t-t) * cur_growth) * cur_size
                cur_growth, cur_t = g,t
        cur_size = exp( (cur_t-time) * cur_growth) * cur_size
        return [('-en',time,pops[pop],cur_size),
                ('-eg',time,pops[pop],cur_growth)]

    events = []
    ret = []
    assert min(sampled_t) == 0.0
    for t in sorted(set(sampled_t)):
        curr = list(array(sampled_n * (sampled_t == t), dtype=int))
        if t == 0.0:
            curr = ["-I", len(sampled_n)] + curr
            ret += [tuple(curr)]
        else:
            curr = ["-eI",t] + curr
            events += [tuple(curr)]

    events += demo.events
    events = sorted(events, key=lambda x:x[1])
    for event in events:
        flag = event[0]
        if flag == '-ep':
            _,t,i,j,pij = event
            ret += [('-es',t,pops[i],1.-pij)]
            npop += 1
            if j not in pops:
                pops[j] = npop
                ret += get_size_events(j,t)                
            else:
                ret += [('-ej',t,npop, pops[j])]
            continue
        elif flag == '-ej':
            _,t,i,j = event
            if j not in pops:
                pops[j] = pops[i]
                ret += get_size_events(j,t)
                continue
            else:
                event = (flag,t,pops[i],pops[j])
        elif flag == '-eg':
            _,t,i,alpha = event
            sizes[i] += [event]
            if i not in pops:
                continue
            event = (flag,t,pops[i],alpha)            
        elif flag == '-en':
            _,t,i,N = event
            sizes[i] += [event]
            if i not in pops:
                continue
            event = (flag,t,pops[i],N)
        elif flag == '-eI':
            event = list(event) + [0]*(npop+2-len(event))
            event = tuple(event)
        else:
            assert False
        ret += [event]
    return " ".join(sum([list(map(str,x)) for x in ret], []))

def _convert_ms_cmd(cmd_list, params):
    cmd_list = list(cmd_list)
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

    pops_map = dict(list(zip(pops_by_time, list(range(1, len(pops_by_time)+1)))))
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
        #cmd[0] = "-" + cmd[0]

    cmd_list = [['d','1']] + cmd_list
    return cmd_list

def _snp_sequence_from_1_ms_sim(lines, pops_by_lin):
    n_at_leaves = [v for k,v in sorted(Counter(pops_by_lin).items())]
    num_pops = len(n_at_leaves)
    #ret = []
    n = len(pops_by_lin)

    assert lines[0] == "//"
    nss = int(lines[1].split(":")[1])
    if nss == 0:
        return
    positions = list(map(float, lines[2].split(":")[1].strip().split()))
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
        yield tuple([(n-d,d) for n,d in zip(n_at_leaves,dd)])
        #ret += []
    #return zip(positions,ret)
    #return positions, ret
    #return ret
