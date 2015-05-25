#!/usr/bin/env python2.7
import time
import argparse
from subprocess import check_output
import os
import sqlite3
import random
import networkx as nx

from sum_product import SumProduct
from huachen_eqs import SumProduct_Chen
from demography import Demography

from collections import Counter, defaultdict
import itertools

import multiprocessing as mp
import numpy as np

conn = sqlite3.connect('.bench.db')
cur = conn.cursor()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_taxa", type=int)
    parser.add_argument("lineages_per_taxon", type=int)
    parser.add_argument("reps", type=int)
    parser.add_argument("cores", type=int)
    parser.add_argument("--reset", action="store_true", help="Reset results database")
    parser.add_argument("--moranOnly", action="store_true", help="Only do Moran SFS")
    args = parser.parse_args()
    if args.reset:
        cur.execute("drop table results")
        create_table()
    
    if args.cores > 1:
        pool = mp.Pool(processes=args.cores)
        results_list = pool.map(time_runs, [(args.n_taxa,args.lineages_per_taxon,args.moranOnly)] * args.reps)
    else:
        results_list = map(time_runs, [(args.n_taxa,args.lineages_per_taxon,args.moranOnly)] * args.reps)
    for results in results_list:
        for entry in results:
            store_result(*entry)
    conn.commit()
    conn.close()

def time_runs(args):
    n_taxa, lineages_per_taxon, moranOnly = args
    # Get a random phylogeny
    tree_str = random_binary_tree(n_taxa)
    tree = Demography.from_newick(tree_str, lineages_per_taxon)
    #print tree.to_newick()
    n = n_taxa * lineages_per_taxon
    results = []
    snp_list = run_simulation(tree, 100, lineages_per_taxon)
    for snp,state in enumerate(snp_list):
        #print(state)
        state_tuple = tuple([v['derived'] for k,v in sorted(state.iteritems())])
        #print(state_tuple)
        tree.update_state(state)
        rid = random.getrandbits(32)

        sp_list = [("moran",SumProduct)]
        if not moranOnly:
            sp_list += [("chen",SumProduct_Chen)]
        for name,method in sp_list:
            #print(name)
            with Timer() as t:
                ret = method(tree).p()
            #print(ret)
            results.append((name,n_taxa, lineages_per_taxon, snp, t.interval, ret, rid, str(state_tuple), tree_str))
    return results

def create_table():
    cur.execute("""create table results (model varchar, n integer, """
                 """lineages integer, site integer, time real, result real, run_id integer, state varchar, tree varchar)""")

def store_result(name, n, l, i, t, res, rid, state, tree):
    cur.execute("insert into results values (?, ?, ?, ?, ?, ?, ?, ?, ?)", (name, n, l, i, t, res, rid, state, tree))

try:
    create_table()
except sqlite3.OperationalError:
    # conn.execute("delete from results)"
    pass

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        #print('Call took %.03f sec.' % self.interval)

def random_binary_tree(n):
    g = nx.DiGraph()
    nodes = ["l%i" % i for i in range(n)]
    i = 0

    lengths = {v: 0.0 for v in nodes}
    for _ in range(n - 1):
        coal = random.sample(nodes, 2)
        int_node = "i%i" % i

        t = random.expovariate(1) / float(n) * 2.0
        lengths = {k: v + t for k, v in lengths.items()}

        g.add_edges_from([(int_node, c, {'edge_length': lengths[c]}) for c in coal])
        
        lengths[int_node] = 0.0

        i += 1
        nodes = list((set(nodes) - set(coal)) | set([int_node]))
    assert len(nodes) == 1
    return newick_helper(g, nodes[0])

def newick_helper(g, node):
    children = g.successors(node)
    el = None
    try:
        parent = g.predecessors(node)[0]
        el = g[parent][node]['edge_length']
    except IndexError:
        parent = None
    if children:
        ret = "(%s,%s)" % tuple([newick_helper(g, c) for c in children])
        if parent is not None:
            ret += ":%f" % el
        return ret
    else:
        ret = "%s" % node
        if el is not None:
            ret += ":%f" % el
        return ret

SCRM_PATH = os.environ['SCRM_PATH']

def build_command_line(demo, L, lineages_per_taxon):
    '''Given a tree, build a scrm command line which will simulate from it.'''
    ejopts = []
    Iopts = []
    
    tfac = 0.5 
    theta = 1.

    lineages = []
    lineage_map = {}
    for i, leaf_node in list(enumerate(sorted(demo.leaves), 1)):
        nsamp = lineages_per_taxon
        Iopts.append(nsamp)
        lineage_map[leaf_node] = i
        lineages += [leaf_node] * nsamp
        age = demo.node_data[leaf_node]['model'].tau * tfac

        p, = demo.predecessors(leaf_node)
        while True:
            if p not in lineage_map:
                lineage_map[p] = i
                tau = demo.node_data[p]['model'].tau
                #if p.edge_length == float("inf"):
                if tau == float('inf'):
                    break
                age += tau * tfac
                old_p = p
                p, = demo.predecessors(p)
            else:
                # We have a join-on time
                ejopts.append((age, i, lineage_map[p]))
                break

    cmdline = ["-I %d %s" % (len(Iopts), " ".join(map(str, Iopts)))]
    for ej in ejopts:
        cmdline.append("-ej %g %d %d" % ej)
    cmdline = ["%s %d 1 -t %g" % (SCRM_PATH, sum(Iopts), theta)] + cmdline
    #print(cmdline)
    return lineages, " ".join(cmdline)

def run_simulation(tree, L, lineages_per_taxon):
    lineages, cmd = build_command_line(tree, L, lineages_per_taxon)
    species = list(set(lineages))
    n_lineages = Counter(lineages)

    #print(cmd)
    output = [l.strip() for l in check_output(cmd, shell=True).split("\n")]
    def f(x):
        if x == "//":
            f.i += 1
        return f.i
    f.i = 0 
    for k, lines in itertools.groupby(output, f):
        if k == 0:
            continue
        # Skip preamble
        next(lines)
        # segsites
        segsites = int(next(lines).split(" ")[1])
        # positions
        next(lines)
        # at haplotypes
        lin_counts = defaultdict(lambda: np.zeros(segsites, dtype=int))
        for hap, lin in zip(lines, lineages):
            hap = list(map(int, hap))
            lin_counts[lin] += hap
    return [{lin: {'derived':lin_counts[lin][i], 
                   'ancestral':n_lineages[lin] - lin_counts[lin][i]}
             for lin in lineages}
            for i in range(segsites)]
   
if __name__=="__main__":
    main()
