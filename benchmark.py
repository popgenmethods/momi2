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

import multiprocessing as mp

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
    
    pool = mp.Pool(processes=args.cores)
    for _ in range(args.reps):
        #time_runs(args.n_taxa, args.lineages_per_taxon, args.moranOnly)
        pool.apply_async(time_runs, args=(args.n_taxa,args.lineages_per_taxon,args.moranOnly))
    conn.close()

def time_runs(n_taxa, lineages_per_taxon, moranOnly):
    # Get a random phylogeny
    tree = Demography.from_newick(random_binary_tree(n_taxa), lineages_per_taxon)
    print tree.to_newick()
    n = n_taxa * lineages_per_taxon
    for snp in range(100):
        state = (0,) * len(tree.leaves)
        while sum(state) == 0 or sum(state) == n:
            state = [random.randint(0, lineages_per_taxon) for l in tree.leaves]
        state = {l: {'derived':d,'ancestral':lineages_per_taxon-d} for l,d in zip(sorted(tree.leaves),
                                                                                       state)}
        print(state)
        tree.update_state(state)
        rid = random.getrandbits(32)

        sp_list = [("moran",SumProduct)]
        if not moranOnly:
            sp_list += [("chen",SumProduct_Chen)]
        for name,method in sp_list:
            print(name)
            with Timer() as t:
                ret = method(tree).p()
            print(ret)
            store_result(name, n_taxa, lineages_per_taxon, snp, t.interval, ret, rid)
    conn.commit()

conn = sqlite3.connect('.bench.db')
cur = conn.cursor()

def create_table():
    cur.execute("""create table results (model varchar, n integer, """
                 """lineages integer, site integer, time real, result real, run_id integer)""")

def store_result(name, n, l, i, t, res, rid):
    cur.execute("insert into results values (?, ?, ?, ?, ?, ?, ?)", (name, n, l, i, t, res, rid))

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
        print('Call took %.03f sec.' % self.interval)

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

if __name__=="__main__":
    main()
