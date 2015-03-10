from size_history import ConstantTruncatedSizeHistory
from demography import Demography
import pytest
import networkx as nx
import random
from sum_product import SumProduct
from test_inference import run_scrm
#scrm = sh.Command(os.environ["SCRM_PATH"])
import numpy as np
from collections import Counter
import scipy, scipy.stats

def test_admixture_demo():
    leaf_lins = {'a' : 2, 'bcd': 2}
    theta = 1.0

    num_scrm_samples = 10000
    
    waitingTimes = np.random.random(5) * 2.0 + 0.1

    absoluteTimes = np.array(waitingTimes)
    for i in range(1,len(absoluteTimes)):
        absoluteTimes[i] += absoluteTimes[i-1]

    bcdSplit,bcSplit,bdJoin,abdJoin,abcdJoin = absoluteTimes

    bcdProb, bcProb = np.random.random(2)

    leaf_pops = ['a','bcd']
    # events: ('bc','d'), ('b','c'), 'bd', 'abd', 'abcd'
    eventList = [(('bc','bcd'), ('d','bcd')), # 'bcd' splits into 'bc','d'
                 (('b','bc'), ('c','bc')), # 'bc' splits into 'b','c'
                 (('bd','b'), ('bd','d')), # b,d coalesces into bd
                 (('abd','bd'),('abd','a')), # a,bd coalesce into abd
                 (('abcd','abd'),('abcd','c')), # abd,c coalesce into abcd
                 ]
    demoEdgeList = []
    for e1,e2 in eventList:
        demoEdgeList += [e1,e2]
    demo = nx.DiGraph(demoEdgeList)
    nd = dict(demo.nodes(data=True))
    nd['a']['lineages'] = leaf_lins['a']
    nd['bcd']['lineages'] = leaf_lins['bcd']
    #bcdProb,bcProb = random.uniform(0,1), random.uniform(0,1)
    nd['bcd']['splitprobs'] = {'bc' : bcdProb, 'd' : 1-bcdProb}
    nd['bc']['splitprobs'] = {'b' : bcProb, 'c' : 1-bcProb}

    popWaitTimes = {'bcd' : bcdSplit, # bcd splits
                    'bc' : bcSplit - bcdSplit, #bc splits
                    'b' : bdJoin - bcSplit, 'd' : bdJoin - bcdSplit, # b,d join
                    'a' : abdJoin, 'bd' : abdJoin - bdJoin, # a,bd join
                    'c' : abcdJoin, 'abd' : abcdJoin - abdJoin, #abd,c join
                    'abcd' : float('inf'), #tmrca
                    }

    demo = Demography(demo) 
    for v in demo:
        nd = demo.node_data[v]
        n_sub = demo.n_lineages_at_node[v]
        nd['model'] = ConstantTruncatedSizeHistory(N=1.0,
                                                   tau= popWaitTimes[v],
                                                   n_max=n_sub)

    empirical_sfs = run_scrm([sum([v for k,v in leaf_lins.iteritems()]), 
                              num_scrm_samples, 
                              '-t', theta, 
                              '-I', 2, leaf_lins['a'], leaf_lins['bcd'], # 1=a, 2=bcd
                              '-es', bcdSplit/2.0, 2, bcdProb, # 1=a, 2=bc, 3=d
                              '-es', bcSplit/2.0, 2, 1-bcProb, # 1=a, 2=c, 3=d, 4=b
                              '-ej', bdJoin/2.0, 4, 3, # 1=a, 2=c, 3=bd
                              '-ej', abdJoin/2.0, 3, 1, # 1=abd, 2=c
                              '-ej', abcdJoin/2.0, 2, 1, # 1=abcd
                              ], 
                             (leaf_lins['a'], leaf_lins['bcd']))
    
    theoretical_sfs = {}
    for sfs_entry in empirical_sfs:
        state = {'a' : {'derived' : sfs_entry[0]},
                 'bcd' : {'derived' : sfs_entry[1]}}
        for v in state:
            state[v].update({'ancestral' : leaf_lins[v] - state[v]['derived']})
        demo.update_state(state)
        theoretical_sfs[sfs_entry] = SumProduct(demo).p() * float(num_scrm_samples) * theta / 2.0

    configs = sorted(empirical_sfs.keys())
    def sfsArray(sfs):
        return np.array([sfs[x] for x in configs])

    assert scipy.stats.chisquare(sfsArray(empirical_sfs), sfsArray(theoretical_sfs)) >= .05
    #assert theoretical_sfs == empirical_sfs
