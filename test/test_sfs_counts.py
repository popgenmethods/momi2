from size_history import ConstantTruncatedSizeHistory, PiecewiseHistory, ExponentialTruncatedSizeHistory
from demography import Demography
import pytest
import networkx as nx
import random
from sum_product import SumProduct
from test_inference import run_scrm, sfs_p_value
#scrm = sh.Command(os.environ["SCRM_PATH"])
import numpy as np
from collections import Counter
import scipy, scipy.stats
import itertools

#theta = 1.0
#num_scrm_samples = 10000
#num_scrm_samples = 5000
num_scrm_samples = 1000
#num_scrm_samples = 10
#theta = .01
#num_scrm_samples = 100000

def expo_demo_func(n, tau, growth_rate, N_top, theta, rho=None, numLoci=None, smcLen = None):
    N_bottom = N_top * np.exp(growth_rate * tau)
    def expo_demo():
        leaf_lins = {'a' : n}
        leaf_pops = ('a',)
        demo = nx.DiGraph([])
        demo.add_node('a')
        nd = dict(demo.nodes(data=True))
        nd['a']['lineages'] = leaf_lins['a']
        
        demo = Demography(demo)
        nd = demo.node_data['a']
        nd['model'] = PiecewiseHistory([ExponentialTruncatedSizeHistory(n, theta/2. * tau, theta/2. * N_top, theta/2. * N_bottom),
                                        ConstantTruncatedSizeHistory(n, float('inf'), theta/2. * N_top),
                                        ])

        scrm_args = [n, num_scrm_samples,
                     '-t', theta * N_bottom,
                     '-G', growth_rate * N_bottom * 2.0,
                     '-eG', tau / N_bottom / 2.0, 0.0,
                     ]
        if rho is not None:
            scrm_args += ['-r', rho * N_bottom, numLoci,
                          '-l', smcLen]
        return demo, scrm_args, leaf_lins, leaf_pops
    return expo_demo

def tree_demo_2():
    leaf_lins = {'a' : 4, 'b': 4}

    waitingTimes = np.random.random(1) * 2.0 + 0.1

    absoluteTimes = np.array(waitingTimes)
    for i in range(1,len(absoluteTimes)):
        absoluteTimes[i] += absoluteTimes[i-1]

    abJoin, = absoluteTimes

    # events: a,b merge at time tau
    eventList = [(('ab','a'), ('ab','b')), # a,b merge
                 ]
    demoEdgeList = []
    for e1,e2 in eventList:
        demoEdgeList += [e1,e2]
    demo = nx.DiGraph(demoEdgeList)
    nd = dict(demo.nodes(data=True))
    nd['a']['lineages'] = leaf_lins['a']
    nd['b']['lineages'] = leaf_lins['b']

    popWaitTimes = {'a' : abJoin, 'b' : abJoin, # a,b join
                    'ab' : float('inf'),
                    }

    demo = Demography(demo) 
    for v in demo:
        nd = demo.node_data[v]
        n_sub = demo.n_lineages_subtended_by[v]
        nd['model'] = ConstantTruncatedSizeHistory(N=1.0,
                                                   tau= popWaitTimes[v],
                                                   n_max=n_sub)    

    scrm_args = [sum([v for k,v in leaf_lins.iteritems()]), 
                 num_scrm_samples, 
                 #'-t', theta, 
                 '-T',
                 '-I', 2, leaf_lins['a'], leaf_lins['b'], # 1=a, 2=b
                 '-ej', abJoin/2.0, 2, 1, # 1=ab
                 ]
    leaf_pops = ('a','b')
    return demo, scrm_args, leaf_lins, leaf_pops

def tree_demo_4():
    leaf_lins = {'a' : 2, 'b': 2, 'c' : 2, 'd' : 2}

    waitingTimes = np.random.random(3) * 2.0 + 0.1

    absoluteTimes = np.array(waitingTimes)
    for i in range(1,len(absoluteTimes)):
        absoluteTimes[i] += absoluteTimes[i-1]

    abJoin,abcJoin,abcdJoin = absoluteTimes

    # events: a,b merge at time tau
    eventList = [(('ab','a'), ('ab','b')), # a,b merge
                 (('abc','ab'), ('abc','c')), # ab,c merge
                 (('abcd','abc'), ('abcd','d')), # abc,d merge
                 ]
    demoEdgeList = []
    for e1,e2 in eventList:
        demoEdgeList += [e1,e2]
    demo = nx.DiGraph(demoEdgeList)
    nd = dict(demo.nodes(data=True))
    for k,v in leaf_lins.iteritems():
        nd[k]['lineages'] = v

    popWaitTimes = {'a' : abJoin, 'b' : abJoin, # a,b join
                    'ab' : abcJoin - abJoin, 'c' : abcJoin, # ab,c join
                    'abc' : abcdJoin - abcJoin, 'd' : abcdJoin, #abc,d join
                    'abcd' : float('inf'),
                    }

    demo = Demography(demo) 
    for v in demo:
        nd = demo.node_data[v]
        n_sub = demo.n_lineages_subtended_by[v]
        nd['model'] = ConstantTruncatedSizeHistory(N=1.0,
                                                   tau= popWaitTimes[v],
                                                   n_max=n_sub)    

    scrm_args = [sum([v for k,v in leaf_lins.iteritems()]), 
                 num_scrm_samples, 
                 #'-t', theta, 
                 '-T',
                 '-I', len(leaf_lins), leaf_lins['a'], leaf_lins['b'], leaf_lins['c'], leaf_lins['d'], # 1=a, 2=b, 3=c, 4=d
                 '-ej', abJoin/2.0, 2, 1, # 1=ab
                 '-ej', abcJoin/2.0, 3, 1, # 1 = abc
                 '-ej', abcdJoin/2.0, 4, 1, #1 = abcd
                 ]
    leaf_pops = ('a','b','c','d')
    return demo, scrm_args, leaf_lins, leaf_pops

test_demos = [tree_demo_2, 
              tree_demo_4, 
              expo_demo_func(n=10, tau=.01, growth_rate=random.uniform(-500,500), N_top=random.uniform(0.1,10.0), theta=random.uniform(0.1,10.0))]
#test_demos = [expo_demo_func(n=10, tau=400.0 /2e4, growth_rate=150.0, N_top=1.0, theta=10000., rho=10000., numLoci=int(1e7), smcLen=int(1e4))] # test for Matthias

@pytest.mark.parametrize("demo_func", test_demos)
def test_sfs_counts(demo_func):
    demo, scrm_args, leaf_lins, leaf_pops = demo_func()

    scrm_args += ['-seed', random.randint(0,999999999)]

    empirical_sfs,sqCounts,nonzeroCounts = run_scrm(scrm_args, tuple([leaf_lins[v] for v in leaf_pops]))
    
    theoretical_sfs = {}
    ranges = [range(leaf_lins[v]+1) for v in leaf_pops]
    total_lins = sum([leaf_lins[v] for v in leaf_pops])
    #for sfs_entry in empirical_sfs:
    for sfs_entry in itertools.product(*ranges):
        sfs_entry = tuple(sfs_entry)
        if sum(sfs_entry) == 0 or sum(sfs_entry) == total_lins:
            continue # skip polymorphic sites
        state = {leaf_pops[i] : {'derived' : sfs_entry[i]} for i in range(len(sfs_entry))}
        for v in state:
            state[v].update({'ancestral' : leaf_lins[v] - state[v]['derived']})
        demo.update_state(state)
        theoretical_sfs[sfs_entry] = SumProduct(demo).p()
        #theoretical_sfs[sfs_entry] = SumProduct(demo).p() * float(num_scrm_samples) * theta / 2.0
    #p_val = sfs_p_value(empirical_sfs, sqCounts, theoretical_sfs, num_scrm_samples, theta)
    p_val = sfs_p_value(nonzeroCounts, empirical_sfs, sqCounts, theoretical_sfs, num_scrm_samples)
    print(p_val)
    cutoff = 0.05
    #cutoff = 1.0
    assert p_val > cutoff

    #configs = sorted(empirical_sfs.keys())
    #assert scipy.stats.chisquare(sfsArray(empirical_sfs), sfsArray(theoretical_sfs))[1] >= .05
    #assert theoretical_sfs == empirical_sfs
