from __future__ import division
import pytest
from momi import make_demography, expected_sfs, expected_total_branch_len, sum_sfs_list, simulate_ms, sfs_list_from_ms

from test_sims import simple_admixture_demo
from test_gradient import simple_two_pop_demo, piecewise_constant_demo, simple_five_pop_demo, simple_five_pop_demo, exp_growth_model
from test_gradient import log_within

import autograd.numpy as np
import sys, os
import cPickle as pickle

def exp_growth_0_model(x, n_lins):
    x0 = np.array([x[0], 0.0, x[1]])
    return exp_growth_model(x0, n_lins)

MODELS = [{'demo':simple_admixture_demo,'nlins':{'1':5,'2':5},'params':7},
          {'demo':simple_two_pop_demo,'nlins':{'1':5,'2':8},'params':4},
          {'demo':piecewise_constant_demo,'nlins':{'a':10},'params':9},
          {'demo':simple_five_pop_demo,'nlins':{str(i):i for i in range(1,6)},'params':30},
          {'demo':exp_growth_model,'nlins':{'1':10},'params':3},
          {'demo':exp_growth_0_model,'nlins':{'1':10},'params':2},
          ]         
MODELS = {m['demo'].__name__ : m for m in MODELS}
#for m in MODELS.values():
#    m['demofunc'] = lambda x: m['demo'](x, m['nlins'])

PICKLE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_sfs.pickle")

def generate_sfs():
    with open( PICKLE, "rb" ) as sfs_dict_file:
        sfs_dict = pickle.load(sfs_dict_file)
        ret = []
        for k,v in sfs_dict.iteritems():
            m_name,params,sampled_sfs = k

            n_lin = MODELS[m_name]['nlins']
            demo = MODELS[m_name]['demo'](np.array(params),n_lin)
            v2 = compute_stats(demo, sampled_sfs)
            yield m_name,v,v2

@pytest.mark.parametrize("m_name,v,v2", generate_sfs())
def test_generated_cases(m_name,v,v2):
    for stat1,stat2 in zip(v,v2):
        assert np.allclose(stat1,stat2)
    
def compute_stats(demo, sampled_sfs):
    sampled_sfs = to_dict(sampled_sfs)
    agg_sfs = sum_sfs_list(sampled_sfs)
    config_list = tuple(sorted(agg_sfs.keys()))
    return expected_sfs(demo,config_list), expected_total_branch_len(demo)

def from_dict(sampled_sfs):
    # make it hashable
    return tuple([tuple(locus.iteritems()) for locus in sampled_sfs])

def to_dict(sampled_sfs):
    # make it a dictionary
    return [dict(locus) for locus in sampled_sfs]

if __name__=="__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "generate":
        results = {}
        for m_name,m_val in MODELS.iteritems():
            print "# GENERATING %s" % m_name
            for i in range(10):
                x = np.random.normal(size=m_val['params'])
                demo = m_val['demo'](x, m_val['nlins'])
                
                sampled_sfs = sfs_list_from_ms(simulate_ms(demo, num_sims=100, mu=10.))
                sampled_sfs = from_dict(sampled_sfs)
                results[(m_name, tuple(x), sampled_sfs)] = compute_stats(demo, sampled_sfs)
        pickle.dump(results, open(PICKLE, "wb"))
    elif len(sys.argv) == 1:
        pass
    else:
        raise Exception("Unrecognized command line options.")
