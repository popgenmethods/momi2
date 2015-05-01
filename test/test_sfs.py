from __future__ import division
import pytest
from demography import make_demography
from sum_product import compute_sfs
from test_sfs_counts import simple_admixture_demo
from test_gradient import simple_two_pop_demo, piecewise_constant_demo, simple_five_pop_demo, simple_five_pop_demo, exp_growth_model
from test_gradient import log_within
import autograd.numpy as np
import sys, os
import cPickle as pickle

def exp_growth_0_model(x, n_lins):
    x0 = np.array([x[0], 0.0, x[1]])
    return exp_growth_model(x0, n_lins)

MODELS = [{'demo':simple_admixture_demo,'nlins':{'1':2,'2':2},'params':7},
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
            m_name,params,configs = k
            sfs,branch_len = v

            n_lin = MODELS[m_name]['nlins']
            demo = MODELS[m_name]['demo'](np.array(params),n_lin)
            sfs2,branch_len2 = compute_sfs(demo, configs)
            yield m_name,sfs,sfs2,branch_len,branch_len2

@pytest.mark.parametrize("m_name,sfs,sfs2,branch_len,branch_len2", generate_sfs())
def test_generated_cases(m_name,sfs,sfs2,branch_len,branch_len2):
    log_within( np.array(branch_len, ndmin=1), np.array(branch_len2, ndmin=1) , eps=1e-6)
    log_within( sfs, sfs2, eps=1e-6)
    

if __name__=="__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "generate":
        results = {}
        for m_name,m_val in MODELS.iteritems():
            print "# GENERATING %s" % m_name
            for i in range(10):
                x = np.random.normal(size=m_val['params'])
                demo = m_val['demo'](x, m_val['nlins'])
                
                sampled_sfs,_,_ = demo.simulate_sfs(10)
                config_list = tuple(sorted(sampled_sfs.keys()))
                
                results[(m_name, tuple(x), config_list)] = compute_sfs(demo, config_list)
        pickle.dump(results, open(PICKLE, "wb"))
    elif len(sys.argv) == 1:
        pass
    else:
        raise Exception("Unrecognized command line options.")
