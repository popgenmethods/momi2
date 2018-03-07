
import pytest
import momi
from momi.demo_model import DemographicModel
from momi.events import get_event_from_old, LeafEvent, SizeEvent, JoinEvent, PulseEvent, GrowthEvent

from demo_utils import simple_admixture_demo, simple_two_pop_demo, piecewise_constant_demo, simple_five_pop_demo, simple_five_pop_demo, exp_growth_model, exp_growth_0_model

import autograd.numpy as np
import sys
import os
import pickle as pickle

# TODO add a test with archaic leafs

MODELS = [{'demo': simple_admixture_demo, 'nlins': (5, 5), 'params': 7},
          {'demo': simple_two_pop_demo, 'nlins': (5, 8), 'params': 4},
          {'demo': piecewise_constant_demo, 'nlins': (10,), 'params': 9},
          {'demo': simple_five_pop_demo, 'nlins': tuple(
              range(1, 6)), 'params': 30},
          {'demo': exp_growth_model, 'nlins': (10,), 'params': 3},
          {'demo': exp_growth_0_model, 'nlins': (10,), 'params': 2},
          ]
MODELS = {m['demo'].__name__: m for m in MODELS}
# for m in MODELS.values():
#    m['demofunc'] = lambda x: m['demo'](x, m['nlins'])

PICKLE = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "test_sfs.pickle")


def generate_sfs():
    with open(PICKLE, "rb") as sfs_dict_file:
        sfs_dict = pickle.load(sfs_dict_file)
        ret = []
        for k, v in sfs_dict.items():
            m_name, params, sampled_sfs = k

            n_lin = MODELS[m_name]['nlins']
            demo = MODELS[m_name]['demo'](np.array(params))
            yield m_name, v, demo, sampled_sfs


@pytest.mark.parametrize("m_name,v,demo,sampled_sfs", generate_sfs())
def test_generated_cases(m_name, v, demo, sampled_sfs):
    compute_stats(demo, sampled_sfs, *v)


def compute_stats(demo, sampled_sfs, true_sfs=None, true_branch_len=None):
    sampled_sfs = momi.site_freq_spectrum(demo.leafs, to_dict(sampled_sfs))
    demo.set_data(sampled_sfs, length=1)
    demo.set_mut_rate(1)

    exp_branch_len = demo.expected_branchlen()
    exp_sfs = demo.expected_sfs()

    configs = sorted([tuple(map(tuple, c)) for c in sampled_sfs.configs])
    exp_sfs = np.array([exp_sfs[c] for c in configs])

    # use ms units
    exp_branch_len = exp_branch_len / 4.0 / demo.N_e
    exp_sfs = exp_sfs / 4.0 / demo.N_e

    if true_sfs is not None:
        assert np.allclose(true_sfs, exp_sfs, rtol=1e-4)
    if true_branch_len is not None:
        assert np.allclose(true_branch_len, exp_branch_len, rtol=1e-4)

    return exp_sfs, exp_branch_len


def from_dict(sampled_sfs):
    # make it hashable
    return tuple([tuple(locus.items()) for locus in sampled_sfs])


def to_dict(sampled_sfs):
    # make it a dictionary
    return [dict(locus) for locus in sampled_sfs]

if __name__ == "__main__":
    # TODO check this simulation code still works!
    results = {}
    for m_name, m_val in MODELS.items():
        print("# GENERATING %s" % m_name)
        for i in range(10):
            x = np.random.normal(size=m_val['params'])
            demo = m_val['demo'](x, m_val['nlins'])

            demo.demo_hist = demo.demo_hist.rescaled()

            #seg_sites = simulate_ms(
            #    ms_path, demo.demo_hist._get_multipop_moran(demo.pops, demo.n), num_loci=100, mut_rate=1.0)

            # TODO fix this simulation code!!!
            num_bases = 1000
            mu = 1.
            n_loci = 100
            sfs = demo.demo_hist.simulate_data(
                demo.pops, demo.n,
                mutation_rate=mu/num_bases,
                recombination_rate=0,
                length=num_bases,
                num_replicates=n_loci).sfs

            sampled_sfs = from_dict(sfs.to_dict(vector=True))
            results[(m_name, tuple(x), sampled_sfs)
                    ] = compute_stats(demo, sampled_sfs)
    if len(sys.argv) == 2 and sys.argv[1] == "generate":
        pickle.dump(results, open(PICKLE, "wb"))
