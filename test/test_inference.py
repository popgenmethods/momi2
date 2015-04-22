import sh
import os
import scipy.optimize
import math
import re
from pprint import pprint
import random
import autograd.numpy as np
import newick
from autograd.numpy import sum
from autograd import grad
import networkx as nx
from size_history import ConstantTruncatedSizeHistory

from sum_product import SumProduct
from demography import Demography


def get_exact_jsfs(demo, theta=2.0):
    leafs = sorted(list(demo.leaves))
    n_ders = [range(demo.node_data[l]['lineages']+1) for l in leafs]
    total_n = sum([demo.node_data[l]['lineages'] for l in leafs])
    jsfs = {}
    for comb in itertools.product(*n_ders):
        assert len(comb) == len(leafs)
        state = {}
        total_der = 0
        for l,n_d in zip(leafs,comb):
            total_der += n_d
            state[l] = {'derived' : n_d, 'ancestral' : demo.node_data[l]['lineages'] - n_d}
        if total_der == 0 or total_der == total_n:
            continue
        demo.update_state(state)
        weight = SumProduct(demo).p(normalized=False) * theta / 2.0
        state = tuple([state[l]['derived'] for l in leafs])
        jsfs[state] = weight
    return jsfs

def test_joint_sfs_inference():
    N0=1.0
    theta=1.0
    t0=random.uniform(.25,2.5)
    t1= t0 + random.uniform(.5,5.0)
    num_runs = 10000

    def get_demo(join_time):
        return Demography.from_ms("-I 3 1 1 1 -ej $0 1 2 -ej $1 2 3",
                                  join_time / 2. * N0,
                                  t1 / 2. * N0)

    true_demo = get_demo(t0)

    jsfs,sqCounts,nonzero = true_demo.simulate_sfs(num_runs, theta=theta)
    totalSnps = sum([v for k,v in jsfs.items()])
    logFactorialTotalSnps = sum([math.log(x) for x in range(1,totalSnps)])

    print(t0,t1)
    def f(join_time):
        assert join_time.shape == (1,)
        return -get_demo(join_time[0]).log_likelihood_prf(theta * num_runs, jsfs)

    x0 = np.array([random.uniform(0,t1)])
    g = grad(f)
    res = scipy.optimize.minimize(f, x0, method='L-BFGS-B', jac=g, bounds=((0,t1),))
    print res.jac
    assert abs(res.x - t0) / t0 < .05

if __name__ == "__main__":
    test_joint_sfs_inference()
