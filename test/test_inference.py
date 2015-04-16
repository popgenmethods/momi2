from __future__ import division
import sh
import os
import scipy.optimize
import math
import re
from pprint import pprint
import random
import numpy as np

from sum_product import SumProduct
from demography import Demography


def test_joint_sfs_inference():
    N0=1.0
    theta=1.0
    t0=random.uniform(.25,2.5)
    t1= t0 + random.uniform(.5,5.0)
    num_runs = 10000

    def scrm_cmd(join_time):
        return "-I 3 1 1 1 -ej %f 1 2 -ej %f 2 3" % (join_time / 2. * N0, t1 / 2. * N0)

    true_demo = Demography.from_ms(scrm_cmd(t0))

    jsfs,sqCounts,nonzero = true_demo.simulate_sfs(num_runs, theta=theta)
    totalSnps = sum([v for k,v in jsfs.items()])
    logFactorialTotalSnps = sum([math.log(x) for x in range(1,totalSnps)])

    pprint(dict(jsfs))
    print(t0,t1)
    def f(join_time):
        demo = Demography.from_ms(scrm_cmd(join_time))
        lambd = theta / 2.0 * num_runs * demo.totalSfsSum
        # poisson probability for total # snps is e^-lambd * lambd^totalSnps / totalSnps!
        ret = lambd + logFactorialTotalSnps - totalSnps * math.log(lambd)
        for states, weight in sorted(jsfs.items()):
            st = {a: {'derived': b, 'ancestral': 1 - b} for a, b in zip("123", states)}
            demo.update_state(st)
            sp = SumProduct(demo)
            #print(weight, states, sp.p(normalized=True))
            ret -= weight * math.log(sp.p(normalized=True))
            #ret -= weight * math.log(sp.p())
        return ret
    #f(t0)
    res = scipy.optimize.fminbound(f, 0, t1, xtol=.01)
    #res = scipy.optimize.minimize(f, random.uniform(0,t1), bounds=((0,t1),))
    #assert res == t0
    assert abs(res - t0) / t0 < .05
#    print (res, t0, t1)
#    print(res.x, t0, t1)


if __name__ == "__main__":
    # test_jeff()
    test_joint_sfs_inference()
