import os, random
import scipy.optimize
import autograd.numpy as np
from autograd import grad

from momi import make_demography, simulate_ms, sfs_list_from_ms, unlinked_log_likelihood, sum_sfs_list

def test_joint_sfs_inference():
    N0=1.0
    mu=1.0
    t0=random.uniform(.25,2.5)
    t1= t0 + random.uniform(.5,5.0)
    num_runs = 10000

    def get_demo(join_time):
        join_time, = join_time
        return make_demography("-I 3 1 1 1 -ej $0 1 2 -ej $1 2 3",
                               join_time / 2. * N0,
                               t1 / 2. * N0)

    true_demo = get_demo([t0])

    sfs = sum_sfs_list(sfs_list_from_ms(simulate_ms(true_demo, num_sims=num_runs, mu=mu)))
    neg_log_lik = lambda t: -unlinked_log_likelihood(sfs, get_demo(t), mu * num_runs)
    
    print(t0,t1)
    x0 = np.array([random.uniform(0,t1)])
    res = scipy.optimize.minimize(neg_log_lik, x0, method='L-BFGS-B', jac=grad(neg_log_lik), bounds=((0,t1),))
    print res.jac
    assert abs(res.x - t0) / t0 < .05

if __name__ == "__main__":
    test_joint_sfs_inference()
