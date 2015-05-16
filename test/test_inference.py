import os, random
import scipy.optimize
import autograd.numpy as np
from autograd import grad

from momi import make_demography, simulate_ms, sfs_list_from_ms
from momi import CompositeLogLikelihood as LogLik

def test_joint_sfs_inference():
    N0=1.0
    theta=1.0
    t0=random.uniform(.25,2.5)
    t1= t0 + random.uniform(.5,5.0)
    num_runs = 10000

    def get_demo(join_time):
        join_time, = join_time
        return make_demography("-I 3 1 1 1 -ej $0 1 2 -ej $1 2 3",
                               join_time / 2. * N0,
                               t1 / 2. * N0)

    true_demo = get_demo([t0])

    sfs_list = sfs_list_from_ms(simulate_ms(true_demo, num_sims=num_runs, theta=theta),
                                true_demo.n_at_leaves)

    log_lik = LogLik(sfs_list, demo_func=get_demo, theta=theta)

    print(t0,t1)
    def f(join_time):
        assert join_time.shape == (1,)
        return -log_lik.log_likelihood(join_time)
        #return -log_likelihood_prf(get_demo(join_time[0]), theta * num_runs, jsfs)

    x0 = np.array([random.uniform(0,t1)])
    g = grad(f)
    res = scipy.optimize.minimize(f, x0, method='L-BFGS-B', jac=g, bounds=((0,t1),))
    print res.jac
    assert abs(res.x - t0) / t0 < .05

if __name__ == "__main__":
    test_joint_sfs_inference()
