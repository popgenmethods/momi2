from __future__ import division
from util import aggregate_sfs
import autograd.numpy as np
import scipy
from sum_product import compute_sfs

## TODO: rename to LogLikelihoodSurface
class LogLikelihoodPRF(object):
    ## TODO: add EPSILON parameter for underflow issues
    ## TODO: add comments/documentation
    def __init__(self, get_demo_theta, sfs_list):
        self.raw_get_demo_theta = get_demo_theta
        self.sfs_list = sfs_list
        self.sfs_agg = aggregate_sfs(sfs_list)

        self.config_list, self.counts_agg = zip(*sorted(self.sfs_agg.iteritems()))
        self.counts_agg = np.array(self.counts_agg)
        self.factorials_agg = scipy.special.gammaln(self.counts_agg + 1)

    def get_demo_theta(self, x):
        demo,theta = self.raw_get_demo_theta(x)
        if theta is None:
            theta = 1.0
        theta = theta * np.ones(shape=(len(self.sfs_list),))
        assert len(theta) == len(self.sfs_list)
        return demo,theta

    def log_likelihood(self, x):
        demo,theta = self.get_demo_theta(x)
        theta = np.sum(theta)

        sfs_vals, branch_len = compute_sfs(demo, self.config_list)
        ret = -branch_len * theta + np.sum(np.log(sfs_vals * theta) * self.counts_agg - self.factorials_agg)

        assert ret < 0.0
        return ret        
