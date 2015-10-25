from __future__ import division
import networkx as nx
import autograd.numpy as np
from collections import Counter, defaultdict
from tensor import sfs_eval_dirs
from likelihood_surface import MEstimatorSurface
from random import shuffle
from util import aggregate_sfs
from sum_product import raw_compute_sfs

class CompressedLikelihoodSurface(MEstimatorSurface):
    def __init__(self, n_components, n_per_pop, sfs_list, theta, demo_func=lambda demo: demo, eps=1e-80):
        super(CompressedLikelihoodSurface, self).__init__(theta, demo_func)
        self.n_loci = len(sfs_list)
        
        n_per_pop = np.array(n_per_pop)
        ordering = []
        while sum(n_per_pop) > 0:
            to_add = list(np.arange(len(n_per_pop))[n_per_pop > 0])
            shuffle(to_add)
            for i in to_add:
                ordering += [i]
                n_per_pop[i] -= 1

        self.compressed_sfs = CompressedOrderedSfs(aggregate_sfs(sfs_list),
                                                   n_components, ordering, init_draws = len(n_per_pop))
        self.eps = eps

    def evaluate(self, params, vector=False):
        if vector:
            raise Exception("Vectorized likelihood not implemented")
        
        demo = self.demo_func(params)
        theta = np.sum(self._get_theta(params))

        probs = raw_compute_sfs(self.compressed_sfs.dirs, demo)
        branch_len, probs = probs[0], probs[1:]
        probs = probs / branch_len * self.compressed_sfs.n_paths + self.eps
        
        theta = self._get_theta(params)
        theta = np.ones(self.n_loci) * theta
        theta = np.sum(theta)

        ret = -branch_len * theta + self.compressed_sfs.n_snps * np.log(branch_len * theta) + np.sum(self.compressed_sfs.weights * np.log(probs))
        return -ret
       
        

        
class CompressedOrderedSfs(object):
    def __init__(self, sfs, components, ordering, init_draws=0):
        self.ordering = ordering
        sfs = self._sfs = dict({k:v for k,v in sfs.iteritems() if v != 0})
        components = min(components, len(self._sfs))
        
        ## the vector of samples in each deme
        self._n = Counter()
        for i in ordering:
            self._n[i] += 1
        assert set(self._n.keys()) == set(range(len(self._n)))
        self._n = tuple([v for _,v in sorted(self._n.iteritems())])

        self._subsamples = {} # all the subsamples encountered so far
        self._curr_mass = Counter() # the subsamples that currently have some mass        

        # add in the empty sample
        self._add_subsample((tuple((0,) * len(self._n)),)*2, 1)
        # add in all possible samples, for the first init_draws
        for _ in range(init_draws):
            # the configs currently with mass
            prev_s = list(self._curr_mass.keys())
            # subsample each of these configs
            for s in prev_s:
                # check if s was removed from curr_mass
                if s in self._curr_mass:
                    self._subsample_next(s)

        self.n_snps = sum(self._sfs.values())
        while len(self._curr_mass) < components:
            sorted_mass = self._curr_mass.most_common()
            for s,m in sorted_mass:
                assert m > 0.0
                
                if tuple(np.array(s[0]) + np.array(s[1])) == self._n:
                    continue
                else:
                    self._subsample_next(s)
                    assert np.isclose(sum(self._curr_mass.values()), self.n_snps)
                    break

        sorted_mass = self._curr_mass.most_common()
        s_list, m_list = zip(*sorted_mass)
        
        self.weights = np.array(m_list)
        self.dirs = defaultdict(list)
        for s in s_list:
            for k,v in self._subsamples[s]['vecs'].iteritems():
                self.dirs[k] += [v]

        for k in self.dirs:
            self.dirs[k] = np.vstack(self.dirs[k])
            self.dirs[k] = np.vstack((np.ones(self.dirs[k].shape[1]), self.dirs[k]))

        self.n_paths = np.array([self._subsamples[s]['n_paths'] for s in s_list])
        #self.ordered_mass = np.array([self._subsamples[s]['ordered_mass'] for s in s_list])
        #assert np.allclose(self.ordered_mass * self.n_paths, self.weights)
        #assert len(self.dirs) == len(set(ordering))
        #assert all(x.shape[0] in self.dirs[0].shape[0] for x in self.dirs.values())
                
    def _children(self, s):
        n_s = sum(map(sum,s)) 
        ret = []
        if n_s >= len(self.ordering):
            return ret
        pop = self.ordering[n_s]
        for i in (0,1):
            new_s = map(list, s)
            new_s[i][pop] += 1
            new_s = tuple(map(tuple, new_s))
            ret += [new_s]
        return ret

    def _parents(self, s):
        n_s = sum(map(sum,s))
        ret = []
        if n_s == 0:
            return ret
        pop = self.ordering[n_s-1]
        for i in (0,1):
            new_s = map(list, s)
            new_s[i][pop] -= 1
            if new_s[i][pop] < 0:
                continue
            new_s = tuple(map(tuple, new_s))
            ret += [new_s]
        return ret
            
    def _add_subsample(self, s, n_paths):
        """add some paths to a subsample s"""
        assert n_paths > 0

        ## if parents have any mass, propagate it down
        for p in self._parents(s):
            if p in self._curr_mass:
                self._subsample_next(p)

        ## if not yet encountered s, add it
        if s not in self._subsamples:
            vecs = get_subsample_vectors(s, self._n)
            self._subsamples[s] = {'vecs' : vecs, 'ordered_mass' : np.squeeze(sfs_eval_dirs(self._sfs, vecs)), 'n_paths' : 0}

        ## add paths and mass to s
        self._subsamples[s]['n_paths'] += n_paths
        self._curr_mass[s] += n_paths * self._subsamples[s]['ordered_mass']
        if self._curr_mass[s] == 0.0:
            del self._curr_mass[s]

        ## if any children of s been encountered, propagate down to them
        if s in self._curr_mass and any([c in self._subsamples for c in self._children(s)]):
            self._subsample_next(s)

    def _subsample_next(self, s):
        """draw an additional sample for the subsample s"""
        # should only be further sampling s if it has some mass
        assert s in self._curr_mass

        # store and remove all paths going thru s
        n_paths = self._subsamples[s]['n_paths']
        self._subsamples[s]['n_paths'] = 0
        del self._curr_mass[s]

        # add paths to the children of s
        for new_s in self._children(s):
            self._add_subsample(new_s, n_paths)


def get_subsample_vectors(subsample, total):
    configs = list(subsample) + [total]
    configs = map(np.array, configs)

    ancestral,derived,total = configs
    
    assert len(ancestral) == len(derived) and len(ancestral) == len(total)
    assert all(ancestral >= 0) and all(derived >= 0) and all(ancestral + derived <= total)

    ret = []
    for a,d,t in zip(ancestral, derived, total):
        total_d = np.array(np.arange(t+1), ndmin=2)
        curr = np.ones((1,t+1))

        for i in range(d):
            curr = curr * (total_d / t)
            total_d = total_d - 1
            t = t-1
        for i in range(a):
            curr = curr * ((t - total_d) / t)
            t = t-1
        ret += [curr]
    return dict(zip(range(1,len(ret)+1), ret))
            
