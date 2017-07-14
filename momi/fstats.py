import collections as co
from cached_property import cached_property
import autograd.numpy as np
from .compute_sfs import _expected_sfs_tensor_prod


class Fstats(object):
    def __init__(self, sampled_n_dict):
        self.sampled_n_dict = {p: n for p, n in sampled_n_dict.items()
                               if n > 0}

    def tensor_prod(self, derived_weights_dict):
        raise NotImplementedError

    @cached_property
    def denom(self):
        return self.tensor_prod({})

    def ordered_prob(self, subsample_dict,
                     fold=False):
        """
        The ordered probability for the subsample given by
        subsample_dict.

        Parameters:
        subsample_dict: dict of list
           dict mapping population to a list of 0s and 1s giving the
              ordered subsample within that population.
        """
        if fold:
            rev_subsample = {p: 1 - np.array(s)
                             for p, s in subsample_dict.items()}

            return (self.ordered_prob(subsample_dict)
                    + self.ordered_prob(rev_subsample))

        derived_weights_dict = {}
        for pop, pop_subsample in subsample_dict.items():
            n = self.sampled_n_dict[pop]
            arange = np.arange(n+1)

            cnts = co.Counter(pop_subsample)

            prob = np.ones(n+1)
            for i in range(cnts[0]):
                prob *= (n - arange - i)
            for i in range(cnts[1]):
                prob *= (arange - i)
            for i in range(cnts[0] + cnts[1]):
                prob /= float(n - i)

            derived_weights_dict[pop] = prob

        return self.tensor_prod(derived_weights_dict) / self.denom

    def count_1100(self, A, B, C, O=None):
        # O=None -> O is the root population
        subsample_dict = co.defaultdict(list)
        subsample_dict[A].append(1)
        subsample_dict[B].append(1)
        subsample_dict[C].append(0)
        if O is not None:
            subsample_dict[O].append(0)

        return self.ordered_prob(
            subsample_dict,
            fold=(O is not None))

    def baba(self, A, B, C, *O):
        return self.count_1100(A, C, B, *O)

    def abba(self, A, B, C, *O):
        return self.count_1100(B, C, A, *O)

    def abba_baba(self, *args):
        return self.baba(*args) - self.abba(*args)

    def f4(self, *args):
        return self.abba_baba(*args)

    def f3(self, A, B, O):
        return self.f4(O, A, O, B)

    def f2(self, A, B):
        return self.f4(A, B, A, B)

    def pattersons_d(self, *args):
        abba = self.abba(*args)
        baba = self.baba(*args)
        return (baba - abba) / (baba + abba)

    def greens_f(self, A, B, C, *O):
        return self.abba_baba(A, B, C, *O) / self.abba_baba(A, C, C, *O)


class EmpiricalFstats(Fstats):
    def __init__(self, sfs, sampled_n_dict):
        self.sfs = sfs
        super(EmpiricalFstats, self).__init__(sampled_n_dict)

    def tensor_prod(self, derived_weights_dict):
        return JackknifeArray.from_chunks(
            self.sfs.freqs_matrix.T.dot(
                self.sfs.configs.count_subsets(derived_weights_dict,
                                               self.sampled_n_dict)))

    @property
    def n_subsets(self):
        return self.denom.est

    @property
    def n_jackknife_blocks(self):
        return self.sfs.n_loci


class JackknifeArray(object):
    @classmethod
    def from_chunks(cls, x):
        tot = np.sum(x)
        return cls(tot, tot - x)

    def __init__(self, est, jackknife):
        self.est = est
        self.jackknife = jackknife

    def __add__(self, other):
        return JackknifeArray(self.est + other.est,
                              self.jackknife + other.jackknife)

    def __sub__(self, other):
        return JackknifeArray(self.est - other.est,
                              self.jackknife - other.jackknife)

    def __mul__(self, other):
        return JackknifeArray(self.est * other.est,
                              self.jackknife * other.jackknife)

    def __truediv__(self, other):
        return JackknifeArray(self.est / other.est,
                              self.jackknife / other.jackknife)

    @property
    def resids(self):
        return self.jackknife - self.est

    @property
    def var(self):
        return np.mean(self.resids**2) * (len(self.jackknife) - 1)

    @property
    def sd(self):
        return np.sqrt(self.var)

    @property
    def z_score(self):
        return self.est / self.sd

    def __str__(self):
        return "JackknifeArray(est={}, sd={}, z_score={})".format(
            self.est, self.sd, self.z_score)


class ExpectedFstats(Fstats):
    def __init__(self, demo, ascertainment_pops, sampled_n_dict):
        self.demo = demo
        self.ascertainment_pops = ascertainment_pops
        super(ExpectedFstats, self).__init__(sampled_n_dict)

    def tensor_prod(self, derived_weights_dict):
        sampled_pops, sampled_n = zip(*sorted(self.sampled_n_dict.items()))
        demo = self.demo._get_multipop_moran(sampled_pops, sampled_n)

        vecs = []
        for p, n in zip(sampled_pops, sampled_n):
            v = []
            try:
                row = derived_weights_dict[p]
            except KeyError:
                row = np.ones(n+1)
            assert len(row) == n+1

            if p in self.ascertainment_pops:
                v.append([row[0]] + [0.0] * n)  # all ancestral state
                v.append([0.0] * n + [row[-1]])  # all derived state
            else:
                for _ in range(2):
                    v.append(row)
            v.append(row)

            vecs.append(np.array(v))

        res = _expected_sfs_tensor_prod(vecs, demo)
        return res[2] - res[0] - res[1]


class ModelFitFstats(Fstats):
    def __init__(self, sfs, demo, ascertainment_pop, sampled_n_dict):
        self.empirical = EmpiricalFstats(sfs, sampled_n_dict)
        self.sampled_n_dict = self.empirical.sampled_n_dict
        self.expected = ExpectedFstats(demo, ascertainment_pop,
                                       self.sampled_n_dict)

    def tensor_prod(self, derived_weights_dict):
        return ModelFitArray(self.expected.tensor_prod(derived_weights_dict),
                             self.empirical.tensor_prod(derived_weights_dict))

    @property
    def n_subsets(self):
        return self.empirical.n_subsets

    @property
    def n_jackknife_blocks(self):
        return self.empirical.n_jackknife_blocks


class ModelFitArray(object):
    def __init__(self, theoretical, jackknife_arr):
        self.theoretical = theoretical
        self.jackknife_arr = jackknife_arr

    def __add__(self, other):
        return ModelFitArray(self.theoretical + other.theoretical,
                             self.jackknife_arr + other.jackknife_arr)

    def __sub__(self, other):
        return ModelFitArray(self.theoretical - other.theoretical,
                             self.jackknife_arr - other.jackknife_arr)

    def __mul__(self, other):
        return ModelFitArray(self.theoretical * other.theoretical,
                             self.jackknife_arr * other.jackknife_arr)

    def __truediv__(self, other):
        return ModelFitArray(self.theoretical / other.theoretical,
                             self.jackknife_arr / other.jackknife_arr)

    @property
    def expected(self):
        return self.theoretical

    @property
    def observed(self):
        return self.jackknife_arr.est

    @property
    def sd(self):
        return self.jackknife_arr.sd

    @property
    def z_score(self):
        return (self.observed - self.expected) / self.sd

    def __str__(self):
        return ("ModelFitArray(expected={}, observed={},"
                " sd={}, z_score={})").format(self.expected, self.observed,
                                              self.sd, self.z_score)
