import collections as co
import functools as ft
from cached_property import cached_property
import autograd.numpy as np
from .compute_sfs import _expected_sfs_tensor_prod


class SfsStats(object):
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
        # Estimate for the admixture of C into B in tree (((A,B),C),O)
        return self.abba_baba(A, B, C, *O) / self.abba_baba(A, C, C, *O)

    def f4_ratio(self, A, B, C, X, *O):
        # For tree (((A,B),C),O), and X admixed between B,C
        # an estimate for the admixture proportion from B
        # ref: Patterson et al 2012, Ancient Admixture in Human History, eq (4)
        return self.f4(X, C, A, *O) / self.f4(B, C, A, *O)

    def singleton_probs(self, pops):
        denom = None
        probs = {}
        for pop in pops:
            prob = self.ordered_prob(dict([
                (p, [1]) if p == pop else (p, [0])
                for p in pops]), fold=True)
            probs[pop] = prob
            if denom is None:
                denom = prob
            else:
                denom = denom + prob
        return {"probs": probs, "denom": 1-denom}


class ObservedSfsStats(SfsStats):
    def __init__(self, sfs, sampled_n_dict):
        is_ascertained = dict(zip(sfs.sampled_pops, sfs.ascertainment_pop))
        if sum(n for p, n in sampled_n_dict.items()
               if is_ascertained[p]) < 2:
            raise ValueError("sampled_n_dict must contain at least 2 ascertained alleles")
        self.sfs = sfs
        super(ObservedSfsStats, self).__init__(sampled_n_dict)

    def tensor_prod(self, derived_weights_dict):
        weighted_counts = self.sfs.configs.count_subsets(derived_weights_dict,
                                                         self.sampled_n_dict)

        # subtract out weights of monomorphic
        mono_anc = {}
        mono_der = {}
        for pop, asc in zip(self.sfs.sampled_pops, self.sfs.ascertainment_pop):
            try:
                v = derived_weights_dict[pop]
            except KeyError:
                try:
                    v = [1] * (self.sampled_n_dict[pop]+1)
                except KeyError:
                    continue
            if asc:
                mono_anc[pop] = [v[0]] + [0]*(len(v)-1)
                mono_der[pop] = [0]*(len(v)-1) + [v[-1]]
            else:
                mono_anc[pop] = v
                mono_der[pop] = v
        mono_anc = self.sfs.configs.count_subsets(
            mono_anc, self.sampled_n_dict)
        mono_der = self.sfs.configs.count_subsets(
            mono_der, self.sampled_n_dict)

        return JackknifeArray.from_chunks(
            self.sfs.freqs_matrix.T.dot(
                weighted_counts - mono_anc - mono_der))

    @property
    def n_subsets(self):
        return self.denom.est

    @property
    def n_jackknife_blocks(self):
        return self.sfs.n_loci


def jackknife_arr_op(wrapped_op):
    @ft.wraps(wrapped_op)
    def wraps_op(self, other):
        try:
            other.est, other.jackknife
        except AttributeError:
            return wrapped_op(self, JackknifeArray(other, other))
        else:
            return wrapped_op(self, other)
    return wraps_op

class JackknifeArray(object):
    @classmethod
    def from_chunks(cls, x):
        tot = np.sum(x)
        return cls(tot, tot - x)

    def __init__(self, est, jackknife):
        self.est = est
        self.jackknife = jackknife

    def apply(self, fun):
        return JackknifeArray(fun(self.est),
                              fun(self.jackknife))

    @jackknife_arr_op
    def __add__(self, other):
        return JackknifeArray(self.est + other.est,
                              self.jackknife + other.jackknife)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return JackknifeArray(-self.est, -self.jackknife)

    @jackknife_arr_op
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    @jackknife_arr_op
    def __mul__(self, other):
        return JackknifeArray(self.est * other.est,
                              self.jackknife * other.jackknife)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return (self**-1) * other

    @jackknife_arr_op
    def __pow__(self, other):
        return JackknifeArray(self.est ** other.est,
                              self.jackknife ** other.jackknife)

    @jackknife_arr_op
    def __rpow__(self, other):
        return JackknifeArray(other.est ** self.est,
                              other.jackknife ** self.jackknife)

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

    def __repr__(self):
        return "JackknifeArray(est={}, sd={}, z_score={}) at {}".format(
            self.est, self.sd, self.z_score, hex(id(self)))


class ExpectedSfsStats(SfsStats):
    def __init__(self, demo, ascertainment_pops):
        self.demo = demo
        self.ascertainment_pops = ascertainment_pops
        super(ExpectedSfsStats, self).__init__(dict(zip(demo.sampled_pops,
                                                      demo.sampled_n)))

    def tensor_prod(self, derived_weights_dict):
        #sampled_pops, sampled_n = zip(*sorted(self.sampled_n_dict.items()))
        #demo = self.demo._get_multipop_moran(sampled_pops, sampled_n)
        demo = self.demo

        vecs = []
        for p, n in zip(demo.sampled_pops, demo.sampled_n):
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


class ModelFitStats(SfsStats):
    def __init__(self, sfs, demo, ascertainment_pop):
        sampled_n_dict = dict(zip(demo.sampled_pops, demo.sampled_n))
        self.empirical = ObservedSfsStats(sfs, sampled_n_dict)
        self.sampled_n_dict = self.empirical.sampled_n_dict
        self.expected = ExpectedSfsStats(demo, ascertainment_pop)

    def tensor_prod(self, derived_weights_dict):
        return ModelFitArray(self.expected.tensor_prod(derived_weights_dict),
                             self.empirical.tensor_prod(derived_weights_dict))

    @property
    def n_subsets(self):
        return self.empirical.n_subsets

    @property
    def n_jackknife_blocks(self):
        return self.empirical.n_jackknife_blocks

def get_theoretical_jackknifeArr(other):
    # Helper function for ModelFitArray binary operations
    try:
        return (other.theoretical, other.jackknife_arr)
    except AttributeError:
        return (other, other)


class ModelFitArray(object):
    def __init__(self, theoretical, jackknife_arr):
        self.theoretical = theoretical
        self.jackknife_arr = jackknife_arr

    def apply(self, fun):
        return ModelFitArray(fun(self.theoretical),
                             self.jackknife_arr.apply(fun))

    def __add__(self, other):
        o_th, o_ja = get_theoretical_jackknifeArr(other)
        return ModelFitArray(self.theoretical + o_th,
                             self.jackknife_arr + o_ja)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1)*other

    def __rsub__(self, other):
        return (self * (-1)) + other

    def __mul__(self, other):
        o_th, o_ja = get_theoretical_jackknifeArr(other)
        return ModelFitArray(self.theoretical * o_th,
                             self.jackknife_arr * o_ja)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        o_th, o_ja = get_theoretical_jackknifeArr(other)
        return ModelFitArray(self.theoretical ** o_th,
                             self.jackknife_arr ** o_ja)

    def __rpow__(self, other):
        o_th, o_ja = get_theoretical_jackknifeArr(other)
        return ModelFitArray(o_th ** self.theoretical,
                             o_ja ** self.jackknife_arr)


    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return (self**-1) * other

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
