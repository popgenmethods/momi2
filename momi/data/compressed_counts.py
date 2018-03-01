from cached_property import cached_property
# autograd.numpy.array() inefficient, so use vanilla numpy here
import numpy as np
import scipy.sparse


# to hash configs, represent it as a str
# (this seems to be more memory efficient than representing it as a tuple)
def _config2hashable(config):
    return "\t".join("%d,%d" % (a, d) for a, d in config)


# the inverse function of _config2hashable
def _hashed2config(config_str):
    return tuple((int(a), int(d))
                 for a, d in (x.split(",")
                              for x in config_str.strip().split()))


class _CompressedList(object):
    def __init__(self, items=None):
        self.uniq_values = []
        self.value2uniq = {}
        self.index2uniq = []
        self._array = None

        if items:
            for i in items:
                self.append(i)

    def __eq__(self, other):
        return list(self) == list(other)

    def __len__(self):
        return len(self.index2uniq)

    def __getitem__(self, index):
        if isinstance(index, slice) or isinstance(index, np.ndarray):
            ret = _CompressedList()
            ret.value2uniq = dict(self.value2uniq)
            ret.uniq_values = list(self.uniq_values)
            ret.index2uniq = self._get_array()[index]
            return ret
        else:
            return self.uniq_values[self.index2uniq[index]]

    def _get_array(self):
        if self._array is None:
            self._array = np.array(self.index2uniq, dtype=int)
        return self._array

    def append(self, value):
        self._array = None
        try:
            uniq_idx = self.value2uniq[value]
        except KeyError:
            uniq_idx = len(self.uniq_values)
            self.value2uniq[value] = uniq_idx
            self.uniq_values.append(value)
        self.index2uniq.append(uniq_idx)

    def extend(self, iterator):
        for i in iterator:
            self.append(i)


class _CompressedHashedCounts(object):
    def __init__(self, npops):
        self.compressed_list = _CompressedList()
        self.npops = npops

    def __len__(self):
        return len(self.compressed_list)

    def append(self, config):
        self.compressed_list.append(_config2hashable(config))

    def index2uniq(self, i=None):
        if i is None:
            return self.compressed_list.index2uniq
        else:
            return self.compressed_list.index2uniq[i]

    def config_array(self):
        ret = np.zeros((len(self.compressed_list.uniq_values), self.npops, 2),
                       dtype=int)
        for i, config_str in enumerate(self.compressed_list.uniq_values):
            ret[i, :, :] = _hashed2config(config_str)
        return ret

    def compressed_allele_counts(self):
        return CompressedAlleleCounts(self.config_array(),
                                      self.index2uniq())


class CompressedAlleleCounts(object):
    @classmethod
    def from_iter(cls, config_iter, npops, sort=True):
        compressed_hashes = _CompressedHashedCounts(npops)
        for config in config_iter:
            compressed_hashes.append(config)
        return cls(compressed_hashes.config_array(),
                   compressed_hashes.index2uniq(),
                   sort=sort)

    def __init__(self, config_array, index2uniq,
                 sort=True):
        self.config_array = config_array
        self.index2uniq = np.array(index2uniq, dtype=int)
        if sort:
            self.sort_configs()

    def __getitem__(self, i):
        return self.config_array[self.index2uniq[i], :, :]

    def __eq__(self, other):
        try:
            return (np.all(self.config_array == other.config_array)
                    and np.all(self.index2uniq == other.index2uniq))
        except AttributeError:
            return False

    def __len__(self):
        return len(self.index2uniq)

    def filter(self, idxs):
        to_keep = self.index2uniq[idxs]
        uniq_to_keep, uniq_to_keep_inverse = np.unique(
            to_keep, return_inverse=True)
        return CompressedAlleleCounts(self.config_array[uniq_to_keep, :, :],
                                      uniq_to_keep_inverse)

    def sort_configs(self):
        # sort configs so that "(very) similar" configs are next to each other
        # and will end up in the same batch,
        # thus avoiding redundant computation
        # "similar" == configs have same num missing alleles
        # "very similar" == configs are folded copies of each other
        a = self.config_array[:, :, 0]  # ancestral counts
        d = self.config_array[:, :, 1]  # derived counts
        n = a + d  # totals

        n = list(map(tuple, n))
        a = list(map(tuple, a))
        d = list(map(tuple, d))

        folded = list(map(min, list(zip(a, d))))

        keys = list(zip(n, folded))
        sorted_idxs = sorted(range(len(n)), key=lambda i: keys[i])
        sorted_idxs = np.array(sorted_idxs, dtype=int)

        unsorted_idxs = [None] * len(sorted_idxs)
        for i, j in enumerate(sorted_idxs):
            unsorted_idxs[j] = i
        unsorted_idxs = np.array(unsorted_idxs, dtype=int)

        self.config_array = self.config_array[sorted_idxs, :, :]
        self.index2uniq = unsorted_idxs[self.index2uniq]

    def count_configs(self):
        return np.bincount(self.index2uniq, minlength=len(self.config_array))

    @cached_property
    def n_samples(self):
        return np.max(np.sum(self.config_array, axis=2), axis=0)

    @property
    def index2uniq_mat(self):
        return scipy.sparse.coo_matrix(
            (np.ones(len(self)), (self.index2uniq, np.arange(len(self)))),
            shape=(len(self.config_array), len(self)))
