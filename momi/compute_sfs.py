import networkx as nx
import autograd.numpy as np
from .data.configurations import ConfigList
from .math_functions import (hypergeom_quasi_inverse,
                             binom_coeffs,
                             _apply_error_matrices,
                             convolve_trailing_axes,
                             sum_trailing_antidiagonals)
from .moran_model import moran_transition


def expected_sfs(
        demography, configs, mut_rate=1.0, normalized=False,
        folded=False, error_matrices=None):
    """
    Expected sample frequency spectrum (SFS) entries for the specified
    demography and configs. The expected SFS is the expected number of
    observed mutations for a configuration. If mutation rate=1, it is
    equivalent to the expected branch length subtended by a configuration.

    Parameters
    ----------
    demography : Demography
    configs : ConfigList
        if configs.folded == True, returns the folded SFS entries
    mut_rate : float
         mutation rate per unit time
    normalized : optional, bool
         if True, mut_rate is ignored, and the SFS is divided by the
         expected total branch length.
         The returned values then represent probabilities, that a given
         mutation will segregate according to the specified configurations.

    Returns
    -------
    sfs : 1d numpy.ndarray
         sfs[j] is the SFS entry corresponding to configs[j]


    Other Parameters
    ----------------
    folded: optional, bool
         if True, return the folded SFS value for each entry
         Default is False.
    error_matrices : optional, sequence of 2-dimensional numpy.ndarray
         length-D sequence, where D = number of demes in demography.
         error_matrices[i] describes the sampling error in deme i as:
         error_matrices[i][j,k] = P(observe j mutants in deme i | k mutants in deme i)
         If error_matrices is not None, then the returned value is adjusted
         to account for this sampling error, in particular the effect it
         has on the total number of observed mutations.

    See Also
    --------
    expected_total_branch_len : sum of all expected SFS entries
    expected_sfs_tensor_prod : compute summary statistics of SFS
    """
    sfs, denom = _expected_sfs(demography, configs, folded, error_matrices)
    if normalized:
        sfs = sfs / denom
    else:
        sfs = sfs * mut_rate
    return sfs


def _expected_sfs(demography, configs, folded, error_matrices):
    if np.any(configs.sampled_n != demography.sampled_n) or np.any(configs.sampled_pops != demography.sampled_pops):
        raise ValueError(
            "configs and demography must have same sampled_n, sampled_pops. Use Demography.copy() or ConfigList.copy() to make a copy with different sampled_n.")

    vecs, idxs = configs._vecs_and_idxs(folded)

    if error_matrices is not None:
        vecs = _apply_error_matrices(vecs, error_matrices)

    vals = expected_sfs_tensor_prod(vecs, demography)

    sfs = vals[idxs['idx_2_row']]
    if folded:
        sfs = sfs + vals[idxs['folded_2_row']]

    denom = vals[idxs['denom_idx']]
    for i in (0, 1):
        denom = denom - vals[idxs[("corrections_2_denom", i)]]

    #assert np.all(np.logical_or(vals >= 0.0, np.isclose(vals, 0.0)))

    return sfs, denom


def expected_total_branch_len(demography, error_matrices=None, ascertainment_pop=None,
                              sampled_pops=None, sampled_n=None):
    """
    The expected total branch length of the sample genealogy.
    Equivalently, the expected number of observed mutations when
    mutation rate=1.

    Parameters
    ----------
    demography : Demography

    Returns
    -------
    total : float-like
         the total expected number of SNPs/branch length

    Other Parameters
    ----------------
    error_matrices : optional, sequence of 2-dimensional numpy.ndarray
         length-D sequence, where D = number of demes in demography.
         error_matrices[i] describes the sampling error in deme i as:
         error_matrices[i][j,k] = P(observe j mutants in deme i | k mutants in deme i)
         If error_matrices is not None, then the returned value is adjusted
         to account for this sampling error, in particular the effect it
         has on the total number of observed mutations.

    See Also
    --------
    expected_sfs : individual SFS entries
    expected_tmrca, expected_deme_tmrca : other interesting statistics
    expected_sfs_tensor_prod : compute general class of summary statistics
    """
    if ascertainment_pop is None:
        ascertainment_pop = [True] * len(demography.sampled_n)
    ascertainment_pop = np.array(ascertainment_pop)

    vecs = [[np.ones(n + 1), [1] + [0] * n, [0] * n + [1]]
            if asc else np.ones((3, n + 1), dtype=float)
            for asc, n in zip(ascertainment_pop, demography.sampled_n)]
    if error_matrices is not None:
        vecs = _apply_error_matrices(vecs, error_matrices)

    ret = expected_sfs_tensor_prod(vecs, demography)
    return ret[0] - ret[1] - ret[2]


def expected_tmrca(demography, sampled_pops=None, sampled_n=None):
    """
    The expected time to most recent common ancestor of the sample.

    Parameters
    ----------
    demography : Demography

    Returns
    -------
    tmrca : float-like

    See Also
    --------
    expected_deme_tmrca : tmrca of subsample within a deme
    expected_sfs_tensor_prod : compute general class of summary statistics
    """
    vecs = [np.ones(n + 1) for n in demography.sampled_n]
    n0 = len(vecs[0]) - 1.0
    vecs[0] = np.arange(n0 + 1) / n0
    return np.squeeze(expected_sfs_tensor_prod(vecs, demography))


def expected_heterozygosity(demography, restrict_to_pops=None,
                            error_matrices=None):
    if restrict_to_pops is None:
        restrict_to_pops = demography.sampled_pops
    configs = np.zeros(
        (len(restrict_to_pops), len(demography.sampled_pops), 2),
        dtype=int)
    for i, pop in enumerate(restrict_to_pops):
        if pop in restrict_to_pops:
            configs[i, demography.sampled_pops.index(pop), :] = 1
    configs = ConfigList(demography.sampled_pops, configs,
                         sampled_n=demography.sampled_n)
    return expected_sfs(demography, configs,
                        error_matrices=error_matrices)


def expected_deme_tmrca(demography, deme, sampled_pops=None, sampled_n=None):
    """
    The expected time to most recent common ancestor, of the samples within
    a particular deme.

    Parameters
    ----------
    demography : Demography
    deme : the deme

    Returns
    -------
    tmrca : float

    See Also
    --------
    expected_tmrca : the tmrca of the whole sample
    expected_sfs_tensor_prod : compute general class of summary statistics
    """
    deme = list(demography.sampled_pops).index(deme)
    vecs = [np.ones(n + 1) for n in demography.sampled_n]

    n = len(vecs[deme]) - 1
    vecs[deme] = np.arange(n + 1) / (1.0 * n)
    vecs[deme][-1] = 0.0

    return np.squeeze(expected_sfs_tensor_prod(vecs, demography))


def expected_sfs_tensor_prod(vecs, demography, mut_rate=1.0, sampled_pops=None):
    """
    Viewing the SFS as a D-tensor (where D is the number of demes), this
    returns a 1d array whose j-th entry is a certain summary statistic of the
    expected SFS, given by the following tensor-vector multiplication:

    res[j] = \sum_{(i0,i1,...)} E[sfs[(i0,i1,...)]] * vecs[0][j,i0] * vecs[1][j, i1] * ...

    where E[sfs[(i0,i1,...)]] is the expected SFS entry for config
    (i0,i1,...), as given by expected_sfs

    Parameters
    ----------
    vecs : sequence of 2-dimensional numpy.ndarray
         length-D sequence, where D = number of demes in the demography.
         vecs[k] is 2-dimensional array, with constant number of rows, and
         with n[k]+1 columns, where n[k] is the number of samples in the
         k-th deme. The row vector vecs[k][j,:] is multiplied against
         the expected SFS along the k-th mode, to obtain res[j].
    demo : Demography
    mut_rate : float
         the rate of mutations per unit time

    Returns
    -------
    res : numpy.ndarray (1-dimensional)
        res[j] is the tensor multiplication of the sfs against the vectors
        vecs[0][j,:], vecs[1][j,:], ... along its tensor modes.

    See Also
    --------
    sfs_tensor_prod : compute the same summary statistics for an observed SFS
    expected_sfs : compute individual SFS entries
    expected_total_branch_len, expected_tmrca, expected_deme_tmrca :
         examples of coalescent statistics that use this function
    """
    # NOTE cannot use vecs[i] = ... due to autograd issues
    sampled_n = [np.array(v).shape[-1] - 1 for v in vecs]
    vecs = [np.vstack([np.array([1.0] + [0.0] * n),  # all ancestral state
                       np.array([0.0] * n + [1.0]),  # all derived state
                       v])
            for v, n in zip(vecs, demography.sampled_n)]

    res = _expected_sfs_tensor_prod(vecs, demography, mut_rate=mut_rate)

    # subtract out mass for all ancestral/derived state
    for k in (0, 1):
        res = res - res[k] * np.prod([l[:, -k] for l in vecs], axis=0)
        assert np.isclose(res[k], 0.0)
    # remove monomorphic states
    res = res[2:]

    return res


def _expected_sfs_tensor_prod(vecs, demography, mut_rate=1.0):
    leaf_states = dict(list(zip(demography.sampled_pops, vecs)))

    res = LikelihoodTensorList.compute_sfs(
        leaf_states, demography)

    return res * mut_rate


class LikelihoodTensorList(object):
    @classmethod
    def compute_sfs(cls, leaf_states, demo):
        liklist = cls(leaf_states, demo)
        for event in nx.dfs_postorder_nodes(
                demo._event_tree):
            liklist._process_event(event)
        assert len(liklist.likelihood_list) == 1
        lik, = liklist.likelihood_list
        return lik.sfs

    def __init__(self, leaf_liks_dict, demo):
        self.likelihood_list = [
            LikelihoodTensor(l, 0, [p])
            for p, l in leaf_liks_dict.items()
        ]
        self.demo = demo

    def _get_likelihoods(self, pop):
        for lik in self.likelihood_list:
            if pop in lik.pop_labels:
                return lik

    def _process_event(self, event):
        e_type = self.demo._event_type(event)
        if e_type == 'leaf':
            self._process_leaf_likelihood(event)
        elif e_type == 'merge_subpops':
            self._process_merge_subpops_likelihood(event)
        elif e_type == 'merge_clusters':
            self._process_merge_clusters_likelihood(event)
        elif e_type == 'pulse':
            self._process_pulse_likelihood(event)
        else:
            raise Exception("Unrecognized event type.")

        for newpop in self.demo._parent_pops(event):
            lik = self._get_likelihoods(newpop)
            lik.make_last_axis(newpop)
            n = lik.get_last_axis_n()
            if n > 0:
                lik.add_last_axis_sfs(self.demo._truncated_sfs(newpop))
                if event != self.demo._event_root:
                    lik.matmul_last_axis(
                        np.transpose(moran_transition(
                            self.demo._scaled_time(newpop), n)))

    def _rename_pop(self, oldpop, newpop):
        self._get_likelihoods(oldpop).rename_pop(
            oldpop, newpop)

    def _process_leaf_likelihood(self, event):
        (pop, idx), = self.demo._parent_pops(event)
        if idx == 0:
            self._rename_pop(pop, (pop, idx))
        else:
            # ghost population
            batch_size = self.likelihood_list[0].liks.shape[0]
            self.likelihood_list.append(LikelihoodTensor(
                np.ones((batch_size, 1)), 0,
                [(pop, idx)]
            ))

    def _in_same_lik(self, pop1, pop2):
        return self._get_likelihoods(pop1) is self._get_likelihoods(pop2)

    def _merge_pops(self, newpopname, child_pops, n=None):
       child_liks = [self._get_likelihoods(p)
                     for p in child_pops]

       for lik, pop in zip(child_liks, child_pops):
           lik.make_last_axis(pop)
           lik.mul_trailing_binoms()

       pop1, pop2 = child_pops
       lik1, lik2 = child_liks
       if lik1 is lik2:
           lik1.sum_trailing_antidiagonals()
       else:
           self.likelihood_list.remove(lik2)
           lik1.convolve_trailing_axes(lik2)

       lik1.mul_trailing_binoms(divide=True)
       lik1.rename_pop(pop1, newpopname)

       if n is not None:
            N = lik1.get_last_axis_n()
            if n < N:
                lik1.matmul_last_axis(
                    hypergeom_quasi_inverse(N, n))

    def _process_merge_clusters_likelihood(self, event):
        child_pops = list(self.demo._child_pops(
            event).keys())
        parent_pop, = self.demo._parent_pops(event)

        assert not self._in_same_lik(*child_pops)
        self._merge_pops(parent_pop, child_pops)

    def _process_merge_subpops_likelihood(self, event):
        child_pops = list(self.demo._child_pops(
            event).keys())
        parent_pop, = self.demo._parent_pops(event)
        n = self.demo._n_at_node(parent_pop)

        assert self._in_same_lik(*child_pops)
        self._merge_pops(parent_pop, child_pops, n=n)

    def _process_pulse_likelihood(self, event):
        parent_pops = self.demo._parent_pops(event)
        child_pops_events = self.demo._child_pops(event)
        assert len(child_pops_events) == 2
        child_pops, child_events = list(zip(*list(child_pops_events.items())))

        recipient, non_recipient, donor, non_donor = self.demo._pulse_nodes(event)
        assert set(parent_pops) == set([donor, non_donor])
        assert set(child_pops) == set([recipient, non_recipient])
        if len(set(child_events)) == 2:
            ## more memory-efficient to do split then join
            admixture_probs, admixture_idxs = self.demo._admixture_prob(recipient)
            admixture_probs_dims = [recipient, non_donor, donor]
            assert set(admixture_probs_dims) == set(admixture_idxs)
            admixture_probs = np.transpose(
                admixture_probs,
                [admixture_idxs.index(i)
                 for i in admixture_probs_dims])

            recipient_lik = self._get_likelihoods(recipient)
            donor_lik = self._get_likelihoods(non_recipient)
            assert donor_lik is not recipient_lik

            recipient_lik.make_last_axis(recipient)
            donor_lik.make_last_axis(non_recipient)

            recipient_lik.admix_trailing_pop(admixture_probs, donor)
            self._merge_pops(donor, [donor, non_recipient])
            self._rename_pop(recipient, non_donor)
        else:
            # in this case, (typically) more memory-efficient to multiply likelihood by transition 4-tensor
            # (if only 2 populations, and much fewer SFS entries than samples, it may be more efficient to replace -ep with -es,-ej)
            child_event, = set(child_events)
            lik = self._get_likelihoods(recipient)
            assert lik is self._get_likelihoods(non_recipient)
            pulse_probs, pulse_idxs = self.demo._pulse_prob(event)
            pulse_probs_dims = [recipient, non_recipient, non_donor, donor]
            assert set(pulse_probs_dims) == set(pulse_idxs)
            pulse_probs = np.transpose(pulse_probs, [
                pulse_idxs.index(i)
                for i in pulse_probs_dims])

            lik.make_last_axis(recipient)
            lik.make_last_axis(non_recipient)

            lik.matmul_last_axis(pulse_probs, axes=2)

            lik.rename_pop(recipient, non_donor)
            lik.rename_pop(non_recipient, donor)


class LikelihoodTensor(object):
    def __init__(self, liks, sfs, pop_labels):
        self.liks = liks
        self.sfs = sfs
        self.pop_labels = pop_labels
        # extra leading dimension for batch
        assert len(self.pop_labels) + 1 == len(
            self.liks.shape)

    def copy(self):
        return LikelihoodTensor(self.liks, self.sfs, list(self.pop_labels))

    def admix_trailing_pop(self, admix_probs_3tensor,
                           newpop_name):
        """
        admix_probs_3tensor should be array returned by demography._admixture_prob_helper
        """
        self.matmul_last_axis(admix_probs_3tensor)
        self.pop_labels.append(newpop_name)

    def sum_trailing_antidiagonals(self):
        trailing_shape = list(self.liks.shape[-2:])
        lik = np.reshape(self.liks, [-1] + trailing_shape)
        lik = sum_trailing_antidiagonals(lik)
        self.liks = np.reshape(lik, [-1] + list(self.liks.shape[1:-2]) + [sum(trailing_shape) - 1])
        self.pop_labels.pop()

    def convolve_trailing_axes(self, other):
        def within_pop_sfs(a, b):
            return a.sfs * b.liks[
                [slice(None)] + [0] * len(b.pop_labels)]
        self.sfs = within_pop_sfs(
            self, other) + within_pop_sfs(other, self)

        class reshape_to_3tensor(object):
            def __init__(self, lik):
                self.lik = np.reshape(lik, [
                    lik.shape[0], -1, lik.shape[-1]])
                self.old_shape = lik.shape

        reshaped0 = reshape_to_3tensor(other.liks)
        reshaped1 = reshape_to_3tensor(self.liks)

        assert reshaped0.lik.shape[0] == reshaped1.lik.shape[0]
        convolved = convolve_trailing_axes(reshaped0.lik, reshaped1.lik)
        self.liks = np.reshape(
            convolved,
            [reshaped0.lik.shape[0]] +
            list(reshaped0.old_shape[1:-1]) +
            list(reshaped1.old_shape[1:-1]) +
            [-1])
        self.pop_labels = list(
            other.pop_labels[:-1]) + list(self.pop_labels)

    @property
    def n_pops(self):
        return len(self.pop_labels)

    @property
    def n_axes(self):
        # extra dimension for the batches (data)
        return self.n_pops + 1

    def pop_axis(self, pop):
        # extra leading dimension for batch (data)
        return self.pop_labels.index(pop) + 1

    def rename_pop(self, oldpop, newpop):
        self.pop_labels[
            self.pop_labels.index(oldpop)] = newpop

    def make_last_axis(self, pop):
        axis = self.pop_axis(pop)
        perm = [i for i in range(self.n_axes)
                if i != axis] + [axis]
        self.liks = np.transpose(self.liks, perm)
        self.pop_labels = [p for p in self.pop_labels if p != pop] + [pop]
        assert len(self.pop_labels) + 1 == len(self.liks.shape)

    def add_last_axis_sfs(self, truncated_sfs):
        self.sfs = self.sfs + np.dot(
            self.liks[[slice(None)] +
                      [0] * (self.n_pops - 1) +
                      [slice(None)]],
            truncated_sfs)

    def get_last_axis_n(self):
        return self.liks.shape[-1] - 1

    def get_last_axis_binom_coeffs(self):
        return binom_coeffs(self.get_last_axis_n())

    def mul_trailing_binoms(self, divide=False):
        coeffs = self.get_last_axis_binom_coeffs()
        if divide:
            coeffs = 1.0 / coeffs
        self.mul_trailing(coeffs)

    def matmul_last_axis(self, mat, axes=1):
        reshaped_liks = np.reshape(self.liks, [-1] + [np.prod(
            self.liks.shape[-axes:])])
        reshaped_mat = np.reshape(mat, [np.prod(
            mat.shape[:axes], dtype=int)] + [-1])
        reshaped_liks = np.dot(reshaped_liks, reshaped_mat)
        self.liks = np.reshape(reshaped_liks, list(self.liks.shape[:-axes]) + list(mat.shape[axes:]))

    def mul_trailing(self, to_mult):
        self.liks = self.liks * to_mult
