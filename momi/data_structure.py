
from .util import memoize_instance, _hashable_config, folded_sfs
import autograd.numpy as np
import scipy, scipy.misc
from scipy.misc import comb
from .math_functions import _apply_error_matrices
import warnings

class ConfigList(object):
    """
    Constructs and stores the necessary likelihood vectors,
    to pass to expected_sfs_tensor_prod().

    Can be passed into expected_sfs(observed_sfs,...), in lieu of a list of tuples.
    This can be a little bit faster if calling expected_sfs() many times, 
    otherwise expected_sfs() will create one each time it is called.
    """
    def __init__(self, configs, sampled_n):
        self.sampled_n = sampled_n
        
        self.config_array = np.array(configs, ndmin=3)
        if self.config_array.shape[2] != 2 or self.config_array.shape[1] != len(sampled_n):
            raise Exception("Incorrectly formatted configs")
        self.configs = map(_hashable_config, self.config_array)
        
    def __len__(self):
        return len(self.configs)

    def __iter__(self):
        for config in self.configs:
            yield config

    def __getitem__(self,k):
        return self.configs[k]

    def _apply_to_vecs(self, f, folded=False, normalized=False):
        vecs = dict(self._get_vecs(folded=folded))
        vals = f(vecs['vecs'])
        ret = vals[vecs['idx_2_row']]
        if folded:
            ret = ret + vals[vecs['folded_2_row']]
        if normalized:
            denom = vals[vecs['denom_idx']]
            for i,corr_idxs in enumerate(vecs["corrections_2_denom"]):
                denom = denom - vals[corr_idxs]
            ret = ret / denom
        return ret
   
    @memoize_instance
    def _get_vecs(self, folded=False):       
        # get row indices for each config
        n_rows = 0
        n_rows += 1 # initial row is a "zero" config

        denom_idx = n_rows # next row is for the normalization constant
        n_rows += 1
        
        config_2_row = {} # maps config -> row in vecs
        for config in set(self.configs):
            config_2_row[config] = n_rows
            n_rows += 1
        idx_2_row = np.array([config_2_row[c] for c in self.configs],
                             dtype = int)

        ## remove monomorphic configs
        ## (if there is missing data or error matrices,
        ##  expected_sfs_tensor_prod will return nonzero SFS
        ##  for monomorphic configs)
        monomorphic = np.any(np.sum(self.configs, axis=1) == 0, axis=1)
        idx_2_row[monomorphic] = 0
        
        # get row indices for each denominator
        sample_sizes_array = np.sum(self.config_array, axis=2)
        if np.any(sample_sizes_array > self.sampled_n):
            raise Exception("There is a config that is larger than the specified sample size!")
        
        sample_sizes = [tuple(s) for s in sample_sizes_array]
        ssize_2_row = {}
        ssize_2_corrections = [{}, {}] # corrections for monomorphic sites (all ancestral & all derived)
        for s in set(sample_sizes):
            ## add rows for monomorphic correction terms
            for mono_allele in (0,1):
                mono_config = np.array([s, [0]*len(s)], dtype=int, ndmin=2)
                if mono_allele == 1:
                    mono_config = mono_config[::-1,:]
                mono_config = tuple(map(tuple, np.transpose(mono_config)))
                if mono_config not in config_2_row:
                    config_2_row[mono_config] = n_rows
                    n_rows += 1
                ssize_2_corrections[mono_allele][s] = config_2_row[mono_config]
        corrections_2_denom = [np.array([corr_row[s] for s in sample_sizes], dtype=int)
                               for corr_row in ssize_2_corrections]
        
        # get row indices for folded configs
        if folded:
            rev_confs = self.config_array[:,:,::-1]
            is_symm = np.all(self.config_array == rev_confs, axis=(1,2))
            rev_confs = map(_hashable_config, rev_confs)
            folded_2_row = []
            for rc,symm in zip(rev_confs, is_symm):
                if symm:
                    # map to 0 if symmetric                    
                    folded_2_row += [0]
                else:
                    if rc not in config_2_row:
                        config_2_row[rc] = n_rows
                        n_rows += 1
                    folded_2_row += [config_2_row[rc]]
            folded_2_row = np.array(folded_2_row, dtype=int)
            folded_2_row[monomorphic] = 0 ## dont use monomorphic configs
           
        # construct the vecs
        vecs = [np.zeros((n_rows, n+1)) for n in self.sampled_n]
        
        # construct rows for each config
        configs, rows = zip(*config_2_row.items())
        rows = np.array(rows, ndmin=1)
        configs = np.array(configs, ndmin=3)

        for i in range(len(vecs)):
            n = self.sampled_n[i]
            derived = np.einsum("i,j->ji", np.ones(len(rows)), np.arange(n+1))
            curr = comb(derived, configs[:,i,1]) * comb(n-derived, configs[:,i,0]) / comb(n, np.sum(configs[:,i,:], axis=1))
            vecs[i][rows,:] = np.transpose(curr)

            # the normalization constant
            vecs[i][denom_idx,:] = np.ones(n+1)
        
        ret = {'vecs': vecs, 'denom_idx': denom_idx, 'idx_2_row': idx_2_row, 'corrections_2_denom': corrections_2_denom}
        try:
            ret['folded_2_row'] = folded_2_row
        except UnboundLocalError:
            pass
        ## return value is cached; for safety, return an immutable object
        return tuple(ret.items())

class ObservedSfs(object):
    """
    Can be passed into likelihood.composite_log_likelihood(observed_sfs,...),
    in lieu of a dict.

    Can be a little bit faster to do this if evaluating many times,
    as this will store some computations that would otherwise be repeated.
    
    See also: likelihood.composite_log_likelihood(), ConfigList
    """
    def __init__(self, observed_sfs, sampled_n):
        self._list = ObservedSfsList([observed_sfs], sampled_n)
    def _sfs_list(self):
        return self._list
    
class ObservedSfsList(object):
    """
    Can be passed into likelihood.composite_log_lik_vector(observed_sfs_list,...),
    in lieu of a list of dicts.

    Can be a little bit faster to do this if evaluating many times,
    as this will store some computations that would otherwise be repeated.

    See also: likelihood.composite_log_lik_vector(), ConfigList
    """
    def __init__(self, observed_sfs_list, sampled_n):       
        # remove 0 entries
        self._list = [dict([(k,v) for k,v in sfs.items() if v != 0]) for sfs in observed_sfs_list]
        self.sampled_n = sampled_n

    @memoize_instance
    def _sfs_list(self, folded):
        if folded:
            # for correct combinatorial factors
            return [folded_sfs(sfs) for sfs in self._list] # for correct combinatorial factors
        return self._list
    
    @memoize_instance
    def _config_list(self, folded):
        sfs_list = self._sfs_list(folded)
        # the list of all observed configs
        return ConfigList(list(set(sum([list(sfs.keys()) for sfs in sfs_list],[]))), self.sampled_n)

    @memoize_instance
    def _counts_ij(self, folded):
        # counts_ij is a matrix whose [i,j]th entry is the count of config j at locus i
        sfs_list, config_list = self._sfs_list(folded), self._config_list(folded)
        counts_ij = np.zeros((len(sfs_list), len(config_list)))
        for i,sfs in enumerate(sfs_list):
            for j,config in enumerate(config_list):
                try:
                    counts_ij[i,j] = sfs[config]
                except KeyError:
                    pass
        return counts_ij

    
# class MultiLocusConfig(object):
#     def __init__(self, loci_list, sampled_n, folded=False):
#         # make sure all configs are tuples (hashable)
#         loci_list = [[tuple(c) for c in locus] for locus in loci_list]
        

#         # get mapping from loci to uniq_idxs
#         config_2_idx = {c : idx for idx, c in enumerate(self.uniq_configs)}
#         self.loci_2_uidx = [np.array([config_2_idx[c] for c in locus], dtype=int)
#                             for locus in loci_list]

    
#     def loci_vals(self, vals, normalized=False):
#         ret = self.adjust_vals(vals, normalized=normalized)
#         return [np.array(ret)[idxs]
#                 for idxs in self.loci_2_uidx]
