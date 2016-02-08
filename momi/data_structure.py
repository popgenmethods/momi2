
from .util import reversed_configs, folded_sfs, memoize_instance
import autograd.numpy as np
       

class ConfigList(object):
    """
    Constructs and stores the necessary likelihood vectors,
    to pass to expected_sfs_tensor_prod().

    Can be passed into expected_sfs(observed_sfs,...), in lieu of a list of tuples.
    This can be a little bit faster if calling expected_sfs() many times, 
    otherwise expected_sfs() will create one each time it is called.
    """
    def __init__(self, configs, sampled_n):
        self.configs = [tuple(c) for c in configs]
        self.sampled_n = sampled_n
        
    def __len__(self):
        return len(self.configs)

    def __iter__(self):
        for config in self.configs:
            yield config

    def __getitem__(self,k):
        return self.configs[k]

    def _apply_to_vecs(self, f, folded=False, normalized=False):
        vecs = self.get_vecs(folded=folded)           
        vals = f(vecs['vecs'])
        def corrected_vals(idxs):
            # the values at idx, with correction coeffs for monomorphic sites subtracted off
            # (needs to be done in case of subsamples, or error matrices)
            corr_coeffs = [1.,1.]
            ret = 0.0
            for i,corr_idxs in enumerate(vecs["corrections_2_row"]):
                for j,v in enumerate(vecs['vecs']):
                    corr_coeffs[i] = corr_coeffs[i] * np.einsum("ij,ij->i",
                                                                v[idxs,:],
                                                                v[corr_idxs,:])
                ret = ret - corr_coeffs[i] * vals[corr_idxs]
            return vals[idxs] + ret
        ret = corrected_vals(vecs['idx_2_row'])
        if folded:
            ret = ret + corrected_vals(vecs['folded_2_row'])
        if normalized:
            ret = ret / corrected_vals(vecs['denom_2_row'])
        return ret
    
    @memoize_instance
    def get_vecs(self, folded=False):
        # get row indices for each config
        n_rows = 0
        config_2_row = {} # maps config -> row in vecs
        for config in self.configs:
            config_2_row[config] = n_rows
            n_rows += 1
        idx_2_row = np.array([config_2_row[c] for c in self.configs],
                             dtype = int)

        # get row indices for each denominator
        sample_sizes = [tuple(self.sampled_n) for _ in self.configs]
        ssize_2_row = {}
        ssize_2_corr_row = [{}, {}] # corrections for monomorphic sites (all ancestral & all derived)
        for s in set(sample_sizes):
            ssize_2_row[s] = n_rows
            n_rows += 1
            # add rows for correction terms
            for corr_row in ssize_2_corr_row:
                corr_row[s] = n_rows
                n_rows += 1
        denom_2_row = np.array([ssize_2_row[s] for s in sample_sizes],
                               dtype = int)
        corrections_2_row = [np.array([corr_row[s] for s in sample_sizes], dtype=int)
                             for corr_row in ssize_2_corr_row]
        
        # get row indices for folded configs
        if folded:
            rev_confs, is_symm = reversed_configs(self.configs, sample_sizes, True)
            folded_2_row = []
            for rc,symm in zip(rev_confs, is_symm):
                if symm:
                    # map to 0 if symmetric
                    rc = tuple([0]*len(self.sampled_n))
                if rc not in config_2_row:
                    config_2_row[rc] = n_rows
                    n_rows += 1
                folded_2_row += [config_2_row[rc]]
            folded_2_row = np.array(folded_2_row, dtype=int)
           
        # construct the vecs
        vecs = [np.zeros((n_rows, n+1)) for n in self.sampled_n]
        
        # construct rows for each config
        for i in range(len(vecs)):
            rows, derived = zip(*[(row,c[i]) for c,row in config_2_row.items()])
            vecs[i][rows, derived] = 1.0
        
        # construct rows for each sample size
        for i,n in enumerate(self.sampled_n):
            def get_denom_vec(ssize):
                if ssize != tuple(self.sampled_n):
                    ## TODO: implement this
                    raise NotImplementedError("Not yet implemented missing data")
                return np.ones(n+1)
            rows, denom_vecs = zip(*[(row, get_denom_vec(s)) for s,row in ssize_2_row.items()])
            vecs[i][rows,:] = denom_vecs

            # correction rows
            for j,corr_row in enumerate(ssize_2_corr_row):
                def get_correction_vec(ssize):
                    if ssize != tuple(self.sampled_n):
                        ## TODO: implement this
                        raise NotImplementedError("Not yet implemented missing data")
                    ret = np.zeros(n+1)
                    ret[j*n] = 1.0
                    return ret
                rows, corr_vecs = zip(*[(row, get_correction_vec(s)) for s,row in corr_row.items()])
                vecs[i][rows,:] = corr_vecs
            
        ret = {'vecs': vecs, 'denom_2_row': denom_2_row, 'idx_2_row': idx_2_row, 'corrections_2_row': corrections_2_row}
        try:
            ret['folded_2_row'] = folded_2_row
        except UnboundLocalError:
            pass
        return ret

class ObservedSfs(object):
    """
    Can be passed into likelihood.unlinked_log_likelihood(observed_sfs,...),
    in lieu of a dict.

    Can be a little bit faster to do this if evaluating many times,
    as this will store some computations that would otherwise be repeated.
    
    See also: likelihood.unlinked_log_likelihood(), ConfigList
    """
    def __init__(self, observed_sfs, sampled_n):
        self._list = ObservedSfsList([observed_sfs], sampled_n)
    def _sfs_list(self):
        return self._list
    
class ObservedSfsList(object):
    """
    Can be passed into likelihood.unlinked_log_lik_vector(observed_sfs_list,...),
    in lieu of a list of dicts.

    Can be a little bit faster to do this if evaluating many times,
    as this will store some computations that would otherwise be repeated.

    See also: likelihood.unlinked_log_lik_vector(), ConfigList
    """
    def __init__(self, observed_sfs_list, sampled_n):       
        # remove 0 entries
        self._list = [dict([(k,v) for k,v in sfs.items() if v != 0]) for sfs in observed_sfs_list]
        self.sampled_n = sampled_n

    @memoize_instance
    def _sfs_list(self, folded):
        if folded:
            return [folded_sfs(sfs, self.sampled_n) for sfs in self._list] # for correct combinatorial factors
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