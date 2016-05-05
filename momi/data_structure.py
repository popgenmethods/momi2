
from .util import memoize_instance
import autograd.numpy as np
import scipy, scipy.misc
from scipy.misc import comb
from .math_functions import _apply_error_matrices
from collections import Counter
import warnings
import itertools

def config(a=None,d=None,n=None):
    """
    Returns config c, with c[pop][allele] == count of allele in pop

    Parameters
    ----------
    a : ancestral allele counts
    d : derived allele counts
    n : sample size

    Exactly 2 of a,d,n should be non-None
    """
    if sum([x is None for x in (a,d,n)]) != 1:
        raise ValueError("Exactly 1 of a,d,n should be None")
    if a is None:
        a = np.array(n) - np.array(d)
    elif d is None:
        d = np.array(n) - np.array(a)
    if np.any(a < 0) or np.any(d < 0):
        raise ValueError("Negative allele counts")
    return _config_tuple(np.array([a,d]).T)


class Configs(object):
    """
    Stores a list of configs. Important methods/attributes:

    Configs.sampled_pops: the population labels
    Configs[i] : the i-th config in the list
    Configs.sampled_n : the number of alleles sampled per population.
                        used to construct the likelihood vectors for
                        junction tree algorithm.
    """
    def __init__(self, sampled_pops, configs, sampled_n=None):
        """
        Notes
        -----
        If sampled_n=None, Configs.sampled_n will be the max number of observed
        individuals/alleles per population.
        """
        self.sampled_pops = tuple(sampled_pops)
        
        self.config_tuples = tuple(_config_tuple(c) for c in configs)
        if any(len(ctuple) != len(self.sampled_pops) for ctuple in self.config_tuples):
            raise TypeError("configuration length does not match sampled_pops")
        
        self.config_array = _config_array(self.config_tuples, len(self.sampled_pops))
                
        max_n = np.max(np.sum(self.config_array, axis=2), axis=0)
        
        if sampled_n is None: sampled_n = max_n
        sampled_n = np.array(sampled_n)
        if np.any(sampled_n < max_n): raise ValueError("config greater than sampled_n")
        self.sampled_n = sampled_n

        config_sampled_n = np.sum(self.config_array, axis=2)
        self.has_missing_data = np.any(config_sampled_n != self.sampled_n)

        self._index_dict = {c:i for i,c in enumerate(self.config_tuples)}
       
    def __getitem__(self, i): return self.config_tuples[i]
    def __len__(self): return len(self.config_tuples)
    def get_index(self, config): return self._index_dict[_config_tuple(config)]
    def __eq__(self, other):
        config_tuples = self.config_tuples
        try: return config_tuples == other.config_tuples
        except AttributeError: return False
    
    ## TODO: remove this method (and self.sampled_n attribute)
    def _copy(self, sampled_n=None):
        """
        Notes
        -----
        Note that momi.expected_sfs, momi.composite_log_likelihood require
        Demography.sampled_n == Configs.sampled_n.
        If this is not the case, you can use _copy() to create a copy with the correct
        sampled_n.
        Note this has no affect on the actual allele counts, as missing data is allowed.
        sampled_n is just used to construct (and store) certain vectors for the SFS algorithm.
        """
        if sampled_n is None:
            sampled_n = self.sampled_n
        return Configs(self.sampled_pops, self, sampled_n=sampled_n)

    def _vecs_and_idxs(self, folded):
        vecs,idxs = self._build_vecs_and_idxs(folded)
        ## copy idxs to make it safe
        return vecs, dict(idxs)
    
    @memoize_instance
    def _build_vecs_and_idxs(self, folded):
        # get row indices for each config
        n_rows = 0
        n_rows += 1 # initial row is a "zero" config

        denom_idx = n_rows # next row is for the normalization constant
        n_rows += 1
        
        config_2_row = {} # maps config -> row in vecs
        for config in set(self):
            config_2_row[config] = n_rows
            n_rows += 1
        idx_2_row = np.array([config_2_row[c] for c in self],
                             dtype = int)

        ## remove monomorphic configs
        ## (if there is missing data or error matrices,
        ##  expected_sfs_tensor_prod will return nonzero SFS
        ##  for monomorphic configs)
        monomorphic = np.any(np.sum(self, axis=1) == 0, axis=1)
        idx_2_row[monomorphic] = 0
        
        # get row indices for each denominator
        sample_sizes_array = np.sum(self.config_array, axis=2)
        if np.any(sample_sizes_array > self.sampled_n):
            raise Exception("There is a config that is larger than the specified sample size!")
        
        sample_sizes = [tuple(s) for s in sample_sizes_array]
        #ssize_2_row = {}
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
            rev_confs = list(map(_config_tuple, rev_confs))
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
        configs, rows = list(zip(*list(config_2_row.items())))
        rows = np.array(rows, ndmin=1)
        configs = np.array(configs, ndmin=3)

        for i in range(len(vecs)):
            n = self.sampled_n[i]
            derived = np.einsum("i,j->ji", np.ones(len(rows)), np.arange(n+1))
            curr = comb(derived, configs[:,i,1]) * comb(n-derived, configs[:,i,0]) / comb(n, np.sum(configs[:,i,:], axis=1))
            vecs[i][rows,:] = np.transpose(curr)

            # the normalization constant
            vecs[i][denom_idx,:] = np.ones(n+1)

        idxs = {'denom_idx': denom_idx, 'idx_2_row': idx_2_row}
        assert len(corrections_2_denom) == 2
        idxs.update({('corrections_2_denom',0): corrections_2_denom[0],
                     ('corrections_2_denom',1): corrections_2_denom[1]})
        try:
            idxs['folded_2_row'] = folded_2_row
        except UnboundLocalError:
            pass
        return vecs, idxs
    
def make_sfs(sampled_pops, loci):
    """
    Parameters
    ----------
    sampled_pops : list of the population labels
    loci : list of dicts, or list of lists

           if loci[i] is a dict, then loci[i][config]
           is the count of the config at locus i

           if loci[i] is a list, then loci[i][j]
           is the config of the j-th SNP at locus i
    """
    loci = [Counter(loc) for loc in loci]
    total = sum(loci, Counter())
    configs = Configs(sampled_pops, total.keys())
    loci = [{configs.get_index(k): v for k,v in loc.items()} for loc in loci]
    return Sfs(loci, configs)
    
class Sfs(object):
    """
    Represents an observed SFS across several loci.

    Important methods/attributes:
    Sfs.freq(config, locus) : the frequency of config at the locus (or in total, if locus=None)
    Sfs.loci : list of dicts, with Sfs.loci[locus][config] == Sfs.freq(config, locus=locus)
    Sfs.total : dict, with Sfs.total[config] == Sfs.freq(config, locus=None)
    """    
    def __init__(self, loci, configs):
        """
        Use make_sfs() instead of calling this constructor directly
        """
        self.loci = [Counter(loc) for loc in loci]
        self.total = sum(self.loci, Counter())
        self.configs = configs
        
    def freq(self, configuration, locus=None):
        """
        Notes
        -----
        If locus==None, returns the total frequency across all loci
        """
        assert np.array(configuration).shape == (len(self.sampled_pops), 2)
        configuration = self.configs.get_index(configuration)
        if locus is None:
            return self.total[configuration]
        else:
            return self.loci[locus][configuration]

    @property
    def sampled_n(self):
        return self.configs.sampled_n
    @property
    def sampled_pops(self):
        return self.configs.sampled_pops        
    @property
    def n_loci(self):
        return len(self.loci)
    @property
    def n_nonzero_entries(self):
        return len(self.total)
    @memoize_instance
    def n_snps(self, vector=False, locus=None):
        if vector:
            assert locus is None
            return np.array([self.n_snps(locus=loc) for loc in range(self.n_loci)])
        if locus is None:
            return sum(self.total.values())
        else:
            return sum(self.loci[locus].values())        
    def __eq__(self, other):
        loci_dicts = self.get_dict(vector=True)
        try: return loci_dicts == other.get_dict(vector=True)
        except AttributeError: return False
        
    def get_dict(self, vector=False, locus=None):
        if vector:
            assert locus is None
            return [self.get_dict(locus=loc) for loc in range(self.n_loci)]
        elif locus is None:
            return dict(zip(self.configs, self._total_freqs))
        idxs, counts = self._idxs_counts(locus)
        return dict(zip((self.configs[i] for i in idxs), counts))

    @property
    @memoize_instance
    def _entropy(self):
        _,counts = self._idxs_counts(None)
        p = counts / float(self.n_snps())
        return np.sum(p * np.log(p))

        
    def fold(self):
        """
        Returns a copy of the SFS, but with folded entries.
        """
        loci = []
        for l in self.loci:
            loci += [Counter()]
            for k,v in list(l.items()):
                k = np.array(self.configs[k])
                if tuple(k[:,0]) < tuple(k[:,1]):
                    k = k[:,::-1]
                k = _config_tuple(k)
                loci[-1][k] = loci[-1][k] + v
        ret = make_sfs(self.sampled_pops, loci)
        if np.any(ret.sampled_n != self.sampled_n):
            ret = ret._copy(sampled_n = self.sampled_n)
        return ret
           
    def _copy(self, sampled_n=None):
        """
        See also: Configs._copy()
        """
        if sampled_n is None:
            sampled_n = self.sampled_n        
        return Sfs(self.loci, Configs(self.sampled_pops, self.configs, sampled_n=sampled_n))

    def _integrate_sfs(self, weights, vector=False, locus=None):
        if vector:
            assert locus is None
            return np.array([self._integrate_sfs(weights,locus=loc) for loc in range(self.n_loci)])
        idxs, counts = self._idxs_counts(locus)
        return np.sum(weights[idxs] * counts)

    @property
    def _total_freqs(self):
        return self._idxs_counts(None)[1]
    
    @memoize_instance
    def _idxs_counts(self, locus):
        if locus is None:
            return (slice(None), np.array([self.total[i] for i in range(len(self.total))]))

        #idxs, counts = zip(*self.loci[locus].items())
        idxs = np.array(list(self.loci[locus].keys()), dtype=int)
        counts = np.array([self.loci[locus][k] for k in idxs], dtype=float)
        return idxs, counts    

def make_seg_sites_data(sampled_pops, config_sequences):
    """
    Parameters
    ----------
    sampled_pops : sequence of the population labels
    config_sequences : sequence of sequences of configs
                       config_sequences[i][j] is the configuration at the jth SNP of the ith locus
    """
    return SegSites(sampled_pops, config_sequences)
    
class SegSites(object):
    def __init__(self, sampled_pops, config_sequences, sampled_n=None):
        """
        Use make_seg_sites_data() instead of calling this constructor directly
        """
        config2idx, configs = {}, []

        self.idx_list = []
        for locus_configs in config_sequences:
            self.idx_list.append([])
            for config in locus_configs:
                config = _config_tuple(config)
                try:
                    idx = config2idx[config]
                except KeyError:
                    idx = len(configs)
                    configs.append(config)
                    config2idx[config] = idx
                self.idx_list[-1].append(idx)
        #self.idx_list = [np.array(locus_idxs,dtype=int) for locus_idxs in self.idx_list]
        
        self.configs = Configs(sampled_pops, configs, sampled_n=sampled_n)
        self.sfs = Sfs(self.idx_list, self.configs)

        #assert set(self.sfs.total.keys()) == set(self.configs)
   
    def get_config(self, locus, site):
        return self.configs[self.idx_list[locus][site]]

    def __getitem__(self, loc):
        if loc >= self.n_loci: raise IndexError("Locus out of bounds")
        return (self.get_config(loc, site) for site in range(self.n_snps(locus=loc)))
        
    
    @property
    def sampled_pops(self): return self.sfs.sampled_pops
    @property
    def sampled_n(self): return self.sfs.sampled_n
    @property
    def n_loci(self): return self.sfs.n_loci
    def n_snps(self, locus=None): return self.sfs.n_snps(locus=locus)
    
    def __eq__(self, other):
        configs, idx_list = self.configs, self.idx_list
        try:
            return configs == other.configs and idx_list == other.idx_list
        except AttributeError:
            return False
    
def _config_array(configurations, n_pops):
    if len(configurations) == 0:
        configurations = np.zeros((0,n_pops,2))
    else:
        configurations = np.array(configurations, ndmin=3, dtype=int)
    assert configurations.shape[1:] == (n_pops,2)
    configurations.setflags(write=False)                
    return configurations
    
def _config_tuple(config):
    return tuple(map(tuple, config))    

def write_seg_sites(sequences_file, seg_sites):
    sampled_pops = seg_sites.sampled_pops

    sequences_file.write("\t".join(map(str,sampled_pops)) + "\n")
    for locus_configs in seg_sites:
        sequences_file.write("\n//\n\n")
        for config in locus_configs:
            sequences_file.write("\t".join([",".join(map(str,x)) for x in config]) + "\n")

def read_seg_sites(sequences_file):
    #ret = []
    stripped_lines = (line.strip() for line in sequences_file)
    lines = (line for line in stripped_lines if line != "" and line[0] != "#")

    def get_loc(line):
        if line.startswith("//"):
            get_loc.curr += 1
        return get_loc.curr
    get_loc.curr = -1

    loci = itertools.groupby(lines, get_loc)
    
    _,header = next(loci)
    sampled_pops = tuple(next(header).split())

    def get_configs(locus):
        assert next(locus).startswith("//")
        for line in locus:
            yield tuple(tuple(map(int,x.split(","))) for x in line.split())
            
    return SegSites(sampled_pops, (get_configs(loc) for i,loc in loci))

def _configs_from_derived(derived_counts, sampled_n, sampled_pops):
    input_counts = np.array(derived_counts)    
    derived_counts = np.array(input_counts, ndmin=2)
    ret = [config(d=d,n=sampled_n) for d in derived_counts]
    return Configs(sampled_pops, ret)

def _sfs_subset(configs, counts):
    assert len(counts.shape) == 1 and len(counts) == len(configs.config_array)

    subidxs = np.arange(len(counts))[counts != 0]
    sub_configs = _Configs_Subset(configs, subidxs)

    counts = counts[counts != 0]

    return Sfs([{i:c for i,c in enumerate(counts)}], sub_configs)

class _Configs_Subset(object):
    ## Efficient access to subset of configs
    def __init__(self, configs, sub_idxs):
        self.sub_idxs = sub_idxs
        self.full_configs = configs
        for a in ("sampled_n", "sampled_pops", "has_missing_data"):
            setattr(self, a, getattr(self.full_configs, a))

    # @property
    # def config_array(self):
    #     return self.full_configs.config_array[self.sub_idxs,:,:]
            
    def _vecs_and_idxs(self, folded):
        vecs,_ = self.full_configs._vecs_and_idxs(folded)
        old_idxs, idxs = self._build_idxs(folded)

        vecs = [v[old_idxs,:] for v in vecs]
        ## copy idxs to make it safe
        return vecs, dict(idxs)
        
    @memoize_instance
    def _build_idxs(self, folded):
        _,idxs = self.full_configs._vecs_and_idxs(folded)

        denom_idx_key = 'denom_idx'
        denom_idx = idxs[denom_idx_key]
        idxs = {k: v[self.sub_idxs] for k,v in list(idxs.items()) if k != denom_idx_key}

        old_idxs = np.array(list(set(sum(list(map(list, list(idxs.values()))) + [[denom_idx]], []))))
        old_2_new_idxs = {old_id: new_id for new_id, old_id in enumerate(old_idxs)}

        idxs = {k: np.array([old_2_new_idxs[old_id]
                             for old_id in v])
                for k,v in list(idxs.items())}
        idxs[denom_idx_key] = old_2_new_idxs[denom_idx]
        return old_idxs, idxs

# class _AbstractSfs(object):
#     @property
#     @memoize_instance
#     def _counts_i(self):
#         return np.einsum("ij->i", self._counts_ij)

#     @property
#     @memoize_instance
#     def _counts_j(self):
#         return np.einsum("ij->j", self._counts_ij)

#     @property
#     @memoize_instance
#     def _total_count(self):
#         return np.sum(self._counts_j)
    
# class _SubSfs(_AbstractSfs):
#     ## represents a subsample of SFS
#     def __init__(self, configs, counts):
#         assert len(counts.shape) == 1 and len(counts) == len(configs.config_array)
        
#         subidxs = np.arange(len(counts))[counts != 0]
#         self.configs = _SubConfigs(configs, subidxs)
        
#         counts = counts[counts != 0]
#         self._counts_ij = np.array(counts, ndmin=2)
