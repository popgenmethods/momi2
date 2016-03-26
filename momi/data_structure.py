
from .util import memoize_instance
import autograd.numpy as np
import scipy, scipy.misc
from scipy.misc import comb
from .math_functions import _apply_error_matrices
from collections import Counter
import warnings

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
    return tuple(map(tuple, np.array([a,d]).T))

class Configs(tuple):
    """
    Stores a list of configs. Important methods/attributes:

    Configs.sampled_pops: the population labels
    Configs[i] : the i-th config in the list
    Configs.folded : boolean if the configs are folded
    Configs.sampled_n : the number of alleles sampled per population.
                        used to construct the likelihood vectors for
                        junction tree algorithm.
    """
    def __new__(cls, sampled_pops, configs, fold=False, sampled_n=None):
        return tuple.__new__(cls, map(_hashable_config, configs))
    
    def __init__(self, sampled_pops, configs, fold=False, sampled_n=None):
        """
        Notes
        -----
        If sampled_n=None, Configs.sampled_n will be the max number of observed
        individuals/alleles per population.
        """
        self.sampled_pops = sampled_pops
        self.config_array = _format_configurations(configs, len(sampled_pops), fold)
        self.folded = fold
        max_n = np.max(np.sum(configs, axis=2), axis=0)
        if sampled_n is None:
            sampled_n = max_n
        if np.any(sampled_n < max_n):
            raise ValueError("config greater than sampled_n")
        self.sampled_n = sampled_n

    def copy(self, fold=False, sampled_n=None):
        """
        Notes
        -----
        If self.folded=True, then fold has no affect (the copy will always be folded).
        Note that momi.expected_sfs, momi.composite_log_likelihood require
        Demography.sampled_n == Configs.sampled_n.
        If this is not the case, you can use copy() to create a copy with the correct
        sampled_n.
        Note this has no affect on the actual allele counts, as missing data is allowed.
        sampled_n is just used to construct (and store) certain vectors for the SFS algorithm.
        """
        if sampled_n is None:
            sampled_n = self.sampled_n
        return Configs(self.sampled_pops, self, fold=(fold or self.folded), sampled_n=sampled_n)

    def _apply_to_vecs(self, f, normalized=False):
        vecs = dict(self._get_vecs())
        vals = f(vecs['vecs'])
        ret = vals[vecs['idx_2_row']]
        if self.folded:
            ret = ret + vals[vecs['folded_2_row']]
        if normalized:
            denom = vals[vecs['denom_idx']]
            for i,corr_idxs in enumerate(vecs["corrections_2_denom"]):
                denom = denom - vals[corr_idxs]
            ret = ret / denom
        return ret
   
    @memoize_instance
    def _get_vecs(self):       
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
        if self.folded:
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

    
class Sfs(object):
    """
    Represents an observed SFS across several loci.

    Important methods/attributes:
    Sfs.freq(config, locus) : the frequency of config at the locus (or in total, if locus=None)
    Sfs.loci : list of dicts, with Sfs.loci[locus][config] == Sfs.freq(config, locus=locus)
    Sfs.total : dict, with Sfs.total[config] == Sfs.freq(config, locus=None)
    """    
    def __init__(self, sampled_pops, loci, fold=False, sampled_n=None):
        """
        Parameters
        ----------
        sampled_pops : list of the population labels
        loci : list of dicts, or list of lists

               if loci[i] is a dict, then loci[i][config]
               is the count of the config at locus i

               if loci[i] is a list, then loci[i][j]
               is the config of the j-th SNP at locus i
        fold : boolean to fold the SFS
        sampled_n : list of number of sampled alleles per population. (see copy())
        """
        old, loci = loci, []
        for l in old:
            loci += [Counter()]
            try:
                l = l.items()
            except:
                l = zip(l, [1]*len(l))
            if len(l) == 0:
                continue
            configs,freqs = zip(*l)
            configs = map(_hashable_config,
                          _format_configurations(configs, len(sampled_pops), fold))
            for c,f in zip(configs, freqs):
                loci[-1][c] += f
        
        self.folded = fold        
        self.loci = map(_ConfigDict, loci)
        self.total = _ConfigDict(sum(self.loci, Counter()))
        self.sampled_pops = tuple(sampled_pops)
        if np.array(self.total.keys(),ndmin=3).shape[1:] != (len(sampled_pops), 2):
            raise TypeError("len(sampled_pops) != len of individual configs")
        self._sampled_n = sampled_n
        # self.configs = Configs(self.sampled_pops, list(self.total.keys()),
        #                        fold, sampled_n)

    def __getstate__(self):
        return {'sampled_pops' : self.sampled_pops,
                'sampled_n' : self._sampled_n,
                'loci' : self.loci,
                'fold' : self.folded}
    def __setstate__(self, state):
        self.__init__(**state)
        
    @property
    def configs(self):
        return self._configs()
    @memoize_instance
    def _configs(self):
        return Configs(self.sampled_pops, list(self.total.keys()),
                       self.folded, self._sampled_n)
        
    def freq(self, configuration, locus=None):
        """
        Notes
        -----
        If locus==None, returns the total frequency across all loci
        """
        assert np.array(configuration).shape == (len(self.sampled_pops), 2)
        configuration = _hashable_config(configuration)
        if locus is None:
            return self.total[configuration]
        else:
            return self.loci[locus][configuration]

    @property
    def sampled_n(self):
        if self._sampled_n is None:
            return self.configs.sampled_n
        return self._sampled_n
        
    def copy(self, fold=False, sampled_n=None):
        """
        See also: Configs.copy()
        """
        if sampled_n is None:
            sampled_n = self.sampled_n        
        return Sfs(self.sampled_pops, self.loci, fold=(fold or self.folded), sampled_n=sampled_n)
        
    def __eq__(self, other):
        return self.loci == other.loci and self.sampled_pops == other.sampled_pops
    def __ne__(self, other):
        return not self == other
        
    @memoize_instance
    def _counts_ij(self):
        # counts_ij is a matrix whose [i,j]th entry is the count of config j at locus i
        config_list = self.configs
        counts_ij = np.zeros((len(self.loci), len(config_list)))
        for i,sfs in enumerate(self.loci):
            for j,config in enumerate(config_list):
                try:
                    counts_ij[i,j] = sfs[config]
                except KeyError:
                    pass
        return counts_ij

class _ConfigDict(Counter):
    def __init__(self, *args, **kwargs):
        self.immutable = False
        Counter.__init__(self, *args,**kwargs)
        self.immutable = True
    def __setitem__(self, key, item):
        if self.immutable:
           raise NotImplementedError()
        Counter.__setitem__(self, _hashable_config(key),item)    
    
class SegSites(object):
    """
    Represents a bunch of segregating sites by their configurations
    and positions.

    Important attributes/methods:
    SegSites.config_arrays: list of arrays. config_arrays[locus][i, pop, allele] is the allele count in pop at the i-th site in locus
    SegSites.position_arrays: list of lists. position_arrays[locus][i] is the position (between 0 and 1) of the i-th site on the locus.
    """
    def __init__(self, sampled_pops, config_arrays, position_arrays, fold=False, sampled_n=None):
        """
        Parameters
        ----------
        sampled_pops : list of the population labels
        config_arrays : list of array-likes
        position_arrays : list of list
        fold : boolean to fold the configs
        sampled_n : list of number of samples per population (see SegSites.copy())
        """
        if len(position_arrays) != len(config_arrays) or any(len(p) != len(c) for p,c in zip(position_arrays,config_arrays)):
            raise TypeError("position_arrays and config_arrays should have the same lengths")
      
        # make sure they are sorted
        for p in position_arrays:
            p = np.array(p)
            if np.any(p[1:] < p[:-1]):
                raise TypeError("position_arrays are not sorted")
        
        self.position_arrays = tuple(map(tuple, position_arrays))
        self.config_arrays = tuple(_format_configurations(c, len(sampled_pops), fold) for c in config_arrays)
        self.folded = fold

        self._sampled_pops = sampled_pops
        self._sampled_n = sampled_n # could be None instead of the true sampled_n
        
        #self.sfs = Sfs(sampled_pops, self.config_arrays, fold=fold, sampled_n=sampled_n)

    def __getstate__(self):
        return {'position_arrays' : self.position_arrays,
                'config_arrays' : self.config_arrays,
                'fold' : self.folded,
                'sampled_pops' : self._sampled_pops,
                'sampled_n' : self._sampled_n}

    def __setstate__(self, state):
        self.__init__(**state)
        
    @property
    def sfs(self):
        return self._sfs()
    @memoize_instance
    def _sfs(self):
        return Sfs(self.sampled_pops, self.config_arrays, fold=self.folded, sampled_n=self._sampled_n)
    
    def position(self,locus,site):
        return self.position_arrays[locus][site]

    def allele_count(self, locus, site, pop, allele):
        return self.config_arrays[locus][site,pop,allele]

    def config(self, locus, site):
        return self.config_arrays[locus][site,:,:]
        
    @property
    def sampled_pops(self):
        return self._sampled_pops

    @property
    def sampled_n(self):
        if self._sampled_n is None:
            return self.sfs.sampled_n
        return self._sampled_n
    
    def copy(self, fold=False, sampled_n=None):
        if sampled_n is None:
            sampled_n = self.sampled_n        
        return SegSites(self.sampled_pops, self.config_arrays, self.position_arrays, fold=(fold or self.folded), sampled_n=sampled_n)
    
    def __eq__(self, other):
        try:
            return self.position_arrays == other.position_arrays and len(self.config_arrays) == len(other.config_arrays) and all(np.all(s==a) for s,a in zip(self.config_arrays,other.config_arrays)) and self.sampled_pops == other.sampled_pops
        except AttributeError:
            return False
    def __ne__(self, other):
        return not self == other    

def _format_configurations(configurations, n_pops, fold):
    if len(configurations) == 0:
        configurations = np.zeros((0,n_pops,2))
    else:
        configurations = np.array(configurations, ndmin=3, dtype=int)
        assert configurations.shape[1:] == (n_pops,2)

    if fold:
        for i,c in list(enumerate(configurations)):
            if tuple(c[:,0]) < tuple(c[:,1]):
                configurations[i,:,:] = c[:,::-1]

    configurations.setflags(write=False)                
    return configurations
    
def _hashable_config(config):
    return tuple(map(tuple, config))    

def write_seg_sites(sequences_file, seg_sites, sampled_pops=None):
    if sampled_pops is None:
        try:
            sampled_pops = seg_sites.sampled_pops
        except AttributeError:
            raise AttributeError("seg_sites.sampled_pops attribute does not exist; must provide sampled_pops argument")
    elif hasattr(seg_sites, 'sampled_pops'):
        warnings.warn("sampled_pops provided, ignoring seg_sites.sampled_pops attribute")
    sequences_file.write("Position\t:\t" + "\t".join(map(str,sampled_pops)) + "\n")
    #for seq in seg_sites:
    for locus_pos, locus_configs in zip(seg_sites.position_arrays, seg_sites.config_arrays):
        sequences_file.write("\n//\n\n")
        for pos,config in zip(locus_pos,locus_configs):
            sequences_file.write(str(pos) + "\t:\t" + "\t".join([",".join(map(str,x)) for x in config]) + "\n")

def read_seg_sites(sequences_file):
    #ret = []
    linenum = 0
    positions, configs = [], []
    for line in sequences_file:
        line = line.strip()
        if line == "" or line[0] == "#":
            continue

        if linenum == 0:
            _,sampled_pops = line.split(":")
            sampled_pops = tuple(sampled_pops.strip().split())
        elif line[:2] == "//":
            #ret += [[]]
            positions += [[]]
            configs += [[]]
        else:
            pos,config = line.split(":")
            pos = float(pos)
            config = config.strip().split()
            config = tuple(tuple(map(int,x.split(","))) for x in config)
            #ret[-1] += [(pos,config)]
            positions[-1] += [pos]
            configs[-1] += [config]
        
        linenum += 1
    return SegSites(sampled_pops, configs, positions)

def _configs_from_derived(derived_counts, sampled_n, sampled_pops):
    input_counts = np.array(derived_counts)    
    derived_counts = np.array(input_counts, ndmin=2)
    ret = [config(d=d,n=sampled_n) for d in derived_counts]
    return Configs(sampled_pops, ret)
# def _configs_from_derived(derived_counts, sampled_n):
#     input_counts = np.array(derived_counts)
    
#     derived_counts = np.array(input_counts, ndmin=2)
#     ret = np.transpose([sampled_n - derived_counts, derived_counts],
#                        axes=[1,2,0])
#     ret = [tuple(map(tuple,x)) for x in ret]
    
#     if input_counts.shape != derived_counts.shape:
#         assert derived_counts.shape[0] == 1
#         ret = ret[0]
#     return ret   
