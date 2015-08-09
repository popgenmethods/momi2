import numpy as np
import sktensor as skt
from sktensor.tucker import hosvd
import pandas as pd

def sfs_eval_dirs(sfs, dirs):
    projection = 0.
    ## TODO: vectorize for loop?
    for config,val in sfs.iteritems():
        for leaf,i in zip(sorted(dirs.keys()), config):
            val = val * dirs[leaf][:,i]
        projection = projection + val
    return projection

def get_sfs_tensor(sfs, n_per_pop):
    idx, vals = zip(*(sfs.iteritems()))
    idx = tuple(np.array(x) for x in zip(*idx))
    return skt.sptensor(idx, vals, shape=tuple(n+1 for n in n_per_pop), dtype=np.float)

def greedy_hosvd(sfs_tensor, n_entries, verbose=False):
    U_list = hosvd(sfs_tensor, sfs_tensor.shape, compute_core=False)
    total_energy = sfs_tensor.norm()**2
    curr_entries = [(total_energy, [], [], sfs_tensor)]
    
    for d in range(len(sfs_tensor.shape)):
        prev_entries = curr_entries
        curr_entries = []
        
        for prev_energy, prev_dirs, prev_idxs, prev_tens in prev_entries:

            energy_sum = 0.0
            for next_idx, next_dir in enumerate(U_list[d].T):
                next_tens = prev_tens.ttv((next_dir,) , (0,) )
                try:
                    energy = next_tens.norm()**2
                except AttributeError:
                    energy = next_tens**2
                energy_sum += energy
                curr_entries.append( (energy,
                                      prev_dirs + [next_dir],
                                      prev_idxs + [next_idx],
                                      next_tens))

            curr_entries = sorted(curr_entries, key=lambda x: x[0], reverse=True)[:n_entries]
            assert np.isclose(energy_sum, prev_energy)
    if verbose:
        #print "# Selected components:\n", [idx for _,_,idx,_ in curr_entries]
        to_print = pd.DataFrame([(idx, energy / total_energy) for energy,_,idx,_ in curr_entries],
                                columns=['Component','Percent Energy'])
        print "# Selected components:\n", to_print, "\n# Unselected percent energy:", 1.0-sum(to_print['Percent Energy'])
    return [dirs for _,dirs,_,_ in curr_entries]
