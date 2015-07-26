import numpy as np
import sktensor as skt
from sktensor.tucker import hosvd
from Queue import PriorityQueue

def get_sfs_tensor(sfs, n_per_pop):
    idx, vals = zip(*(sfs.iteritems()))
    idx = tuple(np.array(x) for x in zip(*idx))
    return skt.sptensor(idx, vals, shape=tuple(n+1 for n in n_per_pop), dtype=np.float)

def greedy_hosvd(sfs_tensor, n_steps, verbose=False):
    U_list = hosvd(sfs_tensor, sfs_tensor.shape, compute_core=False)

    searched = set([])
    to_search = PriorityQueue()

    def elem_vecs(elem):
        return tuple([U[:,e] for U,e in zip(U_list, elem)])
    
    def elem_priority(elem):
        return -(sfs_tensor.ttv(elem_vecs(elem))**2)
        
    def add_elem(elem):
        if elem in searched or any([e < 0 for e in elem]):
            return
        try:
            to_search.put((elem_priority(elem), elem))
            searched.add(elem)
        except IndexError:
            pass

    add_elem( tuple([0] * len(sfs_tensor.shape)))
    for i in range(n_steps):
        if to_search.empty():
            break
        _,next_elem = to_search.get()
        for j in range(len(next_elem)):
            to_add = list(next_elem)
            to_add[j] += 1
            add_elem(tuple(to_add))

    ret = [x for _,x in sorted([(elem_priority(elem), elem)
                                for elem in searched])[:n_steps]]
    if verbose:
        print "# Selected components:\n", ret
    ret = [elem_vecs(x) for x in ret]
    return ret
