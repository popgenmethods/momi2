import networkx as nx
from cached_property import cached_property
from autograd.numpy import log
from util import memoize_instance, memoize, my_einsum, fft_einsum
import scipy, scipy.misc
import autograd.numpy as np
from sum_product import SumProduct, _lik_axes

import parse_ms
import random
import itertools


class Demography(nx.DiGraph):
    @classmethod
    def from_ms(cls, ms_cmd, *params, **kwargs):
        return cls(parse_ms.to_nx(ms_cmd, *params, **kwargs))

    @classmethod
    def from_newick(cls, newick, default_lineages=None, default_N=1.0):
        ms_cmd,leafs = parse_ms.from_newick(newick, default_lineages, default_N)
        ret = cls.from_ms(ms_cmd, leafs=leafs)
        return ret

    def __init__(self, *args, **kwargs):
        super(Demography, self).__init__(*args, **kwargs)
        nd = self.node_data
        if not all('lineages' in nd[k] for k in self.leaves):
            raise Exception("'lineages' attribute must be set for each leaf node.")
        for v in self:
            if 'model_func' in nd[v] and 'model' not in nd[v]:
                nd[v]['model'] = nd[v]['model_func'](self.n_lineages_at_node[v])
        if not all('model' in nd[k] for k in self):
            raise Exception("'model' attribute must be set for all nodes.")

    @cached_property
    def eventTree(self):
        eventEdgeList = []
        currEvents = {l : (l,) for l in self.leaves}
        eventDict = {e : {'subpops' : (l,), 'parent_pops' : (l,), 'child_pops' : {}} for l,e in currEvents.iteritems()}
        
        for e in self.graph['events']:
            # get the population edges forming the event
            parent_pops, child_pops = map(set, zip(*e))
            child_events = set([currEvents[c] for c in child_pops])
            assert len(e) == 2 and len(parent_pops) + len(child_pops) == 3 and len(child_events) in (1,2)

            sub_pops = set(itertools.chain(*[eventDict[c]['subpops'] for c in child_events]))
            sub_pops.difference_update(child_pops)
            sub_pops.update(parent_pops)

            eventDict[e] = {'parent_pops' : tuple(parent_pops), 'subpops' : tuple(sub_pops), 'child_pops' : {c : currEvents[c] for c in child_pops}}
            currEvents.update({p : e for p in sub_pops})
            for p in child_pops:
                del currEvents[p]
            eventEdgeList += [(e, c) for c in child_events]
        ret = nx.DiGraph(eventEdgeList)
        for e in eventDict:
            ret.add_node(e, **(eventDict[e]))

        assert len(currEvents) == 1
        root, = [v for k,v in currEvents.iteritems()]
        ret.root = root

        return ret
    
    @property
    def event_root(self):
        return self.eventTree.root

    def event_type(self, event):
        if len(event) == 1:
            return 'leaf'
        elif len(self.parent_pops(event)) == 2:
            return 'admixture'
        elif len(self.eventTree[event]) == 2:
            return 'merge_clusters'
        else:
            return 'merge_subpops'

    def sub_pops(self, event):
        return self.eventTree.node[event]['subpops']

    def parent_pops(self, event):
        return self.eventTree.node[event]['parent_pops']

    # returns dict of {childPop : childEvent}
    def child_pops(self, event):
        return self.eventTree.node[event]['child_pops']

    @cached_property
    def totalSfsSum(self):
        return normalizing_constant(self)

    @cached_property
    def root(self):
        nds = [node for node, deg in self.in_degree().items() if deg == 0]
        assert len(nds) == 1
        return nds[0]
    
    @cached_property
    def node_data(self):
        return dict(self.nodes(data=True))

    @cached_property
    def leaves(self):
        return set([k for k, v in self.out_degree().items() if v == 0])

#     @cached_property
#     def n_lineages_subtended_by(self):
#         nd = self.node_data
#         return {v: sum(nd[l]['lineages'] for l in self.leaves_subtended_by[v]) for v in self}
    @cached_property
    def n_lineages_at_node(self):
        '''Due to admixture events, # lineages at node >= # lineages at leafs'''
        nd = self.node_data
        n_lin_dict = {}
        for v in nx.dfs_postorder_nodes(self, self.root):
            if self.is_leaf(v):
                n_lin_dict[v] = nd[v]['lineages']
            else:
                n_lin_dict[v] = sum([n_lin_dict[c] for c in self[v]])
        return n_lin_dict

#     @cached_property
#     def n_derived_subtended_by(self):
#         nd = self.node_data
#         return {v: sum(nd[l]['derived'] for l in self.leaves_subtended_by[v]) for v in self}

    @cached_property
    def leaves_subtended_by(self):
        return {v: self.leaves & set(nx.dfs_preorder_nodes(self, v)) for v in self}

    '''
    Simulates the SFS from the demography.
    If theta = None, uses total branch lengths for frequencies (ala fastsimcoal)

    returns (sumFreqs,sumSqFreqs,nonzeroFreqs)
    where
    sumFreqs = sum of frequencies across all datasets
    sumSqFreqs = sum of squared frequencies across all datasets
    nonzeroFreqs = # of datasets where frequency was > 0
    '''
    def simulate_sfs(self, num_sims, theta=None, seed=None, additionalParams=""):
        return parse_ms.simulate_sfs(self, num_sims, theta, seed, additionalParams)

    @memoize_instance
    def admixture_prob(self, admixture_node):
        '''
        Returns ndarray with dimensions [child_der, par1_der, par2_der]

        child_der: # derived alleles in admixture_node
        par1_der, par2_der: # derived alleles in parent1, parent2 of admixture_node

        returns probability of child_der given par1_der, par2_der
        '''
        n_node = self.n_lineages_at_node[admixture_node]

        # admixture node must have two parents
        edge1,edge2 = self.in_edges([admixture_node], data=True)
        nd = self.node_data[admixture_node]
        parent1,parent2 = edge1[0], edge2[0]
        prob1,prob2 = nd['splitprobs'][parent1], nd['splitprobs'][parent2]
        assert prob1 + prob2 == 1.0

        n_from_1 = np.arange(n_node+1)
        n_from_2 = n_node - n_from_1
        binom_coeffs = (prob1**n_from_1) * (prob2**n_from_2) * scipy.misc.comb(n_node, n_from_1)
        ret = my_einsum(der_in_admixture_node(n_node), range(4),
                        binom_coeffs, [0],
                        [1,2,3])
        assert ret.shape == tuple([n_node+1] * 3)
        return ret, [admixture_node, parent1, parent2]

    def is_leaf(self, node):
        return node in self.leaves

    def update_state(self, state):
        nd = self.node_data
        for node in state:
            ndn = nd[node]
            ndn.update(state[node])
            if np.any(ndn['lineages'] != ndn['derived'] + ndn['ancestral']):
                raise Exception("derived + ancestral must add to lineages at node %s" % node)
        # Invalidate the caches which depend on node state
        # FIXME: breaks for version 1.0.0 of cached_property module!
#         self.n_derived_subtended_by # make sure cache exists
#         del self.n_derived_subtended_by #reset cache
#         del self.node_data  #reset cache

    def sfs_entries(self, sfs, normalized=False):
        leaves = sorted(self.leaves)
        st = {a: {'derived' : [], 'ancestral' : []} for a in leaves}
        for states in sfs:
            for a, b in zip(leaves, states):
                st[a]['derived'].append(b)
                st[a]['ancestral'].append(self.n_lineages_at_node[a] - b)
        for a in leaves:
            st[a]['derived'],st[a]['ancestral'] = map(lambda x: np.array(st[a][x]),
                                                      ['derived','ancestral'])
        self.update_state(st)
        sp = SumProduct(self)
        return sp.p(normalized=normalized)

    # returns log likelihood under a Poisson random field model
    def log_likelihood_prf(self, theta, sfs):
        #ret = -self.totalSfsSum * theta / 2.0
            #st = {a: {'derived': b, 'ancestral': self.n_lineages_at_node[a] - b} for a, b in zip(leaves, states)}
            #self.update_state(st)
            #sp = SumProduct(self)
            #ret += log(sp.p(normalized=False) * theta / 2.0) * weight - scipy.special.gammaln(weight+1)

        sfs,w = zip(*sorted(sfs.iteritems()))
        w = np.array(w)

        ret = -self.totalSfsSum * theta / 2.0 + np.sum(log(self.sfs_entries(sfs) * theta / 2.0) * w - scipy.special.gammaln(w+1))

        #assert ret < 0.0 and ret > -float('inf')
        assert ret < 0.0
        return ret

@memoize
def der_in_admixture_node(n_node):
    '''
    returns 4d-array, [n_from_parent1, der_in_child, der_in_parent1, der_in_parent2]
    '''
    # axis0=n_from_parent, axis1=der_from_parent, axis2=der_in_parent
    der_in_parent = np.tile(np.arange(n_node+1), (n_node+1,n_node+1,1))
    n_from_parent = np.transpose(der_in_parent, [2,0,1])
    der_from_parent = np.transpose(der_in_parent, [0,2,1])
    
    anc_in_parent = n_node - der_in_parent
    anc_from_parent = n_from_parent - der_from_parent
    
    x = scipy.misc.comb(der_in_parent, der_from_parent) * scipy.misc.comb(anc_in_parent, anc_from_parent) / scipy.misc.comb(n_node, n_from_parent)

    return fft_einsum(x, [0, 1, 2],
                      x[::-1,...], [0, 1, 3],
                      [0,1,2,3], [1])[:,:(n_node+1),:,:]


def normalizing_constant(demography):
    # get the previous state
    ## TODO: remove this, make state a property of SumProduct instead of Demography
    try:
        prev_state = {}
        for v in demography.leaves:
            nd = demography.node_data[v]
            prev_state[v] = {'ancestral' : nd['ancestral'], 'derived' : nd['derived']}
    except:
        prev_state = None

    # to_directed() makes a deep-copy of the nx.DiGraph
    #demography = Demography(demography.to_directed())
    # set all alleles to be of ancestral type
    state = {}
    for v in demography.leaves:
        state[v] = {}
        state[v]['derived'] = 0
        state[v]['ancestral'] = demography.node_data[v]['lineages']
    demography.update_state(state)
    # now create the Sum-Product
    sp = SumProduct(demography)

    ret = 0.0
    for event in sp.G.eventTree:
        bottom_likelihood,_ = sp.partial_likelihood(event)
        #event_subpops = demography.sub_pops(event)
        for newpop in demography.parent_pops(event):
            newpop_idx = _lik_axes(sp.G,event).index(newpop)
            idx = [0] * bottom_likelihood.ndim
            idx[0], idx[newpop_idx] = slice(None), slice(None)
#             newpop_idx = event_subpops.index(newpop)
#             idx = [0] * bottom_likelihood.ndim
#             idx[newpop_idx] = slice(bottom_likelihood.shape[newpop_idx])
            # 1 - partial_likelihood_bottom is probability of at least one derived leaf lineage
            #ret = ret + np.sum((1.0 - bottom_likelihood[idx]) * sp.truncated_sfs(newpop))
            ret = ret + my_einsum(1.0-bottom_likelihood[idx], ['',newpop],
                                  sp.truncated_sfs(newpop), [newpop],
                                  [''])

    # subtract off the term for all alleles derived
    state = {}
    for v in demography.leaves:
        state[v] = {}
        state[v]['derived'] = demography.node_data[v]['lineages']
        state[v]['ancestral'] = 0
    demography.update_state(state)
    # now create the Sum-Product
    sp = SumProduct(demography)

    ret = ret - sp.p(normalized=False)

    ## now reset the state
    ## TODO: remove this, make state a property of SumProduct instead of Demography
    if prev_state is not None:
        demography.update_state(prev_state)
    assert ret > 0.0
    return np.squeeze(ret)
