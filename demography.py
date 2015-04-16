import networkx as nx
from cached_property import cached_property
from sum_product import SumProduct
import parse_ms

import random
import itertools

class Demography(nx.DiGraph):
    @classmethod
    def from_ms(cls, ms_cmd, leafs=None):
        return cls(parse_ms.to_nx(ms_cmd, leafs=leafs))

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
                nd[v]['model'] = nd[v]['model_func'](self.n_lineages_subtended_by[v])
        if not all('model' in nd[k] for k in self):
            raise Exception("'model' attribute must be set for all nodes.")

    @cached_property
    def eventTree(self):
        eventEdgeList = []
        currEvents = {l : (l,) for l in self.leaves}
        eventDict = {e : {'subpops' : (l,), 'newpops' : (l,)} for l,e in currEvents.iteritems()}
        
        for e in self['events']:
            # get the population edges forming the event
            parent_pops, child_pops = map(set, zip(*e))
            child_events = set([currEvents[c] for c in child_pops])
            assert len(e) == 2 and len(parent_pops) + len(child_pops) == 3 and len(child_events) in (1,2)

            sub_pops = set(itertools.chain(*[eventDict[c]['subpops'] for c in child_events]))
            sub_pops.difference_update(child_pops)
            sub_pops.update(parent_pops)

            currEvents.update({p : e for p in sub_pops})
            for p in child_pops:
                del currEvents[p]
            eventDict[e] = {'newpops' : tuple(parent_pops), 'subpops' : tuple(sub_pops)}
            eventEdgeList += [(e, c) for c in child_events]
        ret = nx.DiGraph(eventEdgeList)
        for e in eventDict:
            ret.add_node(e, **(eventDict[e]))
        return ret

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

    @cached_property
    def n_lineages_subtended_by(self):
        nd = self.node_data
        return {v: sum(nd[l]['lineages'] for l in self.leaves_subtended_by[v]) for v in self}

    @cached_property
    def n_derived_subtended_by(self):
        nd = self.node_data
        return {v: sum(nd[l]['derived'] for l in self.leaves_subtended_by[v]) for v in self}

    @cached_property
    def leaves_subtended_by(self):
        return {v: self.leaves & set(nx.dfs_preorder_nodes(self, v)) for v in self}

    def is_leaf(self, node):
        return node in self.leaves

    def update_state(self, state):
        nd = self.node_data
        for node in state:
            ndn = nd[node]
            ndn.update(state[node])
            if ndn['lineages'] != ndn['derived'] + ndn['ancestral']:
                raise Exception("derived + ancestral must add to lineages at node %s" % node)
        # Invalidate the caches which depend on node state
        # FIXME: breaks for version 1.0.0 of cached_property module!
        self.n_derived_subtended_by # make sure cache exists
        del self.n_derived_subtended_by #reset cache
        del self.node_data  #reset cache

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


def normalizing_constant(demography):
    # to_directed() makes a deep-copy of the nx.DiGraph
    demography = Demography(demography.to_directed())
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
    for node in demography:
        # 1 - partial_likelihood_bottom is probability of at least one derived leaf lineage
        ret += ((1.0 - sp.partial_likelihood_bottom(node)) * sp.truncated_sfs(node)).sum()

    # subtract off the term for all alleles derived
    state = {}
    for v in demography.leaves:
        state[v] = {}
        state[v]['derived'] = demography.node_data[v]['lineages']
        state[v]['ancestral'] = 0
    demography.update_state(state)
    # now create the Sum-Product
    sp = SumProduct(demography)

    ret -= sp.p(normalized=False)
    return ret
