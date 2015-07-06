import networkx as nx
from Bio import Phylo
from cStringIO import StringIO
from util import cached_property
from size_history import ConstantTruncatedSizeHistory, ExponentialTruncatedSizeHistory, PiecewiseHistory
import numpy as np

class Demography(nx.DiGraph):
    @classmethod
    def from_newick(cls, newick, default_lineages=None, default_N=1.0):
        t = cls(_newick_to_nx(newick, default_lineages))
        # add models to edges
        for v0, v1, d in t.edges(data=True):
            n_sub = t.n_lineages_subtended_by[v1]
            nd = t.node_data[v1]
            _set_model(nd, n_sub, d['branch_length'], default_N=default_N)
        _set_model(t.node_data[t.root], t.n_lineages_subtended_by[t.root], float('inf'), default_N=default_N)
        return t

    def __init__(self, *args, **kwargs):
        super(Demography, self).__init__(*args, **kwargs)
        nd = self.node_data
        if not all('lineages' in nd[k] for k in self.leaves):
            raise Exception("'lineages' attribute must be set for each leaf node.")

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
        try:
            del self.n_derived_subtended_by
            del self.node_data
        except AttributeError:
            pass

    def to_newick(self):
        return _to_newick(self, self.root)


_field_factories = {
    "N": float, "lineages": int, "ancestral": int,
    "derived": int, "model": str, "n": int,
    "tau": float,
    }
def _extract_momi_fields(comment):
    for field in comment.split("&&"):
        if field.startswith("momi:"):
            attrs = field.split(":")
            assert attrs[0] == "momi"
            attrs = [a.split("=") for a in attrs[1:]]
            attrdict = dict((a, _field_factories[a.split('_')[0]](b)) for a, b in attrs)
            return attrdict
    return {}

def _newick_to_nx(newick, default_lineages=None):
    newick = StringIO(newick)
    phy = Phylo.read(newick, "newick")
    phy.rooted = True
    edges = []
    nodes = []
    node_data = {}
    clades = [phy.root]
    phy.root.name = phy.root.name or "root"
    i = 0
    while clades:
        clade = clades.pop()
        nd = _extract_momi_fields(clade.comment or "")
        if 'lineages' not in nd and default_lineages is not None:
            nd['lineages'] = default_lineages
        nodes.append((clade.name, nd))
        for c_clade in clade.clades:
            clades += clade.clades
            if c_clade.name is None:
                c_clade.name = "node%d" % i
                i += 1
            ed = {'branch_length': c_clade.branch_length}
            edges.append((clade.name, (c_clade.name), ed))
    t = nx.DiGraph(data=edges)
    t.add_nodes_from(nodes)
    tn = dict(t.nodes(data=True))
    for node in node_data:
        tn[node].update(node_data[node])
    return t

def _to_newick(G, root):
    parent = list(G.predecessors(root))
    try:
        edge_length = str(G[parent[0]][root]['branch_length'])
    except IndexError:
        edge_length = None
    if not G[root]:
        assert edge_length is not None
        return root + ":" + edge_length
    else:
        children = list(G[root])
        assert len(children) == 2
        dta = [(_to_newick(G, child),) for child in children]
        ret = "(%s,%s)" % (dta[0] + dta[1])
        if edge_length:
            ret += ":" + edge_length
        return ret

def _set_model(nd, n_sub, branch_length, default_N):
    if 'model' not in nd or nd['model'] == "constant":
        nd['model'] = ConstantTruncatedSizeHistory(
                N=nd.get('N', default_N),
                tau=branch_length,
                n_max=n_sub)
    elif nd['model'] == "exponential":
        nd['model'] = ExponentialTruncatedSizeHistory(n_max=n_sub,
                                                      tau=branch_length,
                                                      N_top=nd['N_top'],
                                                      N_bottom=nd['N_bottom'])
    elif nd['model'] == 'piecewise':
        i = -1
        epochs = []
        while True:
            i += 1
            try:
                model = nd['model_'+str(i)]
            except KeyError:
                break
            if  model == 'constant':
                epochs.append(ConstantTruncatedSizeHistory(n_max=n_sub,
                                                           tau=nd['tau_'+str(i)],
                                                           N=nd['N_'+str(i)]))
            elif model == 'exponential':
                epochs.append(ExponentialTruncatedSizeHistory(n_max=n_sub,
                                                              tau=nd['tau_'+str(i)],
                                                              N_top=nd['N_top_'+str(i)],
                                                              N_bottom=nd['N_bottom_'+str(i)]))
            else:
                raise Exception("Unsupported model type: %s" % model)
        nd['model'] = PiecewiseHistory(epochs)
        if not np.allclose(nd['model'].tau , branch_length):
            raise Exception("Sum of piecewise epoch lengths does not equal branch length")
    else:
        raise Exception("Unsupported model type: %s" % nd['model'])

