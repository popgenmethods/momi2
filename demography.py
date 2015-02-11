import networkx as nx
from Bio import Phylo
from cStringIO import StringIO
from truncated_size_history import ConstantTruncatedSizeHistory
from cached_property import cached_property

def _extract_momi_fields(comment):
    for field in comment.split("&&"):
        if field.startswith("momi:"):
            attrs = field.split(":")
            assert attrs[0] == "momi"
            attrdict = dict(a.split("=") for a in attrs[1:])
            return attrdict

class Demography(nx.DiGraph):
    @classmethod
    def from_newick(cls, newick):
        newick = StringIO(newick)
        phy = Phylo.read(newick, "newick")
        phy.rooted = True
        edges = []
        node_data = {}
        clades = [phy.root]
        phy.root.name = "root"
        i = 0
        while clades:
            clade = clades.pop()
            for c_clade in clade.clades:
                clades += clade.clades
                if c_clade.name is None:
                    c_clade.name = "node%d" % i
                    i += 1
                nd = {}
                ed = {'branch_length': c_clade.branch_length}
                if hasattr(c_clade, 'comment'):
                    nd['comment'] = c_clade.comment
                    attrd = _extract_momi_fields(c_clade.comment)
                    nd.update(attrd)
                    if 'lineages' in nd:
                        nd['lineages'] = int(nd['lineages'])
                edges.append((clade.name, (c_clade.name), ed))
                node_data[c_clade.name] = nd
        t = nx.DiGraph(data=edges)
        tn = dict(t.nodes(data=True))
        for node in node_data:
            tn[node].update(node_data[node])
        return cls(t)

    def __init__(self, *args, **kwargs):
        super(Demography, self).__init__(*args, **kwargs)
        nd = self.node_data
        if not all('lineages' in nd[k] for k in self.leaves):
            raise "'lineages' attribute must be set for each leaf node."

    @property
    def root(self):
        nds = [node for node, deg in self.in_degree().items() if deg == 0]
        assert len(nds) == 1
        return nds[0]
    
    @property
    def node_data(self):
        return dict(self.nodes(data=True))

    @property
    def leaves(self):
        return set([k for k, v in self.in_degree().items() if v == 1])

    @property
    def n_lineages_subtended_by(self):
        nd = self.node_data
        return {v: sum(nd[l]['lineages'] for l in self.leaves_subtended_by[v]) for v in self}

    @property
    def n_leaf_derived(self):
        nd = self.node_data
        return {v: sum(nd[l]['n_derived'] for l in self.leaves_subtended_by[v]) for v in self}

    @property
    def max_leaf_lineages(self):
        nd = self.node_data
        return max([nd[l]['n_derived'] + nd[l]['n_ancestral'] for v in self.leaves])

    @property
    def leaves_subtended_by(self):
        return {v: self.leaves & set(nx.dfs_preorder_nodes(self, v)) for v in self}

    def is_leaf(self, node):
        return node in self.leaves
