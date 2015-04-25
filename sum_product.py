import autograd.numpy as np
from autograd.numpy import sum
import scipy.misc
import bidict as bd
from util import memoize_instance, memoize, fftconvolve, my_trace, swapaxes, my_einsum

class LabeledAxisArray(object):
    def __init__(self, array, axisLabels, copyArray=True):
        self.array = array
        if copyArray:
            self.array = self.array + 0
        self.axes = bd.bidict({x : i for i,x in enumerate(axisLabels)})
        assert len(self.axes) == len(axisLabels) # assert no repeats
        assert len(self.axes) == len(self.array.shape)

    def tensor_multiply(self, other, axis):
        new_array = np.tensordot(self.array, other.array, [[self.axes[axis]], [other.axes[axis]]])
        new_axes = []
        for old in self, other:
            new_axes += [old.axes[:i] for i in range(len(old.axes)) if old.axes[:i] != axis]
        return LabeledAxisArray(new_array, new_axes, copyArray=False)

    def sum_axes(self, old_axes, new_label):
        a0,a1 = old_axes
        self.swap_axis(a0, 0)
        self.swap_axis(a1, 1)

        # sum the antidiagonals of the first two axes
        new_array = self.array[::-1,...]
        new_array = np.array([my_trace(new_array,offset=k) 
                              for k in range(-new_array.shape[0]+1,new_array.shape[1])])

        new_axes = bd.bidict()
        new_axes[new_label] = 0
        for i in range(2,len(self.axes)):
            new_axes[self.axes[:i]] = i-1
        self.array = new_array
        self.axes = new_axes
    
    ## TODO: get rid of this function
    def swap_axis(self, axis, new_pos):
        swapped_axis = self.axes[:new_pos]
        old_pos = self.axes[axis:]
        self.axes.forceput(axis, new_pos)
        self.axes.forceput(swapped_axis, old_pos)
        self.array = swapaxes(self.array, old_pos, new_pos)

    # returns array[0,...,0,:,0,...,0] with : at specified axis
    def get_zeroth_vector(self, axisLabel):
        idx = [0] * len(self.axes)
        axis = self.axes[axisLabel]
        idx[axis] = slice(self.array.shape[axis])
        return self.array[idx]

    def multiply_along_axis(self, axisLabel, vec):
        inds = range(len(self.axes))
        self.array = my_einsum(self.array, inds, vec, [self.axes[axisLabel]], inds)

    def divide_along_axis(self, axisLabel, vec):
        self.multiply_along_axis(axisLabel, 1.0/vec)

    def apply_transition(self, axisLabel, transition):
        self.array = transition(self.array, axis=self.axes[axisLabel])

    def relabel_axis(self, old_label, new_label):
        self.axes[new_label] = self.axes[old_label]
        # del self.axes[old_label]

    def expand_labels(self, new_labels):
        added_labels = [x for x in new_labels if x not in self.axes]
        # new_labels should contain all the old_labels
        assert len(new_labels) == len(added_labels) + len(self.axes)
        self.array = self.array[[slice(None)] * len(self.array.shape) + [np.newaxis] * len(added_labels)]
        #self.array.resize(list(self.array.shape) + [1] * len(added_labels))
        for l in added_labels:
            self.axes[l] = len(self.axes)
        self.reorder_axes(new_labels)

    def reorder_axes(self, new_order):
        assert len(new_order) == len(self.axes)
        label_permutation = [self.axes[l] for l in new_order]
        self.array = np.transpose(self.array, label_permutation)
        self.axes = bd.bidict({x : i for i,x in enumerate(new_order)})

class SumProduct(object):
    ''' 
    compute joint SFS entry via a sum-product like algorithm,
    using Moran model to do transitions up the tree,
    and using Polanski Kimmel to compute frequency of mutations at each node in tree
    '''
    def __init__(self, demography):
        self.G = demography
        
    def p(self, normalized = False):
        '''Return joint SFS entry for the demography'''
        ret = self.joint_sfs(self.G.event_root)
        if normalized:
            ret = ret / self.G.totalSfsSum
        return ret

    @memoize_instance
    def leaf_likelihood_bottom(self, leaf):
        n_node = self.G.node_data[leaf]['lineages']
        ret = np.zeros(n_node + 1)
        ret[self.G.node_data[leaf]['derived']] = 1.0
        return ret

    def combinatorial_factors(self, node):
        n_node = self.G.n_lineages_at_node[node]
        return scipy.misc.comb(n_node, np.arange(n_node + 1))

    ## TODO: make this a property of the demography, instead of the sum_product
    @memoize_instance
    def truncated_sfs(self, node):
        n_node = self.G.n_lineages_at_node[node]

        sfs = [self.G.node_data[node]['model'].freq(n_derived, n_node) for n_derived in range(n_node + 1)]
        if self.G.node_data[node]['model'].tau == float('inf'):
            assert node == self.G.root
            sfs[-1] = 0.0
        #sfs[sfs == float("inf")] = 0.0
        return np.array(sfs)

    @memoize_instance
    def partial_likelihood_top(self, event, popList):
        ''' Partial likelihood of data at top of node, i.e.
        i.e. = P(n_top) P(x | n_derived_top, n_ancestral_top)
        note n_top is fixed in Moran model, so P(n_top)=1
        '''       
        ret = LabeledAxisArray(self.partial_likelihood_bottom(event), self.G.sub_pops(event))
        for pop in popList:
            ret.apply_transition(pop, self.G.node_data[pop]['model'].transition_prob)
        return ret.array

    @memoize_instance
    def partial_likelihood_bottom(self, event):
        '''Partial likelihood of data under Moran model, given alleles at bottom of node
        i.e. = P(n_bottom) P(x | n_derived_bottom, n_ancestral_bottom)
        note n_bottom is fixed in Moran model, so P(n_bottom)=1
        '''
        if self.G.event_type(event) == 'leaf':
            leafpop, = self.G.parent_pops(event)
            return self.leaf_likelihood_bottom(leafpop)
        elif self.G.event_type(event) == 'admixture':
            childpop, = self.G.child_pops(event).keys()
            p1,p2 = self.G.parent_pops(event)

            childEvent, = self.G.eventTree[event]
            childTopLik = self.partial_likelihood_top(childEvent, frozenset([childpop]))
            
            ret = LabeledAxisArray(childTopLik, self.G.sub_pops(childEvent))
            ret = ret.tensor_multiply(self.G.admixture_prob(childpop), childpop)
            ret.reorder_axes(self.G.sub_pops(event)) # make sure axes are in correct order
            return ret.array
        elif self.G.event_type(event) == 'merge_subpops':
            newpop, = self.G.parent_pops(event)
            childPops = self.G[newpop]
            childEvent, = self.G.eventTree[event]
            childTopLik = self.partial_likelihood_top(childEvent, frozenset(childPops))
            
            c1,c2 = childPops
            ret = LabeledAxisArray(childTopLik, self.G.sub_pops(childEvent))
            for c in c1,c2:
                ret.multiply_along_axis(c, self.combinatorial_factors(c))
            ret.sum_axes((c1,c2), newpop)
            ret.divide_along_axis(newpop, self.combinatorial_factors(newpop))
            # make sure axis labels are correctly ordered
            ret.reorder_axes(self.G.sub_pops(event))
            return ret.array
        elif self.G.event_type(event) == 'merge_clusters':
            newpop, = self.G.parent_pops(event)
            liks = []
            for childPop, childEvent in self.G.child_pops(event).iteritems():
                childTopLik = self.partial_likelihood_top(childEvent, frozenset([childPop]))
                childTopLik = LabeledAxisArray(childTopLik, self.G.sub_pops(childEvent))
                childTopLik.multiply_along_axis(childPop, self.combinatorial_factors(childPop))
                # make childTopLik have same axisLabels as the array toReturn
                childTopLik.relabel_axis(childPop, newpop)
                childTopLik.expand_labels(self.G.sub_pops(event))
                liks.append(childTopLik.array)
            ret = LabeledAxisArray(fftconvolve(*liks), self.G.sub_pops(event), copyArray=False)
            ret.divide_along_axis(newpop, self.combinatorial_factors(newpop))
            return ret.array
        else:
            raise Exception("Event type %s not yet implemented" % self.G.event_type(event))
       
    @memoize_instance
    def joint_sfs(self, event):
        '''The joint SFS entry for the configuration under this node'''
        # if no derived leafs, return 0
        if all(self.G.n_derived_subtended_by[subpop] == 0 for subpop in self.G.sub_pops(event)):
            return 0.0
        
        ret = 0.0
        for newpop in self.G.parent_pops(event):
            # term for mutation occurring at the newpop
            labeledArray = LabeledAxisArray(self.partial_likelihood_bottom(event), self.G.sub_pops(event), copyArray=False)
            ret += (labeledArray.get_zeroth_vector(newpop) * self.truncated_sfs(newpop)).sum()

        #if self.G.is_leaf(node):
        if self.G.event_type(event) == 'leaf':
            return ret
        # add on terms for mutation occurring below this node
        # if no derived leafs on right, add on term from the left
        elif self.G.event_type(event) == 'merge_clusters':
            c1, c2 = self.G.eventTree[event]
            for child, other_child in ((c1, c2), (c2, c1)):
                #if self.G.n_derived_subtended_by[child] == 0:
                if all(self.G.n_derived_subtended_by[subpop] == 0 for subpop in self.G.sub_pops(child)):
                    ret += self.joint_sfs(other_child)
            return ret
        elif self.G.event_type(event) == 'merge_subpops' or self.G.event_type(event) == 'admixture':
            childEvent, = self.G.eventTree[event]
            ret += self.joint_sfs(childEvent)
            return ret
        else:
            raise Exception("Event type %s not yet implemented" % self.G.event_type(event))
