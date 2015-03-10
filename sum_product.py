from __future__ import print_function, division
import scipy.misc, scipy.signal
import math
import numpy as np
import bidict as bd

from util import memoize_instance, memoize

class LabeledAxisArray(object):
    def __init__(self, array, axisLabels, copyArray=True):
        self.array = array
        if copyArray:
            self.array = self.array.copy()
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
        #a0,a1 = (self.axes[l] for l in old_axes)
        a0,a1 = old_axes
        self.swap_axis(a0, 0)
        self.swap_axis(a1, 1)

        new_array = np.zeros([self.array.shape[0] + self.array.shape[1] - 1] + list(self.array.shape[2:]))
        for i in range(self.array.shape[0]):
            for j in range(self.array.shape[1]):
                new_array[i+j,...] += self.array[i,j,...]

        new_axes = bd.bidict()
        new_axes[new_label] = 0
        for i in range(2,len(self.axes)):
            new_axes[self.axes[:i]] = i-1
        self.array = new_array
        self.axes = new_axes
    
    def swap_axis(self, axis, new_pos):
        swapped_axis = self.axes[:new_pos]
        old_pos = self.axes[axis:]
        self.axes.forceput(axis, new_pos)
        self.axes.forceput(swapped_axis, old_pos)
        self.array = self.array.swapaxes(old_pos, new_pos)

    # returns array[0,...,0,:,0,...,0] with : at specified axis
    def get_zeroth_vector(self, axisLabel):
        idx = [0] * len(self.axes)
        axis = self.axes[axisLabel]
        idx[axis] = slice(self.array.shape[axis])
        return self.array[idx]

    def apply_along_axis(self, axisLabel, func):
        # this might be slow (like a for loop)
        self.array = np.apply_along_axis(func, self.axes[axisLabel], self.array)

    def multiply_along_axis(self, axisLabel, vec):
        tmpView = np.swapaxes(self.array, self.axes[axisLabel], len(self.array.shape)-1)
        tmpView *= vec
        #np.swapaxes(self.array, self.axes[axisLabel], len(self.array.shape)-1)

    def divide_along_axis(self, axisLabel, vec):
        tmpView = np.swapaxes(self.array, self.axes[axisLabel], len(self.array.shape)-1)
        tmpView /= vec
        #np.swapaxes(self.array, self.axes[axisLabel], len(self.array.shape)-1)        

    def relabel_axis(self, old_label, new_label):
        self.axes[new_label] = self.axes[old_label]
        # del self.axes[old_label]

    def expand_labels(self, new_labels):
        added_labels = [x for x in new_labels if x not in self.axes]
        # new_labels should contain all the old_labels
        assert len(new_labels) == len(added_labels) + len(self.axes)
        self.array.resize(list(self.array.shape) + [1] * len(added_labels))
        for l in added_labels:
            self.axes[l] = len(self.axes)
        self.reorder_axes(new_labels)

    def reorder_axes(self, new_order):
        assert len(new_order) == len(self.axes)
        label_permutation = [self.axes[l] for l in new_order]
        self.array = self.array.transpose(label_permutation)
        self.axes = {x : i for i,x in enumerate(new_order)}

class SumProduct(object):
    ''' 
    compute joint SFS entry via a sum-product like algorithm,
    using Moran model to do transitions up the tree,
    and using Polanski Kimmel to compute frequency of mutations at each node in tree
    '''
    def __init__(self, demography):
        self.G = demography
        # TODO: make eventTree contain the demography rather than vice versa
        self.eventTree = self.G.eventTree
        # assert self.n_derived_leafs(tree) > 0 and self.n_derived_leafs(tree) < self.n_leaf_lins(tree)
        
    def p(self, normalized = False):
        '''Return joint SFS entry for the demography'''
        ret = self.joint_sfs(self.eventTree.root)
        if normalized:
            ret /= self.G.totalSfsSum
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

    @memoize_instance
    def truncated_sfs(self, node):
        n_node = self.G.n_lineages_at_node[node]
        sfs = np.array([self.G.node_data[node]['model'].freq(n_derived, n_node) for n_derived in range(n_node + 1)])
        sfs[sfs == float("inf")] = 0.0
        return sfs

    @memoize_instance
    def partial_likelihood_top(self, event, popList):
        ''' Partial likelihood of data at top of node, i.e.
        i.e. = P(n_top) P(x | n_derived_top, n_ancestral_top)
        note n_top is fixed in Moran model, so P(n_top)=1
        '''       
        #return self.G.node_data[pop]['model'].transition_prob(bottom_likelihood)
        ret = LabeledAxisArray(self.partial_likelihood_bottom(event), event['subpops'])
        for pop in popList:
            ret.apply_along_axis(pop, lambda x: self.G.node_data[pop]['model'].transition_prob(x))
        return ret.array

    @memoize_instance
    def partial_likelihood_bottom(self, event):
        '''Partial likelihood of data under Moran model, given alleles at bottom of node
        i.e. = P(n_bottom) P(x | n_derived_bottom, n_ancestral_bottom)
        note n_bottom is fixed in Moran model, so P(n_bottom)=1
        '''
        #if self.G.is_leaf(node):
        if event['type'] == 'leaf':
            leafpop, = event['newpops']
            return self.leaf_likelihood_bottom(leafpop)
        elif event['type'] == 'admixture':
            childpop = event['childpop']
            p1,p2 = event['newpops']

            childEvent, = self.eventTree[event]
            childTopLik = self.partial_likelihood_top(childEvent, frozenset([childpop]))
            
            ret = LabeledAxisArray(childTopLik, childEvent['subpops'])
            ret = ret.tensor_multiply(self.G.admixture_prob(childpop), childpop)
            ret.reorder_axes(event['subpops']) # make sure axes are in correct order
            return ret.array
        elif event['type'] == 'merge_subpops':
            newpop, = event['newpops']
            childPops = self.G[newpop]
            childEvent, = self.eventTree[event]
            childTopLik = self.partial_likelihood_top(childEvent, frozenset(childPops))
            
            c1,c2 = childPops
            ret = LabeledAxisArray(childTopLik, childEvent['subpops'])
            for c in c1,c2:
                ret.multiply_along_axis(c, self.combinatorial_factors(c))
            ret.sum_axes((c1,c2), newpop)
            ret.divide_along_axis(newpop, self.combinatorial_factors(newpop))
            # make sure axis labels are correctly ordered
            ret.reorder_axes(event['subpops'])
            return ret.array
        elif event['type'] == 'dummy_merge_clusters':
            c1,c2 = self.eventTree[event]
            array = np.multiply.outer(*[self.partial_likelihood_bottom(c) for c in c1,c2])
            labels = list(c1['subpops']) + list(c2['subpops'])
            ret = LabeledAxisArray(array, labels, copyArray=False)
            # make sure array dimensions has correct ordering
            ret.reorder_axes(event['subpops'])
            return ret.array
        elif event['type'] == 'merge_clusters':
            newpop, = event['newpops']
            liks = []
            for parEvent, childEvent, edgeInfo in self.eventTree.edges([event], data=True):
                assert parEvent is event
                childPop = edgeInfo['childPop']
                childTopLik = self.partial_likelihood_top(childEvent, frozenset([childPop]))
                childTopLik = LabeledAxisArray(childTopLik, childEvent['subpops'])
                childTopLik.multiply_along_axis(childPop, self.combinatorial_factors(childPop))
                # make childTopLik have same axisLabels as the array toReturn
                childTopLik.relabel_axis(childPop, newpop)
                childTopLik.expand_labels(event['subpops'])
                liks.append(childTopLik.array)
            ret = LabeledAxisArray(scipy.signal.fftconvolve(*liks), event['subpops'], copyArray=False)
            ret.divide_along_axis(newpop, self.combinatorial_factors(newpop))
            return ret.array
        else:
            raise Exception("Event type %s not yet implemented" % event['type'])
       
    @memoize_instance
    def joint_sfs(self, event):
        '''The joint SFS entry for the configuration under this node'''
        # if no derived leafs, return 0
        if all(self.G.n_derived_subtended_by[subpop] == 0 for subpop in event['subpops']):
            return 0.0
        
        ret = 0.0
        for newpop in event['newpops']:
            # term for mutation occurring at the newpop
            labeledArray = LabeledAxisArray(self.partial_likelihood_bottom(event), event['subpops'], copyArray=False)
            ret += (labeledArray.get_zeroth_vector(newpop) * self.truncated_sfs(newpop)).sum()

        #if self.G.is_leaf(node):
        if event['type'] == 'leaf':
            return ret
        # add on terms for mutation occurring below this node
        # if no derived leafs on right, add on term from the left
        elif event['type'] in ['merge_clusters', 'dummy_merge_clusters']:
            c1, c2 = self.eventTree[event]
            for child, other_child in ((c1, c2), (c2, c1)):
                #if self.G.n_derived_subtended_by[child] == 0:
                if all(self.G.n_derived_subtended_by[subpop] == 0 for subpop in child['subpops']):
                    ret += self.joint_sfs(other_child)
            return ret
        elif event['type'] == 'merge_subpops' or event['type'] == 'admixture':
            childEvent, = self.eventTree[event]
            ret += self.joint_sfs(childEvent)
            return ret
        else:
            raise Exception("Event type %s not yet implemented" % event['type'])
