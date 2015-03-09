# TODO: move this into file sum_product.py
class NormalizingConstant(SumProduct):
    def __init__(self, demography):
        super(NormalizingConstant,self).__init__(demography)

    def leaf_likelihood_bottom(self, leaf):
        n_node = self.G.node_data[leaf]['lineages']
        ret = np.array([0.0] * (n_node + 1))
        ret[0] = 1.0
        return ret        

    def normalizing_constant(self, event=None):
        if event is None:
            event = self.eventTree.root

        ret = 0.0
        for newpop in event['newpops']:
            # term for mutation occurring at the newpop
            # partial_likelihood_bottom is the likelihood of _no_ derived alleles beneath event, given value of derived alleles
            labeledArray = LabeledAxisArray(self.partial_likelihood_bottom(event), event['subpops'], copyArray=False)
            # do 1.0 - partial_likelihood_bottom to get the likelihood of _some_ derived alleles beneath event
            ret += ((1.0 - labeledArray.get_zeroth_vector(newpop)) * self.truncated_sfs(newpop)).sum()        

        for childEvent in self.eventTree[event]:
            ret += self.normalizing_constant(childEvent)
        
        return ret
