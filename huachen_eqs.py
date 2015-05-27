from __future__ import division
import numpy
import scipy.misc
import operator
import math
from util import memoize_instance
import warnings

math_mod = math
myint,myfloat = int,float

## UNCOMMENT FOR HIGHER PRECISION
# import gmpy2
# math_mod = gmpy2
# gmpy2.get_context().precision=100
# myint,myfloat = gmpy2.mpz, gmpy2.mpfr


'''
Formulas from Hua Chen 2012, Theoretical Population Biology
Note that for all formulas from that paper, N = diploid population size
'''

class SumProduct_Chen(object):
    ''' 
    compute sfs of data via Hua Chen's sum-product algorithm
    '''
    def __init__(self, demography):
        self.G = demography
        attach_Chen(self.G)
    
    def p(self):
        '''Return the likelihood for the data'''
        return self.joint_sfs(self.G.root)

    @memoize_instance
    def partial_likelihood_top(self, node, n_ancestral_top, n_derived_top):
        n_top = n_derived_top + n_ancestral_top
        n_leaves = self.G.n_lineages_subtended_by[node]
        ret = 0.0

        for n_bottom in range(n_top,n_leaves+1):
            for n_derived_bottom in range(n_derived_top, self.G.n_derived_subtended_by[node]+1):
                n_ancestral_bottom = n_bottom - n_derived_bottom

                if n_derived_bottom > 0 and n_derived_top == 0:
                    continue

                p_bottom = self.partial_likelihood_bottom(node, n_ancestral_bottom, n_derived_bottom)
                if p_bottom == 0.0:
                    continue
                p_top = p_bottom * self.G.chen[node].g(n_bottom,n_top)

                p_top *= math.exp(log_urn_prob(
                        n_derived_top,
                        n_ancestral_top,
                        n_derived_bottom, 
                        n_ancestral_bottom))
                ret += p_top
        return ret

    @memoize_instance
    def partial_likelihood_bottom(self, node, n_ancestral, n_derived):
        '''Likelihood of data given alleles (state) at bottom of node.'''
        # Leaf nodes are "clamped"
        if self.G.is_leaf(node):
            if n_ancestral + n_derived == self.G.n_lineages_subtended_by[node] and n_derived == self.G.n_derived_subtended_by[node]:
                return 1.0
            else:
                return 0.0
        # Sum over allocation of lineages to left and right branch
        # Left branch gets between 1 and (total - 1) lineages
        ret = 0.0
        total_lineages = n_ancestral + n_derived

        left_node,right_node = self.G[node]

        n_leaves_l = self.G.n_lineages_subtended_by[left_node]
        n_leaves_r = self.G.n_lineages_subtended_by[right_node]
        for n_ancestral_l in range(n_ancestral + 1):
            n_ancestral_r = n_ancestral - n_ancestral_l
            # Sum over allocation of derived alleles to left, right branches
            for n_derived_l in range(n_derived + 1):
                n_derived_r = n_derived - n_derived_l
                n_left = n_ancestral_l + n_derived_l
                n_right = n_ancestral_r + n_derived_r
                if any([n_right == 0, n_right > n_leaves_r, n_left==0, n_left > n_leaves_l]):
                    continue
                p = math.exp(log_binom(n_ancestral, n_ancestral_l) + 
                         log_binom(n_derived, n_derived_l) - 
                         log_binom(total_lineages, n_ancestral_l + n_derived_l))
                assert p != 0.0
                for args in ((right_node, n_ancestral_r, n_derived_r), (left_node, n_ancestral_l, n_derived_l)):
                    p *= self.partial_likelihood_top(*args)
                ret += p
        return ret

    @memoize_instance
    def joint_sfs(self, node):
        n_leaves = self.G.n_lineages_subtended_by[node]
        ret = 0.0
        for n_derived in range(1, self.G.n_derived_subtended_by[node]+1):
            for n_bottom in range(n_derived, n_leaves+1):
                n_ancestral = n_bottom - n_derived
                p_bottom = self.partial_likelihood_bottom(node, n_ancestral, n_derived)
                ret += p_bottom * self.G.chen[node].truncated_sfs(n_derived, n_bottom)

        if self.G.is_leaf(node):
            return ret

        # add on terms for mutation occurring below this node
        # if no derived leafs on right, add on term from the left
        c1, c2 = self.G[node]
        for child, other_child in ((c1, c2), (c2, c1)):
            if self.G.n_derived_subtended_by[child] == 0:
                ret += self.joint_sfs(other_child)
        return ret


def attach_Chen(tree):
    '''Attach Hua Chen equations to each node of tree.
    Does nothing if these formulas have already been added.'''
    if not hasattr(tree, "chen"):
        tree.chen = {}
        for node in tree:
            size_model = tree.node_data[node]['model']
            tree.chen[node] = SFS_Chen(size_model.N / 2.0, size_model.tau, tree.n_lineages_subtended_by[node])

class SFS_Chen(object):
    def __init__(self, N_diploid, timeLen, max_n):
        self.timeLen = timeLen
        self.N_diploid = N_diploid
        # precompute
        for n in range(1,max_n+1):
            for i in range(1,n+1):
                self.truncated_sfs(i,n)
                
            max_m = n
            if timeLen == float('inf'):
                max_m = 1
            for m in range(1,max_m+1):
                self.g(n,m)
            
    @memoize_instance    
    def g(self, n, m):
        return g(n, m, self.N_diploid, self.timeLen)


    @memoize_instance
    def ET(self, i, n, m):
        try:
            return ET(i, n, m, self.N_diploid, self.timeLen)
        except ZeroDivisionError:
            warnings.warn("divide by zero in hua chen formula")
            return 0.0

    @memoize_instance    
    def ES_i(self, i, n, m):
        '''TPB equation 4'''
        assert n >= m
        return math.fsum([p_n_k(i, n, k) * k * self.ET(k, n, m) for k in range(m, n + 1)])

    @memoize_instance
    def truncated_sfs(self, i, n):
        max_m = n-i+1
        if self.timeLen == float('inf'):
            max_m = 1
            
        ret = 0.0
        for m in range(1,max_m+1):
            ret += self.ES_i(i, n, m)
        return ret
        

def log_factorial(n):
    return math_mod.lgamma(n+1)

def log_rising(n,k):
    return log_factorial(n+k-1) - log_factorial(n-1)

def log_falling(n,k):
    return log_factorial(n) - log_factorial(n-k)

def gcoef(k, n, m, N_diploid, tau):
    k, n, m = map(myint, [k, n, m])
    N_diploid = myfloat(N_diploid)
    tau = myfloat(tau)
    return (2*k - 1) * (-1)**(k - m) * math_mod.exp(log_rising(m, k-1) + log_falling(n, k) - log_factorial(m) - log_factorial(k - m) - log_rising(n, k))
    #return (2*k - 1) * (-1)**(k - m) * rising(m, k-1) * falling(n, k) / math_mod.factorial(m) / math_mod.factorial(k - m) / rising(n, k) 


def g_sum(n, m, N_diploid, tau):
    if tau == float("inf"):
        if m == 1:
            return 1.0
        return 0.0
    tau = myfloat(tau)
    return float(sum([gcoef(k, n, m, N_diploid, tau) * math_mod.exp(-k * (k - 1) * tau / 4 / N_diploid) for k in range(m, n + 1)]))


g = g_sum

def formula1(n, m, N_diploid, tau):
    def expC2(k):
        return math_mod.exp(-k * (k - 1) / 4 / N_diploid * tau)
    r = sum(gcoef(k, n, m, N_diploid, tau) * 
            ((expC2(m) - expC2(k)) / (k - m) / (k + m - 1) - (tau / 4 / N_diploid * expC2(m)))
            for k in range(m + 1, n + 1))
    #q = 4 * N_diploid / g(n, m, N_diploid, tau)
    q = 4 * N_diploid
    return float(r * q)


def formula3(j, n, m, N_diploid, tau):
    # Switch argument to j here to stay consistent with the paper.
    j, n, m = map(myint, [j, n, m])
    tau, N_diploid = map(myfloat, [tau, N_diploid])
    def expC2(kk):
        return math_mod.exp(-kk * (kk - 1) / 4 / N_diploid * tau)
    r = sum(gcoef(k, n, j, N_diploid, tau) * # was gcoef(k, n, j + 1, N_diploid, tau) * 
            sum(gcoef(ell, j, m, N_diploid, tau) * ( # was gcoef(ell, j - 1, m, N_diploid, tau) * (
                    (
                        expC2(j) * (tau / 4 / N_diploid - ((k - j) * (k + j - 1) + (ell - j)*(ell + j - 1)) / # tau / 4 / N_diploid was 1 in this
                             (k - j) / (k + j- 1) / (ell - j) / (ell + j - 1))
                    )
                    +
                    (
                        expC2(k) * (ell - j) * (ell + j - 1) / (k - j) / (k + j - 1) / (ell - k) / (ell + k - 1)
                    )
                    -
                    (
                        expC2(ell) * (k - j) * (k + j - 1) / (ell - k) / (ell + k - 1) / (ell - j) / (ell + j - 1)
                    )
                )
                for ell in range(m, j)
                )
            for k in range(j + 1, n + 1)
            )
    #q = 4 * N_diploid / myfloat(g(n, m, N_diploid, tau))
    q = 4 * N_diploid
    return float(q * r)

def formula2(n, m, N_diploid, tau):
    def expC2(k):
        return math_mod.exp(-k * (k - 1) / 4 / N_diploid * tau)
    r = sum(gcoef(k, n, m, N_diploid, tau) * 
            ((expC2(k) - expC2(n)) / (n - k) / (n + k - 1) - (tau / 4 / N_diploid * expC2(n)))
            for k in range(m, n))
    #q = 4 * N_diploid / g(n, m, N_diploid, tau)
    q = 4 * N_diploid
    return float(r * q)

def ET(i, n, m, N_diploid, tau):
    '''Starting with n lineages in a population of size N_diploid,
    expected time when there are i lineages conditional on there
    being m lineages at time tau in the past.'''
    if tau == float("inf"):
        if m != 1 or i == 1:
            return 0.0
        return 2 * N_diploid / float(nChoose2(i)) * g(n, m, N_diploid, tau)
    if n == m:
        return tau * (i == n) * g(n, m, N_diploid, tau)
    if m == i:
        return formula1(n, m, N_diploid, tau)
    elif n == i:
        return formula2(n, m, N_diploid, tau)
    else:
        return formula3(i, n, m, N_diploid, tau)

def p_n_k(i, n, k):
    if k == 1:
        return int(i == n)
    else:
        #return scipy.misc.comb(n-i-1,k-2) / scipy.misc.comb(n-1,k-1)
        return math.exp(log_binom(n - i - 1, k - 2) - log_binom(n - 1, k - 1))

def nChoose2(n):
    return (n * (n-1)) / 2

def log_binom(n, k):
    if k < 0 or k > n:
        return -float('inf')
    return log_factorial(n) - log_factorial(n - k) - log_factorial(k)

def log_urn_prob(n_parent_derived, n_parent_ancestral, n_child_derived, n_child_ancestral):
    n_parent = n_parent_derived + n_parent_ancestral
    n_child = n_child_derived + n_child_ancestral
    if n_child_derived >= n_parent_derived and n_parent_derived > 0 and n_child_ancestral >= n_parent_ancestral and n_parent_ancestral > 0:
        return log_binom(n_child_derived - 1, n_parent_derived - 1) + log_binom(n_child_ancestral - 1, n_parent_ancestral - 1) - log_binom(n_child-1, n_parent-1)
    elif n_child_derived == n_parent_derived == 0 or n_child_ancestral == n_parent_ancestral == 0:
        return 0.0
    else:
        return float("-inf")
