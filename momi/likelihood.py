
from .util import count_calls, force_primitive
from .optimizers import _find_minimum, custom_opts, stochastic_opts
import autograd.numpy as np
from .compute_sfs import expected_sfs, expected_total_branch_len
from .demography import DemographyError, Demography
from .data_structure import _sub_sfs
import scipy
import autograd, numdifftools
from autograd import grad, hessian_vector_product, hessian
from collections import Counter
import random, functools, logging, time

logger = logging.getLogger(__name__)

class SfsLikelihoodSurface(object):
    def __init__(self, data, demo_func=None, mut_rate=None, log_prior=None, folded=False, error_matrices=None, truncate_probs=1e-100, batch_size=200):
        """
        Object for computing composite likelihoods, and searching for the maximum composite likelihood.

        Parameters
        ==========
        data: momi.SegSites or momi.Sfs object
        demo_func: function or None
            function that creates momi.Demography object from a vector of input parameters
            if None, this is the identity function (so the "input parameter" is the Demography object itself)
        mut_rate: float or list or None
            The per-locus mutation rate.
            If an array, the length should be the total number of loci.
            If None, use a multinomial model instead of Poisson model (so the numger of segregating sites is fixed)
        folded, error_matrices:
            see help(momi.expected_sfs)
        log_prior:
            function that adds a log-prior to the log-likelihood
        truncate_probs:
            if truncate_probs=eps, then eps is added to the SFS entries before computing the log-likelihood
            this can be helpful for dealing with very small probabilities, within computer accuracy of 0.
        batch_size:
            controls the memory usage. the SFS will be computed in batches of batch_size.
            Decrease batch_size to decrease memory usage (but add running time overhead)
        """
        self.data = data
        
        try: self.sfs = self.data.sfs
        except AttributeError: self.sfs = self.data
        
        self.demo_func = demo_func

        self.mut_rate = mut_rate
        self.folded = folded
        if self.folded:
            self.sfs = self.sfs.fold() # required for entropy term to be correct
        self.error_matrices = error_matrices

        self.truncate_probs = truncate_probs

        self.batch_size = batch_size
        self.sfs_batches = _build_sfs_batches(self.sfs, batch_size)
        #self.sfs_batches = [self.sfs]

        self.log_prior = log_prior

    def log_lik(self, x):
        """
        Returns the composite log-likelihood of the data at the point x.
        """
        if self.demo_func:
            logger.debug("Computing log-likelihood at x = {0}".format(x))            
            demo = self.demo_func(*x)
        else:
            demo = x
           
        G,(diff_keys,diff_vals) = demo._get_graph_structure(), demo._get_differentiable_part()
        ret = 0.0
        for batch in self.sfs_batches:
            ret = ret + _raw_log_lik(diff_vals, diff_keys, G, batch, self.truncate_probs, self.folded, self.error_matrices)

        if self.mut_rate:
            ret = ret + _mut_factor(self.sfs, demo, self.mut_rate, False)

        if self.log_prior:
            ret = ret + self.log_prior(x)
        logger.debug("log-likelihood = {0}".format(ret))
        return ret

    def kl_div(self, x):
        """
        Returns KL-Divergence(Empirical || Theoretical(x)).
        """        
        log_lik = self.log_lik(x)
        ret = -log_lik + self.sfs.n_snps() * self.sfs._entropy + _entropy_mut_term(self.mut_rate, self.sfs.n_snps(vector=True))

        ret = ret / float(self.sfs.n_snps())
        if not self.log_prior:
            assert ret >= 0, "kl-div: %s, log_lik: %s, total_count: %s" % (str(ret), str(log_lik), str(self.sfs.n_snps()))

        return ret
    
    def find_mle(self, x0, method="tnc", jac=True, hess=False, hessp=False, bounds=None, pieces=None, rgen=np.random, out=None, **kwargs):
        """
        Search for the maximum of the likelihood surface
        (i.e., the minimum of the KL-divergence).
       
        Parameters
        ==========
        x0 : numpy.ndarray
             initial guess
        out : file stream
                   write intermediate progress to file or stream (e.g. sys.stdout)
        method : str
                 Can be any method from scipy.optimize.minimize()
                 (e.g. "tnc","L-BFGS-B",etc.), or another optimization
                 method implemented by momi (currently, "nesterov" or
                 "svrg").

                 If a method of scipy.optimize.minimize(), then this function
                 acts as a wrapper around scipy.optimize.minimize().

                 If "nesterov", uses the deterministic method
                    momi.optimizers.nesterov().

                 If "svrg", use the stochastic gradient method
                    momi.optimizers.svrg()
                 (very experimental still).

        jac : bool
              If True, compute gradient automatically, and pass into the optimization method.
              If False, don't pass in gradient to the optimization method.
        hess, hessp: bool
              Pass hessian/hessian-vector-product into the optimization method.
              Only implemented for some scipy optimizers, and may have high memory cost.
        bounds : list of pairs [(lower0,higher0),...]
              As in scipy.optimize.minimize.
              If None, then do unbounded optimization.
              If one of (lower,higher) is None, then unbounded in that direction.
              
              If an element of the list is a number (instead of a pair),
              then the optimizer will fix the parameter at that value,
              and optimize over the other parameters.
        pieces : int or None
              For stochastic gradient descent methods. The number of pieces or
              "minibatches" to divide the data into.
        rgen : numpy.random.RandomState object
              Random number generator for stochastic gradient descent methods.

        **kwargs : additional arguments to pass to the optimizers
        """
        def printout(items):
            outstr = "\t".join(map(str, items))
            if out is None:
                logging.getLogger(__name__ + ".callback").info(outstr)
            else:
                out.write(outstr + "\n")
                out.flush()
        printout(["Seconds","KLDivergence"] + ["X%d" % i for i in range(len(x0))])
        starttime = time.time()
        def print_progress(x,fx,i):
            items = [time.time()-starttime, fx] + list(x)
            printout(items)
        
        if method.lower() in custom_opts or method.lower() in stochastic_opts:
            if jac is False: raise ValueError("Method %s requires gradient" % method)
            if hess or hessp: raise ValueError("Method %s doesn't use hessian" % method)

            method = method.lower()
            if method in stochastic_opts:
                return self._find_mle_stochastic(x0, pieces=pieces, bounds=bounds,
                                                    rgen=rgen, method=method, callback=print_progress,
                                                    **kwargs)
            elif method in custom_opts:
                return _find_minimum(self.kl_div, x0, custom_opts[method], bounds=bounds, callback=print_progress, opt_kwargs=kwargs, gradmakers={'fun_and_jac':autograd.value_and_grad})
            else: assert False
        else:
            return self._find_mle_scipy(x0, bounds, jac, hess, hessp, print_progress=print_progress, method=method, **kwargs)

    def _find_mle_stochastic(self, x0, pieces, bounds, rgen, method, callback, **kwargs):
        try: assert pieces > 0 and pieces == int(pieces)
        except (TypeError,AssertionError):
            raise ValueError("Stochastic Gradient Descent methods require pieces to be a positive integer")
        
        surface_pieces = self._get_stochastic_pieces(pieces, rgen)
        
        def fun(x,i):
            if i is None:
                return self.kl_div(x)
            else:
                if self.log_prior: lp = self.log_prior(x)
                else: lp = 0.0
                ret = -surface_pieces[i].log_lik(x) + surface_pieces[i].sfs.n_snps() * (self.sfs._entropy - lp/float(self.sfs.n_snps())) + _entropy_mut_term(surface_pieces[i].mut_rate, surface_pieces[i].sfs.n_snps(vector=True))
                return ret / float(self.sfs.n_snps())

        opt_kwargs = dict(kwargs)
        opt_kwargs.update({'pieces': pieces, 'rgen': rgen})

        logger.info("Dataset has %d total SNPs, of which %d are unique" % (self.sfs.n_snps(), len(self.sfs.configs)))
        avg_uniq = np.mean([len(s.sfs.configs) for s in surface_pieces])
        logger.info("Broke dataset up into %d pieces, with an average of %f total SNPs and %f unique SNPs" % (pieces,
                                                                                                              self.sfs.n_snps() / float(pieces),
                                                                                                              avg_uniq))
        
        if method == "svrg":
            iter_per_epoch = opt_kwargs.get('iter_per_epoch',None)
            if iter_per_epoch is None:
                iter_per_epoch = np.min((int(np.ceil(len(self.sfs.configs) / avg_uniq)), pieces))
                opt_kwargs['iter_per_epoch'] = iter_per_epoch
            logger.info("Running SVRG with %d iters per epoch" % iter_per_epoch)
        
        return _find_minimum(fun, x0, optimizer=stochastic_opts[method],
                             bounds=bounds, callback=callback, opt_kwargs=opt_kwargs,
                             gradmakers={'fun_and_jac':autograd.value_and_grad})

    def _get_stochastic_pieces(self, pieces, rgen):
        sfs_pieces = _subsfs_list(self.sfs, pieces, rgen)
        if self.mut_rate is None:
            mut_rate = None
        else:
            mut_rate = self.mut_rate * np.ones(self.sfs.n_loci)
            mut_rate = np.sum(mut_rate) / float(pieces)
            
        return [SfsLikelihoodSurface(sfs, demo_func=self.demo_func, mut_rate=mut_rate,
                                     folded = self.folded, error_matrices = self.error_matrices,
                                     truncate_probs = self.truncate_probs,
                                     batch_size = self.batch_size)
                for sfs in sfs_pieces]        
        
                
    def _find_mle_scipy(self, x0, bounds, jac, hess, hessp, print_progress, options={}, **kwargs):       
        hist = lambda:None
        hist.itr = 0
        hist.recent_vals = []
        @functools.wraps(self.kl_div)
        def fun(x):
            ret = self.kl_div(x)
            hist.recent_vals += [(x,ret)]
            return ret

        starttime = time.time()
        
        user_callback = kwargs.pop("callback", lambda x: None)
        def callback(x):
            user_callback(x)
            for y,fx in reversed(hist.recent_vals):
                if np.allclose(y,x): break
            assert np.allclose(y,x)
            try: fx=fx.value
            except AttributeError: pass
            print_progress(x,fx,hist.itr)
            hist.itr += 1
            hist.recent_vals = [(x,fx)]

        opt_kwargs = dict(kwargs)
        opt_kwargs['options'] = options

        opt_kwargs['jac'] = jac
        if jac: replacefun = autograd.value_and_grad
        else: replacefun = None
        
        gradmakers = {}        
        if hess: gradmakers['hess'] = autograd.hessian
        if hessp: gradmakers['hessp'] = autograd.hessian_vector_product
                  
        return _find_minimum(fun, x0, scipy.optimize.minimize,
                             bounds=bounds, callback=callback,
                             opt_kwargs=opt_kwargs, gradmakers=gradmakers, replacefun=replacefun)           
   
def _composite_log_likelihood(data, demo, mut_rate=None, truncate_probs = 0.0, vector=False, **kwargs):
    try:
        sfs = data.sfs
    except AttributeError:
        sfs = data

    sfs_probs = np.maximum(expected_sfs(demo, sfs.configs, normalized=True, **kwargs),
                           truncate_probs)
    log_lik = sfs._integrate_sfs(np.log(sfs_probs), vector=vector)

    # add on log likelihood of poisson distribution for total number of SNPs
    if mut_rate is not None:
        log_lik = log_lik + _mut_factor(sfs, demo, mut_rate, vector)
        
    if not vector:
        log_lik = np.squeeze(log_lik)
    return log_lik

def _mut_factor(sfs, demo, mut_rate, vector):
    mut_rate = mut_rate * np.ones(sfs.n_loci)

    if sfs.configs.has_missing_data:
        raise NotImplementedError("Poisson model not implemented for missing data.")
    E_total = expected_total_branch_len(demo)
    lambd = mut_rate * E_total
    
    ret = -lambd + sfs.n_snps(vector=True) * np.log(lambd)
    if not vector:
        ret = np.sum(ret)
    return ret  

def _entropy_mut_term(mut_rate, counts_i):
    if mut_rate:
        mu = mut_rate * np.ones(len(counts_i))
        mu = mu[counts_i > 0]
        counts_i = counts_i[counts_i > 0]
        return np.sum(-counts_i + counts_i * np.log(np.sum(counts_i) * mu / float(np.sum(mu))))
    return 0.0
    

@force_primitive
def _raw_log_lik(diff_vals, diff_keys, G, data, truncate_probs, folded, error_matrices):
    ## computes log likelihood from the "raw" arrays and graph objects comprising the Demography,
    ## allowing us to compute gradients directly w.r.t. these values,
    ## and thus apply util.precompute_gradients to save memory
    demo = Demography(G, diff_keys, diff_vals)
    return _composite_log_likelihood(data, demo, truncate_probs=truncate_probs, folded=folded, error_matrices=error_matrices)

def _build_sfs_batches(sfs, batch_size):
    counts = sfs._total_freqs
    sfs_len = len(counts)
    
    if sfs_len <= batch_size:
        return [sfs]

    batch_size = int(batch_size)
    slices = []
    prev_idx, next_idx = 0, batch_size    
    while prev_idx < sfs_len:
        slices.append(slice(prev_idx, next_idx))
        prev_idx = next_idx
        next_idx = prev_idx + batch_size

    assert len(slices) == int(np.ceil(sfs_len / float(batch_size)))

    ## sort configs so that "(very) similar" configs end up in the same batch,
    ## thus avoiding redundant computation
    ## "similar" == configs have same num missing alleles
    ## "very similar" == configs are folded copies of each other
    
    a = sfs.configs.value[:,:,0] # ancestral counts
    d = sfs.configs.value[:,:,1] # derived counts
    n = a+d # totals

    n = list(map(tuple, n))
    a = list(map(tuple, a))
    d = list(map(tuple, d))
    
    folded = list(map(min, list(zip(a,d))))

    keys = list(zip(n,folded))
    sorted_idxs = sorted(range(sfs_len), key=lambda i:keys[i])
    sorted_idxs = np.array(sorted_idxs, dtype=int)

    idx_list = [sorted_idxs[s] for s in slices]
    return [_sub_sfs(sfs.configs, counts[idx], subidxs=idx) for idx in idx_list]


def _subsfs_list(sfs, n_chunks, rnd):
    total_counts = sfs._total_freqs
    
    data = []
    indices = []
    indptr = []
    for cnt_i in total_counts:
        indptr.append(len(data))
        
        curr = rnd.multinomial(cnt_i, [1./float(n_chunks)]*n_chunks)
        indices += list(np.arange(len(curr))[curr != 0])
        data += list(curr[curr != 0])
    indptr.append(len(data))

    # row = SFS entry, column=chunk    
    random_counts = scipy.sparse.csr_matrix((data,indices,indptr), shape=(len(total_counts), n_chunks))
    random_counts = random_counts.tocsc()

    ret = []
    for start,end in zip(random_counts.indptr[:-1], random_counts.indptr[1:]):
        ret.append(_sub_sfs(sfs.configs, random_counts.data[start:end], random_counts.indices[start:end]))
    return ret
