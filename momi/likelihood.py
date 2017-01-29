
from .util import count_calls, rearrange_tuple_gradients
from .optimizers import _find_minimum, stochastic_opts, LoggingCallback
import autograd.numpy as np
from .compute_sfs import expected_sfs, expected_total_branch_len, expected_heterozygosity
from .demography import DemographyError, Demography
from .data_structure import _sub_sfs, site_freq_spectrum, ConfigArray, Sfs
import scipy
import autograd, numdifftools
from autograd import grad, hessian_vector_product, hessian
from collections import Counter
import random, functools, logging, time

logger = logging.getLogger(__name__)

class SfsLikelihoodSurface(object):
    def __init__(self, data, demo_func=None, mut_rate=None, log_prior=None, folded=False, error_matrices=None, truncate_probs=1e-100, batch_size=1000, p_missing=None, use_pairwise_diffs=False):
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
            Decrease batch_size to decrease memory usage (but add running time overhead).
            set batch_size=-1 to compute all SNPs in a single batch. This is required if you wish to compute hessians or higher-order derivatives with autograd.
        processes:
            the number of cores to use.
            if <= 0 (the default), do not use any parallelization.
            if processes > 0, it is recommended to create the SfsLikelihoodSurface()
            using the with...as... construct:

                with SfsLikelihoodSurface(data, demo_func, processes=10) as surface:
                     mle = surface.find_mle(x0)
                print("MLE is ", mle.x)

            as this will automatically take care of closing connections to the parallel subprocesses.
            (Alternatively, you can manually call surface.close(), but care must be taken
            to make sure surface.close() is called in the event of an Error).
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

        self.log_prior = log_prior
        self.batch_size = batch_size

        if batch_size <= 0:
            self.sfs_batches = None
        else:
            self.sfs_batches = _build_sfs_batches(self.sfs, batch_size)

        if p_missing is None:
            p_missing = self.sfs.p_missing
        self.p_missing = p_missing

        self.use_pairwise_diffs = use_pairwise_diffs

        if self.mut_rate and self.sfs.configs.has_missing_data and not self.use_pairwise_diffs:
            raise ValueError("Expected total branch length not implemented for missing data; set use_pairwise_diffs=True to scale total mutations by the pairwise differences instead.")

    def log_lik(self, x):
        """
        Returns the composite log-likelihood of the data at the point x.
        """
        if self.demo_func:
            logger.debug("Computing log-likelihood at x = {0}".format(str(x).replace('\n','')))
            demo = self.demo_func(*x)
        else:
            demo = x
        demo = demo._get_multipop_moran(self.sfs.sampled_pops,
                                        self.sfs.sampled_n)

        if self.sfs_batches:
            G,(diff_keys,diff_vals) = demo._get_graph_structure(), demo._get_differentiable_part()
            ret = 0.0
            for batch in self.sfs_batches:
                ret = ret + _raw_log_lik(diff_vals, diff_keys, G, batch, self.truncate_probs, self.folded, self.error_matrices)
        else:
            ret = _composite_log_likelihood(self.data, demo, truncate_probs=self.truncate_probs, folded=self.folded, error_matrices=self.error_matrices, use_pairwise_diffs=self.use_pairwise_diffs)

        if self.mut_rate is not None:
            ret = ret + _mut_factor(self.sfs, demo, self.mut_rate, False, p_missing=self.p_missing, use_pairwise_diffs=self.use_pairwise_diffs)

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

    def find_mle(self, x0, method="tnc", jac=True, hess=False, hessp=False, bounds=None, callback=None, **kwargs):
        """
        Search for the maximum of the likelihood surface
        (i.e., the minimum of the KL-divergence).

        Parameters
        ==========
        x0 : numpy.ndarray
             initial guess
        method : str
                 Can be any method from scipy.optimize.minimize()
                 (e.g. "tnc","L-BFGS-B",etc.)
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
        callback: callable, optional
              Called after each iteration as callback(x), like in
              scipy.optimize.minimize. However, x is given 2 additional
              attributes, x.iteration and x.fun, that allow the callback
              function to access the current iteration number and the current
              objective function value.
        **kwargs : additional arguments to pass to scipy.optimize.minimize()

        Notes
        =====
        This is just a wrapper around scipy.optimize.minimize, and takes the same arguments, with the following exceptions:
        1) no "fun" param (this is set to be self.kl_div)
        2) jac, hess, hessp are bools. If True, their respective derivatives are defined using autograd and passed into scipy.optimize.minimize; otherwise, "None" is passed in for the derivatives (in which case scipy may use a numerical derivative if needed)
        """
        print_progress = LoggingCallback(user_callback=callback).callback
        hist = lambda:None
        hist.itr = 0
        hist.recent_vals = []
        starttime = time.time()

        def callback(x):
            for y,fx in reversed(hist.recent_vals):
                if np.allclose(y,x): break
            assert np.allclose(y,x)
            try: fx=fx.value
            except AttributeError: pass
            print_progress(x,fx,hist.itr)
            hist.itr += 1
            hist.recent_vals = [(x,fx)]

        opt_kwargs = dict(kwargs)
        opt_kwargs["method"] = method

        opt_kwargs['jac'] = jac
        if jac: replacefun = autograd.value_and_grad
        else: replacefun = None

        gradmakers = {}
        if hess: gradmakers['hess'] = autograd.hessian
        if hessp: gradmakers['hessp'] = autograd.hessian_vector_product

        @functools.wraps(self.kl_div)
        def fun(x):
            ret = self.kl_div(x)
            hist.recent_vals += [(x,ret)]
            return ret

        return _find_minimum(fun, x0, scipy.optimize.minimize,
                             bounds=bounds, callback=callback,
                             opt_kwargs=opt_kwargs, gradmakers=gradmakers, replacefun=replacefun)

    def _get_stochastic_pieces(self, pieces, rgen, exact):
        sfs_pieces, is_exact = _subsfs_list(self.sfs, pieces, rgen, exact)
        if self.mut_rate is None:
            mut_rate = None
        else:
            mut_rate = self.mut_rate * np.ones(self.sfs.n_loci)
            mut_rate = np.sum(mut_rate) / float(pieces)

        return [SfsLikelihoodSurface(sfs, demo_func=self.demo_func, mut_rate=mut_rate,
                                     folded = self.folded, error_matrices = self.error_matrices,
                                     truncate_probs = self.truncate_probs,
                                     batch_size = self.batch_size)
                for sfs in sfs_pieces], is_exact

    def stochastic_surfaces(self, n_minibatches=None, snps_per_minibatch=None, rgen=np.random, exact=0):
        """
        Partitions the data into n_minibatches random subsets ("minibatches") of roughly equal size. It returns a StochasticSfsLikelihoodSurface object, which can be used for stochastic gradient descent.

        Useful methods of StochasticSfsLikelihoodSurface are:
        1) StochasticSfsLikelihoodSurface.find_mle(...): search for the MLE using stochastic gradient descent or SVRG
        2) StochasticSfsLikelihoodSurface.get_minibatch(i): the Sfs corresponding to the i-th minibatch
        3) StochasticSfsLikelihoodSurface.n_minibatches: the number of minibatches
        """
        if (n_minibatches is None) == (snps_per_minibatch is None):
            raise ValueError("Exactly one of n_minibatches, snps_per_minibatch should be specified")
        if snps_per_minibatch is not None:
            n_minibatches = int( np.ceil( self.sfs.n_snps() / float(snps_per_minibatch) ) )
        return StochasticSfsLikelihoodSurface(self, n_minibatches, rgen, exact)

class StochasticSfsLikelihoodSurface(object):
    def __init__(self, full_surface, pieces, rgen, exact):
        try: assert pieces > 0 and pieces == int(pieces)
        except (TypeError,AssertionError):
            raise ValueError("pieces should be a positive integer")

        self.pieces,self.exact_snps = full_surface._get_stochastic_pieces(pieces, rgen, exact)
        self.total_snp_counts = full_surface.sfs._total_freqs
        logger.info("Created {n_batches} minibatches, with an average of {n_snps} SNPs and {n_sfs} unique SFS entries per batch".format(n_batches=len(self.pieces), n_snps=full_surface.sfs.n_snps() / float(len(self.pieces)), n_sfs=np.mean([len(piece.sfs.configs) for piece in self.pieces])))
        if exact:
            total = np.sum(self.total_snp_counts)
            nexact = np.sum(self.total_snp_counts[self.exact_snps])
            logger.info("Using exact frequencies for %d most frequent entries, accounting for %f of SNPs (%d out of %d)" % (exact, nexact / float(total), nexact, total))

        self.rgen = rgen
        self.full_surface = full_surface

    def get_minibatch(self, i): return self.pieces[i].sfs
    @property
    def n_minibatches(self): return len(self.pieces)

    def find_mle(self, x0, method="adam", bounds=None, rgen=None, callback=None, **kwargs):
        if not rgen:
            rgen = self.rgen
        callback = LoggingCallback(user_callback=callback).callback

        full_surface = self.full_surface
        def fun(x,i):
            if i is None:
                return full_surface.kl_div(x)
            else:
                if full_surface.log_prior: lp = full_surface.log_prior(x)
                else: lp = 0.0
                ret = -self.pieces[i].log_lik(x) + self.pieces[i].sfs.n_snps() * (full_surface.sfs._entropy - lp/float(full_surface.sfs.n_snps())) + _entropy_mut_term(self.pieces[i].mut_rate, self.pieces[i].sfs.n_snps(vector=True))
                return ret / float(full_surface.sfs.n_snps()) * self.n_minibatches

        opt_kwargs = dict(kwargs)
        opt_kwargs.update({'pieces': self.n_minibatches, 'rgen': rgen})

        return _find_minimum(fun, x0, optimizer=stochastic_opts[method],
                             bounds=bounds, callback=callback, opt_kwargs=opt_kwargs,
                             gradmakers={'fun_and_jac':autograd.value_and_grad})

def _make_likelihood_fun(sampled_pops, conf_arr, sampled_n, counts_arr, *args, **kwargs):
    configs = ConfigArray(sampled_pops, conf_arr, sampled_n, None)
    sfs = Sfs([dict(zip(range(len(counts_arr)), counts_arr))], configs)
    surface = SfsLikelihoodSurface(sfs, *args, **kwargs)
    return surface.log_lik

def _composite_log_likelihood(data, demo, mut_rate=None, truncate_probs = 0.0, vector=False, p_missing=None, use_pairwise_diffs=False, **kwargs):
    try:
        sfs = data.sfs
    except AttributeError:
        sfs = data

    sfs_probs = np.maximum(expected_sfs(demo, sfs.configs, normalized=True, **kwargs),
                           truncate_probs)
    log_lik = sfs._integrate_sfs(np.log(sfs_probs), vector=vector)

    # add on log likelihood of poisson distribution for total number of SNPs
    if mut_rate is not None:
        log_lik = log_lik + _mut_factor(sfs, demo, mut_rate, vector, p_missing=p_missing, use_pairwise_diffs=use_pairwise_diffs)

    if not vector:
        log_lik = np.squeeze(log_lik)
    return log_lik

def _mut_factor(sfs, demo, mut_rate, vector, p_missing=None, use_pairwise_diffs=False):
    if use_pairwise_diffs:
        return _mut_factor_het(sfs, demo, mut_rate, vector, p_missing)
    else:
        return _mut_factor_total(sfs, demo, mut_rate, vector)

def _mut_factor_het(sfs, demo, mut_rate, vector, p_missing=None):
    mut_rate = mut_rate * np.ones(sfs.n_loci)
    E_het = expected_heterozygosity(demo, sampled_pops = sfs.sampled_pops)[sfs.ascertainment_pop]
    if p_missing is None:
        p_missing = sfs.p_missing
    lambd = np.einsum("i,j->ij", mut_rate, E_het*(1.0-p_missing))

    ret = -lambd + sfs.avg_pairwise_hets * np.log(lambd)
    #ret = ret / sum(sfs.ascertainment_pop)
    if not vector:
        ret = np.sum(ret)
    else:
        ret = np.sum(ret, axis=1)
    return ret

def _mut_factor_total(sfs, demo, mut_rate, vector):
    mut_rate = mut_rate * np.ones(sfs.n_loci)

    if sfs.configs.has_missing_data:
        raise ValueError("Expected total branch length not implemented for missing data; set use_pairwise_diffs=True to scale total mutations by the pairwise differences instead.")
    E_total = expected_total_branch_len(demo, sampled_pops = sfs.sampled_pops, sampled_n = sfs.sampled_n, ascertainment_pop = sfs.ascertainment_pop)
    lambd = mut_rate * E_total

    ret = -lambd + sfs.n_snps(vector=True) * np.log(lambd)
    if not vector:
        ret = np.sum(ret)
    return ret

def _entropy_mut_term(mut_rate, counts_i):
    if mut_rate is not None:
        mu = mut_rate * np.ones(len(counts_i))
        mu = mu[counts_i > 0]
        counts_i = counts_i[counts_i > 0]
        return np.sum(-counts_i + counts_i * np.log(np.sum(counts_i) * mu / float(np.sum(mu))))
    return 0.0


@rearrange_tuple_gradients()
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

    idxs = np.arange(sfs_len, dtype=int)
    idx_list = [idxs[s] for s in slices]
    return [_sub_sfs(sfs.configs, counts[idx], subidxs=idx) for idx in idx_list]

def _subsfs_list(sfs, n_chunks, rnd, exact):
    random_counts, is_exact = _get_random_counts(sfs, n_chunks, rnd, exact)

    ret = []
    for start,end in zip(random_counts.indptr[:-1], random_counts.indptr[1:]):
        ret.append(_sub_sfs(sfs.configs, random_counts.data[start:end], random_counts.indices[start:end]))
    return ret, is_exact

def _get_random_counts(sfs, n_chunks, rnd, exact):
    total_counts = sfs._total_freqs
    is_exact = np.zeros(len(total_counts), dtype='bool')
    if exact:
        sorted_idxs = np.argsort(total_counts, kind="heapsort")
        exact = sorted_idxs[::-1][:exact]
        is_exact[exact] = True

    data = []
    indices = []
    indptr = []
    for cnt_i,use_exact in zip(total_counts, is_exact):
        indptr.append(len(data))

        if use_exact:
            curr = cnt_i / float(n_chunks) * np.ones(n_chunks, dtype=float)
        else:
            curr = rnd.multinomial(cnt_i, [1./float(n_chunks)]*n_chunks)
        indices += list(np.arange(len(curr))[curr != 0])
        data += list(curr[curr != 0])
    indptr.append(len(data))

    # row = SFS entry, column=chunk
    random_counts = scipy.sparse.csr_matrix((data,indices,indptr), shape=(len(total_counts), n_chunks))
    random_counts = random_counts.tocsc()

    return random_counts, is_exact
