import json
import functools
import logging
import time
import autograd.numpy as np
import scipy
import autograd as ag
from autograd.extend import primitive, defvjp
from .optimizers import _find_minimum, stochastic_opts, LoggingCallback
from .compute_sfs import expected_sfs, expected_total_branch_len, expected_heterozygosity
from .demography import Demography
from .data.configurations import _ConfigList_Subset
from .data.sfs import Sfs

logger = logging.getLogger(__name__)


class SfsLikelihoodSurface(object):
    def __init__(self, data, demo_func=None, mut_rate=None, length=1, log_prior=None, folded=False, error_matrices=None, truncate_probs=1e-100, batch_size=1000, p_missing=0.0, use_pairwise_diffs=False):
        """
        Object for computing composite likelihoods, and searching for the maximum composite likelihood.

        Parameters
        ==========
        data: momi.SegSites or momi.Sfs object
        demo_func: function or None
            function that creates momi.Demography object from a vector of input parameters
            if None, this is the identity function (so the "input parameter" is the Demography object itself)
        mut_rate: float or list or None
            The mutation rate.
            If an array, the length should be the total number of loci.
            If None, use a multinomial model instead of Poisson model (so the numger of segregating sites is fixed)
        length: float or array
            The length of each locus.
            The per-locus mutation rate is mut_rate * length.
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

        try:
            self.sfs = self.data.sfs
        except AttributeError:
            self.sfs = self.data

        self.demo_func = demo_func

        self.mut_rate = mut_rate
        if self.mut_rate is not None:
            self.mut_rate = self.mut_rate * np.array(length)
        self.folded = folded
        if self.folded:
            self.sfs = self.sfs.fold()  # required for entropy term to be correct
        self.error_matrices = error_matrices

        self.truncate_probs = truncate_probs

        self.log_prior = log_prior
        self.batch_size = batch_size

        if batch_size <= 0:
            self.sfs_batches = None
        else:
            self.sfs_batches = _build_sfs_batches(self.sfs, batch_size)

        self.p_missing = p_missing

        self.use_pairwise_diffs = use_pairwise_diffs

        if self.mut_rate and self.sfs.configs.has_missing_data and not self.use_pairwise_diffs:
            raise ValueError(
                "Expected total branch length not implemented for missing data; set use_pairwise_diffs=True to scale total mutations by the pairwise differences instead.")

    def log_lik(self, x, vector=False):
        """
        Returns the composite log-likelihood of the data at the point x.
        """
        ret = self._log_lik(x, vector=vector)
        logger.debug("log-likelihood = {0}".format(ret))
        return ret

    def _score(self, x):
        return ag.grad(self.log_lik)(x)

    def _fisher(self, x):
        return -ag.hessian(self.log_lik)(x)

    def _score_cov(self, params):
        params = np.array(params)

        def f_vec(x):
            ret = self._log_lik(x, vector=True)
            # centralize
            return ret - np.mean(ret)

        j = ag.jacobian(f_vec)(params)
        return np.einsum('ij, ik', j, j)

    def _log_lik(self, x, vector):
        demo = self._get_multipop_moran(x)
        ret = self._get_multinom_loglik(demo, vector=vector) + self._mut_factor(demo, vector=vector)
        if vector:
            ret = ret + self._log_prior(x) / len(ret)
        else:
            ret = ret + self._log_prior(x)
        return ret

    def _get_multipop_moran(self, x):
        if self.demo_func:
            logger.debug(
                "Computing log-likelihood at x = {0}".format(str(x).replace('\n', '')))
            demo = self.demo_func(*x)
        else:
            demo = x
        return demo

    def _get_multinom_loglik(self, demo, vector):
        if self.sfs_batches:
            G = demo._get_graph_structure()
            cache = demo._get_differentiable_part()
            ret = 0.0
            for batch in self.sfs_batches:
                ret = ret + _raw_log_lik(
                    cache, G, batch,
                    self.truncate_probs, self.folded,
                    self.error_matrices, vector)
        else:
            ret = _composite_log_likelihood(
                self.data, demo, truncate_probs=self.truncate_probs,
                folded=self.folded, error_matrices=self.error_matrices,
                use_pairwise_diffs=self.use_pairwise_diffs,
                vector=vector)
        return ret

    def _mut_factor(self, demo, vector):
        if self.mut_rate is not None:
            return _mut_factor(
                self.sfs, demo, self.mut_rate,
                vector, self.p_missing, self.use_pairwise_diffs)
        else:
            return 0

    def _log_prior(self, x):
        if self.log_prior:
            return self.log_prior(x)
        else:
            return 0

    def kl_div(self, x):
        """
        Returns KL-Divergence(Empirical || Theoretical(x)).
        """
        log_lik = self.log_lik(x)
        #ret = -log_lik + self.sfs.n_snps() * self.sfs._entropy + _entropy_mut_term(self.mut_rate, self.sfs, self.p_missing, self.use_pairwise_diffs)
        ret = -log_lik + self.sfs.n_snps() * self.sfs._entropy
        if self.mut_rate:
            ret = ret + \
                self.sfs._get_muts_poisson_entropy(self.use_pairwise_diffs)
        ret = ret / float(self.sfs.n_snps())
        if not self.log_prior:
            assert ret >= 0, "kl-div: %s, log_lik: %s, total_count: %s" % (
                str(ret), str(log_lik), str(self.sfs.n_snps()))
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
        hist = lambda: None
        hist.itr = 0
        hist.recent_vals = []
        starttime = time.time()

        def callback(x):
            for y, fx in reversed(hist.recent_vals):
                if np.allclose(y, x):
                    break
            assert np.allclose(y, x)
            try:
                fx = fx.value
            except AttributeError:
                pass
            print_progress(x, fx, hist.itr)
            hist.itr += 1
            hist.recent_vals = [(x, fx)]

        opt_kwargs = dict(kwargs)
        opt_kwargs["method"] = method

        opt_kwargs['jac'] = jac
        if jac:
            replacefun = ag.value_and_grad
        else:
            replacefun = None

        gradmakers = {}
        if hess:
            gradmakers['hess'] = ag.hessian
        if hessp:
            gradmakers['hessp'] = ag.hessian_vector_product

        @functools.wraps(self.kl_div)
        def fun(x):
            ret = self.kl_div(x)
            hist.recent_vals += [(x, ret)]
            return ret

        return _find_minimum(fun, x0, scipy.optimize.minimize,
                             bounds=bounds, callback=callback,
                             opt_kwargs=opt_kwargs, gradmakers=gradmakers, replacefun=replacefun)


    def stochastic_find_mle(
            self, x0, snps_per_minibatch, stepsize, num_iters,
            bounds=None, callback=None,
            checkpoint_file=None, checkpoint_iter=10,
            svrg_epoch=-1, b1=0.9, b2=0.999, eps=10**-8,
            rgen=np.random):
        """
        Search for maximum likelihood using ADAM-style
        stochastic gradient descent.

        Parameters
        ==========
        x0: numpy.array or str
            The starting point for the optimization.
            If a string, a path to the checkpoint file of
            a previous run.
        snps_per_minibatch: int
            The number of SNPs per minibatch
        stepsize: float
            The stepsize for each step
        bounds: None or list of (lower, upper) pairs
            Bounds for the parameter space
        callback: function
            function to call at every step
        checkpoint_file: None or str
            File to save intermediate progress
        checkpoint_iter: int
            Number of iterations between saving intermediate progress
        logging_freq: int
            Frequency to output current iteration to logging
        svrg_epoch: int
            If positive, use SVRG with given epoch length

            SVRG computes the full likelihood every epoch and uses
            this to improve the accuracy of the stochastic gradient
        b1, b2, eps: float
            Parameters for ADAM algorithm. See ADAM paper.
            Recommended to leave these at the default values.
        rgen: random generator
            By default the numpy random generator, which
            can be set with numpy.random.seed().

            Alternatively, use numpy.random.RandomState to create
            a separate random generator and pass it in here.
        """
        kwargs = {}
        kwargs["stepsize"] = stepsize
        kwargs["num_iters"] = num_iters
        kwargs["b1"] = b1
        kwargs["b2"] = b2
        kwargs["eps"] = eps
        kwargs["svrg_epoch"] = svrg_epoch
        kwargs["checkpoint_file"] = checkpoint_file
        kwargs["checkpoint_iter"] = checkpoint_iter
        kwargs["callback"] = callback
        kwargs["bounds"] = bounds

        if isinstance(x0, str):
            with open(x0) as f:
                kwargs.update(json.load(f))
        else:
            kwargs["x0"] = x0

        return self._stochastic_surfaces(
            snps_per_minibatch=snps_per_minibatch,
            rgen=rgen).find_mle(**kwargs)

    def _get_stochastic_pieces(self, pieces, rgen):
        sfs_pieces = _subsfs_list(self.sfs, pieces, rgen)
        if self.mut_rate is None:
            mut_rate = None
        else:
            mut_rate = self.mut_rate * np.ones(self.sfs.n_loci)
            mut_rate = np.sum(mut_rate) / float(pieces)

        return [SfsLikelihoodSurface(sfs, demo_func=self.demo_func, mut_rate=None,
                                     folded=self.folded, error_matrices=self.error_matrices,
                                     truncate_probs=self.truncate_probs, batch_size=self.batch_size)
                for sfs in sfs_pieces]

    def _stochastic_surfaces(self, n_minibatches=None, snps_per_minibatch=None, rgen=np.random):
        """
        Partitions the data into n_minibatches random subsets ("minibatches") of roughly equal size. It returns a StochasticSfsLikelihoodSurface object, which can be used for stochastic gradient descent.

        Useful methods of StochasticSfsLikelihoodSurface are:
        1) StochasticSfsLikelihoodSurface.find_mle(...): search for the MLE using stochastic gradient descent or SVRG
        2) StochasticSfsLikelihoodSurface.get_minibatch(i): the Sfs corresponding to the i-th minibatch
        3) StochasticSfsLikelihoodSurface.n_minibatches: the number of minibatches
        """
        if (n_minibatches is None) == (snps_per_minibatch is None):
            raise ValueError(
                "Exactly one of n_minibatches, snps_per_minibatch should be specified")
        if snps_per_minibatch is not None:
            n_minibatches = int(
                np.ceil(self.sfs.n_snps() / float(snps_per_minibatch)))
        return StochasticSfsLikelihoodSurface(self, n_minibatches, rgen)


class StochasticSfsLikelihoodSurface(object):

    def __init__(self, full_surface, pieces, rgen):
        try:
            assert pieces > 0 and pieces == int(pieces)
        except (TypeError, AssertionError):
            raise ValueError("pieces should be a positive integer")

        self.pieces = full_surface._get_stochastic_pieces(pieces, rgen)
        self.total_snp_counts = full_surface.sfs._total_freqs
        logger.info("Created {n_batches} minibatches, with an average of {n_snps} SNPs and {n_sfs} unique SFS entries per batch".format(n_batches=len(
            self.pieces), n_snps=full_surface.sfs.n_snps() / float(len(self.pieces)), n_sfs=np.mean([len(piece.sfs.configs) for piece in self.pieces])))

        self.rgen = rgen
        self.full_surface = full_surface

    def get_minibatch(self, i): return self.pieces[i].sfs

    @property
    def n_minibatches(self): return len(self.pieces)

    def avg_neg_log_lik(self, x, i):
        if i is None:
            return -self.full_surface.log_lik(x) / self.full_surface.sfs.n_snps()
        demo = self.full_surface._get_multipop_moran(x)
        ret = -self.pieces[i]._get_multinom_loglik(demo, False) * self.n_minibatches
        ret = ret - self.full_surface._mut_factor(demo, False) - self.full_surface._log_prior(x)
        return ret / self.full_surface.sfs.n_snps()

    def find_mle(self, x0, method="adam", bounds=None, rgen=None, callback=None, **kwargs):
        if not rgen:
            rgen = self.rgen
        callback = LoggingCallback(user_callback=callback).callback

        full_surface = self.full_surface

        opt_kwargs = dict(kwargs)
        opt_kwargs.update({'pieces': self.n_minibatches, 'rgen': rgen})

        return _find_minimum(self.avg_neg_log_lik, x0, optimizer=stochastic_opts[method],
                             bounds=bounds, callback=callback, opt_kwargs=opt_kwargs,
                             gradmakers={'fun_and_jac': ag.value_and_grad})


def _composite_log_likelihood(data, demo, mut_rate=None, truncate_probs=0.0, vector=False, p_missing=None, use_pairwise_diffs=False, **kwargs):
    try:
        sfs = data.sfs
    except AttributeError:
        sfs = data

    sfs_probs = np.maximum(expected_sfs(demo, sfs.configs, normalized=True, **kwargs),
                           truncate_probs)
    log_lik = sfs._integrate_sfs(np.log(sfs_probs), vector=vector)

    # add on log likelihood of poisson distribution for total number of SNPs
    if mut_rate is not None:
        log_lik = log_lik + \
            _mut_factor(sfs, demo, mut_rate, vector,
                        p_missing, use_pairwise_diffs)

    if not vector:
        log_lik = np.squeeze(log_lik)
    return log_lik


def _mut_factor(sfs, demo, mut_rate, vector, p_missing, use_pairwise_diffs):
    if use_pairwise_diffs:
        return _mut_factor_het(sfs, demo, mut_rate, vector, p_missing)
    else:
        return _mut_factor_total(sfs, demo, mut_rate, vector)


def _mut_factor_het(sfs, demo, mut_rate, vector, p_missing):
    mut_rate = mut_rate * np.ones(sfs.n_loci)
    E_het = expected_heterozygosity(
        demo,
        restrict_to_pops=np.array(
            sfs.sampled_pops)[sfs.ascertainment_pop])

    p_missing = p_missing * np.ones(len(sfs.ascertainment_pop))
    p_missing = p_missing[sfs.ascertainment_pop]
    lambd = np.einsum("i,j->ij", mut_rate, E_het * (1.0 - p_missing))

    counts = sfs.avg_pairwise_hets[:, sfs.ascertainment_pop]
    ret = -lambd + counts * np.log(lambd) - scipy.special.gammaln(counts + 1)
    ret = ret * sfs.sampled_n[sfs.ascertainment_pop] / float(
        np.sum(sfs.sampled_n[sfs.ascertainment_pop]))
    if not vector:
        ret = np.sum(ret)
    else:
        ret = np.sum(ret, axis=1)
    return ret


def _mut_factor_total(sfs, demo, mut_rate, vector):
    mut_rate = mut_rate * np.ones(sfs.n_loci)

    if sfs.configs.has_missing_data:
        raise ValueError(
            "Expected total branch length not implemented for missing data; set use_pairwise_diffs=True to scale total mutations by the pairwise differences instead.")
    E_total = expected_total_branch_len(
        demo, sampled_pops=sfs.sampled_pops, sampled_n=sfs.sampled_n, ascertainment_pop=sfs.ascertainment_pop)
    lambd = mut_rate * E_total

    counts = sfs.n_snps(vector=True)
    ret = -lambd + counts * np.log(lambd) - scipy.special.gammaln(counts + 1)
    if not vector:
        ret = np.sum(ret)
    return ret

def rearrange_dict_grad(fun):
    """
    Decorator that allows us to save memory on the forward pass,
    by precomputing the gradient
    """
    @primitive
    def wrapped_fun_helper(xdict, dummy):
        ## ag.value_and_grad() to avoid second forward pass
        ## ag.checkpoint() ensures hessian gets properly checkpointed
        val, grad = ag.checkpoint(ag.value_and_grad(fun))(xdict)
        assert len(val.shape) == 0
        dummy.cache = grad
        return val

    def wrapped_fun_helper_grad(ans, xdict, dummy):
        def grad(g):
            #print("foo")
            return {k:g*v for k,v in dummy.cache.items()}
        return grad
    defvjp(wrapped_fun_helper, wrapped_fun_helper_grad, None)

    @functools.wraps(fun)
    def wrapped_fun(xdict):
        return wrapped_fun_helper(ag.dict(xdict), lambda:None)
    return wrapped_fun

def _raw_log_lik(cache, G, data, truncate_probs, folded, error_matrices, vector=False):
    def wrapped_fun(cache):
        demo = Demography(G, cache=cache)
        return _composite_log_likelihood(data, demo, truncate_probs=truncate_probs, folded=folded, error_matrices=error_matrices, vector=vector)
    if vector:
        return ag.checkpoint(wrapped_fun)(cache)
    else:
        ## avoids second forward pass, and has proper
        ## checkpointing for hessian,
        ## but doesn't work for vectorized output
        return rearrange_dict_grad(wrapped_fun)(cache)


#def _build_sfs_batches(sfs, batch_size):
#    counts = sfs._total_freqs
#    sfs_len = len(counts)
#
#    if sfs_len <= batch_size:
#        return [sfs]
#
#    batch_size = int(batch_size)
#    slices = []
#    prev_idx, next_idx = 0, batch_size
#    while prev_idx < sfs_len:
#        slices.append(slice(prev_idx, next_idx))
#        prev_idx = next_idx
#        next_idx += batch_size
#
#    assert len(slices) == int(np.ceil(sfs_len / float(batch_size)))
#
#    idxs = np.arange(sfs_len, dtype=int)
#    idx_list = [idxs[s] for s in slices]
#    return [_sub_sfs(sfs.configs, counts[idx], subidxs=idx) for idx in idx_list]

def _build_sfs_batches(sfs, batch_size):
    sfs_len = len(sfs.configs)
    if sfs_len <= batch_size:
        return [sfs]

    ret = []
    batch_size = int(batch_size)
    for batch_start in range(0, sfs_len, batch_size):
        next_start = min(batch_start + batch_size, sfs_len)
        ret.append(sfs._subset_configs(np.arange(batch_start,
                                                 next_start)))
    return ret


def _subsfs_list(sfs, n_chunks, rnd):
    n_snps = int(sfs.n_snps())
    logger.debug("Splitting {} SNPs into {} minibatches".format(n_snps, n_chunks))

    logger.debug("Building list of length {}".format(n_snps))
    idxs = np.zeros(n_snps, dtype=int)
    total_counts = np.array(sfs._total_freqs, dtype=int)
    curr = 0
    for i, cnt in enumerate(total_counts):
        idxs[curr:(curr+cnt)] = i
        curr += cnt

    logger.debug("Permuting list of {} SNPs".format(n_snps))
    idxs = rnd.permutation(idxs)

    logger.debug("Splitting permuted SNPs into {} minibatches".format(n_chunks))
    ret = []
    for chunk in range(n_chunks):
        chunk_idxs, chunk_cnts = np.unique(idxs[chunk::n_chunks],
                                           return_counts=True)
        sub_configs = _ConfigList_Subset(sfs.configs, chunk_idxs)
        ret.append(Sfs.from_matrix(
            np.array([chunk_cnts]).T, sub_configs,
            folded=sfs.folded, length=None))
    return ret
