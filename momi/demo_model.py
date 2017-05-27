
from cached_property import cached_property
import autograd as ag
import autograd.numpy as np
import logging
import collections as co
import networkx as nx
import pandas as pd
import itertools as it
from .demography import demographic_history
from .likelihood import SfsLikelihoodSurface
from .confidence_region import _ConfidenceRegion
from .compute_sfs import expected_sfs_tensor_prod


def demographic_model(default_N, gen_time=1):
    """
    Arguments
    ---------
    default_N: the default population size
        every population is assumed to have its size
        as default_N, unless manually changed by
        set_size().

        the optimizer also internally uses this value
        to rescale the problem. for best optimizer
        performance, default_N should be on the same
        order magnitude as the "typical" effective size
        (e.g. 1e4 in humans)
    gen_time: units of time per generation
        (e.g. if generation is 29 years, set gen_time=29
            to use years as the time unit)
    """
    return DemographicModel(
        N_e=default_N, gen_time=gen_time, parameters=[],
        event_funs=[], sample_t_funs={}, leafs=[],
        data=None, muts_per_gen=None, folded=None,
        mem_chunk_size=None, use_pairwise_diffs=None)


class DemographicModel(object):
    def __init__(self, N_e, gen_time, parameters,
                 event_funs, sample_t_funs, leafs,
                 data, muts_per_gen, folded,
                 mem_chunk_size, use_pairwise_diffs):
        self.N_e = N_e
        self.gen_time = gen_time
        self.parameters = [p.copy() for p in parameters]
        self.event_funs = list(event_funs)
        self.sample_t_funs = dict(sample_t_funs)
        self.leafs = list(leafs)

        self._set_data(data=data, muts_per_gen=muts_per_gen,
                       folded=folded, mem_chunk_size=mem_chunk_size,
                       use_pairwise_diffs=use_pairwise_diffs)

    def copy(self):
        return DemographicModel(
            N_e=self.N_e, gen_time=self.gen_time,
            parameters=self.parameters, event_funs=self.event_funs,
            sample_t_funs=self.sample_t_funs, leafs=self.leafs,
            data=self._data, muts_per_gen=self._muts_per_gen,
            folded=self._folded, mem_chunk_size=self._mem_chunk_size,
            use_pairwise_diffs=self._use_pairwise_diffs)


    def add_param(self, name, x0,
                  lower_x=1e-12, upper_x=None,
                  transform_x=None,
                  opt_scale=None):
        """
        Add a parameter to the demographic model.

        Arguments
        ---------
        name: str, should be unique for each parameter
        x0: float, starting value
        lower_x: float or None
        upper_x: float or None
             lower, upper boundaries. None means no boundary in that direction
        transform_x: function or None
            transformation to obtain the parameter
            as a function of x (the untransformed parameter)
            and all previously added parameters (passed as keyword
            arguments).

            for example, if we are adding parameter t1,
            and want to ensure it is larger than the previously
            added parameter t2, we can use transform_x to add
            on the value of t2:

            model.add_param("t1", ..., transform_x=lambda x, t2, **kw: x+t2)

        Other Arguments
        ---------------
        opt_scale: one of None,"linear","log","logit"
            scaling that the optimizer internally uses
            during optimization. If None, determines
            the scaling as follows:

            if lower_x > 0 and upper_x < 1: "logit"
            elif lower_x > 0: "log"
            else: "linear"
        """
        self._conf_region = None

        if transform_x is None:
            transform_x = lambda x, **kw: x

        bounds = (lower_x, upper_x)
        if opt_scale is None:
            if lower_x is None:
                lower_x=-float("inf")
            if upper_x is None:
                upper_x=float("inf")

            if lower_x > 0 and upper_x < 1:
                opt_scale="logit"
            elif lower_x > 0:
                opt_scale="log"
            else:
                opt_scale="linear"

        if opt_scale == "logit":
            opt_trans = lambda x: 1./(1.+np.exp(-x))
            inv_opt_trans = lambda p: np.log(p/(1.-p))
        elif opt_scale == "log":
            opt_trans=np.exp
            inv_opt_trans = np.log
        elif opt_scale == "linear":
            opt_trans = inv_opt_trans = lambda x: x
        else:
            raise ValueError("Unrecognized opt_scale")

        param = Parameter(
            name, x0, opt_trans, inv_opt_trans,
            transform_x=transform_x,
            x_bounds=bounds)
        self.parameters.append(param)

    def add_leaf(self, pop, t=0, N=None):
        """
        Add a leaf (sampled) population to the
        model.

        Arguments
        ---------
        pops: str, the name of the population
        t: float or str or function
           the time the population was sampled.
           can be a constant, the name of a parameter,
           or a function of the parameters
        N: None or float or str or function
           the population size.
           If None, the starting size is N_e.
           Otherwise, this is a constant, or the name
           of a parameter, or a function of the parameters
        """
        self.leafs.append(pop)

        if t != 0:
            self.sample_t_funs[pop] = TimeValue(
                t, self.N_e, self.gen_time)

        if N is not None:
            self.set_size(pop, t, N)

    def move_lineages(self, pop1, pop2, t, p=1, N=None):
        """
        Move each lineage in pop1 to pop2 at time t
        with probability p.

        Arguments
        ---------
        pop1, pop2: str
        t: float or str or function
           either a constant, or the name of a parameter,
           or a function of the parameters
        p: float or str or function
           either a constant, or the name of a parameter,
           or a function of the parameters
        N: None or float or str or function
           if non-None, set the size of pop2 to N
        """
        if p == 1:
            self.event_funs.append((JoinEventFun(
                t, pop1, pop2, self.N_e, self.gen_time)))
        else:
            self.event_funs.append(PulseEventFun(
                t, p, pop1, pop2, self.N_e,
                self.gen_time))

        if N is not None:
            self.event_funs.append((SizeEventFun(
                t, N, pop2, self.N_e, self.gen_time)))

    def set_size(self, pop, t, N=None, g=0):
        """
        Set the size of pop at t to N,
        and the population size to be exponentially
        growing before t, at rate g per unit time

        Arguments
        ---------
        pop: str
        t: float or str or function
           constant, or parameter name, or function of params
        N: None or float or str or function
           constant, or parameter name, or function of params
           if None, leaves the size unchanged
        g: float or str or function
           constant, or parameter name, or function of params
        """
        if N is not None:
            self.event_funs.append(SizeEventFun(
                t, N, pop, self.N_e, self.gen_time))
        if g != 0:
            self.event_funs.append(GrowthEventFun(
                t, g, pop, self.N_e, self.gen_time))

    def get_params(self):
        """
        Return a dictionary with the current parameter
        values.
        """
        params_dict = co.OrderedDict()
        for param in self.parameters:
            param.update_params_dict(params_dict)
        return params_dict

    def get_x(self, param=None):
        """
        Return the current value of x (the untransformed parameters).
        """
        if param is None:
            return np.array([p.x for p in self.parameters])
        else:
            for p in self.parameters:
                if p.name == param:
                    return p.x
            return ValueError("Unrecognized parameter {}".format(param))

    def set_x(self, x, param=None):
        """
        Set the value of x (the untransformed parameters).
        """
        if param is None:
            if len(x) != len(self.parameters):
                raise ValueError(
                    "len(x) != {}".format(len(self.parameters)))

            for p_i, x_i in zip(self.parameters, x):
                p_i.x = x_i
        else:
            for p in self.parameters:
                if p.name == param:
                    p.x = x
                    return
            raise ValueError("Unrecognized parameter {}".format(param))

    def simulate_data(self, length, recombination_rate,
                      mutation_rate, num_replicates,
                      sampled_n_dict=None):
        demo = self._get_demo()
        if sampled_n_dict is None:
            if self._data is None:
                raise ValueError("Need to set data or supply sample sizes")
            sampled_n_dict = dict(zip(self._data.configs.sampled_pops,
                                      self._data.configs.sampled_n))
        demo = demo._get_multipop_moran(self.leafs, [sampled_n_dict[k] for k in self.leafs])
        return demo.simulate_data(length=length, recombination_rate=recombination_rate,
                                  mutation_rate=4*self.N_e*mutation_rate,
                                  num_replicates=num_replicates)

    def _get_demo(self):
        params_dict = self.get_params()

        events = []
        for f in self.event_funs:
            events.extend(f(params_dict))

        archaic_times_dict = {}
        for k, f in self.sample_t_funs.items():
            archaic_times_dict[k] = f(params_dict)

        demo = demographic_history(
            events, archaic_times_dict=archaic_times_dict)
        demo.params = co.OrderedDict(sorted(
            params_dict.items()))

        def printable_params():
            for k, v in demo.params.items():
                try:
                    v = v.value
                except:
                    pass
                yield (k, v)
        logging.debug("Demographic parameters = {}".format(
            co.OrderedDict(printable_params())))

        return demo

    def _get_opt_x(self):
        return np.array([p.inv_opt_trans(p.x)
                         for p in self.parameters])

    def _x_from_opt_x(self, opt_x):
        return [p.opt_trans(ox)
                for p, ox in zip(self.parameters, opt_x)]

    def _opt_x_from_x(self, x):
        return np.array([
            p.inv_opt_trans(x_i)
            for p, x_i in zip(self.parameters, x)
        ])

    def _opt_demo_fun(self, *opt_x):
        x = self._x_from_opt_x(opt_x)
        return self._demo_fun(*x)

    def _demo_fun(self, *x):
        prev_x = self.get_x()
        try:
            self.set_x(x)
            return self._get_demo()
        except:
            raise
        finally:
            self.set_x(prev_x)

    def set_data(
            self, data, muts_per_gen=None, folded=False,
            mem_chunk_size=1000, use_pairwise_diffs=None,
            n_blocks_jackknife=100):
        """
        Sets data, and optionally the mutation rate,
        in order to compute likelihoods and fit parameters

        Arguments
        ---------
        data: data object as read in by momi.vcf2momi, or
            as simulated by DemographicModel.simulate_data()
        muts_per_gen: float or None
            the number of de novo mutations per generation,
            i.e. the genome length * per base mutation rate.

            if None, assumes the number of observed SNPs
            is fixed (i.e. if using a SNP chip instead of
            whole genome sequencing)
        folded:
            whether the SFS should be folded
        mem_chunk_size:
            controls memory usage by computing likelihood
            in chunks of SNPs.
            if mem_chunk_size=-1 then the data is not broken up
            into chunks

        Other Arguments
        ---------------
        use_pairwise_diffs: None or bool
            Only has an effect if muts_per_gen is not None
            if False, the likelihood incorporates a term for the total number
            of mutations (corresponding to the total tree length).
            This requires there to be no missing data
            if True, the likelihood instead uses a term for the average
            heterozygosity within populations (corresponding to the pairwise
            coalescence time within a population)
            if None, uses the pairwise heterozygosity if there is missing data;
            else, if there is no missing data, use the total number of mutations
        """
        self._set_data(
            data=data._chunk_data(n_blocks_jackknife),
            muts_per_gen=muts_per_gen, folded=folded,
            mem_chunk_size=mem_chunk_size,
            use_pairwise_diffs=use_pairwise_diffs)

    def _set_data(self, data, muts_per_gen, folded,
            mem_chunk_size, use_pairwise_diffs):
        self._opt_surface = None
        self._conf_region = None
        self._data = data
        self._folded = folded
        self._mem_chunk_size = mem_chunk_size
        self._muts_per_gen = muts_per_gen
        self._use_pairwise_diffs = use_pairwise_diffs

    def _get_opt_surface(self):
        if self._opt_surface is None or list(
                self._opt_surface.data.sampled_pops) != list(self.leafs):
            self._conf_region = None
            if self._data is None:
                raise ValueError("Need to call DemographicModel.set_data()")
            logging.info("Constructing likelihood surface...")

            sfs = self._data.subset_populations(
                self.leafs).sfs
            self._opt_surface = self._make_surface(
                sfs, opt_surface=True)

            logging.info("Finished constructing likelihood surface")

        return self._opt_surface

    def _get_conf_region(self):
        opt_surface = self._get_opt_surface()
        if self._conf_region is None or not np.allclose(
                self.get_x(), self._conf_region.point):
            opt_x = self._get_opt_x()
            opt_score = opt_surface._score(opt_x)
            opt_score_cov = opt_surface._score_cov(opt_x)
            opt_fisher = opt_surface._fisher(opt_x)

            self._conf_region = _ConfidenceRegion(
                opt_x, opt_score, opt_score_cov, opt_fisher,
                psd_rtol=1e-4)
        return self._conf_region

    def marginal_wald(self):
        marginal_wald_df = co.OrderedDict()
        marginal_wald_df["Param"] = [p.name for p in self.parameters]
        marginal_wald_df["Value"] = list(self.get_params().values())
        marginal_wald_df["x"] = list(self.get_x())
        marginal_wald_df["std_x"] = np.sqrt(np.diag(self.godambe(inverse=True)))
        return pd.DataFrame(marginal_wald_df)

    def godambe(self, inverse=False):
        # use delta method
        G = self._get_conf_region().godambe(inverse=inverse)
        opt_x = self._get_opt_x()
        dx_do = np.array([
            p.opt_trans(ox) for p, ox in zip(self.parameters, opt_x)
        ])
        if not inverse:
            dx_do = 1./dx_do
        return np.einsum("i,ij,j->ij", dx_do, G, dx_do)

    def test(self, null_point=None, sims=int(1e3), test_type="ratio", alt_point=None, *args, **kwargs):
        if null_point is None:
            null_point = self.get_x()
        null_point = self._opt_x_from_x(null_point)
        if alt_point is not None:
            alt_point = self._opt_x_from_x(alt_point)
        return self._get_conf_region.test(null_point=null_point, sims=sims, test_type=test_type, alt_point=alt_point, *args, **kwargs)

    def _make_surface(self, sfs, opt_surface):
        use_pairwise_diffs = self._use_pairwise_diffs
        if use_pairwise_diffs is None:
            use_pairwise_diffs = sfs.configs.has_missing_data

        muts_per_gen = self._muts_per_gen
        if muts_per_gen is None:
            mut_rate = None
        else:
            mut_rate = 4 * self.N_e * muts_per_gen / sfs.n_loci

        if opt_surface:
            demo_fun = self._opt_demo_fun
        else:
            demo_fun = self._demo_fun

        p_miss = self._data._p_missing
        p_miss = {pop: pm for pop, pm in zip(
            self._data.populations, p_miss)}
        p_miss = np.array([p_miss[pop] for pop in sfs.sampled_pops])
        return SfsLikelihoodSurface(
            sfs, demo_fun, mut_rate=mut_rate,
            folded=self._folded, batch_size=self._mem_chunk_size,
            use_pairwise_diffs=use_pairwise_diffs,
            p_missing = p_miss)

    def check_fit_pairwise_diffs(self):
        return self._make_pairwise_diffs_modelfit().folded_pairwise_diffs_df()

    def check_excess_hets(self):
        return self._make_pairwise_diffs_modelfit().excess_het_df()

    def _make_pairwise_diffs_modelfit(self):
        opt_surface = self._get_opt_surface()
        sfs = opt_surface.sfs
        pairwise_missingness = self._data._pairwise_missingness
        pairwise_missingness = {
            (pop_i, pop_j): pairwise_missingness[i,j]
            for i, pop_i in enumerate(self._data.populations)
            for j, pop_j in enumerate(self._data.populations)}
        pairwise_missingness = np.array([[
            pairwise_missingness[(pop_i, pop_j)]
            for pop_j in sfs.sampled_pops] for pop_i in sfs.sampled_pops])
        return PairwiseDiffsModelFit(
            sfs, self._get_demo(),
            opt_surface.mut_rate * sfs.n_loci,
            pairwise_missingness)


    def log_likelihood(self):
        """
        The log likelihood at the current parameter values
        """
        return self._get_opt_surface().log_lik(self._get_opt_x())

    def kl_div(self):
        """
        The KL-divergence at the current parameter values
        """
        return self._get_opt_surface().kl_div(self._get_opt_x())

    def optimize(self, method="tnc", jac=True,
                 hess=False, hessp=False, **kwargs):
        """
        Search for the maximum likelihood value of the
        parameters.

        This is just a wrapper around scipy.optimize.minimize
        on the KL-divergence

        Arguments
        ---------
        method: str
            any method from scipy.optimize.minimize
            (e.g. "tnc", "L-BFGS-B", etc)
        jac : bool
              If True, use autograd to compute the gradient
              and pass it into scipy.optimize.minimize
        hess, hessp: bool
              If True, use autograd to compute the hessian or
              hessian-vector-product, and pass into
              scipy.optimize.minimize.

              Requires that the data was set with
              DemographicModel.set_data(..., mem_chunk_size=-1),
              and may incur a high memory cost.
        **kwargs: additional arguments passed to
              scipy.optimize.minimize
        """
        bounds = [p.opt_x_bounds for p in self.parameters]
        if all([b is None for bnd in bounds for b in bnd]):
            bounds = None

        res = self._get_opt_surface().find_mle(
            self._get_opt_x(), method=method,
            jac=jac, hess=hess, hessp=hessp,
            bounds=bounds, **kwargs)

        res.x = self._x_from_opt_x(res.x)
        self.set_x(res.x)
        return res


class SizeEventFun(object):
    def __init__(self, t, N, pop, N_e, gen_time):
        self.t = TimeValue(t, N_e, gen_time)
        self.N = SizeValue(N, N_e)
        self.pop = pop

    def __call__(self, prm_dict):
        return [("-en", self.t(prm_dict), self.pop, self.N(prm_dict))]

class JoinEventFun(object):
    def __init__(self, t, pop1, pop2, N_e, gen_time):
        self.t = TimeValue(t, N_e, gen_time)
        self.pop1 = pop1
        self.pop2 = pop2

    def __call__(self, prm_dict):
        return [("-ej", self.t(prm_dict), self.pop1, self.pop2)]

class PulseEventFun(object):
    def __init__(self, t, p, pop1, pop2, N_e, gen_time):
        self.t = TimeValue(t, N_e, gen_time)
        self.p = EventValue(p)
        self.pop1 = pop1
        self.pop2 = pop2

    def __call__(self, prm_dict):
        return [("-ep", self.t(prm_dict), self.pop1,
                 self.pop2, self.p(prm_dict))]

class GrowthEventFun(object):
    def __init__(self, t, g, pop, N_e, gen_time):
        self.t = TimeValue(t, N_e, gen_time)
        self.pop = pop
        self.g = RateValue(g, N_e, gen_time)

    def __call__(self, prm_dict):
        return [("-eg", self.t(prm_dict), self.pop,
                 self.g(prm_dict))]

class EventValue(object):
    def __init__(self, x):
        self.x = x
        self.scale = 1.0

    def __call__(self, params_dict):
        if isinstance(self.x, str):
            x = params_dict[self.x]
        else:
            try:
                x = self.x(**params_dict)
            except TypeError:
                x = self.x
        return x / self.scale

class SizeValue(EventValue):
    def __init__(self, N, N_e):
        self.x = N
        self.scale = N_e

class TimeValue(EventValue):
    def __init__(self, t, N_e, gen_time):
        self.x = t
        self.scale = 4.0 * N_e * gen_time

class RateValue(EventValue):
    def __init__(self, r, N_e, gen_time):
        self.x = r
        self.scale = .25 / N_e / gen_time

class Parameter(object):
    def __init__(self, name, x0, opt_trans, inv_opt_trans,
                 transform_x, x_bounds):
        self.name = name
        self.x = x0
        self.opt_trans = opt_trans
        self.inv_opt_trans = inv_opt_trans
        self.x_bounds = list(x_bounds)
        self.transform_x = transform_x

    def copy(self):
        return Parameter(name=self.name, x0=self.x, opt_trans=self.opt_trans,
                         inv_opt_trans=self.inv_opt_trans, transform_x=self.transform_x,
                         x_bounds=self.x_bounds)

    @property
    def opt_x_bounds(self):
        opt_x_bounds = []
        for bnd in self.x_bounds:
            if bnd is None:
                opt_x_bounds.append(None)
            else:
                opt_x_bounds.append(self.inv_opt_trans(bnd))
        return opt_x_bounds

    def update_params_dict(self, params_dict):
        params_dict[self.name] = self.transform_x(self.x, **params_dict)


class PairwiseDiffsModelFit(object):
    def __init__(self, sfs, demo, mut_rate, pairwise_missingness):
        config_arr = sfs.configs.value
        self.sampled_pops = np.array(sfs.sampled_pops)
        n_pops = len(self.sampled_pops)

        pairwise_diffs = np.zeros((len(config_arr), n_pops, n_pops))
        for i, derived_pop in enumerate(sfs.sampled_pops):
            for j, anc_pop in enumerate(sfs.sampled_pops):
                n_i = sfs.sampled_n[i]
                n_j = sfs.sampled_n[j]
                if i == j:
                    denom = n_i * (n_i-1)
                else:
                    denom = n_i * n_j
                if denom > 0:
                    pairwise_diffs[:,i,j] = config_arr[:,i,1] * config_arr[:,j,0] / float(denom)

        freqs_mat = sfs.freqs_matrix.T
        pairwise_diffs = [[
            freqs_mat.dot(pairwise_diffs[:, i, j])
            for j in range(n_pops)] for i in range(n_pops)]
        pairwise_diffs = np.transpose(pairwise_diffs, (2, 0, 1))
        assert pairwise_diffs.shape == (sfs.n_loci, n_pops, n_pops)

        melted_idxs = np.array([(i, j) for j in range(n_pops)
                                for i in range(n_pops)])
        self.unmelted_idxs = {(i, j): k for k, (i, j) in enumerate(
            melted_idxs)}
        self.melted_der_idx = melted_idxs[:,0]
        self.melted_anc_idx = melted_idxs[:,1]

        self.melted_der_pop = self.sampled_pops[self.melted_der_idx]
        self.melted_anc_pop = self.sampled_pops[self.melted_anc_idx]

        self.melted_diffs = np.array([pairwise_diffs[:, i, j]
                                      for i, j in melted_idxs])
        self.melted_pairwise_missing = np.array([
            pairwise_missingness[i, j] for i, j in melted_idxs])

        self.melted_corrected_cov = np.cov(
            np.einsum("i,ij->ij", 1./(1.-self.melted_pairwise_missing), self.melted_diffs), rowvar=True) * sfs.n_loci
        assert self.melted_corrected_cov.shape == (n_pops**2, n_pops**2)

        self.melted_sum = np.sum(self.melted_diffs, axis=1)
        self.corrected = self.melted_sum / (1.0-self.melted_pairwise_missing)

        lik_arrs = [np.ones((len(melted_idxs), n+1))
                    for n in sfs.sampled_n]
        for i, (j, k) in enumerate(melted_idxs):
            if j == k:
                a = lik_arrs[j]
                n_j = a.shape[1]-1
                i_j = np.arange(n_j+1)
                a[i,:] = i_j * (n_j-i_j) / float(
                    n_j * (n_j-1))
            else:
                a_j = lik_arrs[j]
                n_j = a_j.shape[1]-1
                i_j = np.arange(n_j+1)
                a_j[i,:] = i_j / float(n_j)

                a_k = lik_arrs[k]
                n_k = a_k.shape[1]-1
                i_k = np.arange(n_k+1)
                a_k[i,:] = 1 - i_k / float(n_k)

        self.expected = expected_sfs_tensor_prod(
            lik_arrs, demo, mut_rate=mut_rate,
            sampled_pops = sfs.sampled_pops)

    def linear_stats(self, mat=None):
        if mat is None:
            mat = np.eye(len(self.expected))
        ret = {
            "expected": np.dot(mat, self.expected),
            "observed": np.dot(mat, self.melted_sum),
            "corrected": np.dot(mat, self.corrected),
            "cov": np.dot(mat, np.dot(
                self.melted_corrected_cov, mat.T))
        }
        ret["stddev"] = np.sqrt(np.diag(ret["cov"]))
        ret["z"] = (ret["corrected"] - ret["expected"]) / ret["stddev"]
        return ret

    def folded_pairwise_diffs_df(self):
        allele1 = []
        allele2 = []
        folded_idxs = []
        p_missing = []
        for i, j, p in zip(
                self.melted_der_idx, self.melted_anc_idx,
                self.melted_pairwise_missing):
            if i > j:
                continue
            allele1.append(self.sampled_pops[i])
            allele2.append(self.sampled_pops[j])
            folded_idxs.append((self.unmelted_idxs[(i,j)],
                                self.unmelted_idxs[(j,i)]))
            p_missing.append(p)
        mat = np.zeros((len(folded_idxs), len(self.unmelted_idxs)))
        for k, (i, j) in enumerate(folded_idxs):
            mat[k, i] += 1
            mat[k, j] += 1

        stats = self.linear_stats(mat)
        return pd.DataFrame(co.OrderedDict([
            ("Allele1", allele1),
            ("Allele2", allele2),
            ("AvgDiffs", stats["observed"]),
            ("EstMissingProb", p_missing),
            ("CorrectedAvgDiffs", stats["corrected"]),
            ("Expected", stats["expected"]),
            ("StdDev", stats["stddev"]),
            ("ZScore", stats["z"])
        ]))

    def excess_het_df(self):
        allele1 = []
        allele2 = []
        folded_idxs = []
        p_missing = []
        avg_diffs = []
        n_der_1 = []
        n_der_2 = []
        for idx in range(len(self.melted_der_idx)):
            i = self.melted_der_idx[idx]
            j = self.melted_anc_idx[idx]
            p = self.melted_pairwise_missing[idx]

            if i >= j:
                continue
            allele1.append(self.sampled_pops[i])
            allele2.append(self.sampled_pops[j])

            idx1 = self.unmelted_idxs[(i,j)]
            idx2 = self.unmelted_idxs[(j,i)]
            assert idx1 == idx
            folded_idxs.append((idx1, idx2))
            p_missing.append(p)

            n_der_1.append(self.melted_sum[idx1])
            n_der_2.append(self.melted_sum[idx2])
            avg_diffs.append(n_der_1[-1] + n_der_2[-1])
        mat = np.zeros((len(folded_idxs), len(self.unmelted_idxs)))
        for k, (i, j) in enumerate(folded_idxs):
            mat[k, i] = 1
            mat[k, j] = -1

        stats = self.linear_stats(mat)
        return pd.DataFrame(co.OrderedDict([
            ("Allele1", allele1),
            ("Allele2", allele2),
            ("AvgHet", avg_diffs),
            ("Derived1", n_der_1),
            ("Derived2", n_der_2),
            ("Difference", stats["observed"]),
            ("EstMissingProb", p_missing),
            ("CorrectedDifference", stats["corrected"]),
            ("Expected", stats["expected"]),
            ("StdDev", stats["stddev"]),
            ("ZScore", stats["z"])
        ]))


    def pairwise_diffs_df(self):
        der = self.melted_der_pop
        anc = self.melted_anc_pop

        stats = self.linear_stats()

        ret = list(zip(
            der, anc, stats["observed"],
            self.melted_pairwise_missing,
            stats["corrected"],
            stats["expected"],
            stats["stddev"], stats["z"]))
        ret = sorted(ret, key=lambda x: np.abs(x[-1]), reverse=True)

        return pd.DataFrame(ret, columns=[
            "Derived", "Ancestral", "Observed",
            "EstMissingProb", "CorrectedObs", "Expected",
            "StdDev", "ZScore"])

