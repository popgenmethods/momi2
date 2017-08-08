import autograd as ag
import autograd.numpy as np
import logging
import collections as co
import pandas as pd
from .demography import demographic_history
from .likelihood import SfsLikelihoodSurface
from .confidence_region import _ConfidenceRegion
from .events import LeafEvent, SizeEvent, JoinEvent, PulseEvent, GrowthEvent
from .events import Parameter, ParamsDict
from .demo_plotter import DemographyPlotter
from .fstats import ModelFitFstats


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
        topology_events=[], size_events=[],
        leaf_events=[], leafs=[],
        data=None, muts_per_gen=None, folded=None,
        mem_chunk_size=None, use_pairwise_diffs=None,
        non_ascertained_pops=None)


class DemographicModel(object):
    def __init__(self, N_e, gen_time, parameters,
                 topology_events, size_events,
                 leaf_events, leafs,
                 data, muts_per_gen, folded,
                 mem_chunk_size, use_pairwise_diffs,
                 non_ascertained_pops):
        self.N_e = N_e
        self.gen_time = gen_time
        self.parameters = [p.copy() for p in parameters]
        self.topology_events = list(topology_events)
        self.size_events = list(size_events)
        self.leaf_events = list(leaf_events)
        self.leafs = list(leafs)

        self._set_data(data=data, muts_per_gen=muts_per_gen,
                       folded=folded, mem_chunk_size=mem_chunk_size,
                       use_pairwise_diffs=use_pairwise_diffs,
                       non_ascertained_pops=non_ascertained_pops)

    def copy(self):
        return DemographicModel(
            N_e=self.N_e, gen_time=self.gen_time,
            parameters=self.parameters,
            topology_events=self.topology_events,
            size_events=self.size_events,
            leaf_events=self.leaf_events, leafs=self.leafs,
            data=self._data, muts_per_gen=self._muts_per_gen,
            folded=self._folded,
            mem_chunk_size=self._mem_chunk_size,
            use_pairwise_diffs=self._use_pairwise_diffs,
            non_ascertained_pops=self._non_ascertained_pops)

    def draw(self, additional_times, pop_x_positions):
        demo_plt = self._demo_plotter(additional_times, pop_x_positions)
        demo_plt.draw()
        return demo_plt

    def _demo_plotter(self, additional_times, pop_x_positions):
        try:
            pop_x_positions.items()
        except:
            pop_x_positions = dict(zip(pop_x_positions,
                                       range(len(pop_x_positions))))

        params_dict = self.get_params()
        return DemographyPlotter(
            params_dict, self.N_e,
            sorted(list(self.leaf_events) +
                   list(self.size_events) +
                   list(self.topology_events),
                   key=lambda e: e.t(params_dict)),
            additional_times, pop_x_positions)

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
            and a second argument p, which contains all
            the previously added parameters as attributes.

            for example, if we are adding parameter t1,
            and want to ensure it is larger than the previously
            added parameter t2, we can use transform_x to add
            on the value of t2:

            model.add_param("t1", ..., transform_x=lambda x,p: x+p.t2)

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
            def transform_x(x, p):
                return x

        bounds = (lower_x, upper_x)
        if opt_scale is None:
            if lower_x is None:
                lower_x = -float("inf")
            if upper_x is None:
                upper_x = float("inf")

            if lower_x > 0 and upper_x < 1:
                opt_scale = "logit"
            elif lower_x > 0:
                opt_scale = "log"
            else:
                opt_scale = "linear"

        if opt_scale == "logit":
            def opt_trans(x):
                return 1./(1.+np.exp(-x))

            def inv_opt_trans(p):
                return np.log(p/(1.-p))
        elif opt_scale == "log":
            opt_trans = np.exp
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

    def add_leaf(self, pop, t=0, N=None, g=None):
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

        self.leaf_events.append(LeafEvent(
            t, pop, self.N_e, self.gen_time))

        if N is not None:
            self.set_size(pop, t, N=N)
        if g is not None:
            self.set_size(pop, t, g=g)

    def move_lineages(self, pop1, pop2, t, p=1, N=None, g=None):
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
            self.topology_events.append((JoinEvent(
                t, pop1, pop2, self.N_e, self.gen_time)))
        else:
            self.topology_events.append(PulseEvent(
                t, p, pop1, pop2, self.N_e,
                self.gen_time))

        if N is not None:
            self.set_size(pop2, t, N=N)
        if g is not None:
            self.set_size(pop2, t, g=g)

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
            self.size_events.append(SizeEvent(
                t, N, pop, self.N_e, self.gen_time))
        if g != 0:
            self.size_events.append(GrowthEvent(
                t, g, pop, self.N_e, self.gen_time))

    def get_params_df(self):
        return pd.DataFrame(list(self.get_params().items()),
                            columns=["Param", "Value"])

    def get_params(self):
        """
        Return a dictionary with the current parameter
        values.
        """
        params_dict = ParamsDict()
        for param in self.parameters:
            param.update_params_dict(params_dict)
        return params_dict

    def _get_params_opt_x_jacobian(self):
        def fun(opt_x):
            x = self._x_from_opt_x(opt_x)
            params_dict = ParamsDict()
            for x_i, param in zip(x, self.parameters):
                param.update_params_dict(params_dict, x_i)
            return np.array(list(params_dict.values()))
        return ag.jacobian(fun)(self._get_opt_x())

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
            raise ValueError("Unrecognized parameter {}".format(param))

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
        demo = demo._get_multipop_moran(
            self.leafs, [sampled_n_dict[k] for k in self.leafs])
        return demo.simulate_data(length=length,
                                  recombination_rate=4*self.N_e*recombination_rate,
                                  mutation_rate=4*self.N_e*mutation_rate,
                                  num_replicates=num_replicates)

    def fstats(self, sampled_n_dict=None):
        sfs = self._get_sfs()

        if not sampled_n_dict:
            sampled_n_dict = dict(zip(sfs.sampled_pops,
                                      sfs.sampled_n))
        if not (set(sampled_n_dict.keys()) <= set(sfs.sampled_pops)):
            raise ValueError("{} not in leaf populations".format(
                set(sampled_n_dict.keys()) - set(sfs.sampled_pops)))

        ascertainment_pops = [
            pop for pop, is_asc in zip(
                sfs.sampled_pops, sfs.ascertainment_pop)
            if is_asc]

        return ModelFitFstats(
            sfs, self._get_demo(),
            ascertainment_pops, sampled_n_dict)

    def _get_demo(self):
        params_dict = self.get_params()

        events = []
        for eventlist in (self.size_events,
                          self.topology_events):
            for e in eventlist:
                events.extend(e.oldstyle_event(params_dict))

        archaic_times_dict = {}
        for e in self.leaf_events:
            archaic_times_dict[e.pop] = e.t(params_dict)

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
        return np.array([
            p.opt_trans(ox)
            for p, ox in zip(self.parameters, opt_x)])

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
            n_blocks_jackknife=100, non_ascertained_pops=None):
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
            else, if there is no missing data, use total number of mutations
        """
        self._set_data(
            data=data.chunk_data(n_blocks_jackknife),
            muts_per_gen=muts_per_gen, folded=folded,
            mem_chunk_size=mem_chunk_size,
            use_pairwise_diffs=use_pairwise_diffs,
            non_ascertained_pops=non_ascertained_pops)

    def _set_data(self, data, muts_per_gen, folded,
                  mem_chunk_size, use_pairwise_diffs, non_ascertained_pops):
        self._opt_surface = None
        self._conf_region = None
        self._sfs = None
        self._data = data
        self._folded = folded
        self._mem_chunk_size = mem_chunk_size
        self._muts_per_gen = muts_per_gen
        self._use_pairwise_diffs = use_pairwise_diffs
        self._non_ascertained_pops = non_ascertained_pops

    def _get_sfs(self):
        if self._sfs is None or list(
                self._sfs.sampled_pops) != list(self.leafs):
            self._sfs = self._data.subset_populations(
                self.leafs,
                non_ascertained_pops=self._non_ascertained_pops).sfs
        return self._sfs

    def _get_opt_surface(self):
        if self._opt_surface is None or list(
                self._opt_surface.data.sampled_pops) != list(self.leafs):
            self._conf_region = None
            if self._data is None:
                raise ValueError("Need to call DemographicModel.set_data()")
            logging.info("Constructing likelihood surface...")

            sfs = self._get_sfs()
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
        marginal_wald_df["StdDev"] = np.sqrt(np.diag(self.mle_cov()))
        return pd.DataFrame(marginal_wald_df)

    def mle_cov(self):
        # use delta method
        G = self._get_conf_region().godambe(inverse=True)
        dp_do = self._get_params_opt_x_jacobian()
        return np.dot(dp_do, np.dot(G, dp_do.T))

    def test(self, null_point=None, sims=int(1e3), test_type="ratio",
             alt_point=None, *args, **kwargs):
        if null_point is None:
            null_point = self.get_x()
        null_point = self._opt_x_from_x(null_point)
        if alt_point is not None:
            alt_point = self._opt_x_from_x(alt_point)
        return self._get_conf_region.test(
            null_point=null_point, sims=sims,
            test_type=test_type, alt_point=alt_point, *args, **kwargs)

    def _make_surface(self, sfs, opt_surface):
        use_pairwise_diffs = self._use_pairwise_diffs
        if use_pairwise_diffs is None:
            use_pairwise_diffs = True

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
            p_missing=p_miss)

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

    def stochastic_optimize(self, n_minibatches=None, snps_per_minibatch=None, rgen=np.random,
                            printfreq=1, **kwargs):
        def callback(opt_x):
            self.set_x(self._x_from_opt_x(opt_x))
            if opt_x.iteration % printfreq == 0:
                msg = [("it", opt_x.iteration), ("LogLikelihood", -opt_x.fun)]
                msg.extend(list(self.get_params().items()))
                msg = ", ".join(["{}: {}".format(k, v) for k, v in msg])
                logging.info("{" + msg + "}")

        bounds = [p.opt_x_bounds for p in self.parameters]
        if all([b is None for bnd in bounds for b in bnd]):
            bounds = None

        kwargs = dict(kwargs)
        kwargs["callback"] = callback
        kwargs["bounds"] = bounds

        res = self._get_opt_surface().stochastic_surfaces(
            n_minibatches=n_minibatches, snps_per_minibatch=snps_per_minibatch,
            rgen=rgen).find_mle(self._get_opt_x(), **kwargs)

        res.x = self._x_from_opt_x(res.x)
        self.set_x(res.x)
        return res

    def optimize(self, method="tnc", jac=True,
                 hess=False, hessp=False, printfreq=1, **kwargs):
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

        def callback(opt_x):
            self.set_x(self._x_from_opt_x(opt_x))
            if opt_x.iteration % printfreq == 0:
                msg = [("it", opt_x.iteration), ("KLDivergence", opt_x.fun)]
                msg.extend(list(self.get_params().items()))
                msg = ", ".join(["{}: {}".format(k, v) for k, v in msg])
                logging.info("{" + msg + "}")

        res = self._get_opt_surface().find_mle(
            self._get_opt_x(), method=method,
            jac=jac, hess=hess, hessp=hessp,
            bounds=bounds, callback=callback,
            **kwargs)

        res.x = self._x_from_opt_x(res.x)
        self.set_x(res.x)
        return res
