import json
import autograd as ag
import autograd.numpy as np
import scipy, scipy.stats
import logging
import collections as co
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn
from .data.config_array import config_array
from .demography import demographic_history, Demography
from .likelihood import SfsLikelihoodSurface
from .compute_sfs import expected_total_branch_len, expected_sfs
from .confidence_region import _ConfidenceRegion
from .events import LeafEvent, SizeEvent, JoinEvent, PulseEvent, GrowthEvent
from .events import Parameter, ParamsDict
from .events import _build_demo_graph
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
        N_e=default_N, gen_time=gen_time, parameters={},
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
        self.parameters = co.OrderedDict((k, p.copy()) for k, p in parameters.items())
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

    def draw(self, pop_x_positions, additional_times=None, tree_only=False, rad=-.1, legend_kwargs={}, xlab_rotation=-30, x_leafs_only=False, pop_marker_kwargs=None, adjust_pulse_labels={}, add_to_existing=None, cm_scalar_mappable=None, alpha=1.0, **kwargs):
        if x_leafs_only:
            exclude_xlabs = [p for p in pop_x_positions
                             if p not in self.leafs]
        else:
            exclude_xlabs = []
        if add_to_existing is None:
            ax = None
            min_N = None
            no_ticks_legend=False
        else:
            ax = add_to_existing.ax
            min_N = add_to_existing.min_N
            no_ticks_legend=True

        if additional_times is None:
            additional_times = []

        demo_plt = self._demo_plotter(
            additional_times, pop_x_positions,
            legend_kwargs=legend_kwargs, xlab_rotation=xlab_rotation,
            exclude_xlabs=exclude_xlabs,
            pop_marker_kwargs=pop_marker_kwargs,
            adjust_pulse_labels=adjust_pulse_labels,
            ax=ax, min_N=min_N, cm_scalar_mappable=cm_scalar_mappable, alpha=alpha, **kwargs)
        demo_plt.draw(tree_only=tree_only, rad=rad, no_ticks_legend=no_ticks_legend)
        return demo_plt

    def draw_with_bootstraps(self, bootstrap_x, pop_x_positions,
                             linthreshy=None, minor_yticks=None, major_yticks=None,
                             p_min=0, p_max=1, p_cmap=plt.cm.hot,
                             p_rad=.2, p_rad_rand=True,
                             additional_times=[], factr=2):
        cm_scalar_mappable = plt.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=p_min,vmax=p_max), cmap=p_cmap)

        demo_plt = self.draw(pop_x_positions, additional_times=additional_times,
                             x_leafs_only=True, tree_only=True, alpha=0,
                             cm_scalar_mappable=cm_scalar_mappable)

        if linthreshy:
            demo_plt.ax.set_yscale('symlog', linthreshy=linthreshy)
        if minor_yticks:
            demo_plt.ax.set_yticks(minor_yticks, minor=True)
        if major_yticks:
            demo_plt.ax.set_yticks(major_yticks)
        if linthreshy:
            demo_plt.ax.get_yaxis().set_major_formatter(
                mpl.ticker.LogFormatterSciNotation(labelOnlyBase=False,
                                                   minor_thresholds=(float("inf"),float("inf")),
                                                   linthresh=linthreshy))

        curr_x = self.get_x()
        try:
            for i, x in enumerate(bootstrap_x):
                logging.info("Adding {}-th bootstrap to plot".format(i))
                self.set_x(x)
                rad = p_rad
                if p_rad_rand:
                    rad *= np.random.uniform()
                self.draw(pop_x_positions, additional_times=additional_times,
                          add_to_existing=demo_plt,
                          cm_scalar_mappable=cm_scalar_mappable,
                          rad=rad*2*np.random.uniform(),
                          alpha=np.min([1.0, factr/len(bootstrap_x)]),
                          pop_line_color="gray", plot_leafs=False)
        except:
            self.set_x(curr_x)
            raise
        self.set_x(curr_x)
        self.draw(pop_x_positions, additional_times=additional_times,
                  add_to_existing=demo_plt,
                  pulse_line_color='black',
                  cm_scalar_mappable=cm_scalar_mappable,
                  plot_pulse_nums=True, rad=rad)

    def _demo_plotter(self, additional_times, pop_x_positions, **kwargs):
        try:
            pop_x_positions.items()
        except:
            pop_x_positions = dict(zip(
                pop_x_positions, range(len(pop_x_positions))))

        params_dict = self.get_params()
        return DemographyPlotter(
            params_dict, self.N_e,
            sorted(list(self.leaf_events) +
                   list(self.size_events) +
                   list(self.topology_events),
                   key=lambda e: e.t(params_dict)),
            additional_times, pop_x_positions, **kwargs)

    def get_parameter(self, name=None, scaled=False):
        pass

    def set_parameter(self, x, name=None, scaled=False):
        pass

    def add_parameter(self, name, start_value=None,
                      scaled_lower=None,
                      scaled_upper=None,
                      scale_transform=None,
                      unscale_transform=None,
                      rgen=None):
        self._conf_region = None

        assert (scale_transform is None) == (unscale_transform is None)
        if scale_transform is None:
            unscale_transform = scale_transform = lambda x, p: x

        curr_params = self.get_params()
        x_lower = scaled_lower
        x_upper = scaled_upper

        if rgen is None and start_value is None:
            raise ValueError("At least one of rgen, start_value must be specified")
        elif rgen is None:
            def rgen(params):
                return start_value
        elif start_value is None:
            start_value = rgen(curr_params)

        x0 = scale_transform(start_value, curr_params)

        param = Parameter(
            name, x0,
            transform_x=unscale_transform,
            inv_transform_x=scale_transform,
            x_bounds=(x_lower, x_upper),
            rgen=rgen)
        assert name not in self.parameters
        self.parameters[name] = param

    def add_size_param(self, name, N0=None, lower=1, upper=1e10, rgen=None):
        if rgen is None:
            scale = self.N_e
            truncexpon = scipy.stats.truncexpon(b=(upper-lower)/scale,
                                                loc=lower, scale=scale)
            def rgen(params):
                return truncexpon.rvs()

        def scale_transform(x, p):
            return np.log(x)

        self.add_parameter(name, N0,
                           scaled_lower=scale_transform(lower, None),
                           scaled_upper=scale_transform(upper, None),
                           scale_transform=scale_transform,
                           unscale_transform=lambda x, p: np.exp(x),
                           rgen=rgen)

    def add_time_param(self, name, t0=None,
                       lower=0.0, upper=None,
                       lower_constraints=[], upper_constraints=[],
                       rgen=None):
        def lower_bound(params):
            constraints = [params[k] for k in lower_constraints]
            constraints.append(lower)
            return np.max(np.array(constraints))

        def upper_bound(params):
            constraints = [params[k] for k in upper_constraints]
            if upper is not None:
                constraints.append(upper)
            return np.min(np.array(constraints))

        has_upper = (len(upper_constraints) > 0) or (upper is not None)

        def scale_transform(t, params):
            l = lower_bound(params)
            if t < l:
                raise ValueError(
                    "t = {} < {} = max({})".format(
                        t, l, [lower] + list(lower_constraints)))
            if has_upper:
                u = upper_bound(params)
                if t > u:
                    raise ValueError(
                        "t = {} > {} = min({})".format(
                            t, u, [upper] + list(upper_constraints)))

                x = (t - l) / (u - l)
                x = np.log(x/(1.-x))
                return x
            else:
                x = t - l
                return x

        def unscale_transform(x, params):
            l = lower_bound(params)
            if has_upper:
                x = 1./(1.+np.exp(-x))
                u = upper_bound(params)
                return (1-x)*l + x*u
            else:
                return l + x

        if rgen is None:
            scale = self.N_e * self.gen_time  # average time to coalescence

            def rgen(params):
                l = lower_bound(params)
                if has_upper:
                    u = upper_bound(params)
                    b = (u-l)/scale
                else:
                    b = float("inf")
                truncexpon = scipy.stats.truncexpon(b=b, loc=l, scale=scale)
                return truncexpon.rvs()

        if has_upper:
            self.add_parameter(
                name, t0,
                scale_transform=scale_transform,
                unscale_transform=unscale_transform,
                rgen=rgen)
        else:
            self.add_parameter(
                name, t0,
                scaled_lower=1e-16,  # prevent optimizer from having a slightly negative number
                scale_transform=scale_transform,
                unscale_transform=unscale_transform,
                rgen=rgen)

    def add_pulse_param(self, name, p0=None, lower=0.0, upper=1.0, rgen=None):
        if rgen is None:
            def rgen(params):
                return np.random.uniform(lower, upper)

        def scale_transform(x, params):
            # FIXME gives divide by 0 warning when x=0
            return np.log(x/np.array(1-x))

        self.add_parameter(name, p0,
                           scaled_lower=scale_transform(lower, None),
                           scaled_upper=scale_transform(upper, None),
                           scale_transform=scale_transform,
                           unscale_transform=lambda x, params: 1/(1+np.exp(-x)),
                           rgen=rgen)

    def add_growth_param(self, name, g0=None, lower=-.001, upper=.001, rgen=None):
        if rgen is None:
            def rgen(params):
                return np.random.uniform(lower, upper)
        self.add_parameter(name, g0, scaled_lower=lower, scaled_upper=upper,
                           rgen=rgen)

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
        for param in self.parameters.values():
            param.update_params_dict(params_dict)
        # TODO return a pd.Series?
        return params_dict

    def _get_params_opt_x_jacobian(self):
        def fun(opt_x):
            x = self._x_from_opt_x(opt_x)
            params_dict = ParamsDict()
            for x_i, param in zip(
                    x, self.parameters.values()):
                param.update_params_dict(params_dict, x_i)
            return np.array(list(params_dict.values()))
        return ag.jacobian(fun)(self._get_opt_x())

    def get_x(self, param=None):
        """
        Return the current value of x (the untransformed parameters).
        """
        if param is None:
            return np.array([
                p.x for p in self.parameters.values()])
        else:
            for p in self.parameters.values():
                if p.name == param:
                    return p.x
            raise ValueError("Unrecognized parameter {}".format(param))

    def set_random_parameters(self):
        params_dict = ParamsDict()
        for param in self.parameters.values():
            param.resample(params_dict)
            param.update_params_dict(params_dict)

    def set_x(self, x, param=None):
        """
        Set the value of x (the untransformed parameters).
        """
        if param is None:
            if len(x) != len(self.parameters):
                raise ValueError(
                    "len(x) != {}".format(len(self.parameters)))

            for p_i, x_i in zip(self.parameters.values(), x):
                p_i.x = x_i
        else:
            for p in self.parameters.values():
                if p.name == param:
                    p.x = x
                    return
            raise ValueError("Unrecognized parameter {}".format(param))

    # TODO: note these are PER-GENERATION mutation/recombination rates...
    def simulate_data(self, length, recombination_rate,
                      mutation_rate, num_replicates,
                      sampled_n_dict=None, **kwargs):
        demo = self._get_demo(sampled_n_dict)
        return demo.simulate_data(length=length,
                                  recombination_rate=4*self.N_e*recombination_rate,
                                  mutation_rate=4*self.N_e*mutation_rate,
                                  num_replicates=num_replicates,
                                  **kwargs)

    def simulate_vcf(self, outfile, mutation_rate, recombination_rate,
                     length, chrom_names=[1], ploidy=1, random_seed=None,
                     sampled_n_dict=None, **kwargs):
        demo = self._get_demo(sampled_n_dict)
        return demo.simulate_vcf(outfile=outfile,
                                 mutation_rate=4*self.N_e*mutation_rate,
                                 recombination_rate=4*self.N_e*recombination_rate,
                                 length=length, chrom_names=chrom_names,
                                 ploidy=ploidy, random_seed=random_seed,
                                 **kwargs)

    def fstats(self, sampled_n_dict=None):
        demo = self._get_demo(sampled_n_dict)
        sfs = self._get_sfs()

        ascertainment_pops = [
            pop for pop, is_asc in zip(
                sfs.sampled_pops, sfs.ascertainment_pop)
            if is_asc]

        return ModelFitFstats(
            sfs, demo,
            ascertainment_pops)

    def _get_demo(self, sampled_n_dict):
        sampled_n_dict = self._get_sample_sizes(sampled_n_dict)

        params_dict = self.get_params()

        events = []
        for sub_events in (self.leaf_events,
                           self.size_events,
                           self.topology_events):
            for e in sub_events:
                events.append(e)

        events = sorted(events, key=lambda e: e.t(params_dict))
        G = _build_demo_graph(events, sampled_n_dict, params_dict, default_N=1.0)
        demo = Demography(G)

        def printable_params():
            for k, v in params_dict.items():
                v = str(v).replace('\n', '')
                yield (k, v)
        logging.debug("Demographic parameters = {}".format(
            co.OrderedDict(printable_params())))

        return demo

    def _get_opt_x(self):
        return np.array([p.inv_opt_trans(p.x)
                         for p in self.parameters.values()])

    def _x_from_opt_x(self, opt_x):
        return np.array([
            p.opt_trans(ox)
            for p, ox in zip(
                    self.parameters.values(), opt_x)])

    def _opt_x_from_x(self, x):
        return np.array([
            p.inv_opt_trans(x_i)
            for p, x_i in zip(self.parameters.values(), x)
        ])

    def _opt_demo_fun(self, *opt_x):
        opt_x = np.array(opt_x)
        logging.debug("opt_x = {}".format(str(opt_x)))
        x = self._x_from_opt_x(opt_x)
        logging.debug("x = {}".format(str(x)))
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
            n_blocks_jackknife=None, non_ascertained_pops=None):
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
        if n_blocks_jackknife:
            data = data.chunk_data(n_blocks_jackknife)
        self._set_data(
            data=data,
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
        opt_x = self._get_opt_x()
        if self._conf_region is None or not np.allclose(
                opt_x, self._conf_region.point):
            opt_score = opt_surface._score(opt_x)
            opt_score_cov = opt_surface._score_cov(opt_x)
            opt_fisher = opt_surface._fisher(opt_x)

            self._conf_region = _ConfidenceRegion(
                opt_x, opt_score, opt_score_cov, opt_fisher,
                lik_fun=opt_surface.log_lik,
                psd_rtol=1e-4)
        return self._conf_region

    def marginal_wald(self):
        marginal_wald_df = co.OrderedDict()
        marginal_wald_df["Param"] = [
            p.name for p in self.parameters.values()]
        marginal_wald_df["Value"] = list(
            self.get_params().values())
        marginal_wald_df["StdDev"] = np.sqrt(np.diag(self.mle_cov()))
        return pd.DataFrame(marginal_wald_df)

    def mle_cov(self, x_scale=False):
        # use delta method
        G = self._get_conf_region().godambe(inverse=True)
        if x_scale:
            return G
        else:
            dp_do = self._get_params_opt_x_jacobian()
            return np.dot(dp_do, np.dot(G, dp_do.T))

    def test(self, null_point=None, sims=int(1e3), test_type="ratio",
             alt_point=None, *args, **kwargs):
        if null_point is None:
            null_point = self.get_x()
        null_point = self._opt_x_from_x(null_point)
        if alt_point is not None:
            alt_point = self._opt_x_from_x(alt_point)
        return self._get_conf_region().test(
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

    # TODO note these are in PER-GENERATION units
    # TODO allow to pass folded parameter (if passing separate configs?)
    def expected_sfs(self, configs=None, normalized=False):
        configs = self._get_configs(configs)
        demo = self._get_demo(dict(zip(configs.sampled_pops,
                                       configs.sampled_n)))
        ret = expected_sfs(demo, configs, normalized=normalized, folded=self._folded) * self.N_e * 4.0
        return co.OrderedDict(zip(configs.as_tuple(), ret))

    def _get_configs(self, configs):
        if configs is not None:
            return config_array(self.leafs, configs)

        if self._data is None:
            raise ValueError(
                "Need to call set_data() or provide configs")

        sfs = self._get_sfs()
        return sfs.configs

    # TODO note these are in PER-GENERATION units
    def expected_branchlen(self, sampled_n_dict=None):
        demo = self._get_demo(sampled_n_dict)
        return expected_total_branch_len(demo) * self.N_e * 4.0

    def _get_sample_sizes(self, sampled_n_dict):
        if sampled_n_dict is not None:
            sampled_pops_set = set(sampled_n_dict.keys())
            leaf_set = set(self.leafs)
            if not sampled_pops_set <= leaf_set:
                raise ValueError("{} not in leaf populations".format(
                    sampled_pops_set - leaf_set))
            # make sure it is sorted in correct order
            return co.OrderedDict((k, sampled_n_dict[k]) for k in self.leafs
                                  if k in sampled_pops_set)
        if self._data is None:
            raise ValueError(
                "Need to call set_data() or provide dict of sample sizes.")
        sfs = self._get_sfs()
        return co.OrderedDict(zip(sfs.sampled_pops, sfs.sampled_n))

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

    def pairwise_diffs(self, exclude_pops=[],
                       exclude_singletons=False, plot=True):
        pops = [p for p in self.leafs if p not in exclude_pops]
        fstats = self.fstats(sampled_n_dict={
            p: 1 for p in pops})

        if exclude_singletons:
            s_probs = fstats.singleton_probs(pops)

        df = []
        for pop1 in pops:
            for pop2 in pops:
                if pop1 < pop2:
                    prob = fstats.ordered_prob({
                        pop1: [1], pop2: [0]}, fold=True)
                    if exclude_singletons:
                        prob = (
                            prob - s_probs["probs"][pop1] -
                            s_probs["probs"][pop2]) / s_probs[
                                "denom"]

                    z = prob.z_score
                    penalty = np.log(prob.observed / prob.expected)
                    line = [pop1, pop2, penalty, prob.z_score]
                    print(*line)
                    df.append(line)
        ret = pd.DataFrame(sorted(df, key=lambda x: abs(x[-1]),
                                   reverse=True),
                            columns=["Pop1", "Pop2", "Penalty", "Z"])
        if plot:
            pivoted = ret.pivot(index="Pop1", columns="Pop2",
                                values="Z")
            plt.gcf().clear()
            seaborn.heatmap(pivoted, annot=True, center=0)
            plt.title("Residual (Observed-Expected) Z-scores")
            pass
        return ret



    def stochastic_optimize(
            self, n_minibatches=None, snps_per_minibatch=None,
            rgen=np.random, printfreq=1, start_from_checkpoint=None,
            **kwargs):
        def callback(opt_x):
            self.set_x(self._x_from_opt_x(opt_x))
            if opt_x.iteration % printfreq == 0:
                msg = [("it", opt_x.iteration), ("LogLikelihood", -opt_x.fun)]
                msg.extend(list(self.get_params().items()))
                msg = ", ".join(["{}: {}".format(k, v) for k, v in msg])
                logging.info("{" + msg + "}")

        bounds = [p.opt_x_bounds
                  for p in self.parameters.values()]
        if all([b is None for bnd in bounds for b in bnd]):
            bounds = None

        kwargs = dict(kwargs)
        kwargs["callback"] = callback
        kwargs["bounds"] = bounds


        if start_from_checkpoint:
            with open(start_from_checkpoint) as f:
                kwargs.update(json.load(f))
        else:
            kwargs["x0"] = self._get_opt_x()

        res = self._get_opt_surface()._stochastic_surfaces(
            n_minibatches=n_minibatches,
            snps_per_minibatch=snps_per_minibatch,
            rgen=rgen).find_mle(**kwargs)

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
        bounds = [p.opt_x_bounds
                  for p in self.parameters.values()]
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
        res["parameters"] = pd.Series(self.get_params())
        return res
