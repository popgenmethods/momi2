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
from .data.sfs import Sfs
from .demography import Demography
from .likelihood import SfsLikelihoodSurface
from .compute_sfs import expected_total_branch_len, expected_sfs
from .confidence_region import _ConfidenceRegion
from .events import LeafEvent, SizeEvent, JoinEvent, PulseEvent, GrowthEvent
from .events import Parameter, ParamsDict
from .events import _build_demo_graph
from .demo_plotter import DemographyPlotter
from .sfs_stats import ModelFitStats


class DemographicModel(object):
    """Object for representing and inferring a demographic history.

    :param float N_e: the population size, unless manually changed by \
    :meth:`DemographicModel.set_size()`

    :param float gen_time: units of time per generation. \
    For example, if you wish to specify time in years, and a \
    generation is 29 years, set this to 29. Default value is 1.
    """
    def __init__(self, N_e, gen_time=1, muts_per_gen=None):
        self.N_e = N_e
        self.gen_time = gen_time
        self.muts_per_gen = muts_per_gen

        self.parameters = co.OrderedDict()
        self.topology_events = []
        self.size_events = []
        self.leaf_events = []
        self.leafs = []

        self._set_data(sfs=None, length=None,
                       mem_chunk_size=None,
                       use_pairwise_diffs=None,
                       non_ascertained_pops=None)

    def copy(self):
        ret = DemographicModel(self.N_e, self.gen_time,
                               self.muts_per_gen)
        for k, v in self.parameters.items():
            ret.parameters[k] = v.copy()
        ret.topology_events.extend(self.topology_events)
        ret.size_events.extend(self.size_events)
        ret.leaf_events.extend(self.leaf_events)
        ret.leafs.extend(self.leafs)
        ret._set_data(sfs=self._fullsfs, length=self._length,
                      mem_chunk_size=self._mem_chunk_size,
                      use_pairwise_diffs=self._use_pairwise_diffs,
                      non_ascertained_pops=self._non_ascertained_pops)
        return ret

    def set_params(self, new_params=None, randomize=False,
                   scaled=False):
        """Set the current parameter values

        :param dict/list new_params: dict mapping parameter names to \
        new values, or list of new values whose length is the current \
        number of parameters

        :param bool randomize: if True, parameters not in \
        ``new_params`` get randomly sampled new values

        :param bool scaled: if True, values in ``new_params`` have \
        been pre-scaled according to the internal representation used \
        by momi during optimization \
        (see also: :meth:`DemographicModel.add_parameter`)
        """
        if new_params is None:
            new_params = {}

        try:
            new_params.items()
        except AttributeError:
            if len(new_params) != len(self.parameters):
                raise ValueError("New parameters should either be a dict, or be a list with length current number of parameters")
            new_params = dict(zip(self.parameters.keys(), new_params))

        new_params = dict(new_params)
        curr_params = ParamsDict()
        for name, param in self.parameters.items():
            try:
                val = new_params.pop(name)
            except KeyError:
                if randomize:
                    param.resample(curr_params)
            else:
                if scaled:
                    param.x = val
                else:
                    param.x = param.inv_transform_x(val, curr_params)

            param.update_params_dict(curr_params)

        if new_params:
            raise ValueError("Unrecognized parameters: {}".format(
                list(new_params.keys())))

    def add_parameter(self, name, start_value=None,
                      rgen=None,
                      scale_transform=None,
                      unscale_transform=None,
                      scaled_lower=None,
                      scaled_upper=None):
        """Low-level method to add a new parameter. Most users should instead call
        :meth:`DemographicModel.add_size_param`,
        :meth:`DemographicModel.add_pulse_param`,
        :meth:`DemographicModel.add_time_param`, or
        :meth:`DemographicModel.add_growth_param`.

        In order for internal gradients to be correct, ``scale_transform`` and ``unscale_transform`` should be constructed using the ``autograd`` package (see `tutorial <https://github.com/HIPS/autograd/blob/master/docs/tutorial.md>`_).

        :param str name: Name of the parameter.
        :param float start_value: Starting value. If None, use \
        ``rgen`` to sample a random starting value.

        :param function rgen: Function to get a random starting value.
        :param function scale_transform: Function for internally \
        transforming and rescaling the parameter during optimization.

        :param function unscale_transform: Inverse function of \
        ``scale_transform``
        :param float scaled_lower: Lower bound after scaling by \
        ``scale_transform``
        :param float scaled_upper: Upper bound after scaling by \
        ``scale_transform``
        """
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
        """Add a size parameter to the demographic model.

        :param str name: Parameter name
        :param float N0: Starting value. If None, use ``rgen`` to \
        randomly sample
        :param float lower: Lower bound
        :param float upper: Upper bound
        :param function rgen: Function to sample a random starting \
        value. If None, a truncated exponential with rate ``1 / N_e``
        """
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
        """Add a time parameter to the demographic model.

        :param str name: Parameter name
        :param float t0: Starting value. If None, use ``rgen`` to randomly sample
        :param float lower: Lower bound
        :param float upper: Upper bound
        :param list lower_constraints: List of parameter names that \
        are lower bounds
        :param list upper_constraints: List of parameter names that \
        are upper bounds
        :param rgen: Function to sample a random starting value. If \
        None, a truncated exponential with rate \
        ``1 / (N_e * gen_time)`` constrained to satisfy the bounds \
        and constraints.
        """
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
        """Add a pulse parameter to the demographic model.

        :param str name: Parameter name.
        :param float p0: Starting value. If None, randomly sample \
        with ``rgen``
        :param float lower: Lower bound
        :param float upper: Upper bound
        :param function rgen: Function to sample random value. \
        If None, use a uniform distribution.
        """
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

    def add_growth_param(self, name, g0=None, lower=-.001, upper=.001,
                         rgen=None):
        """Add growth rate parameter to the demographic model.

        :param str name: Parameter name
        :param float g0: Starting value. If None, randomly sample \
        with ``rgen``
        :param float lower: Lower bound
        :param float upper: Upper bound
        :param function rgen: Function to sample random value. \
        If None, use uniform distribution
        """
        if rgen is None:
            def rgen(params):
                return np.random.uniform(lower, upper)
        self.add_parameter(name, g0, scaled_lower=lower, scaled_upper=upper,
                           rgen=rgen)

    def add_leaf(self, pop_name, t=0, N=None, g=None):
        """Add a sampled leaf population to the model.

        The arguments t, N, g can be floats or parameter names (strings).

        If N or g are specified, then the size or growth rate are also set at
        the sampling time t. Otherwise they remain at their previous (default)
        values.

        Note that this does not affect the population size and growth below
        time t, which may be an issue if lineages are moved in from other
        populations below t. If you need to set the size or growth below t, use
        :meth:`DemographicModel.set_size()`.

        :param str pop_name: Name of the population
        :param float,str t: Time the population was sampled
        :param float,str N: Population size
        :param float,str g: Population growth rate

        """
        self.leafs.append(pop_name)

        self.leaf_events.append(LeafEvent(
            t, pop_name, self.N_e, self.gen_time))

        if N is not None:
            self.set_size(pop_name, t, N=N)
        if g is not None:
            self.set_size(pop_name, t, g=g)

    def move_lineages(self, pop_from, pop_to, t, p=1, N=None, g=None):
        """Move each lineage in pop_from to pop_to at time t with probability p.

        The arguments t, p, N, g can be floats or parameter names (strings).

        If N or g are specified, then the size or growth rate of pop_to is also
        set at this time, otherwise these parameters remain at their previous
        values.

        :param str pop_from: Population lineages are moved from \
        (backwards in time)
        :param str pop_to: Population lineages are moved to \
        (backwards in time)
        :param float,str t: Time of the event
        :param float,str p: Probability that lineage in pop_from \
        moves to pop_to
        :param float,str N: Population size of pop_to
        :param float,str g: Growth rate of pop_to
        """
        if p == 1:
            self.topology_events.append((JoinEvent(
                t, pop_from, pop_to, self.N_e, self.gen_time)))
        else:
            self.topology_events.append(PulseEvent(
                t, p, pop_from, pop_to, self.N_e,
                self.gen_time))

        if N is not None:
            self.set_size(pop_to, t, N=N)
        if g is not None:
            self.set_size(pop_to, t, g=g)

    def set_size(self, pop_name, t, N=None, g=0):
        """Set population size and/or growth rate at time t.

        The arguments t, N, g can be floats or parameter names (strings).

        If N is not specified then only the growth rate is changed.

        If N is specified and g is not then the growth rate is reset to 0.
        Currently it is not possible to change the size without also setting
        the growth rate.

        :param str pop_name: Population name
        :param float,str t: Time of event
        :param float,str N: Population size
        :param float,str g: Growth rate
        """
        if N is not None:
            self.size_events.append(SizeEvent(
                t, N, pop_name, self.N_e, self.gen_time))
        self.size_events.append(GrowthEvent(
            t, g, pop_name, self.N_e, self.gen_time))

    def get_params(self, scaled=False):
        """Return an ordered dictionary with the current parameter
        values.

        If ``scaled=True``, returns the parameters scaled in the internal
        representation used by momi during optimization (see also:
        :meth:`DemographicModel.add_parameter`)
        """
        params_dict = ParamsDict()
        for param in self.parameters.values():
            if scaled:
                params_dict[param.name] = param.x
            else:
                param.update_params_dict(params_dict)
        return params_dict

    def _get_x(self, param=None):
        if param is None:
            return np.array([
                p.x for p in self.parameters.values()])
        else:
            for p in self.parameters.values():
                if p.name == param:
                    return p.x
            raise ValueError("Unrecognized parameter {}".format(param))

    def _set_x(self, x, param=None):
        if param:
            x = {param: x}
        self.set_params(x, scaled=True)

    def simulate_data(self, length, recoms_per_gen,
                      num_replicates, muts_per_gen=None,
                      sampled_n_dict=None, **kwargs):
        """Simulate data, using msprime as backend

        :param int length: Length of each locus in bases
        :param float recoms_per_gen: Recombination rate per generation per base
        :param float muts_per_gen: Mutation rate per generation per base
        :param int num_replicates: Number of loci to simulate
        :param dict sampled_n_dict: Number of haploids per population. \
        If None, use sample sizes from the current dataset as set by \
        :meth:`DemographicModel.set_data`

        :returns: Dataset of SNP allele counts
        :rtype: :class:`SnpAlleleCounts`
        """
        demo = self._get_demo(sampled_n_dict)
        if muts_per_gen is None:
            if not self.muts_per_gen:
                raise ValueError("Need to provide mutation rate")
            muts_per_gen = self.muts_per_gen
        return demo.simulate_data(
            length=length,
            recombination_rate=4*self.N_e*recoms_per_gen,
            mutation_rate=4*self.N_e*muts_per_gen,
            num_replicates=num_replicates,
            **kwargs)

    def simulate_vcf(
            self, out_prefix,
            recoms_per_gen, length,
            muts_per_gen=None, chrom_name="1",
            ploidy=1, random_seed=None,
            sampled_n_dict=None, **kwargs):
        """Simulate a chromosome using msprime and write it to VCF

        :param str,file outfile: Output VCF file. If a string ending in ".gz", gzip it.
        :param float muts_per_gen: Mutation rate per generation per base
        :param float recoms_per_gen: Recombination rate per generation per base
        :param int length: Length of chromosome in bases
        :param str chrom_name: Name of chromosome
        :param int ploidy: Ploidy
        :param int random_seed: Random seed
        :param dict sampled_n_dict: Number of haploids per population. \
        If None, use sample sizes from the current dataset as set by \
        :meth:`DemographicModel.set_data`
        """
        demo = self._get_demo(sampled_n_dict)
        if muts_per_gen is None:
            if not self.muts_per_gen:
                raise ValueError("Need to provide mutation rate")
            muts_per_gen = self.muts_per_gen
        return demo.simulate_vcf(
            out_prefix=out_prefix,
            mutation_rate=4*self.N_e*muts_per_gen,
            recombination_rate=4*self.N_e*recoms_per_gen,
            length=length, chrom_name=chrom_name,
            ploidy=ploidy, random_seed=random_seed,
            **kwargs)

    # TODO rename to get_multipop_moran?
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
        logging.getLogger(__name__).debug("Demographic parameters = {}".format(
            co.OrderedDict(printable_params())))

        return demo

    def _demo_fun(self, *x):
        logging.getLogger(__name__).debug("x = {}".format(str(x)))
        prev_x = self._get_x()
        try:
            self._set_x(x)
            return self._get_demo(None)
        except:
            raise
        finally:
            self._set_x(prev_x)

    def set_data(
            self, sfs, length=None,
            mem_chunk_size=1000,
            use_pairwise_diffs=True,
            non_ascertained_pops=None):
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
        use_folded_likelihood:
            whether to use the folded SFS when computing likelihood.
            Default is to check the use_folded_likelihood property of the data
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
        if not length:
            length = sfs.length

        self._set_data(
            sfs=sfs, length=length,
            mem_chunk_size=mem_chunk_size,
            use_pairwise_diffs=use_pairwise_diffs,
            non_ascertained_pops=non_ascertained_pops)

    def _set_data(self, sfs, length,
                  mem_chunk_size, use_pairwise_diffs,
                  non_ascertained_pops):
        self._lik_surface = None
        self._conf_region = None
        self._subsfs = None
        self._fullsfs = sfs
        self._length = length
        self._mem_chunk_size = mem_chunk_size
        self._use_pairwise_diffs = use_pairwise_diffs
        self._non_ascertained_pops = non_ascertained_pops

    def _get_sfs(self):
        if self._subsfs is None or list(
                self._subsfs.sampled_pops) != list(self.leafs):
            self._subsfs = self._fullsfs.subset_populations(
                self.leafs,
                non_ascertained_pops=self._non_ascertained_pops).sfs
        return self._subsfs

    #def _get_conf_region(self):
    #    opt_surface = self._get_surface()
    #    opt_x = self._get_opt_x()
    #    if self._conf_region is None or not np.allclose(
    #            opt_x, self._conf_region.point):
    #        opt_score = opt_surface._score(opt_x)
    #        opt_score_cov = opt_surface._score_cov(opt_x)
    #        opt_fisher = opt_surface._fisher(opt_x)

    #        self._conf_region = _ConfidenceRegion(
    #            opt_x, opt_score, opt_score_cov, opt_fisher,
    #            lik_fun=opt_surface.log_lik,
    #            psd_rtol=1e-4)
    #    return self._conf_region

    #def marginal_wald(self):
    #    marginal_wald_df = co.OrderedDict()
    #    marginal_wald_df["Param"] = [
    #        p.name for p in self.parameters.values()]
    #    marginal_wald_df["Value"] = list(
    #        self.get_params().values())
    #    marginal_wald_df["StdDev"] = np.sqrt(np.diag(self.mle_cov()))
    #    return pd.DataFrame(marginal_wald_df)

    #def mle_cov(self, x_scale=False):
    #    # use delta method
    #    G = self._get_conf_region().godambe(inverse=True)
    #    if x_scale:
    #        return G
    #    else:
    #        dp_do = self._get_params_opt_x_jacobian()
    #        return np.dot(dp_do, np.dot(G, dp_do.T))

    #def test(self, null_point=None, sims=int(1e3), test_type="ratio",
    #         alt_point=None, *args, **kwargs):
    #    if null_point is None:
    #        null_point = self._get_x()
    #    null_point = self._opt_x_from_x(null_point)
    #    if alt_point is not None:
    #        alt_point = self._opt_x_from_x(alt_point)
    #    return self._get_conf_region().test(
    #        null_point=null_point, sims=sims,
    #        test_type=test_type, alt_point=alt_point, *args, **kwargs)

    def _get_surface(self):
        if self._lik_surface is not None and (
                list(self._lik_surface.data.sampled_pops) ==
                list(self.leafs)):
            return self._lik_surface

        self._conf_region = None
        if self._fullsfs is None:
            raise ValueError("Need to call DemographicModel.set_data()")
        # TODO better message (e.g. "Building SFS...")
        logging.getLogger(__name__).info("Constructing likelihood surface...")

        sfs = self._get_sfs().combine_loci()
        use_pairwise_diffs = self._use_pairwise_diffs

        muts_per_gen = self.muts_per_gen
        if muts_per_gen is None:
            mut_rate = None
        elif self._length is None:
            raise ValueError("SFS is missing length attribute, need to manually set length in set_data(), or set mutation rate to None")
        else:
            mut_rate = 4 * self.N_e * muts_per_gen * self._length

        demo_fun = self._demo_fun

        p_miss = self._fullsfs._p_missing
        p_miss = {pop: pm for pop, pm in zip(
            self._fullsfs.populations, p_miss)}
        p_miss = np.array([p_miss[pop] for pop in sfs.sampled_pops])

        self._lik_surface = SfsLikelihoodSurface(
            sfs, demo_fun, mut_rate=mut_rate,
            folded=sfs.folded, batch_size=self._mem_chunk_size,
            use_pairwise_diffs=use_pairwise_diffs, p_missing=p_miss)

        logging.getLogger(__name__).info("Finished constructing likelihood surface")

        return self._lik_surface

    # TODO note these are in PER-GENERATION units
    # TODO allow to pass folded parameter (if passing separate configs?)
    def expected_sfs(self, configs=None, normalized=False, folded=None):
        configs = self._get_configs(configs)
        demo = self._get_demo(dict(zip(configs.sampled_pops,
                                       configs.sampled_n)))
        if folded is None:
            folded = self._folded
        ret = expected_sfs(demo, configs, normalized=normalized, folded=folded)
        if not normalized:
            ret = ret * self.N_e * 4.0
        return co.OrderedDict(zip(configs.as_tuple(), ret))

    def _get_configs(self, configs):
        if configs is not None:
            return config_array(self.leafs, configs)

        if self._fullsfs is None:
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
        if self._fullsfs is None:
            raise ValueError(
                "Need to call set_data() or provide dict of sample sizes.")
        sfs = self._get_sfs()
        return co.OrderedDict(zip(sfs.sampled_pops, sfs.sampled_n))

    def log_likelihood(self):
        """
        The log likelihood at the current parameter values
        """
        return self._get_surface().log_lik(self._get_x())

    def kl_div(self):
        """
        The KL-divergence at the current parameter values
        """
        return self._get_surface().kl_div(self._get_x())

    def stochastic_optimize(
            self, n_minibatches=None, snps_per_minibatch=None,
            rgen=np.random, printfreq=1, start_from_checkpoint=None,
            **kwargs):
        def callback(x):
            self._set_x(x)
            if x.iteration % printfreq == 0:
                msg = [("it", x.iteration), ("LogLikelihood", -x.fun)]
                msg.extend(list(self.get_params().items()))
                msg = ", ".join(["{}: {}".format(k, v) for k, v in msg])
                logging.getLogger(__name__).info("{" + msg + "}")

        bounds = [p.x_bounds
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
            kwargs["x0"] = self._get_x()

        res = self._get_surface()._stochastic_surfaces(
            n_minibatches=n_minibatches,
            snps_per_minibatch=snps_per_minibatch,
            rgen=rgen).find_mle(**kwargs)

        self._set_x(res.x)
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
        bounds = [p.x_bounds
                  for p in self.parameters.values()]
        if all([b is None for bnd in bounds for b in bnd]):
            bounds = None

        def callback(x):
            self._set_x(x)
            if x.iteration % printfreq == 0:
                msg = [("it", x.iteration), ("KLDivergence", x.fun)]
                msg.extend(list(self.get_params().items()))
                msg = ", ".join(["{}: {}".format(k, v) for k, v in msg])
                logging.getLogger(__name__).info("{" + msg + "}")

        res = self._get_surface().find_mle(
            self._get_x(), method=method,
            jac=jac, hess=hess, hessp=hessp,
            bounds=bounds, callback=callback,
            **kwargs)

        self._set_x(res.x)
        res["parameters"] = self.get_params()
        return res
