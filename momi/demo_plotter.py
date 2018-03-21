import heapq as hq
import collections as co
import autograd.numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
import logging


class DemographyPlot(object):
    """Object for plotting a demography.

    After creating this object, call :meth:`DemographyPlot.draw`
    to draw the demography.

    :param DemographicModel model: \
    model to plot

    :param list,dict pop_x_positions: \
    list ordering populations along the x-axis, or dict mapping \
    population names to x-axis positions

    :param matplotlib.axes.Axes ax: \
    Canvas to draw the figure on. Defaults to \
    :func:`matplotlib.pyplot.gca`

    :param tuple figsize: \
    If non-None, calls :func:`matplotlib.pyplot.figure` to \
    create a new figure with this width and height. \
    Ignored if ``ax`` is non-None.

    :param float linthreshy: \
    Scale y-axis linearly below this value \
    and logarithmically above it. (Default: use linear scaling only)

    :param list minor_yticks:  Minor ticks on y axis
    :param list major_yticks: Major ticks on y axis
    :param str,matplotlib.colors.Colormap color_map: \
    Colormap mapping pulse probabilities to colors

    :param tuple pulse_color_bounds: pair of ``(lower, upper)`` bounds \
    for mapping pulse probabilities to colors
    """
    def __init__(self, model, pop_x_positions,
                 ax=None, figsize=None,
                 linthreshy=None, minor_yticks=None,
                 major_yticks=None,
                 color_map="cool",
                 pulse_color_bounds=(0, 1),
                 draw=True):
        self.leafs = model.leafs
        self.model = model.copy()
        try:
            pop_x_positions.items()
        except AttributeError:
            pop_x_positions = {p: i for i, p in enumerate(
                pop_x_positions)}
        self.x_pos = dict(pop_x_positions)
        if ax:
            self.ax = ax
        else:
            if figsize:
                plt.figure(figsize=figsize)
            fig = plt.gcf()
            fig.clf()
            self.ax = fig.gca()

        if linthreshy:
            self.ax.set_yscale('symlog', linthreshy=linthreshy)
            self.ax.get_yaxis().set_major_formatter(
                LogFormatterSciNotation(labelOnlyBase=False,
                                        minor_thresholds=(100, 100),
                                        linthresh=5e4))
            self.ax.axhline(y=linthreshy, linestyle=":",
                            color="black", zorder=1)

        self.minor_yticks = minor_yticks
        self.major_yticks = major_yticks
        self.additional_times = []
        if minor_yticks:
            self.additional_times.extend(minor_yticks)
        if major_yticks:
            self.additional_times.extend(major_yticks)
        self.additional_times = sorted(self.additional_times)

        self._init_plot(model)

        self.all_N = [p.N for popline in self._plot.pop_lines.values()
                      for p in popline.points]
        self.base_N = min(self.all_N)
        self.pmin, self.pmax = pulse_color_bounds
        self.cm_scalar_mappable = plt.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=self.pmin, vmax=self.pmax),
            cmap=color_map)

        self.cbar = None
        if draw:
            self.draw()

    def _init_plot(self, model):
        self._plot = _DemographyPlot(
            self.x_pos, model.N_e, self.additional_times)

        params_dict = model.get_params()
        event_list = sorted(list(model.leaf_events) +
                            list(model.size_events) +
                            list(model.topology_events),
                            key=lambda e: e.t(params_dict))

        for e in event_list:
            e.add_to_plot(params_dict, self._plot)

        for pop in self._plot.pop_lines.values():
            pop.goto_time(float('inf'))

    def draw(self, alpha=1.0, tree_color="C0", draw_frame=True,
             rad=0, pulse_label=True):
        """Draw the demography.

        This is method draws the demography by calling
        :meth:`DemographyPlot.draw_tree`,
        :meth:`DemographyPlot.draw_leafs`,
        :meth:`DemographyPlot.draw_pulse`, and
        :meth:`DemographyPlot.draw_frame`.
        Call those methods directly for finer control.

        :param float alpha: Level of transparency (0=transparent, 1=opaque)
        :param str tree_color: Color of tree
        :param bool draw_frame: If True, call \
        :meth:`DemographyPlot.draw_frame` to draw tickmarks and legends

        :param float rad: Arc of pulse arrows in radians.
        :param bool pulse_label: If True, label each pulse with its strength
        """
        self.draw_tree(alpha=alpha, tree_color=tree_color)
        self.draw_leafs(alpha=alpha)
        for pop1, pop2, t, p in self.iter_pulses():
            self.draw_pulse(pop1, pop2, t, p, alpha=alpha,
                            rad=rad, pulse_label=pulse_label)
        if draw_frame:
            self.draw_frame()

    def draw_frame(self, pops=None, rename_pops=None, rotation=-30):
        """Draw tickmarks, legend, and colorbar.

        :param list pops: Populations to label on x-axis. \
        If None, add all populations

        :param dict rename_pops: Dict to give new names to \
        populations on x-axis.

        :param float rotation: Degrees to rotate x-axis population labels by.
        """
        self.draw_xticks(pops=pops, rename_pops=rename_pops,
                         rotation=rotation)
        self.draw_N_legend()
        self.draw_pulse_colorbar()
        if self.minor_yticks:
            self.ax.set_yticks(self.minor_yticks, minor=True)
        if self.major_yticks:
            self.ax.set_yticks(self.major_yticks)
        self.ax.set_ylabel("Time")

    def draw_pulse_colorbar(self):
        self.cm_scalar_mappable.set_array([])
        if not self.cbar:
            self.cbar = self.ax.get_figure().colorbar(
                self.cm_scalar_mappable, fraction=0.046, pad=0.04,
                ax=self.ax)
            self.cbar.set_label("Pulse Probability")

    def add_bootstrap(self, params, alpha,
                      rad=-.1, rand_rad=True):
        """Add an inferred bootstrap demography to the plot.

        :param dict params: Inferred bootstrap parameters
        :param float alpha: Transparency
        :param float rad: Arc of pulse arrows in radians
        :param bool rand_rad: Add random jitter to the arc of the pulse arrows
        """
        model = self.model.copy()
        model.set_params(params)
        additional_plot = AdditionalDemographyPlot(self, model)
        if rand_rad:
            rad *= 2 * np.random.uniform()
        additional_plot.draw(alpha=alpha, tree_color="gray",
                             pulse_label=False, rad=rad)

    def draw_xticks(self, pops=None, rename_pops=None, rotation=-30):
        if pops is None:
            pops = list(self._plot.pop_lines.keys())
        if rename_pops is None:
            rename_pops = {}
        x_pos = {rename_pops.get(p, p): self.x_pos[p] for p in pops}
        xtick_labs, xticks = zip(*sorted(
            x_pos.items(), key=lambda itm: itm[1]))

        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xtick_labs,
                                rotation=rotation)

    def draw_leafs(self, leafs=None,
                   facecolors="none", edgecolors="black",
                   s=100, zorder=2, **kwargs):
        """Draw symbols for the leaf populations.

        :param list leafs: Leaf populations to draw symbols for. \
        If None, add all leafs.

        Extra parameters ``facecolors, edgecolors, s, zorder, **kwargs``
        are passed to :meth:`matplotlib.axes.Axes.scatter`.
        """
        if leafs is None:
            leafs = self.leafs

        x = []
        t = []
        for l in leafs:
            popline = self._plot.pop_lines[l]
            for p in popline.points:
                if p.is_leaf:
                    x.append(self.x_pos[l])
                    t.append(p.t)
                    break

        self.ax.scatter(x, t, facecolors=facecolors,
                        edgecolors=edgecolors,
                        s=s, zorder=zorder, **kwargs)

    def N_to_linewidth(self, N):
        return np.log(N/self.base_N) + 2

    def draw_tree(self, tree_color="C0", alpha=1.0):
        """Draw the demographic tree (without pulse migrations)

        :param str tree_color: Color of the tree
        :param float alpha: Transparency

        """
        # draw vertical lines
        for popname, popline in self._plot.pop_lines.items():
            x = [self.x_pos[popname]]*len(popline.points)
            t = [p.t for p in popline.points]
            xt = list(zip(x, t))

            is_ghost = [False for _ in popline.points]
            for i, p in enumerate(popline.points):
                if p.is_leaf:
                    for j in range(i):
                        is_ghost[j] = True

            for bottom, top, N, ghost in zip(
                    xt[:-1], xt[1:],
                    [p.N for p in popline.points[:-1]],
                    is_ghost):
                curr_x, curr_t = zip(bottom, top)

                if ghost:
                    linestyle = ":"
                else:
                    linestyle = "-"

                self.ax.plot(
                    curr_x, curr_t, color=tree_color,
                    linewidth=self.N_to_linewidth(N),
                    linestyle=linestyle,
                    alpha=alpha, zorder=1)

        # draw horizontal lines
        for arrow in self._plot.pop_arrows:
            if arrow.p != 1:
                continue
            self.ax.plot(
                (arrow.from_pop.x, arrow.to_pop.x),
                (arrow.t, arrow.t), color=tree_color,
                linewidth=self.N_to_linewidth(arrow.from_N),
                alpha=alpha)

    def iter_pulses(self):
        """Iterate over the pulse arrows, to pass to :meth:`DemographyPlot.draw_pulse`

        :returns: iterator over tuples ``(pop1, pop2, t, p)``
        """
        for arrow in self._plot.pop_arrows:
            if arrow.p != 1:
                yield (arrow.to_pop.name, arrow.from_pop.name,
                       arrow.t, arrow.p)

    def draw_pulse(self, pop1, pop2, t, p,
                   rad=-.1, alpha=1.0,
                   pulse_label=True, adj_label_x=0, adj_label_y=0):
        """Draw a pulse.

        Use :meth:`DemographyPlot.iter_pulses` to iterate over \
        the pulses, which can then be plotted with this method.

        :param str pop1: Population the arrow is pointing into
        :param str pop2: Population the arrow is pointing away from
        :param float t: Time of the pulse
        :param float p: Strength of the pulse
        :param float rad: Arc of the pulse arrow in radians.
        :param float alpha: Transparency
        :param bool pulse_label: Add a label for pulse strength?
        :param float adj_label_x: Adjust pulse label x position
        :param float adj_label_y: Adjust pulse label y position
        """
        if p < self.pmin or p > self.pmax:
            logging.getLogger(__name__).warn("Trying to plot pulse p {} not in {}".format(p, (self.pmin, self.pmax)))

        frm = self.x_pos[pop2]
        to = self.x_pos[pop1]
        color = self.cm_scalar_mappable.to_rgba(p)

        self.ax.annotate(
            "", xy=(to, t), xytext=(frm, t),
            arrowprops=dict(
                arrowstyle="-|>,head_width=.3,head_length=.6",
                fc=color, ec=color, ls=":", shrinkA=0,
                alpha=alpha,
                connectionstyle="arc3,rad={rad}".format(
                    rad=rad)))

        if pulse_label:
            if -rad*(frm - to) > 0:
                verticalalignment = "top"
            else:
                verticalalignment = "bottom"

            self.ax.annotate(
                "{}".format("{:.1f}%".format(100*p)),
                xy=(.5*(frm+to) + adj_label_x, t + adj_label_y),
                color="black", weight="bold",
                horizontalalignment="center",
                verticalalignment=verticalalignment)

    def get_N_legend_values(self):
        curr_N = 10**np.floor(
            np.log(min(self.all_N)) / np.log(10))
        N_legend_values = [curr_N]
        while curr_N < max(self.all_N):
            curr_N *= 4
            N_legend_values.append(curr_N)

        return N_legend_values

    def draw_N_legend(self, N_legend_values=None, title="N", **kwargs):
        """Draw legend of population sizes.

        ``**kwargs`` are passed onto :meth:`matplotlib.axes.Axes.legend`.

        :param N_legend_values: Values of ``N`` to plot.
        :param title: Title of legend
        :rtype: :class:`matplotlib.legend.Legend`
        """
        if N_legend_values is None:
            N_legend_values = self.get_N_legend_values()

        N_legend = co.OrderedDict(
            ("{:.1e}".format(N), mpl.lines.Line2D(
                [], [], linewidth=self.N_to_linewidth(N)))
            for N in sorted(N_legend_values))

        return self.ax.legend(
            N_legend.values(), N_legend.keys(),
            title=title, **kwargs)


class AdditionalDemographyPlot(DemographyPlot):
    def __init__(self, parent_plot, model):
        self.leafs = model.leafs
        self.x_pos = parent_plot.x_pos
        self.ax = parent_plot.ax
        self.base_N = parent_plot.base_N
        self.pmin = parent_plot.pmin
        self.pmax = parent_plot.pmax
        self.cm_scalar_mappable = parent_plot.cm_scalar_mappable
        self.additional_times = parent_plot.additional_times

        self._init_plot(model)

    def draw_frame(self, *a, **kw):
        pass

    def draw_xticks(self, *a, **kw):
        raise NotImplementedError

    def draw_pulse_colorbar(self, *a, **kw):
        raise NotImplementedError

    def draw_N_legend(self, *a, **kw):
        raise NotImplementedError


class _DemographyPlot(object):
    def __init__(self, x_pos_dict, default_N, additional_times):
        self.pop_lines = {
            p: PopulationLine(p, x, additional_times, default_N)
            for p, x in x_pos_dict.items()}
        self.pop_arrows = []

    def pop_to_t(self, pop, t, add_time=True):
        pop = self.pop_lines[pop]
        pop.goto_time(t, add_time=add_time)
        return pop

    def add_leaf(self, pop, t):
        self.pop_to_t(pop, t).active = True
        self.pop_to_t(pop, t).next_is_leaf = True

    def set_size(self, pop, t, N):
        pop = self.pop_to_t(pop, t, add_time=False)
        if pop.curr_N != N:
            pop.goto_time(t, add_time=True)
        pop.curr_N = N

    def set_growth(self, pop, t, g):
        pop = self.pop_to_t(pop, t, add_time=True)
        pop.curr_g = g

    def move_lineages(self, pop1, pop2, t, p, pulse_name=None):
        pop2 = self.pop_lines[pop2]
        #pop2.goto_time(t, add_time=not pop2.active)
        pop2.goto_time(t)
        pop2.active = True

        pop1 = self.pop_lines[pop1]
        #pop1.goto_time(t, add_time=(p == 1))
        pop1.goto_time(t)
        if p == 1:
            pop1.step_time(hq.heappop(pop1.time_stack))
            pop1.active = False
        self.pop_arrows.append(PopulationArrow(
            pop1, pop2, t, p, pop1.curr_N, pop2.curr_N, pulse_name))


class DemographyPlotter(object):
    def __init__(self, params_dict,
                 default_N, event_list,
                 additional_times, x_pos_dict,
                 legend_kwargs,
                 xlab_rotation=-30,
                 exclude_xlabs=[],
                 pop_marker_kwargs=None,
                 adjust_pulse_labels={},
                 rename_pops={},
                 min_N=None, ax=None,
                 cm_scalar_mappable=None, colornorm=None,
                 alpha=1.0, pop_line_color="C0", pulse_line_color="gray",
                 plot_pulse_nums=None, plot_leafs=True, linthreshy=None):
        self.plot_leafs = plot_leafs
        self.pop_marker_kwargs = pop_marker_kwargs
        self.exclude_xlabs = list(exclude_xlabs)
        self.adjust_pulse_labels = dict(adjust_pulse_labels)
        self.xlab_rotation = xlab_rotation
        self.legend_kwargs = legend_kwargs
        self.additional_times = list(additional_times)
        self.default_N = default_N
        self.pop_lines = {
            p: PopulationLine(p, x, self.additional_times, self.default_N)
            for p, x in x_pos_dict.items()}
        self.pop_arrows = []
        self.rename_pops = rename_pops
        self.x_pos = x_pos_dict
        self.pop_line_color = pop_line_color
        self.pulse_line_color = pulse_line_color
        if plot_pulse_nums is None:
            plot_pulse_nums = not cm_scalar_mappable
        self.plot_pulse_nums = plot_pulse_nums

        for e in event_list:
            e.add_to_plot(params_dict, self)

        for pop in self.pop_lines.values():
            pop.goto_time(float('inf'))

        if ax is None:
            self.fig = plt.gcf()
            self.fig.clf()
            self.ax = self.fig.gca()
        else:
            self.ax = ax
            self.fig = ax.get_figure()

        self.all_N = [p.N for popline in self.pop_lines.values()
                      for p in popline.points]
        if min_N is None:
            self.min_N = min(self.all_N)
        else:
            self.min_N = min_N

        self.cm_scalar_mappable = cm_scalar_mappable
        self.alpha=alpha

        if linthreshy:
            self.ax.set_yscale('symlog', linthreshy=linthreshy)
            self.ax.get_yaxis().set_major_formatter(
                LogFormatterSciNotation(labelOnlyBase=False,
                                        minor_thresholds=(100,100),
                                        linthresh=5e4))


PopulationPoint = co.namedtuple(
    "PopulationPoint", ["t", "N", "g", "is_leaf"])


PopulationArrow = co.namedtuple(
    "PopulationArrow", ["to_pop", "from_pop", "t", "p", "to_N", "from_N", "pulse_name"])


class PopulationLine(object):
    def __init__(self, name, x, times, N):
        self.name = name
        self.x = x
        if not times:
            self.time_stack = []
        else:
            self.time_stack = list(times)
        hq.heapify(self.time_stack)
        self.curr_N = N
        self.curr_g = 0.0
        self.curr_t = 0.0
        self.points = []
        self.active = False
        self.next_is_leaf = False

    def __repr__(self):
        return "{cls}(name={name}, x={x}) at {loc}".format(
            cls=type(self), name=self.name, x=self.x,
            loc=hex(id(self)))

    def step_time(self, nxt_t, add=True):
        assert self.curr_t <= nxt_t
        self.curr_N = self.curr_N * np.exp(
            -self.curr_g * (nxt_t - self.curr_t))
        self.curr_t = nxt_t

        if add and self.active:
            self.points.append(PopulationPoint(
                self.curr_t, self.curr_N, self.curr_g, self.next_is_leaf))
            self.next_is_leaf = False

    def _push_time(self, t):
        assert t >= self.curr_t
        if t not in self.time_stack:
            hq.heappush(self.time_stack, t)

    def goto_time(self, t, add_time=True):
        # if exponentially growing, add extra time points whenever
        # the population size doubles
        if self.curr_g != 0 and t < float('inf'):
            halflife = np.abs(np.log(.5) / self.curr_g)
            add_t = self.curr_t + halflife
            while add_t < t:
                self._push_time(add_t)
                add_t += halflife

        while self.time_stack and self.time_stack[0] < t:
            self.step_time(hq.heappop(self.time_stack))
        self.step_time(t, add=False)
        if add_time:
            # put t on queue to be added when processing next event
            # (allows further events to change population size before plotting)
            self._push_time(t)
