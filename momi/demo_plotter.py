import heapq as hq
import collections as co
import autograd.numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

PopulationPoint = co.namedtuple(
    "PopulationPoint", ["t", "N", "g", "is_leaf"])

PopulationArrow = co.namedtuple(
    "PopulationArrow", ["to_pop", "from_pop", "t", "p", "to_N", "from_N", "pulse_name"])

class PopulationLine(object):
    def __init__(self, name, x, times, N):
        self.name = name
        self.x = x
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

    def draw(self, tree_only=False, rad=-.1, no_ticks_legend=False):
        self.draw_pops()
        self.draw_join_arrows()
        if not tree_only:
            self.draw_pulse_arrows(rad=rad)
        if not no_ticks_legend:
            self.draw_xticks()
            self.draw_N_legend()
            if self.cm_scalar_mappable:
                self.draw_colorbar()

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

    def N_to_linewidth(self, N):
        return np.log(N/self.min_N) + 2

    def draw_pops(self):
        for popname, popline in self.pop_lines.items():
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
                    linestyle=":"
                else:
                    linestyle="-"
                self.ax.plot(
                    curr_x, curr_t, color=self.pop_line_color,
                    linewidth=self.N_to_linewidth(N),
                    linestyle=linestyle,
                    alpha=self.alpha, zorder=1)

            for p in popline.points:
                if p.is_leaf and self.plot_leafs:
                    if not self.pop_marker_kwargs:
                        self.ax.scatter([self.x_pos[popname]], [p.t],
                                        facecolors="none",
                                        edgecolors="black",
                                        s=100,
                                        alpha=self.alpha, zorder=2)
                    else:
                        self.ax.scatter(
                            [self.x_pos[popname]], [p.t],
                            **self.pop_marker_kwargs[popname], alpha=self.alpha)


    @property
    def join_arrows(self):
        for arrow in self.pop_arrows:
            if arrow.p == 1:
                yield arrow

    @property
    def pulse_arrows(self):
        for arrow in self.pop_arrows:
            if arrow.p != 1:
                yield arrow

    def draw_join_arrows(self):
        for arrow in self.join_arrows:
            self.ax.plot(
                (arrow.from_pop.x, arrow.to_pop.x),
                (arrow.t, arrow.t), color=self.pop_line_color,
                linewidth=self.N_to_linewidth(arrow.from_N), alpha=self.alpha)

    def draw_pulse_arrows(self, rad=-.1):
        for arrow in self.pulse_arrows:
            frm = arrow.from_pop.x
            to = arrow.to_pop.x
            if -rad*(frm - to) > 0:
                verticalalignment = "top"
            else:
                verticalalignment = "bottom"

            if self.cm_scalar_mappable:
                col = self.cm_scalar_mappable.to_rgba(arrow.p)
            else:
                col = self.pulse_line_color

            self.ax.annotate(
                "", xy=(to, arrow.t), xytext=(frm, arrow.t),
                arrowprops=dict(
                    arrowstyle="-|>,head_width=.3,head_length=.6",
                    fc=col, ec=col, ls=":", shrinkA=0,
                    alpha=self.alpha,
                    connectionstyle="arc3,rad={rad}".format(
                        rad=rad)))

            if self.plot_pulse_nums:
                try:
                    adj_x, adj_y = self.adjust_pulse_labels[(arrow.from_pop.name, arrow.to_pop.name)]
                except KeyError:
                    adj_x, adj_y = 0, 0
                self.ax.annotate(
                    "{}".format("{:.1f}%".format(100*arrow.p)),
                    xy=(.5*(frm+to) + adj_x, arrow.t + adj_y),
                    color="black", weight="bold", horizontalalignment="center",
                    verticalalignment=verticalalignment)


    def draw_xticks(self):
        x_pos = {self.rename_pops.get(p, p): x for p, x in self.x_pos.items()
                 if p not in self.exclude_xlabs}
        xtick_labs, xticks = zip(*sorted(
            x_pos.items(), key=lambda itm: itm[1]))
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xtick_labs,
                                rotation=self.xlab_rotation)

    def get_N_legend(self):
        curr_N = 10**np.floor(
            np.log(min(self.all_N)) / np.log(10))
        N_legend_keys = [curr_N]
        while curr_N < max(self.all_N):
            curr_N *= 4
            N_legend_keys.append(curr_N)

        N_legend_values = []
        for N in N_legend_keys:
            N_legend_values.append(
                mpl.lines.Line2D(
                    [], [], linewidth=self.N_to_linewidth(N)))

        N_legend_keys = ["{:.1e}".format(N)
                         for N in N_legend_keys]
        return co.OrderedDict(zip(N_legend_keys, N_legend_values))

    def draw_N_legend(self):
        N_legend = self.get_N_legend()
        lgd = self.ax.legend(
            N_legend.values(), N_legend.keys(),
            title="N", **self.legend_kwargs)

    def draw_colorbar(self):
        self.cm_scalar_mappable.set_array([])
        self.fig.colorbar(self.cm_scalar_mappable, fraction=0.046, pad=0.04)

