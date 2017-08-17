import collections as co
import autograd.numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

PopulationPoint = co.namedtuple(
    "PopulationPoint", ["t", "N", "g", "is_leaf"])

PopulationArrow = co.namedtuple(
    "PopulationArrow", ["to_pop", "from_pop", "t", "p", "pulse_name"])

class PopulationLine(object):
    def __init__(self, name, x, times, N):
        self.name = name
        self.x = x
        self.time_stack = list(reversed(sorted(times)))
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

    def peek_time(self):
        return self.time_stack[-1]

    def push_time(self, t):
        assert t >= self.curr_t
        if (not self.time_stack) or t < self.peek_time():
            self.time_stack.append(t)
        else:
            assert t == self.peek_time()

    def goto_time(self, t, push_time=True):
        while self.time_stack and self.peek_time() < t:
            self.step_time(self.time_stack.pop())
        self.step_time(t, add=False)
        if push_time:
            self.push_time(t)

class DemographyPlotter(object):
    def __init__(self, params_dict,
                 default_N, event_list,
                 additional_times, x_pos_dict):
        self.additional_times = list(additional_times)
        self.default_N = default_N
        self.pop_lines = {
            p: PopulationLine(p, x, self.additional_times, self.default_N)
            for p, x in x_pos_dict.items()}
        self.pop_arrows = []
        self.x_pos = x_pos_dict

        for e in event_list:
            e.add_to_plot(params_dict, self)

        for pop in self.pop_lines.values():
            pop.goto_time(float('inf'))

        self.fig = plt.gcf()
        self.fig.clf()
        self.ax = self.fig.add_axes([.2,.1,.6,.8])
        self.N_legend_ax = self.fig.add_axes([.8,.5,.2,.5],
                                             frameon=False)
        self.N_legend_ax.axis('off')

        self.abs_g_max = max([
            abs(p.g) for popline in self.pop_lines.values()
            for p in popline.points])
        if self.abs_g_max:
            if any([p.g < 0 for popline in self.pop_lines.values()
                    for p in popline.points]):
                self.cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    "gMap", ["cyan", "black", "orange"])
                self.norm = plt.Normalize(vmin=-self.abs_g_max,
                                          vmax=self.abs_g_max)
            else:
                self.cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    "gMap", ["black", "orange"])
                self.norm = plt.Normalize(vmin=0, vmax=self.abs_g_max)
            self.sm = plt.cm.ScalarMappable(
                cmap=self.cmap, norm=self.norm)
            self.sm.set_array([])

        self.all_N = [p.N for popline in self.pop_lines.values()
                      for p in popline.points]
        self.min_N = min(self.all_N)

    def draw(self, tree_only=False, rad=-.1):
        self.draw_pops()
        self.draw_join_arrows()
        if not tree_only:
            self.draw_pulse_arrows(rad=rad)
        self.draw_xticks()
        self.draw_N_legend()

        if self.abs_g_max:
            self.g_ax = self.fig.add_axes([.85,.1,.02,.4])
            self.fig.colorbar(
                self.sm, cax=self.g_ax, format='%.2e')
            self.g_ax.set_xlabel("g")

    def pop_to_t(self, pop, t, push_time=True):
        pop = self.pop_lines[pop]
        pop.goto_time(t, push_time=push_time)
        return pop

    def add_leaf(self, pop, t):
        self.pop_to_t(pop, t).active = True
        self.pop_to_t(pop, t).next_is_leaf = True

    def set_size(self, pop, t, N):
        pop = self.pop_to_t(pop, t, push_time=False)
        if pop.curr_N != N:
            pop.push_time(t)
        pop.curr_N = N

    def set_growth(self, pop, t, g):
        pop = self.pop_to_t(pop, t, push_time=False)
        if pop.curr_g != g:
            pop.push_time(t)
        pop.curr_g = g

    def move_lineages(self, pop1, pop2, t, p, pulse_name=None):
        pop2 = self.pop_lines[pop2]
        pop2.goto_time(t, push_time=not pop2.active)
        pop2.active = True

        pop1 = self.pop_lines[pop1]
        pop1.goto_time(t, push_time=(p == 1))
        if p == 1:
            pop1.step_time(pop1.time_stack.pop())
            pop1.active = False
        self.pop_arrows.append(PopulationArrow(pop1, pop2, t, p, pulse_name))

    def N_to_markersize(self, N):
        return 10 * (np.log(N/self.min_N) + 1)**1.5

    def draw_pops(self):
        for popname, popline in self.pop_lines.items():
            x = [self.x_pos[popname]]*len(popline.points)
            t = [p.t for p in popline.points]
            xt = list(zip(x, t))
            for bottom, top, g in zip(xt[:-1], xt[1:], [p.g for p in popline.points[:-1]]):
                curr_x, curr_t = zip(bottom, top)
                if self.abs_g_max:
                    self.ax.plot(curr_x, curr_t,
                                 color=self.cmap(self.norm(g)))
                else:
                    self.ax.plot(curr_x, curr_t, color="black")
            self.ax.scatter(
                x, t,
                [self.N_to_markersize(p.N) for p in popline.points],
                facecolors=["gray" if p.is_leaf else "none"
                            for p in popline.points],
                edgecolors="black")

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
            self.ax.annotate(
                "", xy=(arrow.to_pop.x, arrow.t),
                xytext=(arrow.from_pop.x, arrow.t),
                arrowprops=dict(arrowstyle="->", fc="black", ec="black",
                                ls="-", shrinkA=0))

    def draw_pulse_arrows(self, rad=-.1, size=20):
        for arrow in self.pulse_arrows:
            frm = arrow.from_pop.x
            to = arrow.to_pop.x
            if -rad*(frm - to) > 0:
                verticalalignment = "top"
            else:
                verticalalignment = "bottom"

            self.ax.annotate(
                "", xy=(to, arrow.t), xytext=(frm, arrow.t),
                arrowprops=dict(
                    arrowstyle="-|>,head_width=.3,head_length=.6", fc="gray", ec="gray", ls=":",
                    shrinkA=0,
                    connectionstyle="arc3,rad={rad}".format(
                        rad=rad)))

            self.ax.annotate(
                "{}".format("{:.1f}%".format(100*arrow.p)),
                xy=(.5*(frm+to), arrow.t),
                color="red", horizontalalignment="center",
                verticalalignment=verticalalignment)

    def draw_xticks(self):
        xtick_labs, xticks = zip(*sorted(
            self.x_pos.items(), key=lambda itm: itm[1]))
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xtick_labs, rotation=-30)

    def get_N_legend(self):
        base = 10.0

        log_N = np.log(self.all_N) / np.log(base)
        N_legend_keys = list(base**(np.arange(
            np.floor(min(log_N)), np.ceil(max(log_N))+1)))[1:-1]
        N_legend_keys.extend([min(self.all_N), max(self.all_N)])
        N_legend_keys = sorted(set(N_legend_keys))
        N_legend_values = []

        for N in N_legend_keys:
            N_legend_values.append(self.ax.scatter(
                [], [], [self.N_to_markersize(N)],
                edgecolors="black", facecolors="none"))

        N_legend_keys = ["{:.2e}".format(N) for N in N_legend_keys]
        return co.OrderedDict(zip(N_legend_keys, N_legend_values))

    def draw_N_legend(self):
        N_legend = self.get_N_legend()
        lgd = self.N_legend_ax.legend(
            N_legend.values(), N_legend.keys(),
            loc='center', title="N")

