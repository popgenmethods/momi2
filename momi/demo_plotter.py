import collections as co
import autograd.numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

PopulationPoint = co.namedtuple(
    "PopulationPoint", ["t", "N", "g", "is_leaf"])

PopulationArrow = co.namedtuple(
    "PopulationArrow", ["to", "frm", "t", "p"])

class PopulationLine(object):
    def __init__(self, times, N):
        self.time_stack = list(reversed(sorted(times)))
        self.curr_N = N
        self.curr_g = 0.0
        self.curr_t = 0.0
        self.points = []
        self.active = False
        self.next_is_leaf = False

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

    def goto_time(self, t):
        while self.time_stack and self.peek_time() < t:
            self.step_time(self.time_stack.pop())
        self.step_time(t, add=False)
        self.push_time(t)

class DemographyPlotter(object):
    def __init__(self, params_dict,
                 default_N, event_list,
                 additional_times, x_pos_dict):
        self.additional_times = list(additional_times)
        self.default_N = default_N
        self.pop_lines = co.defaultdict(
            lambda : PopulationLine(self.additional_times, self.default_N))
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
        self.g_ax = self.fig.add_axes([.85,.1,.02,.4])

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

    def draw(self):
        self.draw_pops()
        self.draw_arrows()
        self.draw_xticks()
        self.draw_N_legend()

        if self.abs_g_max:
            self.fig.colorbar(
                self.sm, cax=self.g_ax, format='%.2e')
            self.g_ax.set_xlabel("g")

    def pop_to_t(self, pop, t):
        pop = self.pop_lines[pop]
        pop.goto_time(t)
        return pop

    def add_leaf(self, pop, t):
        self.pop_to_t(pop, t).active = True
        self.pop_to_t(pop, t).next_is_leaf = True

    def set_size(self, pop, t, N):
        self.pop_to_t(pop, t).curr_N = N

    def set_growth(self, pop, t, g):
        self.pop_to_t(pop, t).curr_g = g

    def move_lineages(self, pop1, pop2, t, p):
        self.pop_arrows.append(PopulationArrow(pop1, pop2, t, p))
        pop1 = self.pop_to_t(pop1, t)
        pop2 = self.pop_to_t(pop2, t)
        pop2.active = True
        if p == 1:
            pop1.step_time(pop1.time_stack.pop())
            pop1.active = False

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
                facecolors=["black" if p.is_leaf else "none"
                            for p in popline.points],
                edgecolors="black")

    def draw_arrows(self):
        all_t = [p.t for popline in self.pop_lines.values() for p in popline.points]
        total_y_height = max(all_t) - min(all_t)
        #arrow_head_width = .025*(total_y_height)
        all_x = list(self.x_pos.values())
        total_x_width = max(all_x) - min(all_x)
        #arrow_head_len = .025*total_x_width
        for arrow in self.pop_arrows:
            frm = self.x_pos[arrow.frm]
            to = self.x_pos[arrow.to]
            if arrow.p == 1:
                fc = 'black'
                ec = 'black'
                ls = "-"
            else:
                fc = 'red'
                ec = 'red'
                ls = ":"
                self.ax.annotate(
                    "{}".format("{:.1e}".format(arrow.p)), xy=(.5*(frm+to), arrow.t),
                    color="red", horizontalalignment="center")
            self.ax.annotate("", xy=(to, arrow.t), xytext=(frm, arrow.t),
                             arrowprops=dict(arrowstyle="->", fc=fc, ec=ec, ls=ls))
            #self.ax.arrow(frm, arrow.t, to-frm, 0,
            #              length_includes_head=True,
            #              head_length=arrow_head_len,
            #              head_width=arrow_head_width,
            #              fc=fc, ec=ec, ls=ls)

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

