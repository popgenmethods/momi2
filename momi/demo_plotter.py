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
                 additional_times, x_pos_dict,
                 legend_kwargs,
                 xlab_rotation=-30):
        self.xlab_rotation = xlab_rotation
        self.legend_kwargs = legend_kwargs
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
        self.ax = self.fig.gca()

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
        raise NotImplementedError(
            "plotting exponential growth"
            " not implemented")

    def move_lineages(self, pop1, pop2, t, p, pulse_name=None):
        pop2 = self.pop_lines[pop2]
        pop2.goto_time(t, push_time=not pop2.active)
        pop2.active = True

        pop1 = self.pop_lines[pop1]
        pop1.goto_time(t, push_time=(p == 1))
        if p == 1:
            pop1.step_time(pop1.time_stack.pop())
            pop1.active = False
        self.pop_arrows.append(PopulationArrow(
            pop1, pop2, t, p, pulse_name))

    def N_to_linewidth(self, N):
        return np.log(N/self.min_N) + 2

    def draw_pops(self):
        for popname, popline in self.pop_lines.items():
            x = [self.x_pos[popname]]*len(popline.points)
            t = [p.t for p in popline.points]
            xt = list(zip(x, t))
            for bottom, top, N in zip(
                    xt[:-1], xt[1:],
                    [p.N for p in popline.points[:-1]]):
                curr_x, curr_t = zip(bottom, top)

                self.ax.plot(
                    curr_x, curr_t, color="C0",
                    linewidth=self.N_to_linewidth(N))

            for p in popline.points:
                if p.is_leaf:
                    self.ax.scatter([self.x_pos[popname]], [p.t],
                                    facecolors="none",
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
            self.ax.plot(
                (arrow.from_pop.x, arrow.to_pop.x),
                (arrow.t, arrow.t), color="C0",
                linewidth=1)

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

