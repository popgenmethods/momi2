import autograd.numpy as np
from matplotlib import pyplot as plt
import momi

#%matplotlib

demo_model = momi.demographic_model(1e4, 29)
demo_model.add_leaf("A")
demo_model.add_leaf("B", t=1e4, N=1e6)
demo_model.set_size("B", t=1e4, g=1e-4)
demo_model.move_lineages("A", "B", 5e3, p=.25)
demo_model.move_lineages("B", "A", 5e4, N=1e3)

demo_plt = demo_model.draw([], ["A", "B"])
demo_plt.ax.set_yscale('symlog',linthreshy=1e4)
demo_plt.ax.set_yticks([1e3, 1e4, 3e4, 5e4])
