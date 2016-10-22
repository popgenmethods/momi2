# momi

momi (MOran Models for Inference) is a program that computes
the expected sample frequency spectrum (SFS), a statistic commonly used
in population genetics.

In particular, momi computes the neutral SFS
for multipopulation models with variable population sizes,
population mergers, and pulse admixture events.

momi can also infer demographic histories and construct
confidence intervals using the technique of automatic differentiation.

The algorithm for the case without migration is described in
[this preprint](http://arxiv.org/abs/1503.01133).
A forthcoming preprint describes the generalization to pulse migration,
and the use of automatic differentiation.

momi is still under active development -- please
report bugs/issues [here](https://github.com/jackkamm/momi/issues)

## Installation and Dependencies

Prerequisites:

* [einsum2](https://github.com/jackkamm/einsum2)
* [autograd](https://github.com/HIPS/autograd)
  * autograd will automatically be installed by pip if you don't have it, but it's recommended to install the latest version from github, as it fixes a [memory issue](https://github.com/HIPS/autograd/issues/103) and this patch is not yet in PyPi.
* C compiler that is OpenMP compatible
* Scientific distribution of Python 3 (recommended) or 2.7, e.g. [Anaconda](http://continuum.io/downloads), [Enthought Canopy](https://www.enthought.com/products/canopy/)
  * Alternatively, custom installation of pip, cython, the SciPy stack

Assumming you satisfy all the requirements, you can install by typing
```
pip install .
```
in the top-level directory of momi (where "setup.py" lives).

If you have problem installing or running, it's probably because of your C compiler.
The default C compiler in OSX does not support OpenMP and thus will break.
I've also sometimes run into issues with Anaconda Python on Linux
because of the old gcc-4 used to compile it (however I can no longer reproduce this error).

In case you are having issues with the C compiler, it is recommended
to install and run momi in a virtual environment, using a C
compiler from Anaconda. This will both support OpenMP and be fully compatible
with the Python used in Anaconda. To do this:

1. Create a new virtual environment named `momi2_env` with `conda create -n momi2_env python=3.5 anaconda` (alternatively, you can use `python=2.7 anaconda`).
2. Switch to the environment with `source activate momi2_env`
3. Install the Anaconda distribution of `gcc` with `conda install gcc`. Note this will clobber your system `gcc`, which is why we are doing this in a virtual environment.
4. Install extra dependencies such as `einsum2` and `autograd`.
5. Install momi2 with `CC=gcc pip install .` (the `CC=gcc` is required on OSX, otherwise
the installation will try to use the default `clang` compiler that does not support OpenMP).

## Getting started

See the [tutorial](examples/tutorial.ipynb) notebook.
You can type
```
ipython notebook
```
or
```
jupyter notebook
```
to open the notebook browser.

## Authors

[John Kamm](mailto:jkamm@stat.berkeley.edu), Jonathan Terhorst, Yun S. Song

## License

momi is not yet publicly released; please do not share with others.

When momi is publicly released, it will be free software under conditions of GNU GPL v3.

## TODO

* Pickling data_structures, demography fails because of decorators/descriptors. Very annoying for multiprocessing. Using setstate/getstate doesn't solve the problem.
* Fix text of tutorial.py
* Fix documentation (e.g. in likelihood.py, but also in a lot of other places now)
* Relatively few tests checking `Demography._pulse_probs()`.
* `import logging`
* add composite_log_likelihood back to the API, as an alternative to SfsLikelihoodSurface that allows second-order derivatives
