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

* C compiler that is OpenMP compatible
* Scientific distribution of Python 3 (recommended) or 2.7, e.g. [Anaconda](http://continuum.io/downloads), [Enthought Canopy](https://www.enthought.com/products/canopy/)
  * Alternatively, custom installation of pip, cython, the SciPy stack

Assumming you satisfy all the requirements, you can install by typing
```
pip install .
```
in the top-level directory of momi (where "setup.py" lives).

### Troubleshooting

If you have problem installing on OSX, it's likely due to your C compiler.
The default C compiler in OSX does not support OpenMP and thus will break.
In case you are having issues with the C compiler, it is recommended
to install and run momi in a virtual environment with a compatible
C compiler. For example, if you use Anaconda Python, you can do the
following:

1. Create a new virtual environment named `momi2_env` with `conda create -n momi2_env python=3.6 anaconda`.
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

## A note on parallelization

momi will automatically use all available CPUs to perform
computations in parallel.
You can control the number of threads by setting the
environment variable `OMP_NUM_THREADS`.

To take full advantage of parallelization, it is
recommended to make sure `numpy` is linked against
a parallel BLAS implementation such as MKL
or OpenBlas.
This is automatically taken care of in most
packaged, precompiled versions of numpy, such as
Anaconda Python.

## Authors

[John Kamm](mailto:jkamm@stat.berkeley.edu), Jonathan Terhorst, Yun S. Song

## License

momi is not yet publicly released; please do not share with others.

When momi is publicly released, it will be free software under conditions of GNU GPL v3.
