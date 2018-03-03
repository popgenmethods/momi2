# momi

momi (MOran Models for Inference) is a Python package that computes
the expected sample frequency spectrum (SFS), a statistic commonly used
in population genetics, and uses it to fit demographic history.

## Pre-release install instructions

Prerequisites:

* Python 3.6
* numpy and Cython. These are included in the [Anaconda](http://continuum.io/downloads) Python distribution.
* C compiler with OpenMP support

Assumming you satisfy all the requirements, you can install by typing

```
pip install .
```

in the top-level directory of momi (where "setup.py" lives).

Note that if you are on MacOS, your default C compiler might
not support OpenMP, in which case you will need to install an
alternative C compiler and provide it through the `CC` environment variable,
for example:

    brew install gcc
    CC=gcc-7 pip install .

## Getting started

See the [tutorial](examples/tutorial.ipynb) notebook.
You can type
```
jupyter notebook examples/tutorial.ipynb
```
to try it out.

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
