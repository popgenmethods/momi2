# momi

momi (MOran Models for Inference) is a Python package that computes
the expected sample frequency spectrum (SFS), a statistic commonly used
in population genetics.

In particular, momi computes the neutral SFS
for multipopulation models with variable population sizes,
population mergers, and pulse admixture events.

momi can also infer demographic histories and construct
confidence intervals using the technique of automatic differentiation.

The algorithm for the case without migration is described in
[this paper](http://www.tandfonline.com/doi/abs/10.1080/10618600.2016.1159212).
The generalization to pulse migration, and the use of automatic differentiation,
is discussed in Chapter 3 of Jack Kamm's [thesis](https://jackkamm.github.io/thesis.pdf),
which we are working on turning into a paper.

momi is still under active development -- please
email [Jack Kamm](jackkamm@gmail.com) about any bugs/issues.

## Installation and Dependencies

Prerequisites:

* C compiler with OpenMP support
* Scientific distribution of Python3, e.g. [Anaconda](http://continuum.io/downloads), [Enthought Canopy](https://www.enthought.com/products/canopy/)
  * Alternatively, custom installation of pip, cython, the SciPy stack
  * Note: momi should also work with Python2, but this is less well tested, and it is strongly recommended to use Python3.

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

[Jack Kamm](mailto:jackkamm@gmail.com), Jonathan Terhorst, Yun S. Song

## License

momi is not yet publicly released; please do not share with others.
