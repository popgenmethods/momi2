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
* gcc
* Scientific distribution of Python 2.7, e.g. [Anaconda](http://continuum.io/downloads), [Enthought Canopy](https://www.enthought.com/products/canopy/)
  * Alternatively, custom installation of pip, cython, the SciPy stack

To install, in the top-level directory of momi (where "setup.py" lives), type
```
pip install .
```

## Getting started

See the [tutorial](examples/tutorial.ipynb).

## Authors

[John Kamm](mailto:jkamm@stat.berkeley.edu), Jonathan Terhorst, Yun S. Song

## License

momi is not yet publicly released; please do not share with others.

When momi is publicly released, it will be free software under conditions of GNU GPL v3.

## TODO

* To make mutation work with missing data -- somehow incorporate the coverage probabilities as part of the Multinomial Poisson model, and/or treat the coverage probabilities as a nuisance parameter.
* Pickling data_structures, demography fails because of decorators/descriptors. Very annoying for multiprocessing. Using setstate/getstate doesn't solve the problem.
* Fix text of tutorial.py
* Fix documentation (e.g. in likelihood.py, but also in a lot of other places now)
* Relatively few tests checking `Demography._pulse_probs()`.
* `import logging`
* add composite_log_likelihood back to the API, as an alternative to SfsLikelihoodSurface that allows second-order derivatives