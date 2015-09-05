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
* Scientific Python distribution, e.g. [Anaconda](http://continuum.io/downloads), [Enthought Canopy](https://www.enthought.com/products/canopy/)
  * Alternatively, custom installation of pip, cython, the SciPy stack

To install, in the top-level directory of momi (where "setup.py" lives), type
```
pip install .
```

## Getting started

See [tutorial](examples/tutorial.py).

## Authors

[John Kamm](mailto:jkamm@stat.berkeley.edu), Jonathan Terhorst, Yun S. Song

## License

momi is not yet publicly released; please do not share with others.

When momi is publicly released, it will be free software under conditions of GNU GPL v3.