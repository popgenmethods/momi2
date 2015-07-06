# momi

This repository implements the algorithm and results in this [paper](http://arxiv.org/abs/1503.01133).
In particular, it computes the expected joint site frequency spectrum (SFS) for a tree-shaped demography without migration,
via a multipopulation Moran model.

It also computes the "truncated site frequency spectrum" for a single population, i.e. the frequency
spectrum for mutations arising after a certain point in time. This can be used in both Moran and coalescent
approaches to computing the multipopulation SFS.

See [example.py](example.py) for an example of how to construct a population history and compute its SFS entries.

## Upcoming features

momi is under active development. Upcoming features include:
* Pulse migration/admixture
* Parameter inference via gradient descent, automatic differentiation
* Improved user interface

These extensions to momi will be described in a future paper. Stay tuned!