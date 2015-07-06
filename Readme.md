# momi

This repository implements the algorithm and results in this [paper](http://arxiv.org/abs/1503.01133).
In particular, it computes the expected joint site frequency spectrum (SFS) for a tree-shaped demography without migration,
via a multipopulation Moran model.

It also computes the "truncated site frequency spectrum" for a single population, i.e. the frequency
spectrum for mutations arising after a certain point in time. This can be used in both Moran and coalescent
approaches to computing the multipopulation SFS.

See example.py for an example of how to construct a population history and compute its SFS entries.

## Upcoming features

An upcoming paper will extend the Moran joint SFS algorithm to handle demographic histories with pulse migration, and
will also include a method for fitting models via gradient descent.
The user interface of momi will also be changing when migration and gradient descent are added.
Stay tuned for updates!