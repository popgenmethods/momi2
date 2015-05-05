# momi

momi (MOran Models for Inference) is a program that computes
the expected site frequency spectrum (SFS), a statistic commonly used
in population genetics.

momi computes the SFS for complex demographic scenarios with
multiple populations with pulse migrations, variable population size,
and population mergers.

momi supports automatic differentiation, allowing for inference of demographic
histories via gradient descent, and the construction of asymptotically
correct confidence intervals for estimated parameters.

The algorithm for the case without migration is described in
[this preprint](http://arxiv.org/abs/1503.01133).
A forthcoming preprint describes the generalization to pulse migration,
and the use of automatic differentiation.

## A note on scaling and ms

There are multiple conventions for scaling time and the SFS
in population genetics.

We use the same format as the popular coalescent
simulator [ms](http://home.uchicago.edu/rhudson1/source/mksamples.html),
and in particular scale time in the same manner. 

For a sample of n1,n2,... individuals in populations 1,2,...,
we define the SFS to be an expected branch length in the genealogical
tree relating these individuals. In particular:
* SFS(x1,x2,...) = expected total length of branches with x1,x2,... leafs

where this branch length is scaled in the same time units as in ms.

Note also that momi can use ms (or a similar program like [scrm](https://github.com/scrm/scrm))
to simulate an observed SFS. This can be done in two ways:
* Pass in the MS path to Demography.simulate_sfs.
* Set the environment variable $MS_PATH to point to the appropriate location.

## Installation and Dependencies

momi depends on the following Python packages:
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/)
* [networkx](https://networkx.github.io/)
* [autograd](https://github.com/HIPS/autograd)

numpy, scipy, and networkx are automatically included in the
[Anaconda](https://store.continuum.io/cshop/anaconda/) Python
distribution for scientific computing.

momi is written purely in Python, though many of the functions
in numpy and scipy are actually implemented in C and Fortran.

After installing dependencies, do
```
python setup.py install
```
to install.

## Examples

* [Construct a demography and compute its SFS](examples/example_sfs.py)
* [Infer a demography with gradient descent](examples/example_inference.py)

## Future development

momi is still under active development -- we would be grateful if you
reported bugs at https://github.com/jackkamm/momi/issues

Upcoming features, in roughly the order to be implemented, are:
* Support for ancient DNA
* Parallelization and/or GPU support; code optimization/profiling
 * Compiling numpy with a parallel implementation of BLAS/LAPACK should automatically parallelize all the matrix/tensor operations.
 * To take advantage of this we may need to convert einsum calls to tensordot, which is less flexible, but faster, automatically parallelizable, and interfaces better with GPU packages.
* Support for missing alleles

