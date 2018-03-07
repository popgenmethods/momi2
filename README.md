# momi

momi (MOran Models for Inference) is a Python package that computes
the expected sample frequency spectrum (SFS), a statistic commonly used
in population genetics, and uses it to fit demographic history.

## Build instructions

Prerequisites:

* Python 3.6
* pip, numpy, and Cython.
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

