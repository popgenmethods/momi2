.. _sec-installation:

============
Installation
============

Prerequisites:

* Python 3.6
* pip, numpy, and Cython.
* C compiler with OpenMP support

Assumming you satisfy all the requirements, you can install by typing

.. code:: bash

    pip install .

in the top-level directory of momi (where "setup.py" lives).

--------------
Note for MacOS
--------------

On MacOS the default C compiler does
not support OpenMP. You will need to install an
alternative C compiler and provide it through the `CC` environment variable,
like so:

.. code:: bash

    brew install gcc
    CC=gcc-7 pip install .

