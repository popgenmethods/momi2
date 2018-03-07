.. _sec-installation:

============
Installation
============

``momi`` is a Python3 package and requires Python >= 3.5.

-------------
Conda install
-------------

This is the recommended method.
First make sure you have `conda <https://conda.io/docs/>`_,
then do

.. code:: bash

    conda install momi -c jackkamm -c bioconda -c conda-forge

See `conda-environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to install into a separate virtual environment (this is useful for keeping dependencies isolated).


--------------
Manual install
--------------

Prerequisites:

* pip, numpy, cython
* C compiler with OpenMP support

Assumming you satisfy all the requirements, you can install by typing

.. code:: bash

    pip install .

from the root directory.

Depending on your setup, ``pip`` may have trouble installing some
dependencies (such as ``msprime`` or ``pysam``).
In this case, you should manually install these dependencies and try again.

See  `venv <https://docs.python.org/3/tutorial/venv.html>`_ to install into a virtual environment.

..............
Note for MacOS
..............

On MacOS the default C compiler does
not support OpenMP. You will need to install an
alternative C compiler and provide it through the ``CC`` environment variable,
like so:

.. code:: bash

    brew install gcc
    CC=gcc-7 python setup.py install

