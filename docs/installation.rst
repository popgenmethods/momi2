.. _sec-installation:

============
Installation
============

``momi`` requires Python >= 3.5.

-------------
Conda install
-------------

1. Download `anaconda <https://www.anaconda.com/download/>`_ or `miniconda <https://conda.io/miniconda.html>`_.
2. (Optional) create a separate `conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to install into:

.. code:: bash

    conda create -n momi-env
    source activate momi-env

3. Install:

.. code:: bash

    conda install momi -c jackkamm -c bioconda -c conda-forge

--------------
Manual install
--------------

Prerequisites:

* pip, numpy, cython
* C compiler with OpenMP support

After satisfying the requirements, install with

.. code:: bash

    pip install .

in the root directory.

Depending on your system, ``pip`` may have trouble installing some
dependencies (such as ``msprime`` or ``pysam``).
In this case, you should manually install these dependencies and try again.

See  `venv <https://docs.python.org/3/tutorial/venv.html>`_ to install into a virtual environment.

..............
Building on MacOS
..............

On MacOS the default C compiler does
not support OpenMP. You will need to install an
alternative C compiler and provide it through the ``CC`` environment variable,
like so:

.. code:: bash

    brew install gcc
    CC=gcc-7 python setup.py install

