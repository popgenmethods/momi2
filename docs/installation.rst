.. _sec-installation:

============
Installation
============

``momi`` requires Python >= 3.5. Binaries can be downloaded with `conda <https://conda.io/docs/>`_ or built with `pip <https://pip.readthedocs.io/en/stable/>`_.

---------------------------------------------
Method 1: Install pre-built binary from conda
---------------------------------------------

1. Download `anaconda <https://www.anaconda.com/download/>`_ or `miniconda <https://conda.io/miniconda.html>`_.
2. (Optional) create a separate `conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to install into:

.. code:: bash

    conda create -n momi-env
    source activate momi-env

3. Install:

.. code:: bash

    conda install momi -c jackkamm -c bioconda -c conda-forge

-------------------------------------
Method 2: Build from source using pip
-------------------------------------

Prerequisites:

* pip, numpy, cython
* C compiler with OpenMP support

Clone the git repository, then ``pip install`` the project root:

.. code:: bash

    git clone https://github.com/jackkamm/momi2
    cd momi2/
    pip install .

Depending on your system, ``pip`` may have trouble installing some
dependencies (such as ``msprime`` or ``pysam``).
In this case, you should manually install these dependencies and try again.

See  `venv <https://docs.python.org/3/tutorial/venv.html>`_ to install into a virtual environment.

.................
Building on MacOS
.................

On MacOS the default C compiler does
not support OpenMP. You will need to install an
alternative C compiler and provide it through the ``CC`` environment variable,
like so:

.. code:: bash

    brew install gcc
    CC=gcc-7 pip install .

