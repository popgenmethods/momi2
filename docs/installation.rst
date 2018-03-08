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

--------------------
Note for MacOS users
--------------------

MacOS users are strongly recommended to use the ``conda install`` method, and to install into a separate conda environment as detailed in step 2.

Reasons are:

1. The default C compiler on MacOS does not support OpenMP. To install from source instead of conda, you will need an OpenMP-compliant C compiler, and to make sure appropriate libraries are linked.
2. There is currently a bug when using MKL with momi on MacOS (`details <https://github.com/ContinuumIO/anaconda-issues/issues/8803>`_). As a workaround, conda will disable MKL when it installs momi on Mac. You should install momi in its own environment to avoid affecting other packages.
