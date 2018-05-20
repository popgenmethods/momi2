.. _sec-installation:

============
Installation
============

``momi`` requires Python >= 3.5, and can be installed with `conda <https://conda.io/docs/>`_ or `pip <https://pip.readthedocs.io/en/stable/>`_.

---------------------
Installing with conda
---------------------

1. Download `anaconda <https://www.anaconda.com/download/>`_ or `miniconda <https://conda.io/miniconda.html>`_.
2. (Optional) create a separate `conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ to install into:

.. code:: bash

    conda create -n my-momi-env
    conda activate my-momi-env

3. Install:

.. code:: bash

    conda install momi -c defaults -c conda-forge -c bioconda -c jackkamm

Note the order of the ``-c`` flags matters, it determines the priority of each channel when installing dependencies.


-------------------
Installing with pip
-------------------

The ``momi`` source distribution is provided on PyPi, and can be downloaded, built, and installed with ``pip``.

First, ensure the following non-Python dependencies are installed with your favorite package manager (e.g. ``apt-get``, ``yum``, ``brew``, ``conda``, etc):

1. hdf5
2. gsl
3. (OSX only) OpenMP-enabled clang

   * If using homebrew, do ``brew install llvm libomp``.
   * Or if using conda, do ``conda install llvm-openmp clang``.
   * You will also need to set the environment variable ``CC=/path/to/clang`` during installation.

Then do ``pip install momi`` (or on OSX, ``CC=/path/to/clang pip install momi``).

Depending on your system, ``pip`` may have trouble installing some
dependencies (such as ``numpy``, ``msprime``, ``pysam``).
In this case, you should manually install these dependencies and try again.

See  `venv <https://docs.python.org/3/tutorial/venv.html>`_ to install into a virtual environment.

---------------
Troubleshooting
---------------

"ModuleNotFoundError: No module named 'momi.convolution'"
=========================================================

This is usually caused by trying to import ``momi``
when in the top-level folder of the ``momi2`` project.
In this case, Python will try to import the local, unbuilt copy
of the ``momi`` subdirectory rather than the installed version.

To fix this, simply ``cd`` out of the top-level directory before
importing ``momi``.

"clang: error: unsupported option '-fopenmp'"
=============================================

On macOS the system version of ``clang`` does not support OpenMP,
which causes this error when building ``momi`` with pip.

To solve this, make sure you have OpenMP-enabled LLVM/clang installed,
and set the environment variable ``CC`` as noted in the pip installation
instructions above.

Note: it is NOT recommended to replace ``clang`` with ``gcc`` on macOS,
as this can cause strange numerical errors when used with Intel MKL; for example, see
https://github.com/ContinuumIO/anaconda-issues/issues/8803
