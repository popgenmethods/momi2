.. _sec-installation:

============
Installation
============

``momi`` requires Python >= 3.5. Binaries can be downloaded with `conda <https://conda.io/docs/>`_ (recommended) or built with `pip <https://pip.readthedocs.io/en/stable/>`_.

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

* pip
* C compiler with OpenMP support

Clone the git repository, then ``pip install`` the project root:

.. code:: bash

    git clone https://github.com/jackkamm/momi2
    cd momi2/
    pip install .

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
of the ``momi`` folder rather than the installed version.

To fix this, simply ``cd`` out of the top-level directory before
importing ``momi``. Alternatively, you can build ``momi`` with
``pip install -e .`` which will build ``momi`` locally and place a
symlink in your site-packages, instead of building a copy of ``momi``
there.

"clang: error: unsupported option '-fopenmp'"
=============================================

On macOS the system version of ``clang`` does not support OpenMP,
which is required to build ``momi``.

To solve this, install ``momi`` via conda instead of building with pip.
Alternatively, you can install a version of ``clang`` that supports
OpenMP and tell pip to use that:

    CC=/path/to/clang pip install .

Even though ``gcc`` supports OpenMP, it is not recommended to replace ``clang`` with ``gcc`` on macOS,
as this can cause strange numerical errors when used with Intel MKL; for example, see
https://github.com/ContinuumIO/anaconda-issues/issues/8803
