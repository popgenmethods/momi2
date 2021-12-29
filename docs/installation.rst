.. _sec-installation:

============
Installation
============

------------------------
Recommended method (pip)
------------------------

.. code :: bash

    # (Optional) Create and activate a separate conda environment
    conda create -n my-momi-env
    conda activate my-momi-env

    # Install non-Python-package dependencies, e.g. with conda
    conda install hdf5 gsl python

    # Install momi and Python dependencies with pip
    pip install numpy
    pip install momi

Additional notes:

- Installing on macOS requires additional steps. See `Installing on
  macOS`_.
- Depending on your system, ``pip`` may have trouble installing some
  dependencies (such as ``numpy``, ``msprime``, ``pysam``).  In this
  case, you should manually install these dependencies and try again.
- You can download ``conda`` from `anaconda <https://www.anaconda.com/download/>`_ or
  `miniconda <https://conda.io/miniconda.html>`_.
- If you prefer not to use conda, you can install the non-Python
  dependencies (hdf5, gsl) with another package manager such as apt.

-------------------
Installing on macOS
-------------------

    **Note:** Recent versions of macOS may not work,
     see `#50 <https://github.com/popgenmethods/momi2/issues/50>`_.
     As an alternative, consider
     `Installing with conda (deprecated)`_.
     

Installing on macOS requires OpenMP-enabled clang, which can be
installed via homebrew or conda as follows:

* On homebrew, do: ``brew install llvm libomp``.
* Or if using conda, do: ``conda install llvm-openmp clang``.

You will also need to set the environment variable ``CC=/path/to/clang`` during installation.

For example, if you installed the above dependencies using homebrew, the command to install would be:

``CC=$(brew --prefix llvm)/bin/clang pip install momi``

If you installed clang via conda, the command would look instead like:

``CC=$(which clang) pip install momi``

----------------------------------
Installing with conda (deprecated)
----------------------------------

New versions of momi are no longer released on conda. However, older
versions of momi that were built for conda are still available, and
may be installed with the following command:

.. code:: bash

    conda install momi -c conda-forge -c bioconda -c jackkamm

Note the order of the ``-c`` flags matters, it determines the priority of each channel when installing dependencies.


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
