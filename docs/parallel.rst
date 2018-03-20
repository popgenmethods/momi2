---------------
Parallelization
---------------


momi will automatically use all available CPUs to perform
computations in parallel.
You can control the number of threads by setting the
environment variable ``OMP_NUM_THREADS``.

To take full advantage of parallelization, it is
recommended to make sure ``numpy`` is linked against
a parallel BLAS implementation such as MKL
or OpenBlas.
This is automatically taken care of in most
packaged, precompiled versions of numpy, such as in
Anaconda Python.
