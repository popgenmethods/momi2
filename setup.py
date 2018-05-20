#!/usr/bin/env python

from setuptools import setup, Extension
import os
import sys

extensions = []
install_requires = ['cached_property>=1.3']

use_cython = "USE_CYTHON" in os.environ
if use_cython:
    from Cython.Build import cythonize
    ext = ".pyx"
else:
    ext = ".c"

class numpy_get_include(object):
    def __str__(self):
        import numpy
        return numpy.get_include()
numpy_get_include = numpy_get_include()

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if not on_rtd:
    extra_compile_args=["-fopenmp"]
    extra_link_args=["-fopenmp"]

    extensions = [
        Extension("momi.convolution",
                  sources=["momi/convolution" + ext],
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args,
                  include_dirs=[numpy_get_include]),
        Extension("momi.einsum2.parallel_matmul",
                  sources=["momi/einsum2/parallel_matmul" + ext],
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args,
                  include_dirs=[numpy_get_include])]
    if use_cython:
        extensions = cythonize(extensions)

    install_requires.extend([
        'autograd>=1.2.0', 'numpy>=1.9.0', 'networkx', 'scipy',
        'pandas', 'msprime', "matplotlib", "seaborn", "pysam"])

setup(name='momi',
      version='2.1.10',
      description='MOran Model for Inference',
      author='Jack Kamm, Jonathan Terhorst, Richard Durbin, Yun S. Song',
      author_email='jkamm@stat.berkeley.edu, terhorst@stat.berkeley.edu, yss@eecs.berkeley.edu',
      packages=['momi', 'momi.einsum2', 'momi.data'],
      install_requires=install_requires,
      python_requires='>=3.5',
      keywords=['population genetics', 'statistics',
                'site frequency spectrum', 'coalescent'],
      url='https://github.com/jackkamm/momi2',
      ext_modules=extensions)
