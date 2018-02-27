#!/usr/bin/env python

# Pre-requisites:
# * gcc
# * a scientific Python distribution such as Anaconda or Enthought. alternatively, custom installation of pip, cython, the SciPy stack

# To install, type 'pip install .' from the top-level directory of momi:

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("momi.convolution",
                        sources=["momi/convolution.pyx"],
                        extra_compile_args=['-fopenmp'],
                        extra_link_args=['-fopenmp'],
                        include_dirs=[numpy.get_include()]),
              Extension("momi.einsum2.parallel_matmul",
                        sources=["momi/einsum2/parallel_matmul.pyx"],
                        extra_compile_args=['-fopenmp'],
                        extra_link_args=['-fopenmp'],
                        include_dirs=[numpy.get_include()])]

setup(name='momi',
      version='0.1',
      description='MOran Model for Inference',
      author='Jack Kamm, Jonathan Terhorst, Yun S. Song',
      author_email='jkamm@stat.berkeley.edu, terhorst@stat.berkeley.edu, yss@eecs.berkeley.edu',
      packages=['momi', 'momi.einsum2', 'momi.data'],
      install_requires=[
          'autograd>=1.2.0', 'numpy>=1.9.0', 'networkx', 'scipy',
          'pandas', 'numdifftools', 'cached_property>=1.3',
          'msprime', "matplotlib", "seaborn", "pysam"],
      keywords=['population genetics', 'statistics',
                'site frequency spectrum', 'coalescent'],
      url='https://github.com/jackkamm/momi2',
      ext_modules=cythonize(extensions),
      )
