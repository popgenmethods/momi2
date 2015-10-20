#!/usr/bin/env python

## Pre-requisites:
## * gcc
## * a scientific Python distribution such as Anaconda or Enthought. alternatively, custom installation of pip, cython, the SciPy stack

## To install, type 'pip install .' from the top-level directory of momi:

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [Extension("convolution",
                        sources=["momi/convolution.pyx"],
                        include_dirs=[numpy.get_include()])]

setup(name='momi',
      version='0.1',
      description='MOran Model for Inference',
      author='Jack Kamm, Jonathan Terhorst, Yun S. Song',
      author_email='jkamm@stat.berkeley.edu, terhorst@stat.berkeley.edu, yss@eecs.berkeley.edu',
      packages=['momi'],
      install_requires=['autograd>=1.0.4','numpy>=1.9.0','networkx','scipy','pandas'],
      keywords=['population genetics','statistics','site frequency spectrum','coalescent'],
      url='https://github.com/jackkamm/momi_private',
      ext_modules=cythonize(extensions),
      )
