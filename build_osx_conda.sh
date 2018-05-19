# run this script by typing
# "source build_osx_conda.sh"
# from an activated conda environment

[ -z "$CONDA_PREFIX" ] && echo "Need to activate conda env" && return 1;

# install non-Python libraries with conda

conda install pip
conda install gsl hdf5 # for msprime
conda install llvm-openmp clang

# install Python dependencies with pip

CC=clang pip install .

# testing

conda install pytest
cd test

# NOTE conda Python doesn't like the default matplotlib backend

MPLBACKEND=TkAgg py.test -v test_sfs.py
MPLBACKEND=TkAgg py.test -v test_inference.py test_msprime.py

pip install numdifftools pyvcf
MPLBACKEND=TkAgg py.test -v

cd ..

# to package distribute:
# python setup.py sdist # build source dist
# python setup.py bdist_wheel # build binary dist
# twine upload dist/*
