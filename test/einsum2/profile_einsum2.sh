#!/bin/bash

ipython time_einsum2.ipy -- --npeinsum --length 1000 --nobatch
OMP_NUM_THREADS=1 ipython time_einsum2.ipy -- --length 1000 --nobatch
OMP_NUM_THREADS=2 ipython time_einsum2.ipy -- --length 1000 --nobatch
OMP_NUM_THREADS=4 ipython time_einsum2.ipy -- --length 1000 --nobatch
ipython time_einsum2.ipy -- --length 1000 --nobatch

ipython time_einsum2.ipy -- --npeinsum
OMP_NUM_THREADS=1 ipython time_einsum2.ipy
OMP_NUM_THREADS=2 ipython time_einsum2.ipy
OMP_NUM_THREADS=4 ipython time_einsum2.ipy
ipython time_einsum2.ipy
