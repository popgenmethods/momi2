#!/bin/bash -x
set -e
export PATH="$HOME/miniconda/bin:$PATH"
PKGS=$(cat .packages)
anaconda --token $ANACONDA_TOKEN upload $PKGS
