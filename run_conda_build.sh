# defaults channel MUST be higher priority than conda-forge or bad things happen!
# e.g., https://github.com/conda/conda-build/issues/2523#issuecomment-346963138
conda build conda/ -c defaults -c conda-forge -c bioconda
