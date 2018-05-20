To build for conda, do:

   conda build conda/ -c defaults -c conda-forge -c bioconda


Note that defaults channel MUST be higher priority than conda-forge or bad things happen!
e.g., see https://github.com/conda/conda-build/issues/2523#issuecomment-346963138
