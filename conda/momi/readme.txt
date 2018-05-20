From project root, do:

   conda build conda/momi -c defaults -c conda-forge -c jackkamm


Note that defaults channel MUST be higher priority than conda-forge or bad things happen!
e.g., see https://github.com/conda/conda-build/issues/2523#issuecomment-346963138
