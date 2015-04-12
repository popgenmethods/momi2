TODO:

***
clean up from_newick, to_newick in demography.py

Either:
get rid of it entirely
or
change it to use from_ms
or
implement to_ms, so Demography.simulate_sfs() can work for newick demos

***
for simulate_sfs, make theta=None work when -r is set
(right now it counts up the total branch length, but doesn't know
how to deal with multiple trees)

***
replace cached_property module (v1.0.0 not compatible with using nx.DiGraph)
***
make state of derived counts, a property of SumProduct, instead of Demography
***
don't make n_max a field of size_history;
instead have etjj and sfs take n as a parameter, and return
the appropriate vector
***
make exponential size history work with growth rate 0
***
add option to simulate from ms instead of scrm
