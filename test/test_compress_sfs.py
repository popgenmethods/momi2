from __future__ import division
import pytest
from momi import make_demography, simulate_ms, sfs_list_from_ms, aggregate_sfs, CompressedOrderedSfs


def test_compress_sfs():
    demo = make_demography("-I 6 2 2 10 10 10 10 -ej .1 1 2 -ej .2 2 3 -ej .3 3 4 -ej .4 4 5 -ej .5 5 6")
    sfs = aggregate_sfs(sfs_list_from_ms(simulate_ms(demo, num_sims=1000, theta=1.0),
                                         demo.n_at_leaves))

    print len(sfs)
    compressed_sfs = CompressedOrderedSfs(sfs, 200, range(6) + range(6) + range(2,6) * 8, init_draws=6)
    #compressed_sfs = CompressedOrderedSfs(sfs, 100, range(6) + range(6) + [2] * 8 + [3] * 8 + [4] * 8 + [5] * 8, init_draws=6)
    print "\n".join(map(str, compressed_sfs._curr_mass.most_common()))

if __name__=="__main__":
    test_compress_sfs()
