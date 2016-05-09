import pytest
from momi import simulate_ms
import momi
from test_ms import ms_path, scrm_path
from io import StringIO

from demo_utils import simple_five_pop_demo
import autograd.numpy as np
from collections import Counter

def test_readwrite_segsites_parse_ms_equal():
    demo = simple_five_pop_demo(n_lins=(10,10,10,10,10)).rescaled()
    n_loci = 1000
    
    #data = momi.simulate_ms(scrm_path, demo,
    #                        num_loci=n_loci, mut_rate=.1)
    raw_ms = momi.simulate_ms(scrm_path, demo, num_loci=n_loci, mut_rate=.1,
                              raw_output=True)
    data = momi.parse_ms.seg_sites_from_ms(raw_ms, demo.sampled_pops)
    assert data.n_loci == n_loci

    strio = StringIO()    
    momi.write_seg_sites(strio, data)

    strio = StringIO(strio.getvalue())
    data2 = momi.read_seg_sites(strio)

    assert data == data2
    assert data.sfs == data2.sfs

    raw_ms.seek(0)
    sfs_dict = Counter()
    curr_lines = None
    ind2pop = sum([[i]*n for i,n in enumerate(demo.sampled_n)], [])
    def update_sfs_dict():
        assert len(curr_lines) == np.sum(demo.sampled_n)
        n_snps = len(curr_lines[0])
        config_array = np.zeros((n_snps, len(demo.sampled_pops), 2), dtype=int)
        for i,snp in enumerate(zip(*curr_lines)):
            snp = map(int, snp)
            for pop,allele in zip(ind2pop, snp):
                config_array[i,pop,allele] += 1
        for config in config_array:
            config = tuple(map(tuple, config))
            sfs_dict[config] += 1
    for line in raw_ms:
        line = line.strip()
        if line == '':
            continue
        elif line[0] in ('0','1'):
            if curr_lines is not None:
                curr_lines.append(line)
        elif line=="//":
            if curr_lines is not None and len(curr_lines) != 0:
                update_sfs_dict()
            curr_lines = []
    update_sfs_dict()
    sfs_dict = dict(sfs_dict)
    assert sfs_dict == data.sfs.to_dict()
        

