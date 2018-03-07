import json
import os
import momi
from momi.data.compressed_counts import CompressedAlleleCounts
from io import StringIO

from demo_utils import simple_five_pop_demo
import autograd.numpy as np
from collections import Counter


def test_combine_loci():
    demo = simple_five_pop_demo()
    n_loci = 1000

    num_bases = 1000
    mu = .1
    data = demo.simulate_data(
        num_bases, recoms_per_gen=0,
        num_replicates=n_loci,
        muts_per_gen=mu/num_bases,
        sampled_n_dict=dict(zip(demo.leafs, [10]*5)))

    data.extract_sfs(10).combine_loci()


def test_load_data():
    data_path = "test_data.json"
    if not os.path.exists(data_path):
        with open(data_path, "w") as data_f:
            vcf_path = "test_vcf.vcf"
            with open(vcf_path) as vcf:
                data = momi.SnpAlleleCounts.read_vcf(
                    vcf_path, ind2pop={
                        f"msp_{i}": f"Pop{i%3}"
                        for i in range(1,7)})
                data.dump(data_f)


    with open(data_path) as f:
        data = momi.SnpAlleleCounts.load(f)

    with open(data_path) as f:
        info = json.load(f)

        chrom_pos_config_key = "(chrom_id,position,config_id)"
        chrom_ids, positions, config_ids = zip(
            *info[chrom_pos_config_key])
        del info[chrom_pos_config_key]

        compressed_counts = CompressedAlleleCounts(
            np.array(info["configs"], dtype=int),
            np.array(config_ids, dtype=int))
        del info["configs"]

        #data2 =  momi.SnpAlleleCounts(
        #    chrom_ids, positions, compressed_counts,
        #    non_ascertained_pops=[], **info)
        data2 = momi.snp_allele_counts(
            chrom_ids, positions, info["populations"],
            (config[:,0] for config in compressed_counts),
            (config[:,1] for config in compressed_counts),
            length=info["length"],
            use_folded_sfs=info["use_folded_sfs"])

    assert data._sfs == data2._sfs
