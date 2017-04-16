import pytest
import momi
import demo_utils
import io
import vcf
import collections as co
import subprocess as sp

def test_read_vcf():
    demo = demo_utils.simple_admixture_3pop(n_lins=(4, 4, 6))
    theta = 100.0
    rho = 100.0
    num_loci = 5
    num_bases = 1e5

    tree_seqs = demo.demo_hist.simulate_trees(
        demo.pops, demo.n,
        mutation_rate=theta / num_bases,
        recombination_rate=rho / num_bases,
        length=num_bases,
        num_replicates=num_loci,
    )
    vcf_readers = []
    config_list = []
    for tree_seq in tree_seqs:
        vcf_io = io.StringIO()
        tree_seq.write_vcf(vcf_io, ploidy=2)
        vcf_io.seek(0)
        vcf_readers.append(vcf.Reader(vcf_io))
        config_list.append(list(
            momi.demography.get_treeseq_configs(
                tree_seq, demo.n)
        ))

    data = momi.seg_site_configs(demo.pops, config_list)

    combined_vcf_io = io.StringIO()
    vcf_writer = vcf.Writer(combined_vcf_io,
                            vcf_readers[0])
    for chrom, vcf_reader in enumerate(vcf_readers,
                                        start=1):
        for record in vcf_reader:
            record.CHROM = f"{chrom}"
            vcf_writer.write_record(record)
    combined_vcf_io.seek(0)
    combined_vcf_reader = vcf.Reader(combined_vcf_io)

    pops2samples = co.OrderedDict()
    samples_iter = iter(combined_vcf_reader.samples)
    for pop, n in zip(demo.pops, demo.n):
        pops2samples[pop] = []
        for _ in range(int(n/2)):
            pops2samples[pop].append(next(samples_iter))
    data2 = momi.parse_data.allele_counts_from_vcf(
        combined_vcf_reader, pops2samples, False)

    assert data2.seg_sites == data

    combined_vcf_io.seek(0)
    combined_vcf_reader = vcf.Reader(combined_vcf_io)
    with open("test_vcf.vcf", "w") as f:
        vcf_writer = vcf.Writer(f, vcf_readers[0])
        for record in combined_vcf_reader:
            record.ID = f"{record.CHROM}_{record.POS}"
            vcf_writer.write_record(record)

    with open("test_vcf.within", "w") as f:
        samples_iter = iter(combined_vcf_reader.samples)
        for pop, n in zip(demo.pops, demo.n):
            for _ in range(int(n/2)):
                s = next(samples_iter)
                print(s, s, pop, file=f)

    sp.run("plink --vcf test_vcf.vcf --double-id --within test_vcf.within --freq --out test_vcf".split())

    data3 = momi.read_plink_frq_strat("test_vcf.frq.strat", demo.pops[0])

    combined_vcf_io.seek(0)
    combined_vcf_reader = vcf.Reader(combined_vcf_io)
    data4 = momi.parse_data.allele_counts_from_vcf(combined_vcf_reader, pops2samples, demo.pops[0])

    assert data3.sfs == data4.sfs
