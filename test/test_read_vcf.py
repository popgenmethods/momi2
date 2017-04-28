import pytest
import momi
import demo_utils
import io
import vcf
import collections as co
import subprocess as sp

vcf_header = """
##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral allele">
##contig=<ID=1>
""".strip()


def test_read_vcf():
    demo = demo_utils.simple_admixture_3pop(n_lins=(4, 4, 6))
    theta = 100.0
    rho = 100.0
    num_loci = 5
    num_bases = 1e5

    demo_pops, demo_n = zip(*sorted(zip(demo.pops, demo.n)))
    tree_seqs = demo.demo_hist.simulate_trees(
        demo_pops, demo_n,
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
                tree_seq, demo_n)
        ))

    data = momi.seg_site_configs(demo_pops, config_list)

    combined_vcf_io = io.StringIO()
    vcf_writer = vcf.Writer(combined_vcf_io,
                            vcf_readers[0])
    for chrom, vcf_reader in enumerate(vcf_readers,
                                        start=1):
        for record in vcf_reader:
            record.CHROM = f"{chrom}"
            vcf_writer.write_record(record)
    combined_vcf_io.seek(0)

    pops2samples = co.OrderedDict()
    samples_iter = iter(vcf_readers[0].samples)
    for pop, n in zip(demo_pops, demo_n):
        pops2samples[pop] = []
        for _ in range(int(n/2)):
            pops2samples[pop].append(next(samples_iter))
    ind2pops = {i: p for p, l in pops2samples.items()
                for i in l}
    data2 = momi.SnpAlleleCounts.read_vcf(
        combined_vcf_io, ind2pops)

    assert data2.seg_sites == data

    combined_vcf_io.seek(0)
    combined_vcf_reader = vcf.Reader(combined_vcf_io)
    with open("test_vcf.vcf", "w") as f:
        header = str(vcf_header)
        f.write(header + "\n")
        print(*("#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT".split() + list(combined_vcf_reader.samples)), sep="\t", file=f)
        for record in combined_vcf_reader:
            anc_genotypes = [record.genotype(f"{s}").gt_type for s in pops2samples[demo_pops[0]]]
            if all(a == 0 for a in anc_genotypes):
                aa = record.REF
            elif all(a == 2 for a in anc_genotypes):
                aa = record.ALT[0]
            else:
                aa = "."
            line = [record.CHROM, str(record.POS), f"{record.CHROM}_{record.POS}", record.REF, str(record.ALT[0]), ".", "PASS", f"AA={aa}", "GT"] + [s.data.GT for s in record.samples]
            print(*line, sep="\t", file=f)
    #combined_vcf_io.seek(0)
    #combined_vcf_reader = vcf.Reader(combined_vcf_io)
    #with open("test_vcf.vcf", "w") as f:
    #    vcf_writer = vcf.Writer(f, vcf_readers[0])
    #    for record in combined_vcf_reader:
    #        record.ID = f"{record.CHROM}_{record.POS}"
    #        vcf_writer.write_record(record)

    with open("test_vcf.within", "w") as f:
        samples_iter = iter(combined_vcf_reader.samples)
        for pop, n in zip(demo_pops, demo_n):
            for _ in range(int(n/2)):
                s = next(samples_iter)
                print(s, s, pop, file=f)

    sp.run("plink --vcf test_vcf.vcf --double-id --within test_vcf.within --freq --out test_vcf".split())

    data3 = momi.read_plink_frq_strat("test_vcf.frq.strat", demo_pops[0])

    data4 = momi.SnpAlleleCounts.read_vcf("test_vcf.vcf", ind2pops, ancestral_alleles = demo_pops[0])
    data5 = momi.SnpAlleleCounts.read_vcf("test_vcf.vcf", {i: p for i, p in ind2pops.items() if p != demo_pops[0]}, ancestral_alleles=True)

    assert data3.sfs == data4.sfs
    assert data5.sfs == data4.sfs
