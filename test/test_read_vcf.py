import pytest
import momi
import demo_utils
import io
import vcf
import collections as co
import subprocess as sp

def test_read_vcf():
    sampled_n_dict = {"a":4,"b":4,"c":6}
    demo = demo_utils.simple_admixture_3pop()
    theta = 100.0
    rho = 100.0
    num_loci = 5
    num_bases = 1e5

    demo.simulate_vcf(
        "test_vcf", recoms_per_gen=rho/num_bases,
        length=num_bases, muts_per_gen=theta/num_bases,
        sampled_n_dict=sampled_n_dict, random_seed=1234,
        force=True)

    data = demo.simulate_data(
        recoms_per_gen=rho/num_bases,
        length=num_bases, muts_per_gen=theta/num_bases,
        sampled_n_dict=sampled_n_dict, num_replicates=1,
        random_seed=1234)

    data2 = momi.SnpAlleleCounts.read_vcf(
        'test_vcf.vcf.gz', ind2pop={f"{pop}_{i}": pop for pop, n in sampled_n_dict.items() for i in range(n)})

    assert data._sfs == data2._sfs.subset_populations(data._sfs.sampled_pops)

    # TODO: test that concatenating datasets from multiple vcfs works?

def test_read_vcf_folded():
    sampled_n_dict = {"a":4,"b":4,"c":6}
    demo = demo_utils.simple_admixture_3pop()
    theta = 100.0
    rho = 100.0
    num_loci = 5
    num_bases = 1e5

    demo.simulate_vcf(
        "test_vcf_folded", recoms_per_gen=rho/num_bases,
        length=num_bases, muts_per_gen=theta/num_bases,
        sampled_n_dict=sampled_n_dict, random_seed=1234,
        force=True, print_aa=False)

    data = demo.simulate_data(
        recoms_per_gen=rho/num_bases,
        length=num_bases, muts_per_gen=theta/num_bases,
        sampled_n_dict=sampled_n_dict, num_replicates=1,
        random_seed=1234)

    data2 = momi.SnpAlleleCounts.read_vcf(
        'test_vcf_folded.vcf.gz', ind2pop={f"{pop}_{i}": pop for pop, n in sampled_n_dict.items() for i in range(n)},
        ancestral_alleles=False)

    assert data._sfs.fold() == data2._sfs.subset_populations(data._sfs.sampled_pops).fold()

    # TODO: test that concatenating datasets from multiple vcfs works?
