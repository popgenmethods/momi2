import itertools as it
import pandas as pd
import numpy as np
import logging
from .data_structure import seg_site_configs

logger = logging.getLogger(__name__)

def read_vcftools_counts2(count_files_dict):
    """
    Reads data produced by vcftools --counts2 [--derived --keep]

    Parameters:
    count_files_dict: a dict, whose keys are populations, and values are corresponding filenames produced by vcftools --counts2
         each file contains the allele counts for the subset of individuals in the population,
         and every file should have the same number of lines, each corresponding to the same SNP

    Returns:
    momi.SegSites object
    """
    def get_lines(count_fname):
        with open(count_fname) as f:
            header = next(f).split()
            expected_header = ["CHROM", "POS", "N_ALLELES", "N_CHR",  "{COUNT}"]
            if header != expected_header:
                raise IOError("Header {} does not match expected header {}".format(header, expected_header))
            for line in f:
                yield line.split()

    count_files_dict = {k: get_lines(v) for k,v in count_files_dict.items()}
    sampled_pops, lines_list = zip(*count_files_dict.items())
    def get_counts(lines_list):
        zipped_lines = it.zip_longest(*lines_list)
        for zline in zipped_lines:
            chrom, snp, n_alleles = zline[0][:3]
            if not all(l[:3] == zline[0][:3] for l in zline):
                raise IOError("Non-matching lines {}".format(zline))
            if int(n_alleles) != 2:
                continue
            ancestral = tuple(int(l[4]) for l in zline)
            derived = tuple(int(l[5]) for l in zline)
            assert (a+d == int(l[3]) for l,a,d in zip(ancestral, derived, zline))
            if all(a == 0 for a in ancestral) or all(d == 0 for d in derived):
                continue
            yield chrom, tuple(zip(ancestral, derived))
    lines_list = get_counts(lines_list)
    lines_list = it.groupby(lines_list, key=lambda x: x[0])

    return seg_site_configs(sampled_pops,
                            ((config for c, config in chrom_lines) for chrom, chrom_lines in lines_list))

def read_plink_frq_strat(fname, polarize_pop, chunk_size = 10000):
    """
    Reads data produced by plink --within --freq counts.

    Parameters:
    fname: the filename
    polarize_pop: the population in the dataset representing the ancestral allele
    chunk_size: read the .frq.strat file in chunks of chunk_size snps

    Returns:
    momi.SegSites object
    """
    def get_chunks(locus_id, locus_rows):
        snps_grouped = it.groupby(locus_rows, lambda r: r[1])
        snps_enum = enumerate(list(snp_rows) for snp_id, snp_rows in snps_grouped)

        for chunk_num, chunk in it.groupby(snps_enum, lambda idx_snp_pair: idx_snp_pair[0] // chunk_size):
            chunk = pd.DataFrame(list(it.chain.from_iterable(snp for snp_num, snp in chunk)),
                                columns = header)
            for col_name, col in chunk.ix[:,("MAC","NCHROBS")].items():
                chunk[col_name] = [int(x) for x in col]

            # check A1, A2 agrees for every row of every SNP
            for a in ("A1","A2"):
                assert all(len(set(snp[a])) == 1 for _,snp in chunk.groupby(["CHR","SNP"]))

            # replace allele name with counts
            chunk["A1"] = chunk["MAC"]
            chunk["A2"] = chunk["NCHROBS"] - chunk["A1"]

            # drop extraneous columns, label indices
            chunk = chunk.ix[:,["SNP","CLST","A1","A2"]]
            chunk.set_index(["SNP","CLST"], inplace=True)
            chunk.columns.name = "Allele"

            ## convert to 3d array (panel)
            chunk = chunk.stack("Allele").unstack("SNP").to_panel()
            assert chunk.shape[2] == 2
            populations = list(chunk.axes[1])
            chunk = chunk.values

            ## polarize
            # remove ancestral population
            anc_pop_idx = populations.index(polarize_pop)
            anc_counts = chunk[:,anc_pop_idx,:]
            chunk = np.delete(chunk, anc_pop_idx, axis=1)
            populations.pop(anc_pop_idx)
            # check populations are same as sampled_pops
            if not sampled_pops:
                sampled_pops.extend(populations)
            assert sampled_pops == populations

            is_ancestral = [(anc_counts[:,allele] > 0) & (anc_counts[:,other_allele] == 0)
                           for allele, other_allele in ((0,1),(1,0))]

            assert np.all(~(is_ancestral[0] & is_ancestral[1]))
            chunk[is_ancestral[1],:,:] = chunk[is_ancestral[1],:,::-1]
            chunk = chunk[is_ancestral[0] | is_ancestral[1],:,:]

            # remove monomorphic sites
            polymorphic = (chunk.sum(axis=1) > 0).sum(axis=1) == 2
            chunk = chunk[polymorphic,:,:]

            yield chunk
        logger.info("Finished reading CHR {}".format(locus_id))

    with open(fname) as f:
        rows = (l.split() for l in f)
        header = next(rows)
        assert header[:2] == ["CHR", "SNP"]

        loci = (it.chain.from_iterable(get_chunks(locus_id, locus_rows))
                   for locus_id, locus_rows in it.groupby(rows, lambda r: r[0]))

        # sampled_pops is not read until the first chunk is processed
        sampled_pops = []
        first_loc = next(loci)
        first_chunk = next(first_loc)

        # add the first chunk/locus back onto the iterators
        first_loc = it.chain([first_chunk], first_loc)
        loci = it.chain([first_loc], loci)

        ret = seg_site_configs(sampled_pops, loci)
        logger.info("Finished reading {}".format(fname))
        return ret
