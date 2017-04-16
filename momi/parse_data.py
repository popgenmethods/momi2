import itertools as it
import pandas as pd
import numpy as np
from scipy import sparse
import multiprocessing as mp
import functools as ft
import subprocess
import logging
from .data_structure import seg_site_configs, SegSites, ConfigArray, CompressedAlleleCounts, _CompressedHashedCounts
from collections import defaultdict, Counter
import json
import vcf

logger = logging.getLogger(__name__)

def allele_counts_from_vcf(vcf_reader, population2samples,
                           ancestral_allele,
                           chunk_size=10000, ploidy=2):
    if ploidy != 2:
        raise NotImplementedError("vcf ploidy != 2")

    samples2columns = {s: i for i, s in enumerate(
        vcf_reader.samples)}
    populations = list(population2samples.keys())
    mat = np.zeros((len(population2samples),
                    len(samples2columns)),
                   dtype = int)
    for pop_idx, pop in enumerate(populations):
        for s in population2samples[pop]:
            mat[pop_idx, samples2columns[s]] = 1

    assert np.all(mat.sum(axis=1) == np.array([len(
        population2samples[pop]) for pop in populations]))
    assert np.all(mat.sum(axis=0) <= 1)

    mat = sparse.csr_matrix(mat)

    npops = len(populations)
    chrom = []
    pos = []
    counter = it.count(0)
    use_aa_info = (ancestral_allele is True)
    polarize_outgroup = (ancestral_allele not in (True, False, None))
    if polarize_outgroup:
        anc_pop_idx = populations.index(ancestral_allele)
        data_pops = np.array(populations)[np.arange(len(populations)) != anc_pop_idx]
    else:
        data_pops = populations
    compressed_hashed = _CompressedHashedCounts(len(data_pops))
    for chunk_num, chunk_records in it.groupby(
            vcf_reader,
            lambda x: next(counter) // chunk_size):
        logger.debug("Reading vcf lines {} to {}".format(
            chunk_num * chunk_size, (chunk_num+1) * chunk_size))

        chunk_array = vcf_records_array(chunk_records, use_aa_info)
        chunk_chrom = []
        chunk_pos = []
        for record in chunk_array.records:
            chunk_chrom.append(record.CHROM)
            chunk_pos.append(record.POS)
        gt_array = chunk_array.gt_array()

        allele_counts = [mat.dot(gt_array[:, :, a].T)
                         for a in (0, 1)]
        allele_counts = np.array(allele_counts).transpose(
            2, 1, 0)
        assert allele_counts.shape[1:] == (len(populations), 2)

        if polarize_outgroup:
            anc_pop_ac = allele_counts[:, anc_pop_idx, :]
            anc_pop_nonzero = anc_pop_ac > 0
            anc_is_ref = anc_pop_nonzero[:, 0] & (~anc_pop_nonzero[:, 1])
            anc_is_alt = (~anc_pop_nonzero[:, 0]) & anc_pop_nonzero[:, 1]

            allele_counts[anc_is_alt, :, :] = allele_counts[anc_is_alt, :, ::-1]

            to_keep = anc_is_ref | anc_is_alt
            allele_counts = allele_counts[to_keep, :, :]
            chunk_chrom = np.array(chunk_chrom)[to_keep]
            chunk_pos = np.array(chunk_pos)[to_keep]

            allele_counts = allele_counts[:, np.arange(allele_counts.shape[1]) != anc_pop_idx, :]

        chrom.extend(chunk_chrom)
        pos.extend(chunk_pos)

        for config in allele_counts:
            compressed_hashed.append(config)

    compressed_allele_counts = compressed_hashed.compressed_allele_counts()
    return SnpAlleleCounts(chrom, pos, compressed_allele_counts, data_pops)


class vcf_records_array(object):
    def __init__(self, vcf_records, require_aa):
        self.records = []
        self.require_aa = require_aa
        for record in vcf_records:
            if len(record.alleles) != 2:
                continue
            if len(record.ALT) != 1:
                continue
            if self.require_aa and "AA" not in record.INFO:
                continue
            self.records.append(record)

    def gt_array(self):
        """
        indices = [row, individual, allele]
        """
        raw_genotypes_arr = []
        for record in self.records:
            raw_genotypes_row = []
            for call in record.samples:
                if call.gt_type is None:
                    raw_genotypes_row.append(-1)
                else:
                    raw_genotypes_row.append(call.gt_type)

            raw_genotypes_arr.append(raw_genotypes_row)

        raw_genotypes_arr = np.array(
            raw_genotypes_arr, dtype=int)

        allele_counts = [2-raw_genotypes_arr,
                         np.array(raw_genotypes_arr)]
        for arr in allele_counts:
            arr[raw_genotypes_arr < 0] = 0

        ret = np.array(allele_counts).transpose(1, 2, 0)
        if self.require_aa:
            alt_is_aa = np.array([
                record.INFO["AA"] != record.REF
                for record in self.records
            ])
            ret[alt_is_aa, :, :] = ref[alt_is_aa, :, ::-1]
        return ret


class SnpAlleleCounts(object):
    """
    The allele counts for a list of SNPs.
    Includes methods for easily reading from vcf files (requires vcftools).
    Can be passed as data into SfsLikelihoodSurface to compute site frequency spectrum and likelihoods.

    Important methods:
    read_vcf_list(): read allele counts from a list of vcf files using multiple parallel cores
    dump(): save data in a compressed JSON format that can be quickly read by load()
    load(): load data stored by dump(). Much faster than read_vcf_list()
    """
    @classmethod
    def read_vcf_list(cls, vcf_list, inds2pop, n_cores=1, vcftools_path="vcftools", input_format="--gzvcf", derived=True, additional_options=[]):
        """
        Read in allele counts from a list of vcf (vcftools required).

        Files are allowed to contain multiple chromosomes;
        however, each chromosome should be in at most 1 file.

        Parameters:
        vcf_list: list of filenames
        inds2pop: dict mapping individual ids to populations
        n_cores: number of parallel cores to use
        vcftools_path: path to vcftools binary. Default assumes vcftools is on your $PATH
        input_format: one of "--vcf", "--gzvcf", "--bcf"
        derived: polarizes the allele counts using the "--derived" option in vcftools. Requires AA in INFO filed of vcf.
        additional_options: additional options to pass to vcftools (e.g. filtering by SNP quality). A list whose entries will be concatenated with spaces before being passed as command line options to vcftools

        Returns:
        SnpAlleleCounts

        Example usage:
        SnpAlleleCounts.read_vcf_list(["file1.vcf.gz", "file2.vcf.gz"], {"ID1": "Pop1", "ID2": "Pop2"}, additional_options=["--snp", "snps_to_keep.txt", "--remove-filtered-all"])
        """
        pool = mp.Pool(n_cores)
        read_vcf = ft.partial(cls.read_vcf, inds2pop=inds2pop, vcftools_path=vcftools_path,
                              input_format=input_format, derived=derived,
                              additional_options=additional_options)
        allele_counts_list = pool.map(read_vcf, vcf_list)
        logger.debug("Concatenating vcf files {}".format(vcf_list))
        ret = cls.concatenate(allele_counts_list)
        logger.debug("Finished concatenating vcf files {}".format(vcf_list))
        return ret

    @classmethod
    def read_vcf(cls, vcf, inds2pop, vcftools_path="vcftools", input_format="--gzvcf", derived=True, additional_options=[]):
        """
        Similar to read_vcf_list, but only reads in a single vcf.
        """
        logger.debug("Reading vcf {}".format(vcf))
        ret = cls._read_vcf_helper(
            vcf, inds2pop, vcftools_path, input_format, derived, additional_options)
        logger.debug("Finished reading vcf {}".format(vcf))
        return ret

    @classmethod
    def _read_vcf_helper(cls, vcf, inds2pop, vcftools_path, input_format, derived, additional_options):
        """
        Uses a recursive strategy to read in the data,
        repeatedly splitting the populations in 2,
        in order to save memory.
        """
        pop2inds = defaultdict(list)
        for ind, pop in inds2pop.items():
            pop2inds[pop].append(ind)

        if len(pop2inds) == 1:
            pop, = pop2inds.keys()
            inds, = pop2inds.values()
            cmd = [vcftools_path, input_format, vcf, "--stdout"]
            cmd += ["--counts2"]
            cmd += list(it.chain(*[["--indv", i] for i in inds]))
            cmd += list(additional_options)
            if derived:
                cmd.append("--derived")
            logger.debug("Executing command: {}".format(" ".join(cmd)))
            ret = cls.read_vcftools_counts2(pop,
                                            subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                             universal_newlines=True).stdout)
            logger.debug("Read population {}".format(pop))
            return ret
        else:
            populations = sorted(pop2inds.keys())
            half_npops = len(populations) // 2
            set0, set1 = [cls._read_vcf_helper(vcf,
                                               {ind: pop
                                                for ind, pop in inds2pop.items()
                                                if pop in pop_subset},
                                               vcftools_path, input_format, derived, additional_options)
                          for pop_subset in (populations[:half_npops], populations[half_npops:])]
            return set0.join(set1)

    @classmethod
    def read_vcftools_counts2(cls, popname, f):
        """
        Reads the output produced by
        vcftools --counts2.

        Parameters:
        popname: the name of the population to create
        f: a file-like object containing the output of vcftools --counts2
        """
        def make_stream(f):
            if isinstance(f, str):
                with open(f) as f_opened:
                    for line in f_opened:
                        yield line.split()
            else:
                for line in f:
                    yield line.split()
        f = make_stream(f)
        header = next(f)
        expected_header = ["CHROM", "POS", "N_ALLELES", "N_CHR", "{COUNT}"]
        if header != expected_header:
            raise IOError("Header {} does not match expected header {}".format(
                header, expected_header))

        populations = [popname]
        chrom_ids = []
        positions = []

        def generate_counts(f):
            for line in f:
                chrom, pos, n_alleles = line[:3]
                if int(n_alleles) != 2:
                    continue
                chrom_ids.append(chrom)
                positions.append(int(pos))
                nchrom, cnt0, cnt1 = map(int, line[3:])
                assert cnt0 + cnt1 == nchrom
                yield ((cnt0, cnt1),)
        # this fills up positions and chrom_ids as well iterating thru the
        # configs
        compressed_counts = CompressedAlleleCounts.from_iter(
            generate_counts(f), len(populations))

        return cls(chrom_ids, positions, compressed_counts, populations)

    @classmethod
    def concatenate(cls, to_concatenate):
        first = next(to_concatenate)
        populations = first.populations
        to_concatenate = it.chain([first], to_concatenate)

        chrom_ids = []
        positions = []

        def get_allele_counts(snp_allele_counts):
            if snp_allele_counts.populations != populations:
                raise ValueError(
                    "Datasets must have same populations to concatenate")
            chrom_ids.extend(snp_allele_counts.chrom_ids)
            positions.extend(snp_allele_counts.positions)
            for c in snp_allele_counts:
                yield c
            for k, v in Counter(snp_allele_counts.chrom_ids).items():
                logger.info("Added {} SNPs from Chromosome {}".format(v, k))

        compressed_counts = CompressedAlleleCounts.from_iter(it.chain.from_iterable((get_allele_counts(cnts)
                                                                                     for cnts in to_concatenate)), len(populations))
        ret = cls(chrom_ids, positions, compressed_counts, populations)
        logger.info("Finished concatenating datasets")
        return ret

    @classmethod
    def from_iter(cls, chrom_ids, positions, allele_counts, populations):
        populations = list(populations)
        return cls(list(chrom_ids),
                   list(positions),
                   CompressedAlleleCounts.from_iter(
                       allele_counts, len(populations)),
                   populations)

    @classmethod
    def load(cls, f):
        """
        Reads the compressed JSON produced
        by SnpAlleleCounts.dump().

        Parameters:
        f: a file-like object
        """
        info = json.load(f)
        logger.debug("Read allele counts from file")
        chrom_ids, positions, config_ids = zip(
            *info["(chrom_id,position,config_id)"])
        compressed_counts = CompressedAlleleCounts(
            np.array(info["configs"], dtype=int), np.array(config_ids, dtype=int))
        return cls(chrom_ids, positions, compressed_counts,
                   info["populations"])

    def dump(self, f):
        """
        Writes data in a compressed JSON format that can be
        quickly loaded.

        Parameters:
        f: a file-like object

        See Also:
        SnpAlleleCounts.load()

        Example usage:

        with gzip.open("allele_counts.gz", "wt") as f:
            allele_counts.dump(f)
        """
        print("{", file=f)
        print('\t"populations": {},'.format(
            json.dumps(list(self.populations))), file=f)
        print('\t"configs": [', file=f)
        n_configs = len(self.compressed_counts.config_array)
        for i, c in enumerate(self.compressed_counts.config_array.tolist()):
            if i != n_configs - 1:
                print("\t\t{},".format(c), file=f)
            else:
                # no trailing comma
                print("\t\t{}".format(c), file=f)
        print("\t],", file=f)
        print('\t"(chrom_id,position,config_id)": [', file=f)
        n_positions = len(self)
        for i, chrom_id, pos, config_id in zip(range(n_positions), self.chrom_ids.tolist(),
                                               self.positions.tolist(),
                                               self.compressed_counts.index2uniq.tolist()):
            if i != n_positions - 1:
                print("\t\t{},".format(json.dumps(
                    (chrom_id, pos, config_id))), file=f)
            else:
                # no trailling comma
                print("\t\t{}".format(json.dumps(
                    (chrom_id, pos, config_id))), file=f)
        print("\t]", file=f)
        print("}", file=f)

    def __init__(self, chrom_ids, positions, compressed_counts, populations):
        if len(compressed_counts) != len(chrom_ids) or len(chrom_ids) != len(positions):
            raise IOError(
                "chrom_ids, positions, allele_counts should have same length")

        self.chrom_ids = np.array(chrom_ids)
        self.positions = np.array(positions)
        self.compressed_counts = compressed_counts
        self.populations = populations

    def __getitem__(self, i):
        return self.compressed_counts[i]

    def __len__(self):
        return len(self.compressed_counts)

    def join(self, other):
        combined_pops = list(self.populations) + list(other.populations)
        if len(combined_pops) != len(set(combined_pops)):
            raise ValueError("Overlapping populations: {}, {}".format(
                self.populations, other.populations))
        if np.any(self.chrom_ids != other.chrom_ids) or np.any(self.positions != other.positions):
            raise ValueError("Chromosome & SNP IDs must be identical")
        return SnpAlleleCounts.from_iter(self.chrom_ids,
                                         self.positions,
                                         (tuple(it.chain(cnt1, cnt2))
                                          for cnt1, cnt2 in zip(self, other)),
                                         combined_pops)

    def filter(self, idxs):
        return SnpAlleleCounts(self.chrom_ids[idxs], self.positions[idxs],
                               self.compressed_counts.filter(idxs), self.populations)

    @property
    def is_polymorphic(self):
        return (self.compressed_counts.config_array.sum(axis=1) != 0).all(axis=1)[self.compressed_counts.index2uniq]

    @property
    def seg_sites(self):
        try:
            self._seg_sites
        except:
            filtered = self.filter(self.is_polymorphic)
            idx_list = [
                np.array([i for chrom, i in grouped_idxs])
                for key, grouped_idxs in it.groupby(
                        zip(filtered.chrom_ids, filtered.compressed_counts.index2uniq),
                        key=lambda x: x[0])]
            self._seg_sites = SegSites(ConfigArray(
                self.populations,
                filtered.compressed_counts.config_array
            ), idx_list)
        return self._seg_sites

    @property
    def sfs(self):
        return self.seg_sites.sfs

    @property
    def configs(self):
        return self.sfs.configs


def read_plink_frq_strat(fname, polarize_pop, chunk_size=10000):
    """
    DEPRACATED

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
        snps_enum = enumerate(list(snp_rows)
                              for snp_id, snp_rows in snps_grouped)

        for chunk_num, chunk in it.groupby(snps_enum, lambda idx_snp_pair: idx_snp_pair[0] // chunk_size):
            chunk = pd.DataFrame(list(it.chain.from_iterable(snp for snp_num, snp in chunk)),
                                 columns=header)
            for col_name, col in chunk.ix[:, ("MAC", "NCHROBS")].items():
                chunk[col_name] = [int(x) for x in col]

            # check A1, A2 agrees for every row of every SNP
            for a in ("A1", "A2"):
                assert all(len(set(snp[a])) == 1 for _,
                           snp in chunk.groupby(["CHR", "SNP"]))

            # replace allele name with counts
            chunk["A1"] = chunk["MAC"]
            chunk["A2"] = chunk["NCHROBS"] - chunk["A1"]

            # drop extraneous columns, label indices
            chunk = chunk.ix[:, ["SNP", "CLST", "A1", "A2"]]
            chunk.set_index(["SNP", "CLST"], inplace=True)
            chunk.columns.name = "Allele"

            # convert to 3d array (panel)
            chunk = chunk.stack("Allele").unstack("SNP").to_panel()
            assert chunk.shape[2] == 2
            populations = list(chunk.axes[1])
            chunk = chunk.values

            # polarize
            # remove ancestral population
            anc_pop_idx = populations.index(polarize_pop)
            anc_counts = chunk[:, anc_pop_idx, :]
            chunk = np.delete(chunk, anc_pop_idx, axis=1)
            populations.pop(anc_pop_idx)
            # check populations are same as sampled_pops
            if not sampled_pops:
                sampled_pops.extend(populations)
            assert sampled_pops == populations

            is_ancestral = [(anc_counts[:, allele] > 0) & (anc_counts[:, other_allele] == 0)
                            for allele, other_allele in ((0, 1), (1, 0))]

            assert np.all(~(is_ancestral[0] & is_ancestral[1]))
            chunk[is_ancestral[1], :, :] = chunk[is_ancestral[1], :, ::-1]
            chunk = chunk[is_ancestral[0] | is_ancestral[1], :, :]

            # remove monomorphic sites
            polymorphic = (chunk.sum(axis=1) > 0).sum(axis=1) == 2
            chunk = chunk[polymorphic, :, :]

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


