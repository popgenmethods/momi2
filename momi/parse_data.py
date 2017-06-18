
from cached_property import cached_property
import itertools as it
import pandas as pd
import numpy as np
from scipy import sparse
import multiprocessing as mp
import functools as ft
import subprocess
import logging
from collections import defaultdict, Counter, OrderedDict
import json
import re
import gzip
from .data_structure import seg_site_configs, SegSites, ConfigArray, CompressedAlleleCounts, _CompressedHashedCounts
from .util import memoize_instance

logger = logging.getLogger(__name__)


class SnpAlleleCounts(object):
    """
    The allele counts for a list of SNPs.
    Can be passed as data into SfsLikelihoodSurface to compute site frequency spectrum and likelihoods.

    Important methods:
    read_vcf(): read allele counts from a single vcf
    read_vcf_list(): read allele counts from a list of vcf files using multiple parallel cores
    dump(): save data in a compressed JSON format that can be quickly read by load()
    load(): load data stored by dump(). Much faster than read_vcf_list()
    """
    @classmethod
    def read_vcf_list(cls, vcf_list, ind2pop, n_cores=1, **kwargs):
        """
        Files may contain multiple chromosomes;
        however, chromosomes should not be spread across multiple files.

        Parameters:
        vcf_list: list of filenames
        ind2pop: dict mapping individual ids to populations
        n_cores: number of parallel cores to use

        Returns:
        SnpAlleleCounts
        """
        pool = mp.Pool(n_cores)
        read_vcf = ft.partial(cls.read_vcf, ind2pop=ind2pop, **kwargs)
        allele_counts_list = pool.map(read_vcf, vcf_list)
        logger.debug("Concatenating vcf files {}".format(vcf_list))
        ret = cls.concatenate(allele_counts_list)
        logger.debug("Finished concatenating vcf files {}".format(vcf_list))
        return ret

    @classmethod
    def read_vcf(cls, vcf_file, ind2pop,
                 ancestral_alleles = None,
                 non_ascertained_pops = []):
        """
        Parameters:
        vcf_file: stream or filename
        ind2pop: dict mapping individual IDs to populations
        ancestral_allele: str or bool or None or function
           if the name of a population, then treats that population as the outgroup to determine AA
           if None/False, uses REF to determine ancestral allele
           if True, uses AA info field to determine ancestral allele, skipping records missing this field
           if function, the function should take a vcf._Record (see pyvcf API) and return the ancestral allele
              (or return None, if the record should be skipped)
        non_ascertained_pops: list of str
           list of populations to treat as non-ascertained
        """
        if type(vcf_file) is str:
            if vcf_file.endswith(".gz"):
                openfun = lambda : gzip.open(vcf_file, "rt")
            else:
                openfun = lambda : open(vcf_file)
            with openfun() as f:
                return cls.read_vcf(
                    vcf_file=f,
                    ind2pop=ind2pop, ancestral_alleles=ancestral_alleles,
                    non_ascertained_pops=non_ascertained_pops)

        for linenum, line in enumerate(vcf_file):
            if line.startswith("##"):
                continue
            elif line.startswith("#CHROM"):
                columns = line.split()
                format_idx = columns.index("FORMAT")
                fixed_columns = columns[:(format_idx+1)]
                sample_columns = columns[(format_idx+1):]

                if ancestral_alleles and not isinstance(ancestral_alleles, str):
                    info_aa_re = re.compile(r"AA=(.)")
                    outgroup = None
                elif ancestral_alleles:
                    info_aa_re = None
                    outgroup = ancestral_alleles
                else:
                    info_aa_re = None
                    outgroup = None

                pop2idxs = defaultdict(list)
                for i, p in ind2pop.items():
                    pop2idxs[p].append(columns.index(i))
                sampled_pops = [p for p in sorted(pop2idxs.keys()) if p != outgroup]

                compressed_hashed = _CompressedHashedCounts(len(sampled_pops))
                chrom = []
                pos = []
            else:
                line = line.split()
                fixed_fields = OrderedDict(zip(fixed_columns, line))
                if not fixed_fields["FORMAT"].startswith("GT"):
                    continue

                alt = fixed_fields["ALT"]
                if "," in alt or alt == ".":
                    continue
                alleles = [fixed_fields["REF"], alt]

                aa = None
                if info_aa_re:
                    info_field_list = fixed_fields["INFO"].split(":")
                    for info_field in info_field_list:
                        aa_matched = info_aa_re.match(info_field)
                        if aa_matched:
                            aa = aa_matched.group(1)
                            break
                    if aa is None or aa == "." or aa not in alleles:
                        continue

                pop_allele_counts = {
                    pop: Counter((
                        a for i in idxs for a in line[i].split(":")[0][::2] if a != "."))
                    for pop, idxs in pop2idxs.items()
                }

                if outgroup:
                    outgroup_counts = pop_allele_counts.pop(outgroup)
                    if len(outgroup_counts) != 1:
                        continue
                    aa, = outgroup_counts.keys()
                    aa = alleles[int(aa)]
                    if aa not in alleles:
                        continue

                if not aa or aa == alleles[0]:
                    allele_order = "01"
                else:
                    assert aa == alleles[1]
                    allele_order = "10"

                config = [[pop_allele_counts[pop][a] for a in allele_order]
                          for pop in sampled_pops]
                compressed_hashed.append(config)

                chrom.append(fixed_fields["#CHROM"])
                pos.append(int(fixed_fields["POS"]))

                if linenum % 10000 == 0:
                    logger.info("Read vcf up to CHR {}, POS {}".format(
                        chrom[-1], pos[-1]))

        compressed_allele_counts = compressed_hashed.compressed_allele_counts()
        return cls(chrom, pos, compressed_allele_counts, sampled_pops, non_ascertained_pops = non_ascertained_pops)

    @classmethod
    def concatenate(cls, to_concatenate):
        to_concatenate = iter(to_concatenate)
        first = next(to_concatenate)
        populations = list(first.populations)
        non_ascertained_pops = list(first.non_ascertained_pops)
        to_concatenate = it.chain([first], to_concatenate)

        chrom_ids = []
        positions = []

        def get_allele_counts(snp_allele_counts):
            if list(snp_allele_counts.populations) != populations or list(
                    snp_allele_counts.non_ascertained_pops) != non_ascertained_pops:
                raise ValueError(
                    "Datasets must have same populations with same ascertainment to concatenate")
            chrom_ids.extend(snp_allele_counts.chrom_ids)
            positions.extend(snp_allele_counts.positions)
            for c in snp_allele_counts:
                yield c
            for k, v in Counter(snp_allele_counts.chrom_ids).items():
                logger.info("Added {} SNPs from Chromosome {}".format(v, k))

        compressed_counts = CompressedAlleleCounts.from_iter(it.chain.from_iterable((get_allele_counts(cnts)
                                                                                     for cnts in to_concatenate)), len(populations))
        ret = cls(chrom_ids, positions, compressed_counts, populations, non_ascertained_pops = non_ascertained_pops)
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

        chrom_pos_config_key = "(chrom_id,position,config_id)"
        chrom_ids, positions, config_ids = zip(
            *info[chrom_pos_config_key])
        del info[chrom_pos_config_key]

        compressed_counts = CompressedAlleleCounts(
            np.array(info["configs"], dtype=int), np.array(config_ids, dtype=int))
        del info["configs"]

        return cls(chrom_ids, positions, compressed_counts,
                   **info)

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
        if self.non_ascertained_pops:
            print('\t"non_ascertained_pops": {},'.format(
                json.dumps(list(self.non_ascertained_pops))), file=f)
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

    def __init__(self, chrom_ids, positions, compressed_counts, populations, non_ascertained_pops=[]):
        if len(compressed_counts) != len(chrom_ids) or len(chrom_ids) != len(positions):
            raise IOError(
                "chrom_ids, positions, allele_counts should have same length")

        self.chrom_ids = np.array(chrom_ids)
        self.positions = np.array(positions)
        self.compressed_counts = compressed_counts
        self.populations = populations
        self.non_ascertained_pops = non_ascertained_pops
        self._subset_populations_cache = {}

    def _chunk_data(self, n_chunks):
        chunk_len = len(self.chrom_ids) / float(n_chunks)
        new_pos = list(range(len(self.chrom_ids)))
        new_chrom = [int(np.floor(i / chunk_len)) for i in new_pos]
        return SnpAlleleCounts(new_chrom, new_pos, self.compressed_counts, self.populations, self.non_ascertained_pops)

    #@cached_property
    #def _p_missing(self):
    #    config_arr = self.compressed_counts.config_array
    #    counts = self.compressed_counts.count_configs()
    #    weights = counts / float(np.sum(counts))
    #    sampled_n = self.compressed_counts.n_samples
    #    n_pops = len(self.populations)

    #    p_miss = (sampled_n - np.sum(config_arr, axis=2)) / sampled_n
    #    return np.einsum(
    #        "i, ij->j", weights, p_miss)

    @cached_property
    def _p_missing(self):
        counts = self.compressed_counts.count_configs()
        sampled_n = self.compressed_counts.n_samples
        n_pops = len(self.populations)

        config_arr = self.compressed_counts.config_array
        # augment config_arr to contain the missing counts
        n_miss = sampled_n - np.sum(config_arr, axis=2)
        config_arr = np.concatenate((config_arr, np.reshape(
            n_miss, list(n_miss.shape)+[1])), axis=2)

        ret = []
        for i in range(n_pops):
            n_valid = []
            for allele in (0, 1, -1):
                removed = np.array(config_arr)
                removed[:, i, allele] -= 1
                valid_removed = (removed[:, i, allele] >= 0) & np.all(
                    np.sum(removed[:,:,:2], axis=1) > 0, axis=1)

                n_valid.append(np.sum((counts * config_arr[:, i, allele])[valid_removed]))
            ret.append(n_valid[-1] / float(sum(n_valid)))
        return np.array(ret)

    @memoize_instance
    def _jacknife_pairwise_missingness(self, n_jackknife_blocks):
        config_arr = self.compressed_counts.config_array
        idxs = self.compressed_counts.index2uniq

        blocklen = len(idxs) / float(n_jackknife_blocks)
        block = np.array(np.floor(np.arange(len(idxs)) / blocklen),
                         dtype=int)
        block = iter(block)

        minlength = int(np.max(idxs))+1
        block_counts = np.stack([
            np.bincount(list(block_idxs), minlength=minlength)
            for grp, block_idxs in it.groupby(
                    idxs, lambda x: next(block))
        ])
        total_counts = np.sum(block_counts, axis=0)

        jackknife_counts = total_counts - block_counts
        ret = self._est_pairwise_missing(jackknife_counts)
        #assert np.allclose(np.mean(ret, axis=-1), self._pairwise_missingness)
        return ret

    @cached_property
    def _pairwise_missingness(self):
        return self._est_pairwise_missing()

    def _est_pairwise_missing(self, counts=None):
        if counts is None:
            counts = self.compressed_counts.count_configs()
        sampled_n = self.compressed_counts.n_samples
        n_pops = len(self.populations)

        config_arr = self.compressed_counts.config_array
        # augment config_arr to contain the missing counts
        n_miss = sampled_n - np.sum(config_arr, axis=2)
        config_arr = np.concatenate((config_arr, np.reshape(
            n_miss, list(n_miss.shape)+[1])), axis=2)

        ret = []
        for i in range(n_pops):
            ret.append([])
            for j in range(n_pops):
                n_valid = []
                n_i = sampled_n[i]
                n_j = sampled_n[j]
                if i == j:
                    n_j -= 1
                if n_i + n_j < 2:
                    if len(counts.shape) > 1:
                        assert len(counts.shape) == 2
                        assert counts.shape[1] == config_arr.shape[0]
                        ret[-1].append(np.zeros(counts.shape[0]))
                    else:
                        ret[-1].append(0)
                    continue

                for a_i in (0, 1, -1):
                    n_valid.append([])
                    for a_j in (0, 1, -1):
                        n_ai = config_arr[:, i, a_i]
                        n_aj = config_arr[:, j, a_j]
                        if i == j and a_i == a_j:
                            n_aj = n_aj - 1
                        removed = np.array(config_arr)
                        removed[:, i, a_i] -= 1
                        removed[:, j, a_j] -= 1

                        valid_removed = (n_ai > 0) & (n_aj > 0) & np.all(
                            np.sum(removed[:,:,:2], axis=1) > 0, axis=1)

                        n_valid[-1].append(np.sum(
                            (counts * n_ai * n_aj).T[valid_removed,...],
                            axis=0))
                n_valid = np.array(n_valid, dtype=float)
                ret[-1].append(1.0 - np.sum(
                    n_valid[:2,:,...][:,:2,...], axis=(0,1)) / np.sum(n_valid, axis=(0,1)))

        return np.array(ret)

    #@cached_property
    #def _pairwise_missingness(self):
    #    config_arr = self.compressed_counts.config_array
    #    counts = self.compressed_counts.count_configs()
    #    weights = counts / float(np.sum(counts))
    #    sampled_n = self.compressed_counts.n_samples
    #    n_pops = len(self.populations)

    #    pairwise_missing_probs = np.zeros((
    #        len(config_arr), n_pops, n_pops))
    #    n_miss = sampled_n - np.sum(config_arr, axis=2)
    #    p_miss1 = n_miss / sampled_n
    #    p_miss2 = n_miss / (sampled_n-1)
    #    for i, derived_pop in enumerate(self.populations):
    #        for j, anc_pop in enumerate(self.populations):
    #            p_imiss = p_miss1[:,i]
    #            if i == j:
    #                p_jmiss = p_miss2[:,j]
    #            else:
    #                p_jmiss = p_miss1[:,j]
    #            pairwise_missing_probs[:,i,j] = p_imiss + (1-p_imiss)*p_jmiss
    #    assert np.allclose(pairwise_missing_probs, np.transpose(pairwise_missing_probs, (0, 2, 1)))
    #    return np.einsum(
    #        "i, ijk->jk", weights, pairwise_missing_probs)

    def subset_populations(self, populations, non_ascertained_pops=None):
        if non_ascertained_pops is not None:
            non_ascertained_pops = tuple(non_ascertained_pops)
        return self._subset_populations(tuple(populations), non_ascertained_pops)

    @memoize_instance
    def _subset_populations(self, populations, non_ascertained_pops):
        if non_ascertained_pops is None:
            non_ascertained_pops = [p for p in self.non_ascertained_pops
                                    if p in populations]

        newPopIdx_to_oldPopIdx = np.array([
            self.populations.index(p) for p in populations], dtype=int)

        uniq_new_configs = CompressedAlleleCounts.from_iter(
            self.compressed_counts.config_array[:, newPopIdx_to_oldPopIdx, :],
            len(populations), sort=False)

        new_compressed_configs = CompressedAlleleCounts(
            uniq_new_configs.config_array,
            [uniq_new_configs.index2uniq[i] for i in self.compressed_counts.index2uniq],
            sort=False)

        return SnpAlleleCounts(
            self.chrom_ids, self.positions,
            new_compressed_configs, populations,
            non_ascertained_pops)

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
        return (self.compressed_counts.config_array[:, self.ascertainment_pop, :].sum(axis=1) != 0).all(axis=1)[self.compressed_counts.index2uniq]

    @property
    def ascertainment_pop(self):
        return np.array([(pop not in self.non_ascertained_pops)
                         for pop in self.populations])

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
                filtered.compressed_counts.config_array,
                ascertainment_pop = self.ascertainment_pop
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


