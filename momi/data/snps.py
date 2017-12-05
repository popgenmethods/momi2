import ast
import itertools as it
import numpy as np
from cached_property import cached_property
import multiprocessing as mp
import functools as ft
from collections import defaultdict, Counter, OrderedDict
import json
import re
import gzip
import logging
from .compressed_counts import CompressedAlleleCounts, _CompressedHashedCounts
from .config_array import ConfigArray
from .seg_sites import SegSites
from ..util import memoize_instance


logger = logging.getLogger(__name__)


class SnpAlleleCounts(object):
    """
    The allele counts for a list of SNPs.
    Can be passed as data into SfsLikelihoodSurface
    to compute site frequency spectrum and likelihoods.

    Important methods:
    read_vcf(): read allele counts from a single vcf
    read_vcf_list(): read allele counts from a list of vcf files using
                     multiple parallel cores
    dump(): save data in a compressed JSON format that can be
            quickly read by load()
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
                 ancestral_alleles=None,
                 non_ascertained_pops=[]):
        """
        Parameters:
        vcf_file: stream or filename
        ind2pop: dict mapping individual IDs to populations
        ancestral_allele: str or bool or None or function
           if the name of a population, then treats that population as
              the outgroup to determine AA
           if None/False, uses REF to determine ancestral allele
           if True, uses AA info field to determine ancestral allele,
              skipping records missing this field
           if function, the function should take a vcf._Record (see pyvcf API)
              and return the ancestral allele
              (or return None, if the record should be skipped)
        non_ascertained_pops: list of str
           list of populations to treat as non-ascertained
        """
        if type(vcf_file) is str:
            if vcf_file.endswith(".gz"):
                def openfun():
                    return gzip.open(vcf_file, "rt")
            else:
                def openfun():
                    return open(vcf_file)
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

                if ancestral_alleles and not isinstance(
                        ancestral_alleles, str):
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
                sampled_pops = [p for p in sorted(pop2idxs.keys())
                                if p != outgroup]

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
                    pop: Counter(a for i in idxs
                                 for a in line[i].split(":")[0][::2]
                                 if a != ".")
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
        return cls(chrom, pos, compressed_allele_counts, sampled_pops,
                   non_ascertained_pops=non_ascertained_pops)

    @classmethod
    def concatenate(cls, to_concatenate):
        to_concatenate = iter(to_concatenate)
        first = next(to_concatenate)
        populations = list(first.populations)
        nonascertained = list(first.non_ascertained_pops)
        to_concatenate = it.chain([first], to_concatenate)

        chrom_ids = []
        positions = []
        index2uniq = []

        compressed_hashes = _CompressedHashedCounts(len(populations))

        for snp_cnts in to_concatenate:
            if any([list(snp_cnts.populations) != populations,
                    list(snp_cnts.non_ascertained_pops) != nonascertained]):
                raise ValueError(
                    "Datasets must have same populations with same"
                    " ascertainment to concatenate")
            old2new_uniq = []
            for config in snp_cnts.compressed_counts.config_array:
                compressed_hashes.append(config)
                old2new_uniq.append(compressed_hashes.index2uniq(-1))

            assert len(snp_cnts.chrom_ids) == len(snp_cnts.compressed_counts.index2uniq)
            assert len(snp_cnts.chrom_ids) == len(snp_cnts.positions)
            for chrom, pos, old_uniq in zip(snp_cnts.chrom_ids,
                                            snp_cnts.positions,
                                            snp_cnts.compressed_counts.index2uniq):
                chrom_ids.append(chrom)
                positions.append(pos)
                index2uniq.append(old2new_uniq[old_uniq])

            for k, v in Counter(snp_cnts.chrom_ids).items():
                logger.info("Added {} SNPs from Chromosome {}".format(v, k))

        compressed_counts = CompressedAlleleCounts(
            compressed_hashes.config_array(), index2uniq)
        ret = cls(chrom_ids, positions, compressed_counts, populations,
                  non_ascertained_pops=nonascertained)
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

    #@classmethod
    #def load(cls, f):
    #    """
    #    Reads the compressed JSON produced
    #    by SnpAlleleCounts.dump().

    #    Parameters:
    #    f: a file-like object
    #    """
    #    info = json.load(f)
    #    logger.debug("Read allele counts from file")

    #    chrom_pos_config_key = "(chrom_id,position,config_id)"
    #    chrom_ids, positions, config_ids = zip(
    #        *info[chrom_pos_config_key])
    #    del info[chrom_pos_config_key]

    #    compressed_counts = CompressedAlleleCounts(
    #        np.array(info["configs"], dtype=int),
    #        np.array(config_ids, dtype=int))
    #    del info["configs"]

    #    return cls(chrom_ids, positions, compressed_counts,
    #               **info)

    @classmethod
    def load(cls, f):
        items = {}
        items_re = re.compile(r'\s*"(.*)":\s*(.*),\s*\n')
        config_re = re.compile(r'\s*"configs":\s*\[\s*\n')
        chrom_pos_idx_re = re.compile(
            r'(\s*)"\(chrom_id,position,config_id\)":(\s*)\[(\s*)\n')
        for line in f:
            items_matched = items_re.match(line)
            config_matched = config_re.match(line)
            chrom_pos_idx_matched = chrom_pos_idx_re.match(line)
            if chrom_pos_idx_matched:
                logger.info("Reading SNPs")
                line_re = re.compile(r'\s*\[(.*)\],?\s*\n')
                chrom_ids = []
                positions = []
                config_ids = []
                for i, line in enumerate(f):
                    if i % 100000 == 0 and i > 0:
                        logger.info(i)
                    try:
                        curr = line_re.match(line).group(1)
                    except AttributeError:
                        assert line == "\t]\n"
                        break
                    else:
                        chrom, pos, idx = curr.split(",")
                        chrom = chrom.strip()
                        #assert chrom[0] == chrom[-1] == '"'
                        chrom_ids.append(chrom)
                        #positions.append(int(pos))
                        positions.append(pos)
                        config_ids.append(int(idx))
                logger.info("Finished reading SNPs")
            elif config_matched:
                line_re = re.compile(r'\s*\[(.*)\],?\s*\n')
                configs = []
                logger.info("Reading unique configs")
                for i, line in enumerate(f):
                    if i % 100000 == 0:
                        if i > 0:
                            logger.info(i)
                            configs[-1] = np.array(configs[-1],
                                                   dtype=int)
                        configs.append([])
                    try:
                        conf = line_re.match(line).group(1)
                    except AttributeError:
                        assert line == "\t],\n"
                        break
                    else:
                        #configs[-1].append(ast.literal_eval(conf))
                        # ast.literal_eval is slow
                        conf = conf.replace("[", " ")
                        conf = conf.replace(",", " ")
                        conf = [[int(x_i) for x_i in x.split()]
                                for x in conf.split("]")[:-1]]
                        configs[-1].append(conf)
                configs[-1] = np.array(configs[-1], dtype=int)
                configs = np.concatenate(configs)
                logger.info("Finished reading configs")
            elif items_matched:
                items[items_matched.group(1)] = ast.literal_eval(
                    items_matched.group(2))

        compressed_counts = CompressedAlleleCounts(
            configs, config_ids,
            sort=False)

        return cls(chrom_ids, positions, compressed_counts,
                   **items)



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
        for i, chrom_id, pos, config_id in zip(
                range(n_positions), self.chrom_ids.tolist(),
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

    def __init__(self, chrom_ids, positions,
                 compressed_counts, populations,
                 non_ascertained_pops=[]):
        if any([len(compressed_counts) != len(chrom_ids),
                len(chrom_ids) != len(positions)]):
            raise IOError(
                "chrom_ids, positions, allele_counts should have same length")

        self.chrom_ids = np.array(chrom_ids)
        self.positions = np.array(positions)
        self.compressed_counts = compressed_counts
        self.populations = populations
        self.non_ascertained_pops = non_ascertained_pops
        self._subset_populations_cache = {}

    @memoize_instance
    def chunk_data(self, n_chunks):
        chunk_len = len(self.chrom_ids) / float(n_chunks)
        new_pos = list(range(len(self.chrom_ids)))
        new_chrom = [int(np.floor(i / chunk_len)) for i in new_pos]
        return SnpAlleleCounts(
            new_chrom, new_pos, self.compressed_counts,
            self.populations, self.non_ascertained_pops)

    @cached_property
    def _p_missing(self):
        """
        Estimates the probability that a random allele
        from each population is missing.

        To estimate missingness, first remove random allele;
        if the resulting config is monomorphic, then ignore.
        If polymorphic, then count whether the removed allele
        is missing or not.

        This avoids bias from fact that we don't observe
        some polymorphic configs that appear monomorphic
        after removing the missing alleles.
        """
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
                # configs with removed allele
                removed = np.array(config_arr)
                removed[:, i, allele] -= 1
                # is the resulting config polymorphic?
                valid_removed = (removed[:, i, allele] >= 0) & np.all(
                    np.sum(removed[:, :, :2], axis=1) > 0, axis=1)

                # sum up the valid configs
                n_valid.append(np.sum(
                    (counts * config_arr[:, i, allele])[valid_removed]))
            # fraction of valid configs with missing additional allele
            ret.append(n_valid[-1] / float(sum(n_valid)))
        return np.array(ret)

    def fstats(self, sampled_n_dict=None):
        return self.sfs.fstats(sampled_n_dict)

    def subset_populations(self, populations, non_ascertained_pops=None):
        if non_ascertained_pops is not None:
            non_ascertained_pops = tuple(non_ascertained_pops)
        return self._subset_populations(tuple(populations),
                                        non_ascertained_pops)

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
            [uniq_new_configs.index2uniq[i]
             for i in self.compressed_counts.index2uniq],
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
        if np.any((self.chrom_ids != other.chrom_ids)
                  | (self.positions != other.positions)):
            raise ValueError("Chromosome & SNP IDs must be identical")
        return SnpAlleleCounts.from_iter(self.chrom_ids,
                                         self.positions,
                                         (tuple(it.chain(cnt1, cnt2))
                                          for cnt1, cnt2 in zip(self, other)),
                                         combined_pops)

    def filter(self, idxs):
        return SnpAlleleCounts(self.chrom_ids[idxs], self.positions[idxs],
                               self.compressed_counts.filter(idxs),
                               self.populations)

    def down_sample(self, sampled_n_dict):
        pops, sub_n = zip(*sampled_n_dict.items())
        pop_idxs = [self.populations.index(p) for p in pops]
        def sub_counts():
            for config in self:
                config = list(config)
                for i, n in zip(pop_idxs, sub_n):
                    curr_n = sum(config[i])
                    if curr_n > n:
                        old_anc, old_der = config[i]
                        new_anc = np.random.hypergeometric(
                            old_anc, old_der, n)
                        new_der = n - new_anc
                        config[i] = (new_anc, new_der)
                yield config

        return type(self).from_iter(self.chrom_ids, self.positions,
                                    sub_counts(), self.populations)


    @property
    def is_polymorphic(self):
        configs = self.compressed_counts.config_array
        ascertained_only = configs[:, self.ascertainment_pop, :]
        ascertained_is_poly = (ascertained_only.sum(axis=1) != 0).all(axis=1)
        return ascertained_is_poly[self.compressed_counts.index2uniq]

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
                        zip(filtered.chrom_ids,
                            filtered.compressed_counts.index2uniq),
                        key=lambda x: x[0])]
            self._seg_sites = SegSites(ConfigArray(
                self.populations,
                filtered.compressed_counts.config_array,
                ascertainment_pop=self.ascertainment_pop
            ), idx_list)
        return self._seg_sites

    @property
    def sfs(self):
        return self.seg_sites.sfs

    @property
    def configs(self):
        return self.sfs.configs
