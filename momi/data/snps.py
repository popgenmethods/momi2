import os
import itertools as it
from collections import defaultdict, Counter
import json
import re
import gzip
import logging
import numpy as np
import pysam
from cached_property import cached_property
from .configurations import ConfigList
from .sfs import Sfs
from ..util import memoize_instance
from .compressed_counts import (
    CompressedAlleleCounts, _CompressedHashedCounts, _CompressedList)


logger = logging.getLogger(__name__)


def snp_allele_counts(chrom_ids, positions, populations,
                      ancestral_counts, derived_counts,
                      length=None, use_folded_sfs=False):
    """Create a :class:`SnpAlleleCounts` object.

    :param iterator chrom_ids: the CHROM at each SNP
    :param iterator positions: the POS at each SNP
    :param list populations: the population names

    :param iterator ancestral_counts: iterator over tuples of \
    the ancestral counts at each SNP. tuple length should \
    be the same as the number of populations

    :param iterator derived_counts: iterator over tuples of \
    the derived counts at each SNP. tuple length should \
    be the same as the number of populations

    :param bool use_folded_sfs: whether the folded SFS should \
    be used when computing likelihoods. Set this to True if there is \
    uncertainty about the ancestral allele.

    :rtype: :class:`SnpAlleleCounts`
    """
    config_iter = (tuple(zip(a, d)) for a, d in
                   zip(ancestral_counts, derived_counts))
    chrom_ids = _CompressedList(chrom_ids)
    positions = list(positions)
    return SnpAlleleCounts(chrom_ids, positions,
                           CompressedAlleleCounts.from_iter(
                               config_iter, len(populations)),
                           populations, use_folded_sfs,
                           [], length,
                           len(chrom_ids), 0)


class SnpAlleleCounts(object):
    """
    The allele counts for a list of SNPs.

    To create a :class:`SnpAlleleCounts` object,
    use :func:`SnpAlleleCounts.read_vcf`, :func:`SnpAlleleCounts.load`,
    or :func:`snp_allele_counts`. Do NOT call the class constructor directly,
    it is for internal use only.
    """
    @classmethod
    def read_vcf(cls, vcf_file, ind2pop,
                 bed_file=None, ancestral_alleles=True,
                 info_aa_field="AA"):
        """Read in a VCF file and return the allele counts at biallelic SNPs.

        :param str vcf_file: VCF file to read in. "-" reads from stdin.
        :param dict ind2pop: Maps individual samples to populations.
        :param str,None bed_file: BED accessibility regions file. \
        Only regions in the BED file are read from the VCF. The BED file \
        is also used to determine the size of the data in bases, so the same \
        BED file should NOT be used across multiple VCFs (otherwise regions will be double \
        counted towards the data length). If no BED is provided, all SNPs in the VCF are read, and the length \
        of the data is set to be unknown.
        :param bool,str ancestral_alleles: If True, use the AA INFO field to \
        determine ancestral allele, skipping SNPs missing this field. \
        If False, ignore ancestral allele information, and set the \
        ``SnpAlleleCounts.use_folded_sfs`` property so that \
        the folded SFS is used by default when computing likelihoods. \
        Finally, if ``ancestral_alleles`` is a string that is the name \
        of a population in ``ind2pop``, then treat that population as an outgroup, \
        using its consensus allele as the ancestral allele; SNPs without \
        consensus are skipped.
        :param str info_aa_field: The INFO field to read Ancestral Allele from. \
        Default is "AA". Only has effect if ``ancestral_alleles=True``.

        :rtype: :class:`SnpAlleleCounts`
        """
        bcf_in = pysam.VariantFile(vcf_file)

        # subset samples for faster VCF parsing
        bcf_in.subset_samples(list(ind2pop.keys()))
        samples = list(bcf_in.header.samples)
        assert set(samples) == set(ind2pop.keys())

        # extract populations, samples
        pop2idxs = defaultdict(list)
        for ind, pop in ind2pop.items():
            pop2idxs[pop].append(samples.index(ind))
        sampled_pops = sorted(
            p for p in pop2idxs.keys() if p != ancestral_alleles)

        # objects to store chrom, pos, configs
        chrom_list = _CompressedList()
        pos_list = []
        compressed_hashed = _CompressedHashedCounts(len(sampled_pops))
        excluded = []

        # pre-allocate objects for temporary configs
        pop_allele_counts = {pop: Counter() for pop in pop2idxs.keys()}
        config = np.zeros((len(sampled_pops), 2), dtype=int)

        if bed_file:
            def open_bed():
                if bed_file.endswith(".gz"):
                    return gzip.open(bed_file, "rt")
                else:
                    return open(bed_file)
            length = 0
            with open_bed() as bed:
                for line in bed:
                    line = line.split()
                    contig = line[0]
                    start, end = map(int, line[1:3])
                    length += (end - start)
                    fetcher = bcf_in.fetch(
                        contig, start, end)
                    cls._read_vcf_helper(
                        fetcher, chrom_list, pos_list,
                        compressed_hashed, excluded,
                        ancestral_alleles, pop2idxs,
                        sampled_pops, pop_allele_counts, config,
                        info_aa_field)
        else:
            length = None
            logger.warn("No BED provided, will need to specify length"
                        " manually with mutation rate")
            fetcher = bcf_in.fetch()
            cls._read_vcf_helper(
                fetcher, chrom_list, pos_list, compressed_hashed,
                excluded, ancestral_alleles, pop2idxs, sampled_pops,
                pop_allele_counts, config, info_aa_field)

        if len(compressed_hashed) == 0:
            logger.warn("No valid SNPs read! Try setting "
                        "ancestral_alleles=False.")

        return cls(chrom_list, pos_list,
                   compressed_hashed.compressed_allele_counts(),
                   sampled_pops, not ancestral_alleles, [], length,
                   len(chrom_list), len(excluded))

    @classmethod
    def _read_vcf_helper(
            cls, bcf_in_fetch, chrom, pos, compressed_hashed, excluded,
            ancestral_alleles, pop2idxs, sampled_pops,
            pop_allele_counts, config, info_aa_field):
        for rec in bcf_in_fetch:
            if len(rec.alleles) != 2:
                continue

            for pop, inds in pop2idxs.items():
                pop_allele_counts[pop].clear()
                for i in inds:
                    for a in rec.samples[i].allele_indices:
                        if a is not None:
                            pop_allele_counts[pop][a] += 1

            if ancestral_alleles is True:
                try:
                    aa = rec.info[info_aa_field]
                except KeyError:
                    excluded.append((rec.chrom, rec.pos))
                    continue
                else:
                    try:
                        aa = rec.alleles.index(aa)
                    except ValueError:
                        excluded.append((rec.chrom, rec.pos))
                        continue
            elif ancestral_alleles:
                outgroup_counts = pop_allele_counts[ancestral_alleles]
                if len(outgroup_counts) != 1:
                    excluded.append((rec.chrom, rec.pos))
                    continue
                aa, = outgroup_counts.keys()
            else:
                aa = 0

            config *= 0
            for pop_idx, pop in enumerate(sampled_pops):
                for a, n in pop_allele_counts[pop].items():
                    config[pop_idx, a] = n

            if aa == 1:
                config = config[:, ::-1]

            compressed_hashed.append(config)

            chrom.append(rec.chrom)
            pos.append(rec.pos)

            if len(pos) % 10000 == 0:
                logger.info("Read vcf up to CHR {}, POS {}".format(
                    chrom[-1], pos[-1]))

    @classmethod
    def concatenate(cls, to_concatenate):
        """Combine a list of :class:`SnpAlleleCounts` into a single object.

        :param iterator to_concatenate: Iterator over :class:`SnpAlleleCounts`
        :rtype: :class:`SnpAlleleCounts`
        """
        to_concatenate = iter(to_concatenate)
        first = next(to_concatenate)
        populations = list(first.populations)
        nonascertained = list(first.non_ascertained_pops)
        to_concatenate = it.chain([first], to_concatenate)

        #chrom_ids = []
        chrom_ids = _CompressedList()
        positions = []
        index2uniq = []

        compressed_hashes = _CompressedHashedCounts(len(populations))

        use_folded_sfs = False
        length = 0
        n_read_snps = 0
        n_excluded_snps = 0
        for snp_cnts in to_concatenate:
            use_folded_sfs = (use_folded_sfs or
                                     snp_cnts.use_folded_sfs)

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

            try:
                length += snp_cnts.length
            except TypeError:
                length = None

            n_read_snps += snp_cnts.n_read_snps
            n_excluded_snps += snp_cnts.n_excluded_snps

            for k, v in Counter(snp_cnts.chrom_ids).items():
                logger.info("Added {} SNPs from Chromosome {}".format(v, k))

        # make sure the positions are sorted
        chrom_ids, positions, index2uniq = zip(*sorted(zip(
            chrom_ids, positions, index2uniq)))

        compressed_counts = CompressedAlleleCounts(
            compressed_hashes.config_array(), index2uniq)
        ret = cls(chrom_ids, positions, compressed_counts, populations,
                  use_folded_sfs=use_folded_sfs,
                  non_ascertained_pops=nonascertained,
                  length=length,
                  n_read_snps=n_read_snps,
                  n_excluded_snps=n_excluded_snps)
        logger.info("Finished concatenating datasets")
        return ret

    @classmethod
    def load(cls, f):
        """Load :class:`SnpAlleleCounts` created \
        from :meth:`SnpAlleleCounts.dump` or ``python -m momi.read_vcf ...``

        :param str,file f: file object or file name to read in
        :rtype: :class:`SnpAlleleCounts`
        """
        if isinstance(f, str):
            if f.endswith(".gz"):
                with gzip.open(f, "rt") as gf:
                    return cls.load(gf)
            else:
                with open(f) as gf:
                    return cls.load(gf)

        # default values for back-compatibility
        items = {"use_folded_sfs": False,
                 "non_ascertained_pops": [],
                 "length": None,
                 "n_excluded_snps": 0}
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
                #chrom_ids = []
                chrom_ids = _CompressedList()
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
                        assert chrom[0] == chrom[-1] == '"'
                        chrom_ids.append(chrom[1:-1])
                        positions.append(float(pos))
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
                items[items_matched.group(1)] = json.loads(
                    items_matched.group(2))

        # for back-compatibility
        if "n_read_snps" not in items:
            items["n_read_snps"] = len(chrom_ids)

        logging.debug("Creating CompressedAlleleCounts")
        compressed_counts = CompressedAlleleCounts(
            configs, config_ids,
            sort=False)

        logging.debug("Creating SnpAlleleCounts")
        return cls(chrom_ids, positions, compressed_counts,
                   **items)

    def dump(self, f):
        """Write data in JSON format.

        :param str,file f: filename or file object. \
        If a filename, the resulting file is gzipped.
        """
        if isinstance(f, str):
            if f.endswith(".gz"):
                with gzip.open(f, "wt") as gf:
                    self.dump(gf)
            else:
                with open(f, "w") as gf:
                    self.dump(gf)
            return

        print("{", file=f)
        print('\t"populations": {},'.format(
            json.dumps(list(self.populations))), file=f)
        if self.non_ascertained_pops:
            print('\t"non_ascertained_pops": {},'.format(
                json.dumps(list(self.non_ascertained_pops))), file=f)
        print('\t"use_folded_sfs": {},'.format(
            json.dumps(self.use_folded_sfs)), file=f)
        print('\t"length": {},'.format(
            json.dumps(self.length)), file=f)
        print('\t"n_read_snps": {},'.format(self.n_read_snps), file=f)
        print('\t"n_excluded_snps": {},'.format(self.n_excluded_snps), file=f)
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
                range(n_positions), list(self.chrom_ids),
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
                 use_folded_sfs, non_ascertained_pops, length,
                 n_read_snps, n_excluded_snps):
        if any([len(compressed_counts) != len(chrom_ids),
                len(chrom_ids) != len(positions)]):
            raise IOError(
                "chrom_ids, positions, allele_counts should have same length")

        self.chrom_ids = chrom_ids
        self.positions = np.array(positions)
        self.compressed_counts = compressed_counts
        self.populations = populations
        self.non_ascertained_pops = non_ascertained_pops
        self.use_folded_sfs = use_folded_sfs
        self.length = length
        self.n_read_snps = n_read_snps
        self.n_excluded_snps = n_excluded_snps

    def __eq__(self, other):
        try:
            return (
                self.compressed_counts == other.compressed_counts and
                np.all(self.chrom_ids == other.chrom_ids) and
                np.all(self.positions == other.positions) and
                self.length == other.length and
                (self.use_folded_sfs ==
                 other.use_folded_sfs) and
                self.n_read_snps == other.n_read_snps and
                self.n_excluded_snps == other.n_excluded_snps)
        except AttributeError:
            return False

    @memoize_instance
    def _chunk_data(self, n_chunks):
        chunk_len = len(self.chrom_ids) / float(n_chunks)
        new_pos = list(range(len(self.chrom_ids)))
        new_chrom = _CompressedList(
            int(np.floor(i / chunk_len)) for i in new_pos)
        return SnpAlleleCounts(
            new_chrom, new_pos, self.compressed_counts,
            self.populations, self.use_folded_sfs,
            self.non_ascertained_pops, self.length,
            self.n_read_snps, self.n_excluded_snps)

    def extract_sfs(self, n_blocks):
        """Extracts SFS from data.

        :param int n_blocks: Number of blocks to split SFS into, for jackknifing and bootstrapping
        :rtype: :class:`Sfs`
        """
        if n_blocks is None:
            return self._sfs
        else:
            return self._chunk_data(n_blocks)._sfs

    @property
    def p_missing(self):
        return self._sfs.p_missing

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
            self.use_folded_sfs,
            non_ascertained_pops, self.length,
            self.n_read_snps, self.n_excluded_snps)

    def __getitem__(self, i):
        return self.compressed_counts[i]

    def __len__(self):
        return len(self.compressed_counts)

    def filter(self, idxs):
        return SnpAlleleCounts(self.chrom_ids[idxs],
                               self.positions[idxs],
                               self.compressed_counts.filter(idxs),
                               self.populations,
                               self.use_folded_sfs,
                               self.non_ascertained_pops, self.length,
                               self.n_read_snps, self.n_excluded_snps)

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

        return SnpAlleleCounts(
            self.chrom_ids, self.positions,
            CompressedAlleleCounts.from_iter(
                sub_counts(), len(self.populations)),
            self.populations, self.use_folded_sfs,
            self.non_ascertained_pops, self.length,
            self.n_read_snps, self.n_excluded_snps)

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

    #@property
    #def seg_sites(self):
    #    try:
    #        self._seg_sites
    #    except:
    #        filtered = self.filter(self.is_polymorphic)
    #        idx_list = [
    #            np.array([i for chrom, i in grouped_idxs])
    #            for key, grouped_idxs in it.groupby(
    #                    zip(filtered.chrom_ids,
    #                        filtered.compressed_counts.index2uniq),
    #                    key=lambda x: x[0])]
    #        self._seg_sites = SegSites(ConfigList(
    #            self.populations,
    #            filtered.compressed_counts.config_array,
    #            ascertainment_pop=self.ascertainment_pop
    #        ), idx_list)
    #    return self._seg_sites

    @cached_property
    def _sfs(self):
        filtered = self.filter(self.is_polymorphic)
        idx_list = [
            np.array([i for chrom, i in grouped_idxs])
            for key, grouped_idxs in it.groupby(
                    zip(filtered.chrom_ids,
                        filtered.compressed_counts.index2uniq),
                    key=lambda x: x[0])]
        configs = ConfigList(
            self.populations,
            filtered.compressed_counts.config_array,
            ascertainment_pop=self.ascertainment_pop)
        if self.length:
            length = self.length * (1 - self._p_excluded)
        else:
            length = None
        ret = Sfs(idx_list, configs, folded=False, length=length)
        if self.use_folded_sfs:
            ret = ret.fold()
        return ret

    @property
    def configs(self):
        return self._sfs.configs

    @property
    def _p_excluded(self):
        n_read = self.n_read_snps
        n_exclude = self.n_excluded_snps

        return n_exclude / (n_read + n_exclude)
