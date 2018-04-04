"""Runnable module to convert VCF file to :class:`SnpAlleleCounts`.

Run this from the command line like ``python -m momi.read_vcf ...``. \
See the ``--help`` flag for command line options. \
Use :meth:`SnpAlleleCounts.read_vcf` to read the VCF from within Python.
"""

import argparse
import sys
import logging
from .data.snps import SnpAlleleCounts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_file", help="VCF file to read")
    parser.add_argument("ind2pop",
                        help="File whose first column is individual ID and second column is population ID")
    parser.add_argument("out_file", help="Output file to store counts. If ends with .gz, gzip it.")
    parser.add_argument("--bed", help="Mask file specifying which regions to read. Also used to determine the size of the data in bases. If not provided then user will need to manually specify the length when required. Do NOT use the same BED file across multiple VCFs or the length of those regions will be double-counted!")
    parser.add_argument("--no_aa", action='store_true',
                        help="Ignore AA information entirely; use folded SFS downstream.")
    parser.add_argument("--outgroup", default=None,
                        help="Set this population as outgroup to determine ancestral allele, instead of using the AA info field. Note the outgroup will not appear in the created data (as it always has allele 0).")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--info_aa_field", default="AA", help="INFO field to read ancestral allele from. Default is AA. Has no effect if --outgroup or --no_aa are set.")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


    with open(args.ind2pop) as f:
        ind2pop = dict([l.split() for l in f])

    if args.no_aa:
        ancestral_alleles = False
    elif args.outgroup:
        ancestral_alleles = args.outgroup
    else:
        ancestral_alleles = True

    SnpAlleleCounts.read_vcf(
        args.vcf_file, ind2pop, bed_file=args.bed,
        ancestral_alleles=ancestral_alleles,
        info_aa_field=args.info_aa_field).dump(args.out_file)
