import argparse
import sys
import logging
from .data.snps import SnpAlleleCounts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_file")
    parser.add_argument("ind2pop",
                        help="File whose first column is individual ID and second column is population ID")
    parser.add_argument("--bed")
    parser.add_argument("--no_aa", action='store_true')
    parser.add_argument("--outgroup", default=None,
                        help="Set this population as outgroup to determine ancestral allele")
    parser.add_argument("--verbose", action="store_true")
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
        ancestral_alleles=ancestral_alleles).dump(sys.stdout)
