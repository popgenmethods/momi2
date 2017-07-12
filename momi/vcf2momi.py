import argparse
import sys
import logging
from .data.snps import SnpAlleleCounts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ind2pop",
                        help="File whose first column is individual ID and second column is population ID")
    parser.add_argument("--info_aa", action="store_true", default=False,
                        help="Use INFO/AA field to set ancestral allele, and ignore all SNPs with missing INFO/AA")
    parser.add_argument("--outgroup", default=None,
                        help="Set this population as outgroup to determine ancestral allele")
    parser.add_argument("--non_ascertained_pops", nargs="*", default=[],
                        help="Populations to treat as non-ascertained (only SNPs that are polymorphic within the ascertained populations are considered)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


    with open(args.ind2pop) as f:
        ind2pop = dict([l.split() for l in f])

    if args.outgroup and args.info_aa:
        raise ValueError("At most one of --outgroup, --info_aa may be specified")
    elif args.outgroup:
        ancestral_alleles = args.outgroup
    elif args.info_aa:
        ancestral_alleles = args.info_aa
    else:
        ancestral_alleles = None

    SnpAlleleCounts.read_vcf(sys.stdin, ind2pop, ancestral_alleles=ancestral_alleles, non_ascertained_pops=args.non_ascertained_pops).dump(sys.stdout)
