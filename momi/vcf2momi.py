from .parse_data import SnpAlleleCounts
import argparse
import sys
import gzip
import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ind2pop",
                        help="File whose first column is individual ID and second column is population ID")
    parser.add_argument("outfile", nargs="?", default=None,
                        help="File to write gzipped json to. If not specified, the result is written to stdout without gzipping.")
    parser.add_argument("--info_aa", action="store_true", default=False,
                        help="Use INFO/AA field to set ancestral allele, and ignore all SNPs with missing INFO/AA")
    parser.add_argument("--outgroup", default=None,
                        help="Set this population as outgroup to determine ancestral allele")
    parser.add_argument("--chunk_size", default=10000, type=int,
                        help="Process vcf lines in batches of chunk_size. Larger chunk_size uses more RAM but is faster.")
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

    outfile = args.outfile
    if not outfile:
        outfile = sys.stdout
    else:
        outfile = gzip.open(outfile, "wt")

    SnpAlleleCounts.read_vcf(sys.stdin, ind2pop, ancestral_alleles=ancestral_alleles, chunk_size=args.chunk_size).dump(outfile)
    if args.outfile:
        outfile.close()
