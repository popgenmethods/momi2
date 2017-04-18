from .parse_data import SnpAlleleCounts
import argparse
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inds2pop",
                        help="File whose first column is individual ID and second column is population ID")
    parser.add_argument("--outgroup", default=None,
                        help="Set this population as outgroup to determine ancestral allele")
    parser.add_argument("--chunk_size", default=10000, type=int,
                        help="Process vcf lines in batches of chunk_size. Larger chunk_size uses more RAM but is faster.")
    args = parser.parse_args()

    with open(args.inds2pop) as f:
        inds2pop = dict([l.split() for l in f])

    SnpAlleleCounts.read_vcf(sys.stdin, inds2pop, outgroup=args.outgroup, chunk_size=args.chunk_size).dump(sys.stdout)
