from .parse_data import SnpAlleleCounts
import argparse
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inds2pop",
                        help="File whose first column is individual ID and second column is population ID")
    args = parser.parse_args()

    with open(args.inds2pop) as f:
        inds2pop = dict([l.split() for l in f])

    SnpAlleleCounts.read_vcf(sys.stdin, inds2pop).dump(sys.stdout)
