from .parse_data import SnpAlleleCounts
import gzip
import sys
import logging
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gz", action="store_true",
                        help="Files read in to concatenate are gzipped")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("files", nargs="+",
                        help="Files containing momi json data to concatenate")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.gz:
        def fopen(fname):
            return gzip.open(fname, "rt")
    else:
        fopen = open

    SnpAlleleCounts.concatenate(SnpAlleleCounts.load(
        fopen(fname)) for fname in args.files).dump(sys.stdout)
