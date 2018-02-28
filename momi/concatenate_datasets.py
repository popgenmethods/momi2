import gzip
import sys
import logging
import argparse
from .data.snps import SnpAlleleCounts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("files", nargs="+",
                        help="Files containing momi json data to concatenate")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    SnpAlleleCounts.concatenate(
        SnpAlleleCounts.load(fname)
        for fname in args.files).dump(sys.stdout)
