import gzip
import sys
import logging
import argparse
from .data.snps import SnpAlleleCounts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "files", nargs="+",
        help="Files containing SNP allele counts")
    parser.add_argument(
        "--n_blocks", type=int, required=True,
        help="Number of blocks for jackknife/bootstrap")
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output file")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    SnpAlleleCounts.concatenate(
        SnpAlleleCounts.load(fname)
        for fname in args.files).extract_sfs(
                args.n_blocks).dump(args.out)
