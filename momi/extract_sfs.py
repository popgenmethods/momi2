import gzip
import sys
import logging
import argparse
from .data.snps import SnpAlleleCounts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "out", type=str, help="Output file")
    parser.add_argument(
        "n_blocks", type=int, default=1,
        help="Number of blocks for jackknife/bootstrap")
    parser.add_argument(
        "files", nargs="+",
        help="Files containing SNP allele counts")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if len(args.files) == 1:
        f, = args.files
        counts = SnpAlleleCounts.load(f)
    else:
        counts = SnpAlleleCounts.concatenate(
            SnpAlleleCounts.load(fname)
            for fname in args.files)

    logging.info("Extracting SFS...")
    counts.extract_sfs(args.n_blocks).dump(args.out)
