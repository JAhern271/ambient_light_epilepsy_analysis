# -*- coding: utf-8 -*-
"""
Check that the converted NHANES tables contain the participants they should.

    python scripts/check_data_integrity.py
    python scripts/check_data_integrity.py --cohort H

Exits non-zero if anything is wrong, so it can gate a pipeline run.

This exists because PAXMIN_H was silently truncated in conversion: it held
only the first 2,489 of 7,776 participants, with the rest replaced by zero
rows. The file opened cleanly and reported no nulls, so nothing downstream
noticed until participants turned up missing in an analysis.
"""

import argparse
import sys

from ambient_light_epilepsy import cohort as ch
from ambient_light_epilepsy import integrity, paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["G", "H", "all"], default="all")
    parser.add_argument("--base-path", default=None)
    return parser.parse_args()


def check(year, base_path):
    print(f"\n=== PAXMIN_{year} ===")
    result = integrity.check_paxmin(year, base_path)

    print(f"  rows                 : {result['rows']:,}")
    print(f"  padding rows (SEQN=0): {result['padding_rows']:,}")
    print(f"  participants present : {result['participants_present']:,}"
          f" of {result['participants_expected']:,} expected")
    print(f"  SEQN range           : {result['seqn_min']:.0f} to {result['seqn_max']:.0f}"
          f"  (expected up to {result['expected_seqn_max']:.0f})")

    # How much of the matched cohort is actually usable
    controls, cases = ch.load_freq_matched_control_groups(year, base_path)
    for name, seqns in [("cases", cases), ("controls", controls)]:
        found = integrity.cohort_availability(year, seqns, base_path)
        n = len(found["present"])
        print(f"  cohort {name:9s}     : {n}/{len(seqns)} present")

    if result["problems"]:
        for problem in result["problems"]:
            print(f"  PROBLEM: {problem}")
        return False

    print("  OK")
    return True


def main():
    args = parse_args()
    print(f"Data root: {paths.data_root(args.base_path)}")

    cohorts = ["G", "H"] if args.cohort == "all" else [args.cohort]
    ok = all([check(year, args.base_path) for year in cohorts])

    if not ok:
        print("\nOne or more tables are incomplete.")
        print("Reconvert from the .xpt with:")
        print("    ALE_OVERWRITE=1 sbatch scripts/convert_xpt/convert_xpt.sh <cohort> PAXMIN")
        sys.exit(1)

    print("\nAll checked tables look complete.")


if __name__ == "__main__":
    main()
