# -*- coding: utf-8 -*-
"""
Build the study cohort: identify cases and select frequency-matched controls.

This produces the freq_match_*.csv files that every downstream analysis
depends on. It was previously done by running notebook 03 by hand.

    python scripts/build_cohort.py --dry-run     # report, write nothing
    python scripts/build_cohort.py               # both cycles
    python scripts/build_cohort.py --cohort G

Sampling is seeded, so repeated runs reproduce the same cohort. Changing
--seed or --control-ratio changes the study population: do it deliberately,
and record why in doc/analysis-log.md.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone

import pandas as pd

from ambient_light_epilepsy import matching, paths, provenance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        choices=["G", "H", "all"],
        default="all",
        help="NHANES cycle to build (default: all)",
    )
    parser.add_argument(
        "--control-ratio",
        type=int,
        default=matching.DEFAULT_CONTROL_RATIO,
        help=f"Controls per case (default: {matching.DEFAULT_CONTROL_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=matching.DEFAULT_SEED,
        help=f"Sampling seed (default: {matching.DEFAULT_SEED})",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Override the data root (default: resolved from config.toml)",
    )
    parser.add_argument(
        "--check-lux",
        action="store_true",
        help="Also report cases with no PAXLUX recording on disk",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be produced without writing anything",
    )
    return parser.parse_args()


def write_provenance(save_dir, year, args, n_cases, n_controls):
    """Record how this cohort was produced, next to the files themselves."""
    record = {
        "cohort": year,
        "cases": n_cases,
        "controls": n_controls,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/build_cohort.py",
        "git_commit": provenance.git_commit(),
        "parameters": {
            "control_ratio": args.control_ratio,
            "seed": args.seed,
            "min_age": matching.MIN_AGE,
            "match_cols": matching.MATCH_COLS,
        },
        "data_root": str(paths.data_root(args.base_path)),
        "machine": platform.node(),
        "python": sys.version.split()[0],
        "packages": provenance.package_versions(),
    }

    path = save_dir / f"freq_match_{year}.provenance.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return path


def build(year, args):
    print(f"\n{'=' * 62}\nCycle {year}\n{'=' * 62}")

    df_all, df_pwe = matching.eligible_participants(year, args.base_path)
    print(f"Adults with a valid recording : {len(df_all)}")
    print(f"  of whom identified as PWE   : {len(df_pwe)}")

    controls, cases = matching.find_frequency_matched_controls(
        df_all, df_pwe,
        control_ratio=args.control_ratio,
        seed=args.seed,
    )

    dropped = len(df_pwe) - len(cases)
    print(f"Cases entering matching       : {len(cases)}"
          f"  ({dropped} dropped for incomplete matching data)")
    print(f"Matched controls              : {len(controls)}"
          f"  ({len(controls) / max(len(cases), 1):.2f} per case,"
          f" {args.control_ratio} requested)")

    if controls.index.nunique() < len(controls):
        print("ERROR: duplicate participants among the controls")
    if set(controls.index) & set(cases.index):
        print("ERROR: participants appear as both case and control")

    if args.check_lux:
        missing = matching.missing_lux_files(cases.index, year, args.base_path)
        if missing:
            print(f"WARNING: {len(missing)} cases have no LUX recording: {missing[:10]}")

    print("\nBalance across matching variables (proportions):")
    for name, table in matching.summarise_match(cases, controls).items():
        print(f"\n  {name}")
        print(table.to_string().replace("\n", "\n  "))

    if args.dry_run:
        print("\n[dry run] nothing written")
        return

    control_path, case_path = matching.save_matching_results(
        controls, cases, year, args.base_path
    )
    prov_path = write_provenance(
        control_path.parent, year, args, len(cases), len(controls)
    )

    print(f"\nWrote {case_path.name}, {control_path.name}, {prov_path.name}")
    print(f"  in {control_path.parent}")


def main():
    args = parse_args()

    print(f"Data root: {paths.data_root(args.base_path)}")
    print(f"Commit   : {provenance.git_commit(short=True)}")
    print(f"Seed     : {args.seed}   control ratio: {args.control_ratio}")

    for year in (["G", "H"] if args.cohort == "all" else [args.cohort]):
        build(year, args)


if __name__ == "__main__":
    main()
