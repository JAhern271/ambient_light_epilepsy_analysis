# -*- coding: utf-8 -*-
"""
Build the study cohort: identify cases and select frequency-matched controls.

This produces the freq_match_*.csv files that every downstream analysis
depends on. It was previously done by running notebook 03 by hand.

    python scripts/build_cohort.py --validity spec --dry-run
    python scripts/build_cohort.py --validity spec --cohort H
    python scripts/build_cohort.py --validity legacy --cohort G

`--validity` is required. 'spec' applies methods.md 5.2 and needs
scripts/build_validity.py to have been run first, with the two thresholds
chosen deliberately; 'legacy' applies the superseded PAXLDAY == '9' header
rule and exists to reproduce the cohort files already in data/processed.

Sampling is seeded, so repeated runs reproduce the same cohort. Changing
--seed, --control-ratio or --validity changes the study population: do it
deliberately, and record why in doc/analysis-log.md.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone

import pandas as pd

from ambient_light_epilepsy import matching, paths, provenance, wear


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
        "--validity",
        choices=["spec", "legacy"],
        required=True,
        help="Which accelerometry validity rule decides eligibility. 'spec' is "
             "methods.md 5.2, read from valid_recordings_{cycle}.csv (build it "
             "first with scripts/build_validity.py). 'legacy' is the superseded "
             "PAXSTS == 1 and PAXLDAY == '9' rule, for reproducing the existing "
             "cohort files. Required: the two rules select different study "
             "populations, so the choice is stated, never defaulted",
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
            "validity_rule": args.validity,
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


def resolve_validity(year, args):
    """
    Which participants count as having a valid recording.

    Kept separate so the two rules are visibly alternatives rather than one
    being a special case of the other. methods.md 5.2 replaced the header rule
    on 2026-09-02; the header rule survives only to reproduce what is already
    in results/.
    """
    if args.validity == "spec":
        table = wear.load_validity(year, args.base_path)
        print(f"Validity rule                 : methods.md 5.2, from "
              f"valid_recordings_{year}.csv")
        return wear.valid_seqns(table)

    header = wear.load_header(year, args.base_path)
    print("Validity rule                 : SUPERSEDED PAXLDAY == '9' header rule")
    return wear.header_only_validity(header)


def build(year, args):
    print(f"\n{'=' * 62}\nCycle {year}\n{'=' * 62}")

    valid = resolve_validity(year, args)

    df_all, df_pwe = matching.eligible_participants(
        year, valid_seqns=valid, base_path=args.base_path
    )
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
