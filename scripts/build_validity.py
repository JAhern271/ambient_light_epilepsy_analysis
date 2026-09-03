# -*- coding: utf-8 -*-
"""
Decide which participants have a valid accelerometer recording (methods.md 5.2).

Reads the whole PAXMIN table for a cycle, applies the minute-level exclusions
of 5.1 and the noon-to-noon valid-day rule of 5.2, and writes one row per
participant to data/processed/valid_recordings_{cycle}.csv. Everything
downstream reads that file rather than rescanning 88 million rows.

    python scripts/build_validity.py --cohort H --min-valid-days 4 --min-wear-hours 20

Both thresholds are REQUIRED and have no defaults. The values in
analysis_params.toml follow Xiao 2023 but are provisional -- the researcher
reserved the choice on 2026-09-02 (doc/implementation-status.md) -- and this
script is the point at which the choice actually changes the study population.
Pass them explicitly, and record the decision in doc/analysis-log.md.

The output also carries `header_only_valid`, the verdict of the superseded
PAXSTS == 1 and PAXLDAY == '9' rule, so the change in cohort membership can be
tabulated in both directions: participants admitted who stopped recording
early, and participants excluded who recorded nine days but wore the device
too little on most of them.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone

import pandas as pd

from ambient_light_epilepsy import params, paths, provenance, wear


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        choices=["G", "H", "all"],
        default="H",
        help="NHANES cycle to assess (default: H, the primary analysis cycle)",
    )
    parser.add_argument(
        "--min-valid-days",
        type=int,
        required=True,
        help="Valid days a participant needs (methods.md 5.2). No default: "
             "the value is not settled, see doc/implementation-status.md",
    )
    parser.add_argument(
        "--min-wear-hours",
        type=float,
        required=True,
        help="Hours of retained wear a day needs to be valid (methods.md 5.2). "
             "No default, for the same reason",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Override the data root (default: resolved from config.toml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be produced without writing anything",
    )
    return parser.parse_args()


def write_provenance(save_dir, cycle, args, table):
    """Record the thresholds used, next to the file they produced."""
    record = {
        "cohort": cycle,
        "participants_assessed": int(len(table)),
        "participants_valid": int(table["meets_criterion"].sum()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/build_validity.py",
        "git_commit": provenance.git_commit(),
        "parameters": {
            "min_valid_days": args.min_valid_days,
            "min_wear_hours": args.min_wear_hours,
            # the settled part of the rule, for completeness
            "minute_exclusions": dict(params.section("validity")),
        },
        "data_root": str(paths.data_root(args.base_path)),
        "machine": platform.node(),
        "python": sys.version.split()[0],
        "packages": provenance.package_versions(),
    }

    path = save_dir / f"valid_recordings_{cycle}.provenance.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return path


def report(table, args):
    """Print the cohort change in both directions, before anything is written."""
    valid = table["meets_criterion"].astype(bool)
    old = table["header_only_valid"].astype(bool)

    print(f"Participants assessed              : {len(table)}")
    print(f"Valid under methods.md 5.2         : {int(valid.sum())}"
          f"   ({args.min_valid_days} days at {args.min_wear_hours} h)")
    print(f"Valid under the superseded rule    : {int(old.sum())}"
          "   (PAXSTS == 1 and PAXLDAY == '9')")
    print(f"  admitted by the change           : {int((valid & ~old).sum())}")
    print(f"  excluded by the change           : {int((~valid & old).sum())}")
    print(f"  net change                       : {int(valid.sum()) - int(old.sum()):+d}")

    print("\nCandidate days per participant:")
    print(table["n_candidate_days"].value_counts().sort_index().to_string())
    print("\nValid days per participant:")
    print(table["n_valid_days"].value_counts().sort_index().to_string())


def build(cycle, args):
    print(f"\n{'=' * 62}\nCycle {cycle}\n{'=' * 62}")
    print("Reading PAXMIN. This is the whole table, so it takes a few minutes.")

    table = wear.valid_recordings(
        cycle,
        min_valid_days=args.min_valid_days,
        min_wear_hours=args.min_wear_hours,
        base_path=args.base_path,
    )

    report(table, args)

    if args.dry_run:
        print("\n[dry run] nothing written")
        return

    path = wear.save_validity(table, cycle, args.base_path)
    prov = write_provenance(path.parent, cycle, args, table)

    print(f"\nWrote {path.name} and {prov.name}")
    print(f"  in {path.parent}")


def main():
    args = parse_args()

    print(f"Data root: {paths.data_root(args.base_path)}")
    print(f"Commit   : {provenance.git_commit(short=True)}")
    print(f"Rule     : >= {args.min_valid_days} valid days "
          f"of >= {args.min_wear_hours} h retained wear")

    for cycle in (["G", "H"] if args.cohort == "all" else [args.cohort]):
        build(cycle, args)


if __name__ == "__main__":
    main()
