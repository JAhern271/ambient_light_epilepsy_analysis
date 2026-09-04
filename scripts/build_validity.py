# -*- coding: utf-8 -*-
"""
Decide which participants have a valid accelerometer recording (methods.md 5.2).

Reads the whole PAXMIN table for a cycle, applies the minute-level exclusions
of 5.1 and the noon-to-noon valid-day rule of 5.2, and writes one row per
participant to data/processed/valid_recordings_{cycle}_{rule}.csv. Everything
downstream reads that file rather than rescanning 88 million rows.

    # the primary rule (methods.md 5.2), settled 2026-09-03
    python scripts/build_validity.py --cohort H --min-valid-days 4 --min-wear-hours 20

    # a pre-specified sensitivity rule, written alongside rather than over it
    python scripts/build_validity.py --cohort H --min-valid-days 3 --min-wear-hours 16

Both thresholds are REQUIRED and have no defaults, even though they are now
settled: this script is the point at which the choice changes the study
population, so every run states the rule it applied and the sensitivity runs
read identically to the primary one. Record any decision in
doc/analysis-log.md.

Each rule writes its own file, named after the thresholds: 4 days at 20 h goes
to valid_recordings_H_d04h20.csv and 3 at 16 to valid_recordings_H_d03h16.csv,
so a sensitivity rule can never overwrite the primary one. An existing table is
never replaced silently either -- pass --overwrite to do that deliberately.

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
        help="Valid days a participant needs (methods.md 5.2). 4 for the "
             "primary rule. No default even though it is settled: this is "
             "where the choice changes the study population",
    )
    parser.add_argument(
        "--min-wear-hours",
        type=float,
        required=True,
        help="Hours of retained wear a day needs to be valid (methods.md 5.2). "
             "20 for the primary rule. No default, for the same reason",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing table. Off by default: a validity table "
             "defines the study population, so rewriting one changes what "
             "every downstream result was computed from",
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


def write_provenance(save_dir, cycle, args, table, path):
    """Record the thresholds used, next to the file they produced."""
    record = {
        "cohort": cycle,
        "rule_label": wear.rule_label(args.min_valid_days, args.min_wear_hours),
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

    sidecar = path.with_suffix(".provenance.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return sidecar


PRIMARY_RULE = (4, 20)          # methods.md 5.2, settled 2026-09-03


def compare_with_primary(table, cycle, args):
    """
    For a sensitivity rule, contrast it with the primary table.

    This is the comparison the run exists for: how much of the study
    population turns on the choice of threshold. Skipped when this IS the
    primary rule, and skipped with a note when the primary table has not
    been built -- the sensitivity rule is still valid on its own.
    """
    label = wear.rule_label(args.min_valid_days, args.min_wear_hours)
    primary_label = wear.rule_label(*PRIMARY_RULE)

    if label == primary_label:
        return

    try:
        primary = wear.load_validity(cycle, primary_label, args.base_path)
    except FileNotFoundError:
        print(f"\n(no {primary_label} table to compare against)")
        return

    shared = table.index.intersection(primary.index)
    this = table.loc[shared, "meets_criterion"].astype(bool)
    that = primary.loc[shared, "meets_criterion"].astype(bool)

    print(f"\nAgainst {primary_label}, over {len(shared)} shared participants:")
    print(f"  valid under both                 : {int((this & that).sum())}")
    print(f"  valid only under {label:<16}: {int((this & ~that).sum())}")
    print(f"  valid only under {primary_label:<16}: {int((~this & that).sum())}")
    print(f"  valid under neither              : {int((~this & ~that).sum())}")
    print(f"  disagreement                     : "
          f"{100 * (this != that).mean():.1f}% of participants")


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
    compare_with_primary(table, cycle, args)

    if args.dry_run:
        print("\n[dry run] nothing written")
        return

    path = wear.save_validity(
        table, cycle, wear.rule_label(args.min_valid_days, args.min_wear_hours),
        base_path=args.base_path, overwrite=args.overwrite,
    )
    prov = write_provenance(path.parent, cycle, args, table, path)

    print(f"\nWrote {path.name} and {prov.name}")
    print(f"  in {path.parent}")


def main():
    args = parse_args()

    print(f"Data root: {paths.data_root(args.base_path)}")
    print(f"Commit   : {provenance.git_commit(short=True)}")
    print(f"Rule     : >= {args.min_valid_days} valid days "
          f"of >= {args.min_wear_hours} h retained wear"
          f"   [{wear.rule_label(args.min_valid_days, args.min_wear_hours)}]")

    for cycle in (["G", "H"] if args.cohort == "all" else [args.cohort]):
        build(cycle, args)


if __name__ == "__main__":
    main()
