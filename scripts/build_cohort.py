# -*- coding: utf-8 -*-
"""
Build the study cohort: who is eligible, and which of them are cases.

Writes eligible_{cycle}_{definition}_{rule}.csv -- one row per eligible
participant with an `epilepsy` flag. That is the cohort the analysis uses.

It is deliberately NOT a matched set. methods.md 8.1 matches on the propensity
score with MatchIt in R, and 9 puts the whole statistical layer there, so
Python says who is eligible and R decides who is compared with whom.

    # the specification's primary cohort
    python scripts/build_cohort.py --cohort H --definition primary \
        --validity spec --min-valid-days 4 --min-wear-hours 20

    # the cycle G replication cohort, broad definition (methods.md 4.5)
    python scripts/build_cohort.py --cohort G --definition broad \
        --validity spec --min-valid-days 4 --min-wear-hours 20

    # reproduce the superseded February cohort exactly
    python scripts/build_cohort.py --cohort H --definition legacy \
        --validity legacy --frequency-match

`--definition` and `--validity` are both required, and neither has a default.
Each names a different study population, and the superseded combination was
once reachable by saying nothing at all.

`--frequency-match` additionally runs the RETIRED frequency matching into
freq_match_*.csv. It exists only to reproduce the existing cohort files and
the superseded results that came from them; 8.1 does not use it.

Changing --definition, --validity, --seed or --control-ratio changes the study
population: do it deliberately, and record why in doc/analysis-log.md.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone

import pandas as pd

from ambient_light_epilepsy import cohort as ch
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
        help=f"Controls per case for --frequency-match only "
             f"(default: {matching.DEFAULT_CONTROL_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=matching.DEFAULT_SEED,
        help=f"Sampling seed (default: {matching.DEFAULT_SEED})",
    )
    parser.add_argument(
        "--min-valid-days",
        type=int,
        default=None,
        help="Valid days the validity table was built with (4 for the primary "
             "rule). Required with --validity spec: it names which table to "
             "read, so the cohort records the rule it was built under",
    )
    parser.add_argument(
        "--min-wear-hours",
        type=float,
        default=None,
        help="Retained wear hours per valid day (20 for the primary rule). "
             "Required with --validity spec",
    )
    parser.add_argument(
        "--definition",
        choices=sorted(ch.DEFINITIONS) + [matching.LEGACY_DEFINITION],
        required=True,
        help="Case definition (methods.md 4.1). 'primary' is the "
             "specification's; 'broad' is the cycle G replication definition; "
             "'legacy' is the superseded drug-first list, for reproducing the "
             "existing cohort files. Required: each selects a different study "
             "population, and 'legacy' has a PPV of 38.9% against G40",
    )
    parser.add_argument(
        "--frequency-match",
        action="store_true",
        help="Also run the RETIRED frequency matching into freq_match_*.csv. "
             "methods.md 8.1 uses full matching on the propensity score in R; "
             "this exists only to reproduce the existing cohort files",
    )
    parser.add_argument(
        "--validity",
        choices=["spec", "legacy"],
        required=True,
        help="Which accelerometry validity rule decides eligibility. 'spec' is "
             "methods.md 5.2, read from valid_recordings_{cycle}_{rule}.csv "
             "(build it "
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
        "--overwrite",
        action="store_true",
        help="Replace an existing cohort file. Off by default: it defines the "
             "study population, so rewriting one changes what every downstream "
             "result was computed from",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be produced without writing anything",
    )
    return parser.parse_args()


def write_provenance(save_dir, year, args, n_cases, n_controls, stem):
    """Record how this cohort was produced, next to the files themselves."""
    record = {
        "cohort": year,
        "cases": n_cases,
        "controls": n_controls,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/build_cohort.py",
        "git_commit": provenance.git_commit(),
        "parameters": {
            "case_definition": args.definition,
            "min_age": matching.MIN_AGE,
            "validity_rule": args.validity,
            "validity_min_valid_days": args.min_valid_days,
            "validity_min_wear_hours": args.min_wear_hours,
            "frequency_matched": args.frequency_match,
            "control_ratio": args.control_ratio if args.frequency_match else None,
            "seed": args.seed if args.frequency_match else None,
            "match_cols": matching.MATCH_COLS if args.frequency_match else None,
        },
        "data_root": str(paths.data_root(args.base_path)),
        "machine": platform.node(),
        "python": sys.version.split()[0],
        "packages": provenance.package_versions(),
    }

    path = save_dir / f"{stem}.provenance.json"
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
        if args.min_valid_days is None or args.min_wear_hours is None:
            raise SystemExit(
                "--validity spec needs --min-valid-days and --min-wear-hours, "
                "which name the validity table to read. The primary rule is "
                "4 and 20 (methods.md 5.2)."
            )

        label = wear.rule_label(args.min_valid_days, args.min_wear_hours)
        table = wear.load_validity(year, label, args.base_path)
        print(f"Validity rule                 : methods.md 5.2, "
              f"from valid_recordings_{year}_{label}.csv")
        return wear.valid_seqns(table)

    header = wear.load_header(year, args.base_path)
    print("Validity rule                 : SUPERSEDED PAXLDAY == '9' header rule")
    return wear.header_only_validity(header)


def rule_name(args):
    """
    The validity rule, as it appears in the cohort filename.

    'legacy' for the superseded header rule, otherwise the same d04h20-style
    label the validity tables carry, so a cohort file and the validity table
    behind it name the same rule.
    """
    if args.validity == "legacy":
        return "legacy"

    return wear.rule_label(args.min_valid_days, args.min_wear_hours)


def frequency_match(df_all, df_pwe, year, args):
    """
    The RETIRED frequency matching, kept to reproduce the existing cohort files.

    methods.md 8.1 uses full matching on the propensity score in R. Nothing in
    the current analysis path calls this; it runs only under --frequency-match.
    """
    controls, cases = matching.find_frequency_matched_controls(
        df_all, df_pwe,
        control_ratio=args.control_ratio,
        seed=args.seed,
    )

    dropped = len(df_pwe) - len(cases)
    print(f"\n[retired] frequency matching")
    print(f"  cases entering matching     : {len(cases)}"
          f"  ({dropped} dropped for incomplete matching data)")
    print(f"  matched controls            : {len(controls)}"
          f"  ({len(controls) / max(len(cases), 1):.2f} per case,"
          f" {args.control_ratio} requested)")

    if controls.index.nunique() < len(controls):
        print("  ERROR: duplicate participants among the controls")
    if set(controls.index) & set(cases.index):
        print("  ERROR: participants appear as both case and control")

    print("\n  Balance across matching variables (proportions):")
    for name, table in matching.summarise_match(cases, controls).items():
        print(f"\n    {name}")
        print(table.to_string().replace("\n", "\n    "))

    return controls, cases


def build(year, args):
    print(f"\n{'=' * 62}\nCycle {year}\n{'=' * 62}")

    valid = resolve_validity(year, args)
    rule = rule_name(args)

    df_all, df_pwe = matching.eligible_participants(
        year, valid_seqns=valid, definition=args.definition,
        base_path=args.base_path,
    )

    print(f"Case definition               : {args.definition}")
    print(f"Adults with a valid recording : {len(df_all)}")
    print(f"  of whom identified as PWE   : {len(df_pwe)}")
    print(f"  control pool                : {len(df_all) - len(df_pwe)}")

    if args.check_lux:
        missing = matching.missing_lux_files(df_pwe.index, year, args.base_path)
        if missing:
            print(f"WARNING: {len(missing)} cases have no LUX recording: {missing[:10]}")

    controls = cases = None
    if args.frequency_match:
        controls, cases = frequency_match(df_all, df_pwe, year, args)

    if args.dry_run:
        print("\n[dry run] nothing written")
        return

    path = matching.save_eligible_sample(
        df_all, df_pwe, year, args.definition, rule,
        base_path=args.base_path, overwrite=args.overwrite,
    )
    written = [path.name]

    if args.frequency_match:
        control_path, case_path = matching.save_matching_results(
            controls, cases, year, args.base_path
        )
        written += [case_path.name, control_path.name]

    prov_path = write_provenance(
        path.parent, year, args, len(df_pwe), len(df_all) - len(df_pwe),
        path.stem,
    )
    written.append(prov_path.name)

    print(f"\nWrote {', '.join(written)}")
    print(f"  in {path.parent}")


def main():
    args = parse_args()

    print(f"Data root : {paths.data_root(args.base_path)}")
    print(f"Commit    : {provenance.git_commit(short=True)}")
    print(f"Definition: {args.definition}")
    print(f"Validity  : {args.validity}"
          + (f"  ({rule_name(args)})" if args.validity == "spec" else ""))
    if args.frequency_match:
        print(f"Seed      : {args.seed}   control ratio: {args.control_ratio}"
              "   [RETIRED frequency matching]")

    for year in (["G", "H"] if args.cohort == "all" else [args.cohort]):
        build(year, args)


if __name__ == "__main__":
    main()
