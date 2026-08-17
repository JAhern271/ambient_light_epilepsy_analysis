# -*- coding: utf-8 -*-
"""
Compute LUX summary metrics for the frequency-matched PWE and control groups
in both NHANES cohorts, join on demographics and covariates, and write one
CSV of per-participant results.

Data and output locations come from config.toml (see ambient_light_epilepsy.paths),
so this runs unchanged locally, on the W: drive and on BlueBEAR.

Examples
--------
    python scripts/lux_analysis.py                      # 5 min data, all participants
    python scripts/lux_analysis.py --downsample 1hz     # full 1 Hz data (slow)
    python scripts/lux_analysis.py --limit 10           # quick smoke test
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import ambient_light_epilepsy.nhanes as nhn
import ambient_light_epilepsy.cohort as ch
import ambient_light_epilepsy.lux_metrics as lm
from ambient_light_epilepsy import paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--downsample",
        choices=["5min", "1hz"],
        default="5min",
        help="LUX sampling to analyse (default: 5min)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N participants per group, for testing",
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Override the data root (default: resolved from config.toml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <analysis root>/lux/lux_<downsample>_fmatch_analysis.csv)",
    )
    return parser.parse_args()


def git_commit():
    """Current commit hash, with a -dirty suffix if the tree has changes."""
    try:
        repo = paths.project_root()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        changed = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
        return commit + ("-dirty" if changed else "")
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def write_provenance(save_path, args, n_rows, base_path):
    """
    Record how a results file was produced, alongside the file itself.

    Without this, a CSV cannot be traced back to the code that made it.
    """
    provenance = {
        "output": save_path.name,
        "rows": n_rows,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/lux_analysis.py",
        "git_commit": git_commit(),
        "parameters": {
            "downsample": args.downsample,
            "limit": args.limit,
        },
        "data_root": str(paths.data_root(base_path)),
        "machine": platform.node(),
        "python": sys.version.split()[0],
        "packages": {
            name: __import__(name).__version__ for name in ("pandas", "numpy", "pyarrow")
        },
    }

    provenance_path = save_path.with_suffix(".provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    return provenance_path


def add_labels(df, cohort_label, epilepsy_status):
    """Tag a summary table with its cohort and case/control status."""
    df = df.copy()
    df["cohort"] = cohort_label
    df["epilepsy"] = epilepsy_status  # 1 = PWE, 0 = control
    return df


def main():
    args = parse_args()

    # compute_lux_summary takes None to mean the raw 1 Hz files
    downsample = None if args.downsample == "1hz" else "5min"

    base_path = args.base_path
    print(f"Python executable: {sys.executable}")
    print(f"Data root        : {paths.data_root(base_path)}")

    if args.output is None:
        # Results go in a dated directory so each run is kept distinct rather
        # than silently overwriting the last one.
        today = datetime.now().strftime("%Y-%m-%d")
        out_dir = paths.analysis_root() / "lux" / today
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"lux_{args.downsample}_fmatch_analysis"
        # Never let a truncated test run overwrite a full set of results
        if args.limit is not None:
            name += f"_limit{args.limit}"
        save_path = out_dir / f"{name}.csv"
    else:
        save_path = Path(args.output)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = []

    for year in ["G", "H"]:
        control_seqn, pwe_seqn = ch.load_freq_matched_control_groups(year, base_path)

        if args.limit is not None:
            print(f"WARNING: limiting to {args.limit} participants per group")
            control_seqn = control_seqn[: args.limit]
            pwe_seqn = pwe_seqn[: args.limit]

        df_pwe = lm.compute_lux_summary(pwe_seqn, year, base_path, downsample)
        df_control = lm.compute_lux_summary(control_seqn, year, base_path, downsample)

        summaries.append(add_labels(df_pwe, year, 1))
        summaries.append(add_labels(df_control, year, 0))

    df_all = pd.concat(summaries, ignore_index=True)

    # Join on covariates
    df_all = nhn.add_employment_and_depression_status(df_all, base_path)
    df_all = nhn.add_demographic_data(df_all, base_path)
    df_all = nhn.add_outdoor_time(df_all, base_path)

    df_all.to_csv(save_path)
    provenance_path = write_provenance(save_path, args, len(df_all), base_path)

    print(f"\nWrote {len(df_all)} rows to {save_path}")
    print(f"Provenance recorded in {provenance_path.name}")


if __name__ == "__main__":
    main()
