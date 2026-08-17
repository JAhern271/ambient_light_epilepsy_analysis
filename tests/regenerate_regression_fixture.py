"""
Regenerate the pinned values used by the regression test.

Run this ONLY after an intentional change to a metric definition, and record
in doc/analysis-log.md what changed and why. Running it to make a failing test
pass defeats the purpose of the test.

    python tests/regenerate_regression_fixture.py
"""

from pathlib import Path

from ambient_light_epilepsy import cohort as ch
from ambient_light_epilepsy import lux_metrics as lm

# A fixed handful of cycle G participants: the first few PWE and controls.
N_PER_GROUP = 3


def main():
    control, pwe = ch.load_freq_matched_control_groups("G")
    seqns = list(pwe[:N_PER_GROUP]) + list(control[:N_PER_GROUP])

    summary = lm.compute_lux_summary(seqns, "G")
    summary = summary.drop(columns=["timezone"])

    out = Path(__file__).parent / "data" / "regression_expected.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    print(f"\nWrote {len(summary)} pinned rows to {out}")


if __name__ == "__main__":
    main()
