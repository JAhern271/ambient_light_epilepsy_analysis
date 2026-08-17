"""
End-to-end tests of the summary pipeline.

The synthetic tests always run. The regression test needs the real NHANES data
and skips without it, so the suite still passes on a machine that cannot reach
the W: drive.
"""

import numpy as np
import pandas as pd
import pytest

from ambient_light_epilepsy import cohort as ch
from ambient_light_epilepsy import lux_metrics as lm
from ambient_light_epilepsy import paths


# ---------------------------------------------------------------------------
# Synthetic end to end
# ---------------------------------------------------------------------------

def test_summary_runs_over_a_synthetic_cohort(example_data_root):
    """compute_lux_summary reads a whole cohort and returns one row each."""
    control, pwe = ch.load_freq_matched_control_groups("X", example_data_root)

    summary = lm.compute_lux_summary(pwe, "X", example_data_root)

    assert len(summary) == len(pwe)
    assert set(summary["SEQN"]) == set(pwe)


def test_summary_has_every_expected_column(example_data_root):
    _, pwe = ch.load_freq_matched_control_groups("X", example_data_root)

    summary = lm.compute_lux_summary(pwe, "X", example_data_root)

    expected = {
        "SEQN", "timezone", "duration_hours", "mean_lux", "mean_daytime_lux",
        "mean_nighttime_lux", "time_above_threshold", "M10", "L5", "RA",
        "m10_midpoint", "l5_midpoint", "IS", "IV",
    }
    assert expected <= set(summary.columns)


def test_summary_recovers_a_known_group_difference(example_data_root):
    """
    The fixture gives controls 1000 lux daytime and PWE 200. The summary must
    reproduce that ordering — a basic check that group labels are not crossed
    somewhere in the loading path.
    """
    control, pwe = ch.load_freq_matched_control_groups("X", example_data_root)

    control_summary = lm.compute_lux_summary(control, "X", example_data_root)
    pwe_summary = lm.compute_lux_summary(pwe, "X", example_data_root)

    assert pwe_summary["mean_daytime_lux"].mean() < control_summary["mean_daytime_lux"].mean()


def test_missing_participant_file_is_skipped_not_fatal(example_data_root):
    """One absent recording must not abort a cohort-wide run."""
    summary = lm.compute_lux_summary([1001, 999999], "X", example_data_root)

    assert len(summary) == 1
    assert summary["SEQN"].iloc[0] == 1001


# ---------------------------------------------------------------------------
# Regression against the real data
# ---------------------------------------------------------------------------

EXPECTED = "regression_expected.csv"

METRIC_COLUMNS = [
    "duration_hours", "mean_lux", "mean_daytime_lux", "mean_nighttime_lux",
    "time_above_threshold", "M10", "L5", "RA", "m10_midpoint", "l5_midpoint",
    "IS", "IV",
]


def real_data_available():
    """True when the full LUX dataset can be reached from this machine."""
    try:
        return paths.lux_dir("G", "5min").exists()
    except (FileNotFoundError, RuntimeError):
        return False


@pytest.mark.skipif(
    not real_data_available(),
    reason="full NHANES LUX data not reachable from this machine",
)
def test_metrics_match_pinned_values(request):
    """
    Recompute a handful of real participants and compare against values pinned
    when the test was written, so a refactor cannot silently move results.

    If this fails after an intentional change to a metric, regenerate the
    fixture with tests/regenerate_regression_fixture.py and record why in
    doc/analysis-log.md.
    """
    expected = pd.read_csv(request.path.parent / "data" / EXPECTED)

    actual = lm.compute_lux_summary(expected["SEQN"].to_numpy(), "G", None)
    actual = actual.set_index("SEQN").loc[expected["SEQN"]].reset_index()

    for column in METRIC_COLUMNS:
        np.testing.assert_allclose(
            actual[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=1e-9,
            err_msg=f"{column} changed",
        )
