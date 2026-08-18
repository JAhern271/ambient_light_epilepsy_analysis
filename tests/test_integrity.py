"""
Tests for the data integrity checks.

These guard against the specific way PAXMIN_H failed: a file that opens
cleanly, has a plausible row count and no nulls, but whose participants
simply stop partway through.
"""

import numpy as np
import pandas as pd
import pytest

from ambient_light_epilepsy import integrity


def write_paxmin(path, seqns, minutes_each=10, padding_rows=0):
    """A miniature PAXMIN-shaped parquet, optionally zero-padded."""
    rows = []
    for seqn in seqns:
        rows.append(pd.DataFrame({
            "SEQN": float(seqn),
            "PAXLXMM": np.linspace(0, 100, minutes_each),
            "PAXMTSM": np.linspace(0, 10, minutes_each),
        }))

    if padding_rows:
        rows.append(pd.DataFrame({
            "SEQN": np.zeros(padding_rows),
            "PAXLXMM": np.zeros(padding_rows),
            "PAXMTSM": np.zeros(padding_rows),
        }))

    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
    return path


def test_coverage_counts_real_participants(tmp_path):
    path = write_paxmin(tmp_path / "ok.parquet", seqns=[101, 102, 103])

    result = integrity.participant_coverage(path)

    assert result["participants"] == 3
    assert result["seqn_min"] == 101
    assert result["seqn_max"] == 103
    assert result["padding_rows"] == 0


def test_padding_rows_are_counted_and_excluded(tmp_path):
    """
    SEQN 0 is padding, not a participant. It must not inflate the count, and
    it must be reported — this is what a truncated conversion looks like.
    """
    path = write_paxmin(tmp_path / "padded.parquet", seqns=[101, 102], padding_rows=500)

    result = integrity.participant_coverage(path)

    assert result["participants"] == 2
    assert result["padding_rows"] == 500
    assert 0.0 not in result["seqns"]


def test_a_padded_file_still_looks_fine_by_row_count(tmp_path):
    """
    The failure mode being guarded against: row count alone tells you nothing,
    because padding makes the file look bigger, not smaller.
    """
    good = write_paxmin(tmp_path / "good.parquet", seqns=range(101, 121))
    bad = write_paxmin(tmp_path / "bad.parquet", seqns=[101, 102], padding_rows=1000)

    good_result = integrity.participant_coverage(good)
    bad_result = integrity.participant_coverage(bad)

    assert bad_result["rows"] > good_result["rows"]          # bigger...
    assert bad_result["participants"] < good_result["participants"]  # ...but emptier


def test_empty_file_reports_no_participants(tmp_path):
    path = write_paxmin(tmp_path / "empty.parquet", seqns=[], padding_rows=100)

    result = integrity.participant_coverage(path)

    assert result["participants"] == 0
    assert result["seqn_min"] is None


def test_cohort_availability_splits_present_from_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("ALE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ALE_PROFILE", raising=False)

    (tmp_path / "X").mkdir()
    write_paxmin(tmp_path / "X" / "PAXMIN_X.parquet", seqns=[101, 102])

    found = integrity.cohort_availability("X", [101, 102, 999])

    assert found["present"] == [101, 102]
    assert found["missing"] == [999]
