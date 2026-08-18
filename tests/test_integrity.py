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


# ---------------------------------------------------------------------------
# Checking the raw .xpt
# ---------------------------------------------------------------------------

def write_xpt(path, n_records, real_records, record_length=8):
    """
    A minimal XPT-shaped file: enough header for the layout parser, then
    `n_records` records of which only the first `real_records` carry data.
    Mimics an interrupted download that preallocated its full length.
    """
    header = bytearray()
    namestr = b"HEADER RECORD*******NAMESTR HEADER RECORD"
    header += namestr.ljust(54, b"!") + b"0001" + b"0" * 22
    header = header[:54] + b"0001" + b"0" * (80 - 58)
    header = bytearray(namestr.ljust(54, b"!")) + b"0001" + b"0" * 22

    # one NAMESTR: length lives in the big-endian short at offset 4
    field = bytearray(140)
    field[4:6] = record_length.to_bytes(2, "big")
    field[8:16] = b"SEQN    "

    obs = b"HEADER RECORD*******OBS     HEADER RECORD".ljust(80, b"!")

    body = b"".join(
        (b"\x41" + b"\x10" * (record_length - 1)) if i < real_records
        else b"\x00" * record_length
        for i in range(n_records)
    )

    path.write_bytes(bytes(header) + bytes(field) + obs + body)
    return path


def test_complete_xpt_reports_no_problems(tmp_path):
    path = write_xpt(tmp_path / "good.xpt", n_records=500, real_records=500)

    result = integrity.check_xpt(path, sample_records=100)

    assert result["problems"] == []
    assert result["real_records"] == 500


def test_zero_filled_xpt_is_detected(tmp_path):
    """The PAXMIN_H failure: full-length file, only the first third real."""
    path = write_xpt(tmp_path / "truncated.xpt", n_records=900, real_records=300)

    result = integrity.check_xpt(path, sample_records=100)

    assert result["problems"], "a zero-filled tail should be reported"
    assert result["real_records"] == 300
    assert "did not complete" in result["problems"][0]


def test_xpt_layout_reads_the_record_length(tmp_path):
    path = write_xpt(tmp_path / "layout.xpt", n_records=10, real_records=10, record_length=16)

    layout = integrity.xpt_layout(path)

    assert layout["record_length"] == 16
    assert layout["records"] == 10
