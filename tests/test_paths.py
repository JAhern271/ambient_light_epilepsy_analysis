"""
Tests for path resolution.

These matter because the project runs on three machines with two different
directory layouts, and a silently wrong path produces a confusing
FileNotFoundError far from its cause.
"""

import pandas as pd
import pytest

from ambient_light_epilepsy import paths


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """An empty data root, selected via the environment override."""
    monkeypatch.setenv("ALE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ALE_PROFILE", raising=False)
    return tmp_path


def test_env_var_overrides_config(data_root):
    assert paths.data_root() == data_root


def test_explicit_argument_wins_over_env_var(data_root, tmp_path):
    other = tmp_path / "elsewhere"
    assert paths.data_root(other) == other


def test_unknown_profile_is_rejected(monkeypatch):
    monkeypatch.delenv("ALE_DATA_ROOT", raising=False)
    monkeypatch.setenv("ALE_PROFILE", "not-a-profile")

    with pytest.raises(RuntimeError, match="not-a-profile"):
        paths.data_root()


def test_raw_table_finds_the_flat_layout(data_root):
    """W: drive and HPC layout: data/G/DEMO_G.parquet"""
    (data_root / "G").mkdir()
    expected = data_root / "G" / "DEMO_G.parquet"
    pd.DataFrame({"SEQN": [1]}).to_parquet(expected)

    assert paths.raw_table("G", "DEMO") == expected


def test_raw_table_finds_the_local_raw_parquet_layout(data_root):
    """Local copy layout: data/G/raw_parquet/DEMO_G.parquet"""
    (data_root / "G" / "raw_parquet").mkdir(parents=True)
    expected = data_root / "G" / "raw_parquet" / "DEMO_G.parquet"
    pd.DataFrame({"SEQN": [1]}).to_parquet(expected)

    assert paths.raw_table("G", "DEMO") == expected


def test_missing_table_error_names_what_was_tried(data_root):
    with pytest.raises(FileNotFoundError) as excinfo:
        paths.raw_table("G", "DEMO")

    message = str(excinfo.value)
    assert "DEMO" in message
    assert "raw_parquet" in message  # both candidate locations reported


def test_processed_file_found_at_data_root(data_root):
    (data_root / "processed").mkdir()
    expected = data_root / "processed" / "people_with_epilepsy_G.csv"
    expected.write_text("SEQN\n1\n")

    assert paths.processed_file("people_with_epilepsy_G.csv", "G") == expected


def test_processed_file_found_under_the_cycle_directory(data_root):
    (data_root / "G" / "processed").mkdir(parents=True)
    expected = data_root / "G" / "processed" / "people_with_epilepsy_G.csv"
    expected.write_text("SEQN\n1\n")

    assert paths.processed_file("people_with_epilepsy_G.csv", "G") == expected


def test_processed_dir_always_writes_to_the_canonical_location(data_root):
    """
    Reads tolerate either layout, but writes must go to one place, or the
    per-cycle and root-level copies drift apart again.
    """
    (data_root / "G" / "processed").mkdir(parents=True)

    target = paths.processed_dir("G", create=True)

    assert target == data_root / "processed"
    assert target.exists()


def test_lux_file_naming(data_root):
    assert paths.lux_file(62218, "G").name == "SEQN_62218_5min.parquet"
    assert paths.lux_file(62218, "G", downsample=None).name == "SEQN_62218.parquet"


def test_lux_dir_rejects_an_unknown_downsample(data_root):
    with pytest.raises(ValueError, match="downsample"):
        paths.lux_dir("G", downsample="10min")


def test_raw_xpt_found_without_a_converted_parquet(data_root):
    """
    The source must be checkable when the parquet is absent — exactly the
    situation after deleting a bad conversion.
    """
    (data_root / "H").mkdir()
    expected = data_root / "H" / "PAXMIN_H.xpt"
    expected.write_bytes(b"not really an xpt")

    assert paths.raw_xpt("H", "PAXMIN") == expected
    with pytest.raises(FileNotFoundError):
        paths.raw_table("H", "PAXMIN")
