"""
Tests for code-first epilepsy case ascertainment.

The synthetic table below is small enough that the correct answer under each of
the four definitions can be worked out by hand, which is what lets these tests
check correctness rather than pin current behaviour. Each row exercises one
thing that could plausibly be got wrong.

These tests deliberately run against the committed analysis_params.toml rather
than against lists defined here, so that they check the *specification's* drug
lists produce the hand-derived answer. Editing a name list in that file will
therefore break them — which is the point: the expected sets below must then be
re-derived by hand, not adjusted to match whatever the code now returns.

The last test needs the real NHANES data and skips without it, so the suite
still passes on a machine that cannot reach the W: drive.
"""

import pandas as pd
import pytest

from ambient_light_epilepsy import cohort as ch


# ---------------------------------------------------------------------------
# A synthetic RXQ_RX, and the answers derived from it by hand
# ---------------------------------------------------------------------------

# NHANES stores an absent reason code as an empty string, and stores the codes
# it does have at three-character category level. Both are reproduced here.
#
# SEQN  drug            use  codes            primary narrow narrow_nocode broad
#  1    levetiracetam    1   G40                 y      y         y          y
#  2    topiramate       1   G43                 .      .         .          y   off-label migraine
#  3    lacosamide       1   -, G40              y      y         y          .   code in the 2nd slot; not in the 12-name list
#  4    clobazam         1   G40.909             y      .         .          .   full ICD-10 code, matched by prefix
#  5    allopurinol      1   G40                 .      .         .          .   miscoded non-ASM, blanked
#  6    phenytoin        2   G40                 .      .         .          .   not taken in the past 30 days
#  7    lamotrigine      1   G40                 y      .         .          y   two rows, one participant
#  7    lamotrigine      1   F31.9               "      "         "          "
#  8    gabapentin       1   M79.2               .      .         .          .   an ASM, but prescribed for nerve pain
#  9    carbamazepine    1   F31.9               .      .         y          y   rarely-off-label name, no G40 code
SYNTHETIC_RX = [
    (1, "LEVETIRACETAM", 1.0, "G40",     "",      ""),
    (2, "TOPIRAMATE",    1.0, "G43",     "",      ""),
    (3, "LACOSAMIDE",    1.0, "",        "G40",   ""),
    (4, "CLOBAZAM",      1.0, "G40.909", "",      ""),
    (5, "ALLOPURINOL",   1.0, "G40",     "",      ""),
    (6, "PHENYTOIN",     2.0, "G40",     "",      ""),
    (7, "LAMOTRIGINE",   1.0, "G40",     "",      ""),
    (7, "LAMOTRIGINE",   1.0, "F31.9",   "",      ""),
    (8, "GABAPENTIN",    1.0, "M79.2",   "",      ""),
    (9, "CARBAMAZEPINE", 1.0, "F31.9",   "",      ""),
]

EXPECTED = {
    "primary":       {1, 3, 4, 7},
    "narrow":        {1, 3},
    "narrow_nocode": {1, 3, 9},
    "broad":         {1, 2, 7, 9},
}


def write_rx(rows, root, cycle="H", with_code_cols=True):
    """Write a synthetic RXQ_RX parquet where paths.raw_table will find it."""
    frame = pd.DataFrame(
        rows,
        columns=["SEQN", "RXDDRUG", "RXDUSE", "RXDRSC1", "RXDRSC2", "RXDRSC3"],
    )
    frame["SEQN"] = frame["SEQN"].astype("float64")   # as NHANES stores it

    if not with_code_cols:
        # Cycle G carries no reason-for-use variables at all
        frame = frame.drop(columns=list(ch.REASON_CODE_COLS))

    cycle_dir = root / cycle
    cycle_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cycle_dir / f"RXQ_RX_{cycle}.parquet", index=False)

    return root


@pytest.fixture
def rx_root(tmp_path):
    """A data root holding the synthetic cycle H prescription table."""
    return write_rx(SYNTHETIC_RX, tmp_path)


# ---------------------------------------------------------------------------
# The four definitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("definition", sorted(EXPECTED))
def test_each_definition_selects_the_hand_derived_cases(definition, rx_root):
    cases = ch.find_cases("H", definition, base_path=rx_root, save=False)

    assert set(cases) == EXPECTED[definition]


def test_cases_are_returned_as_sorted_unique_int64(rx_root):
    """SEQN arrives as float and one participant has two rows."""
    cases = ch.find_cases("H", "primary", base_path=rx_root, save=False)

    assert cases.dtype == "int64"
    assert cases.is_unique
    assert list(cases) == sorted(cases)


def test_the_default_definition_is_the_specifications_primary(rx_root):
    """Omitting `definition` must read analysis_params.toml, not guess."""
    default = ch.find_cases("H", base_path=rx_root, save=False)

    assert set(default) == EXPECTED["primary"]


def test_an_unknown_definition_is_refused(rx_root):
    with pytest.raises(ValueError, match="Unknown case definition"):
        ch.find_cases("H", "primaryy", base_path=rx_root, save=False)


# ---------------------------------------------------------------------------
# Code-first is not drug-first: the distinctions that matter
# ---------------------------------------------------------------------------

def test_code_first_recovers_drugs_absent_from_the_name_list(rx_root):
    """
    Lacosamide and clobazam carry G40 but are not in the twelve-name list, so a
    drug-first definition discards them. This is the reason for the rewrite.
    """
    primary = ch.find_cases("H", "primary", base_path=rx_root, save=False)
    broad = ch.find_cases("H", "broad", base_path=rx_root, save=False)

    assert {3, 4}.issubset(set(primary))
    assert {3, 4}.isdisjoint(set(broad))


def test_the_code_requirement_excludes_off_label_use(rx_root):
    """Topiramate for migraine (G43) is in the name list but is not a case."""
    primary = ch.find_cases("H", "primary", base_path=rx_root, save=False)

    assert 2 not in primary
    assert 2 in ch.find_cases("H", "broad", base_path=rx_root, save=False)


def test_the_code_must_be_on_the_same_prescription_row(rx_root):
    """
    Gabapentin is a confirmed ASM and participant 8 is on it, but the
    prescription is coded for nerve pain, so they are not a case.
    """
    assert 8 not in ch.find_cases("H", "primary", base_path=rx_root, save=False)


def test_only_current_use_counts(rx_root):
    """Participant 6 has phenytoin coded G40 but RXDUSE is 2."""
    for definition in EXPECTED:
        cases = ch.find_cases("H", definition, base_path=rx_root, save=False)
        assert 6 not in cases


def test_a_non_asm_carrying_the_code_is_blanked(rx_root):
    """Allopurinol with a G40 code is a miscode, not a case."""
    assert 5 not in ch.find_cases("H", "primary", base_path=rx_root, save=False)


# ---------------------------------------------------------------------------
# The completeness guard
# ---------------------------------------------------------------------------

def test_an_unreviewed_drug_with_the_code_stops_the_run(tmp_path):
    """
    Confirmation by allow-list alone would silently discard a new ASM. Every
    drug carrying the code must have been reviewed, so an unknown name raises
    and names itself rather than vanishing.
    """
    rows = SYNTHETIC_RX + [(10, "NEWDRUGX", 1.0, "G40", "", "")]
    root = write_rx(rows, tmp_path)

    with pytest.raises(ValueError, match="newdrugx"):
        ch.find_cases("H", "primary", base_path=root, save=False)


def test_a_missing_drug_name_with_the_code_stops_the_run(tmp_path):
    rows = SYNTHETIC_RX + [(11, None, 1.0, "G40", "", "")]
    root = write_rx(rows, tmp_path)

    with pytest.raises(ValueError, match="missing drug name"):
        ch.find_cases("H", "primary", base_path=root, save=False)


def test_the_guard_applies_to_narrow_as_well_as_primary(tmp_path):
    """
    Narrow keeps only four drug names, so an unreviewed drug would never have
    been selected anyway — but leaving it unreviewed still means the reason
    codes have not been fully accounted for.
    """
    rows = SYNTHETIC_RX + [(10, "NEWDRUGX", 1.0, "G40", "", "")]
    root = write_rx(rows, tmp_path)

    with pytest.raises(ValueError, match="newdrugx"):
        ch.find_cases("H", "narrow", base_path=root, save=False)


def test_definitions_without_a_code_requirement_need_no_review(tmp_path):
    """
    Broad selects on drug name, so the thousands of other drugs in the file are
    irrelevant to it and an unreviewed G40 drug must not stop it.
    """
    rows = SYNTHETIC_RX + [(10, "NEWDRUGX", 1.0, "G40", "", "")]
    root = write_rx(rows, tmp_path)

    assert set(ch.find_cases("H", "broad", base_path=root, save=False)) \
        == EXPECTED["broad"]


# ---------------------------------------------------------------------------
# A cycle with no reason-for-use variables
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("definition", ["primary", "narrow"])
def test_a_code_requiring_definition_is_refused_without_the_columns(
    definition, tmp_path
):
    """
    RXQ_RX_G has no reason-for-use variables, so it must refuse rather than
    quietly drop the requirement and return a broader cohort under a name that
    promises a narrower one.
    """
    root = write_rx(SYNTHETIC_RX, tmp_path, cycle="G", with_code_cols=False)

    with pytest.raises(ValueError, match="no reason-for-use variables"):
        ch.find_cases("G", definition, base_path=root, save=False)


@pytest.mark.parametrize("definition", ["broad", "narrow_nocode"])
def test_the_codeless_definitions_work_without_the_columns(definition, tmp_path):
    root = write_rx(SYNTHETIC_RX, tmp_path, cycle="G", with_code_cols=False)

    cases = ch.find_cases("G", definition, base_path=root, save=False)

    assert set(cases) == EXPECTED[definition]


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def test_saving_writes_the_case_list_and_a_provenance_sidecar(rx_root):
    cases = ch.find_cases("H", "primary", base_path=rx_root, save=True)

    assert set(ch.load_cases("H", "primary", base_path=rx_root)) == set(cases)

    sidecar = (rx_root / "processed" / "cases_H_primary.provenance.json")
    assert sidecar.exists()

    import json
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["definition"] == "primary"
    assert record["cases"] == len(cases)
    # The sidecar must record which drug names the definition stood for, since
    # the filename records only the definition's name.
    assert "asm_confirm" in record["parameters"]


def test_an_existing_case_file_is_not_overwritten_silently(rx_root):
    ch.find_cases("H", "primary", base_path=rx_root, save=True)

    path = rx_root / "processed" / "cases_H_primary.csv"
    path.write_text("SEQN\n999\n", encoding="utf-8")

    ch.find_cases("H", "primary", base_path=rx_root, save=True)
    assert set(ch.load_cases("H", "primary", base_path=rx_root)) == {999}

    ch.find_cases("H", "primary", base_path=rx_root, save=True, overwrite=True)
    assert set(ch.load_cases("H", "primary", base_path=rx_root)) == EXPECTED["primary"]


def test_the_definitions_are_saved_side_by_side(rx_root):
    """All four must coexist, since the sensitivity analyses compare them."""
    for definition in EXPECTED:
        ch.find_cases("H", definition, base_path=rx_root, save=True)

    for definition, expected in EXPECTED.items():
        assert set(ch.load_cases("H", definition, base_path=rx_root)) == expected


# ---------------------------------------------------------------------------
# The deprecated drug-first entry point
# ---------------------------------------------------------------------------

def test_find_people_on_asm_warns_and_returns_the_broad_definition(rx_root):
    with pytest.warns(DeprecationWarning, match="not the study's primary"):
        pwe = ch.find_people_on_asm("H", base_path=rx_root)

    assert set(pwe) == EXPECTED["broad"]

    # The legacy filename and format, which downstream code still reads
    legacy = rx_root / "processed" / "people_with_epilepsy_H.csv"
    assert legacy.exists()
    assert set(ch.load_pwe_seqn("H", base_path=rx_root)["SEQN"]) == EXPECTED["broad"]


# ---------------------------------------------------------------------------
# Against the real data
# ---------------------------------------------------------------------------

def real_rx_available(cycle="H"):
    from ambient_light_epilepsy import paths
    try:
        paths.raw_table(cycle, "RXQ_RX")
        return True
    except (FileNotFoundError, RuntimeError):
        return False


@pytest.mark.skipif(
    not real_rx_available(), reason="RXQ_RX_H not reachable on this machine"
)
def test_the_real_cycle_h_yields_match_the_analysis_log():
    """
    Pins the counts recorded in doc/analysis-log.md on 2026-09-02, so a change
    in the drug lists or the matching logic that moves the cohort shows up here
    rather than in a manuscript.

    Note these are *identified* counts, before the age and recording-validity
    criteria: primary 70 (not the 72 originally tabulated, which was the count
    before non-ASMs were blanked), narrow 38, broad 157.
    """
    assert len(ch.find_cases("H", "primary", save=False)) == 70
    assert len(ch.find_cases("H", "narrow", save=False)) == 38
    assert len(ch.find_cases("H", "broad", save=False)) == 157
