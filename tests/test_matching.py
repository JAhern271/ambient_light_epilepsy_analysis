"""
Tests for cohort definition.

Matching decides who is in the study, so a fault here changes every downstream
result while still producing a plausible-looking cohort. These use synthetic
demographics so they run without the real data.
"""

import numpy as np
import pandas as pd
import pytest

from ambient_light_epilepsy import matching


@pytest.fixture
def demographics():
    """
    A synthetic population of 600 adults with the labelled columns matching
    expects, spread across every stratum so controls are always available.
    """
    rng = np.random.default_rng(20260817)
    n = 600

    return pd.DataFrame(
        {
            "age": rng.integers(20, 85, n),
            "sex_label": rng.choice(["Male", "Female"], n),
            "race_label": rng.choice(["Non-Hispanic White", "Non-Hispanic Black"], n),
            "season": rng.choice(["Winter", "Summer"], n),
            "PIR_cat": rng.choice(["<1 (Low)", "1–4 (Middle)"], n),
        },
        index=pd.Index(range(1000, 1000 + n), name="SEQN"),
    )


@pytest.fixture
def cases_and_pool(demographics):
    """The first 40 participants are the cases; all 600 are the pool."""
    return demographics, demographics.iloc[:40]


# ---------------------------------------------------------------------------
# Age banding
# ---------------------------------------------------------------------------

def test_age_bands_have_expected_boundaries():
    ages = pd.Series([19, 20, 24, 25, 79, 80, 95])

    banded = matching.bin_age(ages).astype(str).tolist()

    assert banded == ["0-19", "20-24", "20-24", "25-29", "75-79", "80+", "80+"]


def test_age_bands_are_left_inclusive():
    """A participant aged exactly 25 belongs to 25-29, not 20-24."""
    assert str(matching.bin_age(pd.Series([25])).iloc[0]) == "25-29"


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_same_seed_gives_the_same_cohort(cases_and_pool):
    """Reproducibility is the whole reason the sampling is seeded."""
    pool, cases = cases_and_pool

    first, _ = matching.find_frequency_matched_controls(pool, cases, seed=42)
    second, _ = matching.find_frequency_matched_controls(pool, cases, seed=42)

    assert list(first.index) == list(second.index)


def test_different_seed_gives_a_different_cohort(cases_and_pool):
    pool, cases = cases_and_pool

    first, _ = matching.find_frequency_matched_controls(pool, cases, seed=42)
    second, _ = matching.find_frequency_matched_controls(pool, cases, seed=7)

    assert list(first.index) != list(second.index)


def test_cases_are_never_selected_as_their_own_controls(cases_and_pool):
    pool, cases = cases_and_pool

    controls, selected_cases = matching.find_frequency_matched_controls(pool, cases)

    assert not set(controls.index) & set(selected_cases.index)


def test_controls_are_unique(cases_and_pool):
    """A participant must not be sampled as a control more than once."""
    pool, cases = cases_and_pool

    controls, _ = matching.find_frequency_matched_controls(pool, cases)

    assert controls.index.nunique() == len(controls)


def test_control_ratio_is_a_ceiling_not_a_target(cases_and_pool):
    """
    Never more than `control_ratio` controls per case, but often fewer: a
    stratum with too few eligible participants contributes what it has. The
    real cohort achieves 3.37 per case against 4 requested for this reason.
    """
    pool, cases = cases_and_pool

    controls, selected = matching.find_frequency_matched_controls(
        pool, cases, control_ratio=2
    )

    assert 0 < len(controls) <= 2 * len(selected)


def test_ratio_is_met_exactly_when_the_pool_is_deep():
    """
    Where every participant shares one stratum there is no shortfall, so the
    requested ratio is achieved exactly. This separates "the sampling is
    wrong" from "the data were too thin".
    """
    n = 200
    pool = pd.DataFrame(
        {
            "age": [30] * n,
            "sex_label": ["Female"] * n,
            "race_label": ["Non-Hispanic White"] * n,
            "season": ["Winter"] * n,
            "PIR_cat": ["<1 (Low)"] * n,
        },
        index=pd.Index(range(n), name="SEQN"),
    )
    cases = pool.iloc[:10]

    controls, selected = matching.find_frequency_matched_controls(
        pool, cases, control_ratio=3
    )

    assert len(selected) == 10
    assert len(controls) == 30


def test_larger_ratio_selects_more_controls(cases_and_pool):
    pool, cases = cases_and_pool

    few, _ = matching.find_frequency_matched_controls(pool, cases, control_ratio=1)
    many, _ = matching.find_frequency_matched_controls(pool, cases, control_ratio=3)

    assert len(many) > len(few)


def test_every_control_shares_a_stratum_with_some_case(cases_and_pool):
    """
    The point of frequency matching: no control may come from a stratum that
    contains no cases.
    """
    pool, cases = cases_and_pool

    controls, selected = matching.find_frequency_matched_controls(pool, cases)

    def strata(df):
        df = df.copy()
        df["age_bin"] = matching.bin_age(df["age"])
        return set(map(tuple, df[matching.MATCH_COLS].astype(str).values))

    assert strata(controls) <= strata(selected)


def test_cases_missing_a_matching_variable_are_dropped(cases_and_pool):
    """A case with no PIR cannot be matched on it, so it leaves the study."""
    pool, cases = cases_and_pool
    cases = cases.copy()
    cases.loc[cases.index[0], "PIR_cat"] = np.nan

    _, selected = matching.find_frequency_matched_controls(pool, cases)

    assert cases.index[0] not in selected.index
    assert len(selected) == len(cases) - 1


def test_no_eligible_controls_yields_an_empty_result(demographics):
    """A stratum with no available controls is skipped rather than crashing."""
    cases = demographics.iloc[:5]
    pool = cases  # every candidate is a case, so nothing is left to sample

    controls, selected = matching.find_frequency_matched_controls(pool, cases)

    assert len(controls) == 0
    assert len(selected) == 5


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_summary_covers_every_matching_variable(cases_and_pool):
    pool, cases = cases_and_pool
    controls, selected = matching.find_frequency_matched_controls(pool, cases)

    summary = matching.summarise_match(selected, controls)

    assert set(summary) == set(matching.MATCH_COLS)
    for table in summary.values():
        assert list(table.columns) == ["PWE", "Matched controls"]


def test_summary_proportions_sum_to_one(cases_and_pool):
    pool, cases = cases_and_pool
    controls, selected = matching.find_frequency_matched_controls(pool, cases)

    for name, table in matching.summarise_match(selected, controls).items():
        assert table["PWE"].sum() == pytest.approx(1.0, abs=0.01), name
        assert table["Matched controls"].sum() == pytest.approx(1.0, abs=0.01), name


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def test_saved_files_round_trip(cases_and_pool, tmp_path, monkeypatch):
    monkeypatch.setenv("ALE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ALE_PROFILE", raising=False)

    pool, cases = cases_and_pool
    controls, selected = matching.find_frequency_matched_controls(pool, cases)

    control_path, case_path = matching.save_matching_results(controls, selected, "X")

    assert control_path.exists() and case_path.exists()
    assert list(pd.read_csv(control_path, index_col=0).iloc[:, 0]) == list(controls.index)
    assert list(pd.read_csv(case_path, index_col=0).iloc[:, 0]) == list(selected.index)
