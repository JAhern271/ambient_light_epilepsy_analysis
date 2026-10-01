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
# Inclusion criteria
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sources(monkeypatch, demographics):
    """
    Stand in for the NHANES tables, so the inclusion logic can be tested
    without a data root. Ages are forced so that the age criterion and the
    validity criterion can be told apart.
    """
    demo = demographics.copy()
    demo.loc[1000, "age"] = 19            # under age, but a valid recording
    demo.loc[1001, "age"] = 45
    demo.loc[1002, "age"] = 45
    demo.loc[1003, "age"] = 45

    monkeypatch.setattr(matching.nhn, "load_partial_demo",
                        lambda *a, **k: demo)
    monkeypatch.setattr(matching.nhn, "add_demo_labels", lambda df: df)
    # 1000-1002 are cases; 1003 onwards are potential controls. Both case
    # sources are stubbed: the legacy file and a named definition's file.
    monkeypatch.setattr(matching.ch, "load_pwe_seqn",
                        lambda *a, **k: pd.DataFrame([1000, 1001, 1002]))
    monkeypatch.setattr(matching.ch, "load_cases",
                        lambda *a, **k: pd.Index([1000, 1001, 1002], name="SEQN"))

    return demo


def test_only_participants_with_a_valid_recording_are_eligible(fake_sources):
    """
    The validity gate is applied to the control pool and the cases alike.
    1002 has a valid recording but is withheld here; 1000 is age-ineligible.
    """
    df_all, df_pwe = matching.eligible_participants(
        "X", valid_seqns=[1000, 1001, 1003], definition="primary"
    )

    assert list(df_pwe.index) == [1001]          # 1000 too young, 1002 invalid
    assert set(df_all.index) == {1001, 1003}     # 1000 too young


def test_validity_and_definition_are_both_required():
    """
    Each names a different study population, and the superseded combination --
    the legacy drug-first case list and the PAXLDAY == '9' validity rule --
    was once reachable by saying nothing at all.
    """
    with pytest.raises(TypeError):
        matching.eligible_participants("X")

    with pytest.raises(TypeError):
        matching.eligible_participants("X", valid_seqns=[1, 2])


def test_an_unknown_case_definition_raises(fake_sources):
    """A typo must not fall through to some default case list."""
    with pytest.raises(ValueError, match="Unknown case definition"):
        matching.eligible_participants(
            "X", valid_seqns=[1000, 1001], definition="primry"
        )


def test_the_legacy_definition_reads_the_superseded_file(fake_sources):
    """
    'legacy' is the drug-first list behind everything in results/. It stays
    reachable so those files remain reproducible, but only when asked for by
    name -- it has a positive predictive value of 38.9% against G40.
    """
    _, df_pwe = matching.eligible_participants(
        "X", valid_seqns=[1000, 1001, 1003],
        definition=matching.LEGACY_DEFINITION,
    )

    assert list(df_pwe.index) == [1001]


def test_an_empty_validity_set_yields_an_empty_cohort(fake_sources):
    """Not an exception: a cycle with no valid recordings has no cohort."""
    df_all, df_pwe = matching.eligible_participants(
        "X", valid_seqns=[], definition="primary"
    )

    assert df_all.empty
    assert df_pwe.empty


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


# ---------------------------------------------------------------------------
# The eligible analytic sample
# ---------------------------------------------------------------------------

def test_eligible_filename_names_both_choices():
    """
    Two things decide who is in a cohort: the case definition and the
    valid-day rule. Both are in the filename, so a cohort built under
    different choices cannot overwrite another.
    """
    assert matching.eligible_filename("H", "primary", "d04h20") == (
        "eligible_H_primary_d04h20.csv"
    )
    assert matching.eligible_filename("H", "broad", "d03h16") != (
        matching.eligible_filename("H", "primary", "d04h20")
    )


def test_eligible_sample_round_trips_with_case_status(tmp_path, monkeypatch,
                                                      demographics):
    """
    The file is one row per eligible participant with an `epilepsy` flag, not
    two SEQN lists: eligibility and case status are one table.
    """
    monkeypatch.setenv("ALE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ALE_PROFILE", raising=False)

    df_all = demographics
    df_pwe = demographics.iloc[:40]

    path = matching.save_eligible_sample(df_all, df_pwe, "X", "primary", "d04h20")
    assert path.name == "eligible_X_primary_d04h20.csv"

    back = matching.load_eligible_sample("X", "primary", "d04h20")

    assert len(back) == len(df_all)
    assert back["epilepsy"].sum() == len(df_pwe)
    assert set(back.index[back["epilepsy"] == 1]) == set(df_pwe.index)
    assert back.index.name == "SEQN"


def test_an_existing_cohort_is_not_overwritten_silently(tmp_path, monkeypatch,
                                                        demographics):
    """
    Same reasoning as `wear.save_validity`: this file defines the study
    population, so replacing one changes what every downstream result was
    computed from.
    """
    monkeypatch.setenv("ALE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ALE_PROFILE", raising=False)

    df_all, df_pwe = demographics, demographics.iloc[:40]
    matching.save_eligible_sample(df_all, df_pwe, "X", "primary", "d04h20")

    with pytest.raises(FileExistsError, match="study population"):
        matching.save_eligible_sample(df_all, df_pwe, "X", "primary", "d04h20")

    # a different definition or rule writes alongside it, needing no overwrite
    other = matching.save_eligible_sample(df_all, df_pwe, "X", "broad", "d04h20")
    assert other.name == "eligible_X_broad_d04h20.csv"

    matching.save_eligible_sample(
        df_all, df_pwe, "X", "primary", "d04h20", overwrite=True
    )


def test_a_missing_cohort_says_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ALE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ALE_PROFILE", raising=False)

    with pytest.raises(FileNotFoundError, match="build_cohort"):
        matching.load_eligible_sample("X", "primary", "d04h20")
