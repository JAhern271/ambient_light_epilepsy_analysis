"""
Tests for the non-wear and valid-day rules of methods.md 5.1 and 5.2.

A wrong wear threshold produces a cohort of the wrong size and nothing broken
to notice, so nearly every expected value here is derived on paper from the
recording's structure rather than read off what the code currently returns.

The two valid-day thresholds were settled on 2026-09-03 at D=4 valid days and
H=20 hours, following Xiao 2023. They remain REQUIRED arguments with no
defaults, so every run states the rule it used and the sensitivity rules read
identically to the primary one. The tests below are parameterised on
`min_wear_hours` and `min_valid_days` and assert the rule holds for whatever
values it is given, which is why settling them changed no test in this file.

Recording geometry used throughout
----------------------------------
`late_start` is calibrated to real participant SEQN 73557: PAXFTIME 16:30,
PAXLDAY 9, 11,529 minute records.

Absolute minutes are counted from midnight at the start of calendar day 1, so
the first minute of the recording is absolute minute 990 (16:30). Noon-to-noon
day k covers absolute minutes [720 + 1440k, 720 + 1440k + 1439].

    k = 0   12:00 d1 - 11:59 d2   recording covers 990..2159    1,170 min
    k = 1   12:00 d2 - 11:59 d3   covered in full               1,440 min
      ...                                                        ...
    k = 7   12:00 d8 - 11:59 d9   covered in full               1,440 min
    k = 8   12:00 d9 - 16:38 d9   recording ends at 12,518        279 min
                                                       total   11,529 min

Nine slots; dropping the partial first and last leaves seven candidate days.
Candidate day j (1-indexed) begins at minute index 720 + 1440j - 990.
"""

import numpy as np
import pandas as pd
import pytest

from ambient_light_epilepsy import wear

from conftest import (
    clock_to_minutes,
    make_header,
    make_paxmin,
    mark_nonwear,
    set_minutes,
)


# Geometry of the `late_start` recording, all derived in the docstring above
LATE_FIRST_TIME = "16:30:00"
LATE_TOTAL_MINUTES = 11_529
LATE_CANDIDATE_DAYS = 7
LATE_COVERAGE = [1170] + [1440] * 7 + [279]


def candidate_start(j, first_time=LATE_FIRST_TIME):
    """Minute index at which candidate day `j` (1-indexed) begins."""
    return 720 + 1440 * j - clock_to_minutes(first_time)


def days_for(minutes, first_time=LATE_FIRST_TIME, min_wear_hours=20):
    """Run the whole chain and return the day-level table."""
    prepared = wear.prepare_minutes(minutes, first_time)
    return wear.summarise_days(prepared, min_wear_hours=min_wear_hours)


# ---------------------------------------------------------------------------
# Group 1: day construction. No thresholds involved.
# ---------------------------------------------------------------------------

def test_late_start_recording_has_the_expected_length():
    """450 + 1440*7 + 999. If the builder is wrong, everything below is too."""
    assert len(make_paxmin(first_time=LATE_FIRST_TIME)) == LATE_TOTAL_MINUTES


@pytest.mark.parametrize(
    "first_time, last_day, last_day_minutes, coverage, candidates",
    [
        # 16:30 start, complete recording -- the common case
        ("16:30:00", 9, None, [1170] + [1440] * 7 + [279], 7),
        # 09:11 start: the recording opens mid-morning, so the leading partial
        # day is tiny (169 min) and the trailing one nearly complete (1,280)
        ("09:11:00", 9, None, [169] + [1440] * 7 + [1280], 7),
        # Starting exactly at noon aligns the recording to the day boundaries,
        # so there are no partial days at all and dropping the ends costs two
        # complete days. Does not occur in cycle H (nearest PAXFTIME is 12:01)
        # but it is what produces the 6-candidate tail in the real data.
        ("12:00:00", 9, 720, [1440] * 8, 6),
        # Short recordings: the rule that replaces PAXLDAY == '9'
        ("16:30:00", 6, None, [1170] + [1440] * 4 + [279], 4),
        ("16:30:00", 5, None, [1170] + [1440] * 3 + [279], 3),
    ],
)
def test_noon_day_coverage_and_candidate_count(
    first_time, last_day, last_day_minutes, coverage, candidates
):
    """
    Overlaying noon boundaries on the recording must produce exactly the
    coverage derived by hand, and dropping the first and last slot must leave
    the stated number of candidate days.
    """
    minutes = make_paxmin(
        first_time=first_time, last_day=last_day, last_day_minutes=last_day_minutes
    )
    days = days_for(minutes, first_time=first_time)

    assert sorted(days["minutes_present"].to_list(), reverse=True) == sorted(
        coverage, reverse=True
    )
    assert days["minutes_present"].sum() == len(minutes)
    assert int(days["is_candidate"].sum()) == candidates


def test_the_dropped_days_are_the_first_and_last():
    """Not the two smallest, not two arbitrary ones -- the ends."""
    days = days_for(make_paxmin(first_time=LATE_FIRST_TIME)).sort_index()

    assert not days["is_candidate"].iloc[0]
    assert not days["is_candidate"].iloc[-1]
    assert days["is_candidate"].iloc[1:-1].all()


def test_keeping_the_partial_ends_recovers_the_nearly_complete_day():
    """
    The cost of the chosen reading, made explicit. A 09:11 start has a
    1,280-minute (21.3 h) trailing day that the drop rule discards; with the
    rule switched off it survives the 20 h test and the leading 169-minute day
    does not, giving 8 valid days rather than 7.
    """
    minutes = make_paxmin(first_time="09:11:00")
    prepared = wear.prepare_minutes(
        minutes, "09:11:00", validity={"drop_partial_first_last": False}
    )
    days = wear.summarise_days(prepared, min_wear_hours=20)

    assert int(days["is_candidate"].sum()) == 9      # nothing dropped
    assert int(days["is_valid"].sum()) == 8          # the 169-minute day fails


# ---------------------------------------------------------------------------
# Group 2: the four minute-level exclusions of methods.md 5.1
# ---------------------------------------------------------------------------

TARGET = 3                      # inject into candidate day 3
INJECT = 300


@pytest.mark.parametrize(
    "columns",
    [
        {"PAXPREDM": "3"},                      # non-wear
        {"PAXQFM": 1.0, "PAXFLGSM": "P"},       # quality flagged
        {"PAXMTSM": -0.01},                     # activity uncomputable
    ],
    ids=["non_wear", "quality_flag", "uncomputable"],
)
def test_each_exclusion_removes_its_minutes(columns):
    """Each rule on its own must take a 1,440-minute day down to 1,140."""
    minutes = set_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME),
        candidate_start(TARGET), INJECT, **columns,
    )
    days = days_for(minutes).sort_index()

    assert days["minutes_retained"].iloc[TARGET] == 1440 - INJECT


def test_a_clean_day_retains_every_minute():
    days = days_for(make_paxmin(first_time=LATE_FIRST_TIME)).sort_index()

    assert days["minutes_retained"].iloc[TARGET] == 1440


def test_overlapping_exclusions_are_a_union_not_a_sum():
    """
    All three rules on the SAME 300 minutes must still remove 300 minutes.
    Counting them additively would silently shrink every day, and the result
    would look entirely plausible.
    """
    minutes = set_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME),
        candidate_start(TARGET), INJECT,
        PAXPREDM="3", PAXQFM=1.0, PAXFLGSM="P", PAXMTSM=-0.01,
    )
    days = days_for(minutes).sort_index()

    assert days["minutes_retained"].iloc[TARGET] == 1440 - INJECT


def test_disjoint_exclusions_accumulate():
    """Two rules hitting different minutes must remove both sets."""
    minutes = make_paxmin(first_time=LATE_FIRST_TIME)
    start = candidate_start(TARGET)
    minutes = set_minutes(minutes, start, 100, PAXPREDM="3")
    minutes = set_minutes(minutes, start + 200, 100, PAXQFM=1.0, PAXFLGSM="V")

    days = days_for(minutes).sort_index()

    assert days["minutes_retained"].iloc[TARGET] == 1440 - 200


@pytest.mark.parametrize("flag", ["A", "P", "V", "AB", "ABCSUWXY"])
def test_any_quality_flag_letter_excludes(flag):
    """
    methods.md 5.1 excludes a minute whose flag variable holds any letter, not
    one particular letter. PAXFLGSM concatenates codes, so multi-letter values
    must be handled too.
    """
    minutes = set_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME),
        candidate_start(TARGET), INJECT,
        PAXQFM=float(len(flag)), PAXFLGSM=flag,
    )
    days = days_for(minutes).sort_index()

    assert days["minutes_retained"].iloc[TARGET] == 1440 - INJECT


def test_quality_flag_count_and_flag_string_give_the_same_mask():
    """
    PAXQFM is the number of letters in PAXFLGSM, so `PAXQFM > 0` and
    `PAXFLGSM != ''` are the same rule. Verified to agree on all 88,223,479
    rows of PAXMIN_H; pinned here so the equivalence is not assumed silently.
    """
    minutes = set_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME),
        candidate_start(TARGET), INJECT, PAXQFM=2.0, PAXFLGSM="AB",
    )

    by_count = wear.mask_minutes(minutes)["retained"]
    by_string = wear.mask_minutes(minutes.assign(PAXQFM=0.0))["retained"]

    pd.testing.assert_series_equal(by_count, by_string)


def test_sleep_minutes_are_kept():
    """
    PAXPREDM 2 is sleep wear, not non-wear. Masking it would delete every
    night, and with it the entire nighttime light hypothesis (H2).
    """
    minutes = set_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME),
        candidate_start(TARGET), INJECT, PAXPREDM="2",
    )
    days = days_for(minutes).sort_index()

    assert days["minutes_retained"].iloc[TARGET] == 1440


def test_unknown_wear_status_is_kept():
    """
    PAXPREDM 4 is 'unknown' -- 2,946,459 minutes, 3.3% of PAXMIN_H. methods.md
    5.1 masks only code 3, and the researcher settled on keeping code 4 on
    2026-09-03. Pinned so the decision is explicit and reversing it is a
    visible test change rather than a quiet one.
    """
    minutes = set_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME),
        candidate_start(TARGET), INJECT, PAXPREDM="4",
    )
    days = days_for(minutes).sort_index()

    assert days["minutes_retained"].iloc[TARGET] == 1440


def test_a_minute_with_few_valid_seconds_is_kept():
    """
    PAXTSM is NOT an exclusion. Notebook 09 dropped minutes below 45 seconds,
    but that rule was never in methods.md 5.1, and in cycle H it excluded
    nothing the three real rules do not already exclude -- 63 minutes in 88
    million, every one already dropped. The parameter was deleted on
    2026-09-03 rather than written into the spec.

    PAXTSM is at the codebook minimum of 3 seconds here and the minute still
    survives, so re-adding the rule breaks this test.
    """
    minutes = set_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME),
        candidate_start(TARGET), INJECT, PAXTSM=3.0,
    )
    days = days_for(minutes).sort_index()

    assert days["minutes_retained"].iloc[TARGET] == 1440


def test_the_deleted_wear_seconds_parameter_is_gone():
    """
    `validity.min_valid_seconds` was removed from analysis_params.toml. The
    override check in wear._validity rejects unknown keys, so this also
    confirms nothing else still expects it.
    """
    from ambient_light_epilepsy import params

    assert "min_valid_seconds" not in params.section("validity")

    with pytest.raises(KeyError, match="min_valid_seconds"):
        wear.mask_minutes(
            make_paxmin(first_time=LATE_FIRST_TIME),
            validity={"min_valid_seconds": 45},
        )


# ---------------------------------------------------------------------------
# Group 3: light and activity are masked together (methods.md 5.1)
# ---------------------------------------------------------------------------

def masked_recording():
    """
    A candidate day holding two DIFFERENT kinds of excluded minute:

        1,040 wear minutes        100 lux,      1.0 activity
          300 non-wear minutes  9,999 lux,    500.0 activity
          100 quality-flagged   8,888 lux,    400.0 activity

    Two kinds matter. With only non-wear injected, the retained set and the
    not-non-wear set are the same array, so a version of the code that masked
    activity by non-wear alone would pass -- which is precisely the separate
    masking that 5.1 forbids. The flagged block makes the two sets differ.

    Correctly masked, the day's mean lux is exactly 100 and its mean activity
    exactly 1.0. Unmasked, the mean lux is 2,772.57.
    """
    minutes = make_paxmin(first_time=LATE_FIRST_TIME, lux=100.0, activity=1.0)
    start = candidate_start(TARGET)

    minutes = set_minutes(
        minutes, start, INJECT,
        PAXPREDM="3", PAXLXMM=9999.0, PAXMTSM=500.0,
    )
    return set_minutes(
        minutes, start + INJECT, 100,
        PAXQFM=1.0, PAXFLGSM="P", PAXLXMM=8888.0, PAXMTSM=400.0,
    )


def target_day(prepared):
    """The rows of the candidate day the exclusions were injected into."""
    labels = sorted(prepared["analytic_day"].unique())
    return prepared[prepared["analytic_day"] == labels[TARGET]]


def test_only_the_wear_minutes_survive():
    day = target_day(wear.prepare_minutes(masked_recording(), LATE_FIRST_TIME))

    assert int(day["retained"].sum()) == 1440 - INJECT - 100


def test_light_is_masked_over_every_excluded_minute():
    day = target_day(wear.prepare_minutes(masked_recording(), LATE_FIRST_TIME))

    assert day["mean_lux"].mean() == pytest.approx(100.0)
    # what an unmasked mean would have been, for contrast
    assert day["PAXLXMM"].mean() == pytest.approx(2772.5694444, abs=1e-4)


def test_activity_is_masked_over_every_excluded_minute():
    """
    Masking activity by non-wear alone would leave the 100 flagged minutes in,
    giving a mean of 36.0 rather than 1.0.
    """
    day = target_day(wear.prepare_minutes(masked_recording(), LATE_FIRST_TIME))

    assert day["activity"].mean() == pytest.approx(1.0)


def test_light_and_activity_share_one_retained_minute_set():
    """
    methods.md 5.1 masks the channels jointly, so that no participant can
    contribute light from a minute excluded from their activity metrics.
    """
    prepared = wear.prepare_minutes(masked_recording(), LATE_FIRST_TIME)

    pd.testing.assert_series_equal(
        prepared["mean_lux"].notna(), prepared["activity"].notna(),
        check_names=False,
    )
    assert (prepared["mean_lux"].notna() == prepared["retained"]).all()
    assert (prepared["activity"].notna() == prepared["retained"]).all()


# ---------------------------------------------------------------------------
# Group 4: the wear threshold, parameterised on min_wear_hours
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("min_wear_hours", [16, 20, 22])
def test_a_day_is_valid_at_exactly_the_wear_threshold(min_wear_hours):
    """
    '>= H hours' is inclusive (methods.md 5.2). A 1,440-minute day needs
    60*H retained, so it may lose 1440 - 60*H and no more:

        H = 16  ->   960 retained,  480 maskable
        H = 20  -> 1,200 retained,  240 maskable
        H = 22  -> 1,320 retained,  120 maskable
    """
    maskable = 1440 - 60 * min_wear_hours

    at_limit = days_for(
        mark_nonwear(make_paxmin(first_time=LATE_FIRST_TIME),
                     candidate_start(TARGET), maskable),
        min_wear_hours=min_wear_hours,
    ).sort_index()

    assert at_limit["minutes_retained"].iloc[TARGET] == 60 * min_wear_hours
    assert at_limit["is_valid"].iloc[TARGET]


@pytest.mark.parametrize("min_wear_hours", [16, 20, 22])
def test_a_day_is_invalid_one_minute_below_the_wear_threshold(min_wear_hours):
    """One minute past the allowance and the day must fail."""
    maskable = 1440 - 60 * min_wear_hours

    over_limit = days_for(
        mark_nonwear(make_paxmin(first_time=LATE_FIRST_TIME),
                     candidate_start(TARGET), maskable + 1),
        min_wear_hours=min_wear_hours,
    ).sort_index()

    assert over_limit["minutes_retained"].iloc[TARGET] == 60 * min_wear_hours - 1
    assert not over_limit["is_valid"].iloc[TARGET]


def test_a_dropped_partial_day_is_never_valid():
    """
    The leading partial day of `late_start` holds 1,170 minutes, which clears
    a 16 h threshold on its own. It must still be invalid, because it was
    dropped as partial before the threshold was applied.
    """
    days = days_for(
        make_paxmin(first_time=LATE_FIRST_TIME), min_wear_hours=16
    ).sort_index()

    assert days["minutes_retained"].iloc[0] == 1170
    assert not days["is_valid"].iloc[0]


# ---------------------------------------------------------------------------
# Group 5: the participant rule, parameterised on min_valid_days
# ---------------------------------------------------------------------------

def with_valid_days(k, min_wear_hours=20, first_time=LATE_FIRST_TIME):
    """
    A `late_start` recording whose first 7 - k candidate days are pushed one
    minute below the wear threshold, leaving exactly k valid.
    """
    minutes = make_paxmin(first_time=first_time)
    over_limit = 1440 - 60 * min_wear_hours + 1

    for j in range(1, LATE_CANDIDATE_DAYS - k + 1):
        minutes = mark_nonwear(minutes, candidate_start(j, first_time), over_limit)

    return minutes


@pytest.mark.parametrize("min_valid_days", [3, 4, 5])
def test_participant_included_with_exactly_the_required_valid_days(min_valid_days):
    days = days_for(with_valid_days(min_valid_days))
    summary = wear.summarise_participant(days, min_valid_days=min_valid_days)

    assert summary["n_valid_days"] == min_valid_days
    assert summary["meets_criterion"]


@pytest.mark.parametrize("min_valid_days", [3, 4, 5])
def test_participant_excluded_one_valid_day_short(min_valid_days):
    days = days_for(with_valid_days(min_valid_days - 1))
    summary = wear.summarise_participant(days, min_valid_days=min_valid_days)

    assert summary["n_valid_days"] == min_valid_days - 1
    assert not summary["meets_criterion"]


def test_a_six_day_recording_can_still_qualify():
    """
    The case-recovery mechanism. PAXLDAY == '9' excluded this participant
    outright; methods.md 5.2 gives them 4 candidate days, all valid.
    """
    days = days_for(make_paxmin(first_time=LATE_FIRST_TIME, last_day=6))
    summary = wear.summarise_participant(days, min_valid_days=4)

    assert summary["n_candidate_days"] == 4
    assert summary["n_valid_days"] == 4
    assert summary["meets_criterion"]


def test_a_five_day_recording_does_not_qualify():
    """Excluded by the old rule and by the new one, but now for a stated reason."""
    days = days_for(make_paxmin(first_time=LATE_FIRST_TIME, last_day=5))
    summary = wear.summarise_participant(days, min_valid_days=4)

    assert summary["n_candidate_days"] == 3
    assert not summary["meets_criterion"]


def test_thresholds_have_no_defaults():
    """
    min_valid_days and min_wear_hours are unsettled (implementation-status.md),
    so no call site may adopt 4 / 20 by omission. Both must be passed.
    """
    prepared = wear.prepare_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME), LATE_FIRST_TIME
    )

    with pytest.raises(TypeError):
        wear.summarise_days(prepared)

    with pytest.raises(TypeError):
        wear.summarise_participant(wear.summarise_days(prepared, min_wear_hours=20))


# ---------------------------------------------------------------------------
# Group 6: clock time. PAXMIN has none; it comes from PAXHD's PAXFTIME.
# ---------------------------------------------------------------------------

def test_the_first_minute_carries_the_header_start_time():
    prepared = wear.prepare_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME), LATE_FIRST_TIME
    )
    first = prepared["timestamp"].iloc[0]

    assert (first.hour, first.minute) == (16, 30)


def test_midnight_falls_where_the_sample_counter_says_it_does():
    """
    16:30 is minute 990 of calendar day 1, so minute 450 of the recording is
    midnight starting calendar day 2 -- PAXSSNMP 450 * 4800 = 2,160,000.
    """
    minutes = make_paxmin(first_time=LATE_FIRST_TIME)
    prepared = wear.prepare_minutes(minutes, LATE_FIRST_TIME)

    assert minutes["PAXSSNMP"].iloc[450] == 2_160_000
    midnight = prepared["timestamp"].iloc[450]
    assert (midnight.hour, midnight.minute) == (0, 0)
    assert prepared["PAXDAYM"].iloc[450] == "2"


@pytest.mark.parametrize(
    "window, hours_in_window",
    [((23, 6), 7), ((7, 19), 12)],
    ids=["night_23_06", "day_07_19"],
)
def test_clock_windows_land_on_the_right_number_of_minutes(window, hours_in_window):
    """
    Every noon-to-noon day contains each hour of the clock exactly once, so a
    window of `hours_in_window` hours covers 60 * hours_in_window minutes of
    each of the 7 candidate days.

    This is the test that fails for notebook 09's `PAXSSNMP % 1440`, which
    treats the first minute as midnight and so shifts every clock time by
    PAXFTIME -- 990 minutes for this participant, and a different amount for
    every other one.
    """
    prepared = wear.prepare_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME), LATE_FIRST_TIME
    )
    candidates = prepared[prepared["is_candidate"]]
    hour = candidates["timestamp"].dt.hour

    start, end = window
    if start < end:
        in_window = (hour >= start) & (hour < end)
    else:
        in_window = (hour >= start) | (hour < end)

    assert int(in_window.sum()) == LATE_CANDIDATE_DAYS * 60 * hours_in_window


def test_each_candidate_day_holds_one_whole_night():
    """
    The point of noon-to-noon days (methods.md 5.2): the 23:00-06:00 window
    must fall inside a single analytic day, not be split across two.
    """
    prepared = wear.prepare_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME), LATE_FIRST_TIME
    )
    hour = prepared["timestamp"].dt.hour
    night = prepared[(hour >= 23) | (hour < 6)]

    per_day = night[night["is_candidate"]].groupby("analytic_day").size()

    assert (per_day == 420).all()
    assert len(per_day) == LATE_CANDIDATE_DAYS


# ---------------------------------------------------------------------------
# Group 7: degenerate input
# ---------------------------------------------------------------------------

def test_out_of_order_samples_raise_rather_than_being_renumbered():
    """
    Notebook 09 replaced a non-monotonic counter with `np.arange(len(df))`,
    silently fabricating timestamps for the whole recording. Refuse instead.
    """
    minutes = make_paxmin(first_time=LATE_FIRST_TIME)
    minutes.loc[5, "PAXSSNMP"] = minutes.loc[500, "PAXSSNMP"]

    with pytest.raises(ValueError, match="monotonic"):
        wear.prepare_minutes(minutes, LATE_FIRST_TIME)


def test_a_sample_counter_off_the_minute_grid_raises():
    """PAXSSNMP must be a whole number of minutes: 60 * 80 ticks each."""
    minutes = make_paxmin(first_time=LATE_FIRST_TIME)
    minutes.loc[5, "PAXSSNMP"] += 1

    with pytest.raises(ValueError, match="minute"):
        wear.prepare_minutes(minutes, LATE_FIRST_TIME)


def test_absent_rows_are_not_filled_in():
    """
    Rows can simply be missing from PAXMIN. Those minutes must count as absent
    -- not zero-filled, which would read as an hour of darkness and stillness.
    """
    minutes = make_paxmin(first_time=LATE_FIRST_TIME)
    gap = minutes.index[candidate_start(TARGET):candidate_start(TARGET) + 60]
    minutes = minutes.drop(gap)

    days = days_for(minutes).sort_index()

    assert days["minutes_present"].iloc[TARGET] == 1440 - 60
    assert days["minutes_retained"].iloc[TARGET] == 1440 - 60
    assert days["is_valid"].iloc[TARGET]        # 1,380 minutes still clears 20 h


def test_predicted_wear_status_masks_whether_stored_as_text_or_number():
    """
    PAXPREDM is released as a string ('3'), but any caller who has done
    `astype(float)` -- as notebook 09 does -- passes 3.0. Both must mask, or
    the comparison matches nothing and no minutes are excluded at all.
    """
    as_text = set_minutes(
        make_paxmin(first_time=LATE_FIRST_TIME),
        candidate_start(TARGET), INJECT, PAXPREDM="3",
    )
    as_number = as_text.assign(PAXPREDM=as_text["PAXPREDM"].astype(float))

    assert (
        days_for(as_number).sort_index()["minutes_retained"].iloc[TARGET]
        == days_for(as_text).sort_index()["minutes_retained"].iloc[TARGET]
        == 1440 - INJECT
    )


def test_a_participant_missing_from_the_header_raises():
    """No PAXFTIME means no clock time, so there is nothing to fall back on."""
    header = make_header(seqn=1001, first_time=LATE_FIRST_TIME)

    assert wear.header_start_time(header, 1001) == LATE_FIRST_TIME

    with pytest.raises(KeyError, match="9999"):
        wear.header_start_time(header, 9999)


def test_an_empty_recording_gives_no_valid_days():
    """A participant with no minutes is excluded, not an exception."""
    empty = make_paxmin(first_time=LATE_FIRST_TIME).iloc[0:0]
    days = days_for(empty)
    summary = wear.summarise_participant(days, min_valid_days=4)

    assert summary["n_candidate_days"] == 0
    assert summary["n_valid_days"] == 0
    assert not summary["meets_criterion"]


# ---------------------------------------------------------------------------
# The superseded header rule, kept so results/ stays reproducible
# ---------------------------------------------------------------------------

def test_participants_split_across_row_groups_are_reassembled():
    """
    PAXMIN_H is read a row group at a time, and a participant's minutes can
    straddle a boundary. If the trailing rows are assessed on their own the
    participant appears twice, each time truncated, and the loss is silent.
    """
    one = make_paxmin(seqn=1001, first_time=LATE_FIRST_TIME)
    two = make_paxmin(seqn=1002, first_time=LATE_FIRST_TIME)

    # Chunk boundaries mid-way through 1001 and again mid-way through 1002
    stacked = pd.concat([one, two], ignore_index=True)
    chunks = [stacked.iloc[:5000], stacked.iloc[5000:9000], stacked.iloc[9000:]]

    recovered = dict(wear.iter_participants(chunks))

    assert sorted(recovered) == [1001.0, 1002.0]
    assert len(recovered[1001.0]) == LATE_TOTAL_MINUTES
    assert len(recovered[1002.0]) == LATE_TOTAL_MINUTES


def test_row_group_reassembly_respects_the_participant_filter():
    stacked = pd.concat(
        [make_paxmin(seqn=s, first_time=LATE_FIRST_TIME) for s in (1001, 1002)],
        ignore_index=True,
    )
    chunks = [stacked.iloc[:5000], stacked.iloc[5000:]]

    recovered = dict(wear.iter_participants(chunks, wanted={1002.0}))

    assert list(recovered) == [1002.0]
    assert len(recovered[1002.0]) == LATE_TOTAL_MINUTES


def test_a_participant_alone_in_the_final_chunk_is_not_lost():
    """The carry buffer must be flushed once the chunks run out."""
    one = make_paxmin(seqn=1001, first_time=LATE_FIRST_TIME)

    recovered = dict(wear.iter_participants([one]))

    assert list(recovered) == [1001.0]
    assert len(recovered[1001.0]) == LATE_TOTAL_MINUTES


def test_header_only_validity_reproduces_the_superseded_rule():
    """
    PAXSTS == 1 and PAXLDAY == '9'. Retained only so the existing cohort files
    and everything in results/ remain explicable; it is not the study rule.
    """
    header = make_header(
        seqn=[1, 2, 3, 4],
        first_time=["16:30:00"] * 4,
        last_day=[9, 6, 9, 9],
        status=[1, 1, 2, 1],
    )

    valid = wear.header_only_validity(header)

    assert list(valid) == [1.0, 4.0]


# ---------------------------------------------------------------------------
# The validity table on disk
# ---------------------------------------------------------------------------

def a_validity_table():
    """
    A small participant-level table in the shape `valid_recordings` returns.

    `meets_criterion` and `header_only_valid` are genuine booleans, as they are
    coming out of `summarise_participant`.
    """
    return pd.DataFrame(
        {
            "PAXFTIME": ["16:30:00", "16:30:00", " 8:57:00", "12:30:00"],
            "PAXLDAY": ["9", "6", "9", "5"],
            "n_days_recorded": [9, 6, 9, 5],
            "n_candidate_days": [7, 4, 7, 3],
            "n_valid_days": [7, 4, 2, 3],
            "minutes_retained": [10080, 5760, 2880, 4320],
            "meets_criterion": [True, True, False, False],
            "header_only_valid": [True, False, True, False],
        },
        index=pd.Index([73557.0, 73558.0, 73559.0, 73560.0], name="SEQN"),
    )


def test_validity_table_round_trips_through_csv(tmp_path, monkeypatch):
    """
    save_validity -> load_validity -> valid_seqns must preserve exactly which
    participants are admitted.

    This is pinned because the failure mode is silent and catastrophic rather
    than noisy. `meets_criterion` is written as the text "True"/"False", and
    `valid_seqns` calls `.astype(bool)` on it. pandas infers a bool dtype on
    read, so it works — but on an object column of those strings `.astype(bool)`
    returns True for *every* row, because any non-empty string is truthy. The
    whole cohort would be admitted, the file would look right, and nothing
    would raise.

    Note `PAXLDAY` is written as a string and read back as int64, so the
    assertions below are on the admitted SEQN set rather than on that column.
    """
    monkeypatch.setenv("ALE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ALE_PROFILE", raising=False)

    table = a_validity_table()
    expected = set(wear.valid_seqns(table))

    path = wear.save_validity(table, "X")
    assert path.exists()

    reloaded = wear.load_validity("X")

    assert reloaded["meets_criterion"].dtype == bool
    assert set(wear.valid_seqns(reloaded)) == expected == {73557.0, 73558.0}
    assert reloaded.index.name == "SEQN"


def test_valid_seqns_would_catch_a_string_boolean_column():
    """
    Guards the reasoning in the test above rather than the code: it documents
    that a text column really does admit everyone, so the round-trip assertion
    is known to be load-bearing and not decoration.
    """
    table = a_validity_table()
    as_text = table.assign(
        meets_criterion=table["meets_criterion"].map({True: "True", False: "False"})
    )

    assert len(wear.valid_seqns(table)) == 2
    assert len(wear.valid_seqns(as_text)) == 4      # every row, silently


def test_a_missing_validity_table_says_how_to_build_it(tmp_path, monkeypatch):
    """
    The error names the script and the two thresholds, because the thresholds
    are the researcher's decision and this file is the record of which were
    used. Falling back to a rule of its own would hide that.
    """
    monkeypatch.setenv("ALE_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ALE_PROFILE", raising=False)

    with pytest.raises(FileNotFoundError, match="build_validity"):
        wear.load_validity("X")
