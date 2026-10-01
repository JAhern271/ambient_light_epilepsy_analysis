"""
Tests for the circadian light metrics.

These functions are the ones where a wrong window, denominator or index
produces plausible numbers rather than an error, so most tests here check a
value that can be derived by hand rather than a value the code happens to
produce today.
"""

import numpy as np
import pandas as pd
import pytest

from ambient_light_epilepsy import lux_metrics as lm
from ambient_light_epilepsy import params, wear

from conftest import make_paxmin, make_recording, set_minutes, square_wave


# ---------------------------------------------------------------------------
# Day and night windows
# ---------------------------------------------------------------------------

LIGHT = params.section("light")
DAY_START, DAY_END = LIGHT["day_window"]
NIGHT_START, NIGHT_END = LIGHT["night_window"]


def test_light_windows_are_the_specified_ones():
    """
    Pins analysis_params.toml [light] to methods.md 6.1: day 07:00-19:00,
    night 23:00-06:00. Changing either changes every light metric, so the
    change should have to pass through this test, and through a log entry.
    """
    assert LIGHT["day_window"] == [7, 19]
    assert LIGHT["night_window"] == [23, 6]


def test_daytime_window_is_07_to_19():
    """Daytime is hours 07:00-18:59 inclusive; 19:00 is excluded."""
    # One value per hour, equal to the hour number
    values = np.arange(24, dtype=float)
    df = make_recording(values, epoch_minutes=60)

    result = lm.compute_mean_daytime_lux(df, day_start=DAY_START, day_end=DAY_END)

    # mean of 7, 8, ..., 18
    assert result == pytest.approx(12.5)


def test_nighttime_window_is_23_to_06_and_wraps_midnight():
    """Night is 23:00-05:59, so it must cross midnight."""
    values = np.arange(24, dtype=float)
    df = make_recording(values, epoch_minutes=60)

    result = lm.compute_mean_nighttime_lux(
        df, night_start=NIGHT_START, night_end=NIGHT_END
    )

    # mean of 23, 0, 1, 2, 3, 4, 5 = 38 / 7
    assert result == pytest.approx(38 / 7)


def test_dusk_and_dawn_hours_belong_to_neither_window():
    """
    06:00-06:59 and 19:00-22:59 are excluded from both windows (methods.md
    6.1). Light only in those hours must leave both means at zero.
    """
    values = np.zeros(24)
    values[[6, 19, 20, 21, 22]] = 1000.0
    df = make_recording(values, epoch_minutes=60)

    assert lm.compute_mean_daytime_lux(
        df, day_start=DAY_START, day_end=DAY_END
    ) == 0.0
    assert lm.compute_mean_nighttime_lux(
        df, night_start=NIGHT_START, night_end=NIGHT_END
    ) == 0.0


def test_masked_minutes_are_skipped_not_counted_as_dark():
    """
    A masked minute arrives as NaN (wear.mask_minutes). It must drop out of
    the mean rather than count as 0 lux: with 23:00 masked, the night mean is
    the mean of hours 0-5, which is 2.5.
    """
    values = np.arange(24, dtype=float)
    values[23] = np.nan
    df = make_recording(values, epoch_minutes=60)

    assert lm.compute_mean_nighttime_lux(
        df, night_start=NIGHT_START, night_end=NIGHT_END
    ) == pytest.approx(2.5)


def test_windows_have_no_default():
    """The window must come from the caller, i.e. from [light]."""
    df = make_recording(np.arange(24, dtype=float), epoch_minutes=60)

    with pytest.raises(TypeError):
        lm.compute_mean_daytime_lux(df)
    with pytest.raises(TypeError):
        lm.compute_mean_nighttime_lux(df)


@pytest.mark.parametrize("start, end", [(5, 5), (24, 6), (23, -1)])
def test_impossible_windows_raise(start, end):
    """Equal ends, or an hour outside 0-23, cannot be what was meant."""
    df = make_recording(np.arange(24, dtype=float), epoch_minutes=60)

    with pytest.raises(ValueError):
        lm.compute_mean_nighttime_lux(df, night_start=start, night_end=end)


def test_nighttime_window_on_prepared_paxmin_minutes():
    """
    Second route to the night window, through the frame the analysis really
    uses: raw PAXMIN records passed through wear.prepare_minutes.

    100 lux throughout, except 1000 lux from 20:00 to 22:59 on every calendar
    day. Those hours are outside 23:00-06:00, so the night mean is exactly
    100. The superseded 20:00-05:00 window would have taken them in and
    given 400:

        20-22 h: 8 evenings x 180 min = 1,440 min at 1000 lux
        23 h:    8 x 60  =   480 min at 100
        00-04 h: 8 x 300 = 2,400 min at 100
        (1,440 x 1000 + 2,880 x 100) / 4,320 = 400

    The recording starts at 16:30, so 20:00 on calendar day k (from 0) is
    minute 210 + 1440 k; the last calendar day ends at 16:39, giving 8
    evenings.
    """
    minutes = make_paxmin(first_time="16:30:00", lux=100.0)
    for k in range(8):
        minutes = set_minutes(minutes, 210 + 1440 * k, 180, PAXLXMM=1000.0)

    prepared = wear.prepare_minutes(minutes, "16:30:00")

    assert lm.compute_mean_nighttime_lux(
        prepared, night_start=NIGHT_START, night_end=NIGHT_END
    ) == pytest.approx(100.0)
    # The superseded window, to show the fixture distinguishes the two
    assert lm.compute_mean_nighttime_lux(
        prepared, night_start=20, night_end=5
    ) == pytest.approx(400.0)


def test_daytime_returns_nan_when_no_samples_in_window():
    """A recording with no daytime samples gives NaN, not an error or zero."""
    df = make_recording(np.ones(4), start="2013-06-01 00:00:00", epoch_minutes=60)

    assert np.isnan(
        lm.compute_mean_daytime_lux(df, day_start=DAY_START, day_end=DAY_END)
    )


# ---------------------------------------------------------------------------
# Time above threshold
# ---------------------------------------------------------------------------

def test_time_above_threshold_of_square_wave():
    """
    A 12 h/day square wave at 1000 lux spends half the recording above a
    500 lux threshold, so 720 minutes per day.
    """
    df = square_wave(days=7, high=1000.0, low=0.0, on_hour=6, off_hour=18)

    result = lm.time_above_threshold_normalized(df, threshold=500)

    assert result == pytest.approx(720.0)


def test_time_above_threshold_is_strictly_greater():
    """A signal exactly at the threshold does not count as above it."""
    df = make_recording(np.full(288, 1000.0))

    assert lm.time_above_threshold_normalized(df, threshold=1000) == 0.0


def test_time_above_threshold_is_a_daily_rate_not_a_total():
    """
    The value is minutes per day, so doubling the recording length without
    changing the pattern must not change the result.
    """
    short = square_wave(days=3)
    long = square_wave(days=6)

    assert lm.time_above_threshold_normalized(short, 500) == pytest.approx(
        lm.time_above_threshold_normalized(long, 500)
    )


# ---------------------------------------------------------------------------
# Daytime minutes above 100 / 250 / 1000 lux, per valid day (methods.md 6.2)
# ---------------------------------------------------------------------------

def analytic_days(day_specs):
    """
    Build a prepared-minutes frame and a days table directly, so each test
    can say exactly which clock minutes of which analytic day carry what.

    `day_specs` is a list, one entry per noon-to-noon day, of
    (is_valid, [(clock_start, clock_end, lux), ...]) where clock times are
    minutes past midnight and lux=None marks the minutes as masked. Anything
    not mentioned is 0 lux and retained.
    """
    frames = []
    for d, (_, segments) in enumerate(day_specs):
        day = pd.Timestamp("2013-06-02") + pd.Timedelta(days=d)
        timestamps = day + pd.Timedelta(hours=12) + pd.to_timedelta(np.arange(1440), "m")
        # Clock minute of each row: noon is 720, the following 11:59 is 719
        clock = (720 + np.arange(1440)) % 1440

        lux = np.zeros(1440)
        retained = np.ones(1440, dtype=bool)
        for clock_start, clock_end, value in segments:
            rows = (clock >= clock_start) & (clock < clock_end)
            if value is None:
                retained[rows] = False
            else:
                lux[rows] = value

        frames.append(pd.DataFrame({
            "timestamp": timestamps,
            "mean_lux": np.where(retained, lux, np.nan),
            "retained": retained,
            "analytic_day": day,
        }))

    prepared = pd.concat(frames, ignore_index=True)
    days = pd.DataFrame(
        {"is_valid": [valid for valid, _ in day_specs]},
        index=pd.Index(
            [pd.Timestamp("2013-06-02") + pd.Timedelta(days=d)
             for d in range(len(day_specs))],
            name="analytic_day",
        ),
    )
    return prepared, days


def hhmm(hour, minute=0):
    return hour * 60 + minute


# Fixture A. Two valid days; the day window is 07:00-18:59, 720 minutes.
#   day 1: 07:00-11:59 at 1500 (300 min), 12:00-13:59 at 500 (120),
#          14:00-18:59 at 50 (300); plus 02:00-02:59 at 1500, outside the
#          window, which must not count
#   day 2: 07:00-07:59 at 2500 (60), 08:00-11:59 at 200 (240),
#          12:00-18:59 at 50 (420)
#   day 3: NOT valid, 2500 lux all day, which must not count
#
#   > 100:  (420 + 300) / 2 = 360
#   > 250:  (420 +  60) / 2 = 240
#   > 1000: (300 +  60) / 2 = 180
FIXTURE_A = [
    (True, [(hhmm(7), hhmm(12), 1500.0), (hhmm(12), hhmm(14), 500.0),
            (hhmm(14), hhmm(19), 50.0), (hhmm(2), hhmm(3), 1500.0)]),
    (True, [(hhmm(7), hhmm(8), 2500.0), (hhmm(8), hhmm(12), 200.0),
            (hhmm(12), hhmm(19), 50.0)]),
    (False, [(0, 1440, 2500.0)]),
]

THRESHOLDS = LIGHT["day_thresholds"]


def test_day_thresholds_are_the_specified_ones():
    """Pins [light] to methods.md 6.2: 100, 250 and 1,000 lux."""
    assert THRESHOLDS == [100, 250, 1000]
    assert LIGHT["primary_day_threshold"] == 1000


def test_masked_minute_rule_is_the_decided_one():
    """
    Pins light.masked_minutes to "raw", decided 2026-10-01 (methods.md 6.2),
    and checks it is a value the function accepts.
    """
    assert LIGHT["masked_minutes"] == "raw"
    assert LIGHT["masked_minutes"] in lm.MASKED_MINUTE_RULES


@pytest.mark.parametrize("rule", lm.MASKED_MINUTE_RULES)
def test_minutes_above_thresholds_fixture_a(rule):
    """
    Nothing is masked in fixture A, so both masked-minute rules must agree
    on the hand-derived answer.
    """
    prepared, days = analytic_days(FIXTURE_A)

    result = lm.minutes_above_thresholds(
        prepared, days, thresholds=THRESHOLDS,
        day_window=LIGHT["day_window"], masked_minutes=rule,
    )

    assert result == {100: pytest.approx(360.0),
                      250: pytest.approx(240.0),
                      1000: pytest.approx(180.0)}


# Fixture B. One valid day: 07:00-11:59 at 1500 (300 min), 12:00-13:59
# masked (120 min), 14:00-18:59 at 50 (300). 600 of 720 window minutes are
# retained.
#   raw:     300
#   rescale: 300 * 720 / 600 = 360
FIXTURE_B = [
    (True, [(hhmm(7), hhmm(12), 1500.0), (hhmm(12), hhmm(14), None),
            (hhmm(14), hhmm(19), 50.0)]),
]


@pytest.mark.parametrize("rule, expected", [("raw", 300.0), ("rescale", 360.0)])
def test_masked_minutes_rule(rule, expected):
    prepared, days = analytic_days(FIXTURE_B)

    result = lm.minutes_above_thresholds(
        prepared, days, thresholds=[1000],
        day_window=LIGHT["day_window"], masked_minutes=rule,
    )

    assert result[1000] == pytest.approx(expected)


def test_minutes_exactly_at_a_threshold_are_not_above_it():
    """Strictly greater, as for the superseded function."""
    prepared, days = analytic_days([(True, [(hhmm(7), hhmm(19), 1000.0)])])

    result = lm.minutes_above_thresholds(
        prepared, days, thresholds=[1000],
        day_window=LIGHT["day_window"], masked_minutes="raw",
    )

    assert result[1000] == 0.0


def test_no_valid_days_gives_nan():
    prepared, days = analytic_days([(False, [(0, 1440, 2500.0)])])

    result = lm.minutes_above_thresholds(
        prepared, days, thresholds=THRESHOLDS,
        day_window=LIGHT["day_window"], masked_minutes="raw",
    )

    assert all(np.isnan(value) for value in result.values())


def test_masked_minutes_rule_has_no_default():
    """The choice is open (implementation-status.md), so it must be stated."""
    prepared, days = analytic_days(FIXTURE_B)

    with pytest.raises(TypeError):
        lm.minutes_above_thresholds(
            prepared, days, thresholds=[1000], day_window=LIGHT["day_window"]
        )
    with pytest.raises(ValueError):
        lm.minutes_above_thresholds(
            prepared, days, thresholds=[1000],
            day_window=LIGHT["day_window"], masked_minutes="impute",
        )


def test_minutes_above_thresholds_through_the_wear_chain():
    """
    Second route, from raw PAXMIN records through wear.prepare_minutes and
    wear.summarise_days.

    100 lux throughout, except 1500 lux from 07:00 to 08:59 every morning.
    Every candidate noon-to-noon day contains exactly one such morning and is
    fully worn, so all seven are valid and each has 120 minutes above every
    threshold -- including above 100, since the 100 lux minutes are not
    *above* 100.

    The recording starts at 16:30, so 07:00 on calendar day k+1 (k from 0) is
    minute 870 + 1440 k.
    """
    minutes = make_paxmin(first_time="16:30:00", lux=100.0)
    for k in range(8):
        minutes = set_minutes(minutes, 870 + 1440 * k, 120, PAXLXMM=1500.0)

    prepared = wear.prepare_minutes(minutes, "16:30:00")
    days = wear.summarise_days(prepared, min_wear_hours=20)
    assert int(days["is_valid"].sum()) == 7

    result = lm.minutes_above_thresholds(
        prepared, days, thresholds=THRESHOLDS,
        day_window=LIGHT["day_window"], masked_minutes="raw",
    )

    assert result == {100: pytest.approx(120.0),
                      250: pytest.approx(120.0),
                      1000: pytest.approx(120.0)}


# ---------------------------------------------------------------------------
# M10, L5 and relative amplitude
# ---------------------------------------------------------------------------

# Window coverage rule, rest_activity.min_window_coverage (methods.md 6.5)
COVERAGE = params.section("rest_activity")["min_window_coverage"]


def test_window_coverage_is_the_decided_one():
    """
    Pins rest_activity.min_window_coverage to 20/24, decided 2026-10-01. TOML
    holds it as a decimal, so this also checks that the decimal is exactly the
    double 20 / 24 evaluates to, not a nearby one.
    """
    assert COVERAGE == 20 / 24

def test_square_wave_m10_l5_and_ra():
    """
    With 12 h at 1000 lux and 12 h at 0, a 10 h window fits entirely inside
    the light period and a 5 h window entirely inside the dark period.
    """
    df = square_wave(days=7, high=1000.0, low=0.0, on_hour=6, off_hour=18)

    m10, l5, ra, *_ = lm.relative_amplitude(df, min_window_coverage=COVERAGE)

    assert m10 == pytest.approx(1000.0)
    assert l5 == pytest.approx(0.0)
    assert ra == pytest.approx(1.0)


def test_constant_recording_has_zero_relative_amplitude():
    """No day-night difference means M10 == L5 and RA == 0."""
    df = make_recording(np.full(7 * 288, 100.0))

    m10, l5, ra, *_ = lm.relative_amplitude(df, min_window_coverage=COVERAGE)

    assert m10 == pytest.approx(l5)
    assert ra == pytest.approx(0.0)


# The clock-time outputs of relative_amplitude are the START of each window,
# in minutes past midnight (methods.md 6.5, 8.2). The value is circular: 1439
# and 0 are one minute apart. Positions in the returned 7-tuple:
M10_START = 3
L5_START = 5


def test_m10_start_is_five_hours_before_the_peak(sinusoid_recording):
    """
    A sinusoid peaking at midday puts the brightest 10 h window at 07:00-17:00,
    so it starts 420 minutes after midnight, to within one epoch.
    """
    m10_start = lm.relative_amplitude(sinusoid_recording, min_window_coverage=COVERAGE)[M10_START]

    assert m10_start == pytest.approx(420.0, abs=10.0)


def test_l5_start_is_before_midnight_for_a_trough_at_midnight(sinusoid_recording):
    """
    The same sinusoid troughs at midnight, so the darkest 5 h window is
    21:30-02:30 and starts at 1290 minutes: the late end of the scale, for a
    window centred on its early end. That is why the value is circular.
    """
    l5_start = lm.relative_amplitude(sinusoid_recording, min_window_coverage=COVERAGE)[L5_START]

    assert l5_start == pytest.approx(1290.0, abs=10.0)


def test_m10_start_tracks_a_shifted_peak(sinusoid_recording):
    """Shifting the whole profile 3 h later must shift the M10 start with it."""
    values = sinusoid_recording["mean_lux"].to_numpy()
    shifted = make_recording(np.roll(values, 3 * 12))  # 3 h at 5 min epochs

    baseline_start = lm.relative_amplitude(sinusoid_recording, min_window_coverage=COVERAGE)[M10_START]
    shifted_start = lm.relative_amplitude(shifted, min_window_coverage=COVERAGE)[M10_START]

    assert shifted_start - baseline_start == pytest.approx(180.0, abs=10.0)


def test_m10_start_on_a_plateau_picks_the_earliest_window():
    """
    Documents a tie-breaking behaviour rather than asserting correctness.

    A square wave with 12 h of light contains many 10 h windows of identical
    mean, so the M10 position is genuinely ambiguous. idxmax returns the first,
    which starts the window at light onset, 06:00, rather than centring it in
    the light period (a 07:00 start).

    Real recordings rarely tie exactly, so this mostly matters when
    interpreting synthetic or heavily rounded data -- and lux at night, which
    is often exactly 0 (see the midnight tie test below).
    """
    df = square_wave(days=7, on_hour=6, off_hour=18)

    m10_start = lm.relative_amplitude(df, min_window_coverage=COVERAGE)[M10_START]

    assert m10_start == pytest.approx(360.0)  # 06:00


# ---------------------------------------------------------------------------
# M10 and L5 start times on minute-level data
# ---------------------------------------------------------------------------
#
# Shaped like PAXMIN after wear.prepare_minutes: one row per minute, NaN
# where a minute is masked. Each fixture repeats one day's profile, written
# as (start minute, end minute, lux) blocks on a base of 100 lux.

def minute_profile(blocks, base=100.0):
    """
    One day of minute values. Each block is (start, end, lux) in minutes past
    midnight, end exclusive; a block whose start is later than its end wraps
    midnight.
    """
    day = np.full(1440, base)
    for start, end, lux in blocks:
        if start < end:
            day[start:end] = lux
        else:
            day[start:] = lux
            day[:end] = lux
    return day


# Clock fixture A: 1000 lux 08:00-18:00, 0 lux 22:30-03:30, 100 lux otherwise.
# Exactly one 10 h window is all 1000 and exactly one 5 h window is all 0, so
# by hand: M10 = 1000 starting 08:00 (480), L5 = 0 starting 22:30 (1350),
# RA = 1. The L5 window straddles midnight.
CLOCK_FIXTURE_A = minute_profile([
    (hhmm(8), hhmm(18), 1000.0),
    (hhmm(22, 30), hhmm(3, 30), 0.0),
])


def clock_fixture_a_recording(days=7):
    return make_recording(np.tile(CLOCK_FIXTURE_A, days), epoch_minutes=1)


def test_l5_straddling_midnight_starts_before_midnight():
    """Clock fixture A, every output checked against the hand-derived values."""
    m10, l5, ra, m10_start, m10_time, l5_start, l5_time = lm.relative_amplitude(
        clock_fixture_a_recording(), min_window_coverage=COVERAGE
    )

    assert m10 == pytest.approx(1000.0)
    assert l5 == pytest.approx(0.0)
    assert ra == pytest.approx(1.0)
    assert m10_start == 480.0
    assert l5_start == 1350.0
    assert str(m10_time) == "08:00:00"
    assert str(l5_time) == "22:30:00"


def test_l5_tie_across_midnight_resolves_to_the_first_window_after_midnight():
    """
    Clock fixture B. Documents current behaviour rather than asserting correctness;
    see the tie-break item in doc/implementation-status.md.

    0 lux 22:00-06:00 holds every 5 h window starting 22:00-01:00, all tied at
    L5 = 0. The profile is scanned from 00:00, so the tie goes to the window
    starting at 00:00 -- not to the start of the dark period (22:00), nor to
    the window centred in it (23:30).
    """
    day = minute_profile([
        (hhmm(8), hhmm(18), 1000.0),
        (hhmm(22), hhmm(6), 0.0),
    ])
    df = make_recording(np.tile(day, 7), epoch_minutes=1)

    result = lm.relative_amplitude(df, min_window_coverage=COVERAGE)

    assert result[1] == pytest.approx(0.0)
    assert result[L5_START] == 0.0  # 00:00


def test_a_minute_masked_on_one_day_does_not_move_the_windows():
    """
    Clock fixture C1. NaN on day 3 from 00:00 to 01:00: the profile at those minutes
    is the mean of the other six days, still 0, so nothing changes.
    """
    values = np.tile(CLOCK_FIXTURE_A, 7)
    day3 = 2 * 1440
    values[day3:day3 + 60] = np.nan
    df = make_recording(values, epoch_minutes=1)

    result = lm.relative_amplitude(df, min_window_coverage=COVERAGE)

    assert result[0] == pytest.approx(1000.0)
    assert result[1] == pytest.approx(0.0)
    assert result[M10_START] == 480.0
    assert result[L5_START] == 1350.0


def clock_fixture_c2_recording():
    """Clock fixture A with 12:00-12:59 masked (NaN) on every one of 7 days."""
    values = np.tile(CLOCK_FIXTURE_A, 7)
    for day in range(7):
        noon = day * 1440 + hhmm(12)
        values[noon:noon + 60] = np.nan
    return make_recording(values, epoch_minutes=1)


def clock_fixture_keep_only(start, end):
    """Clock fixture A with every minute outside [start, end) NaN on every day."""
    day = CLOCK_FIXTURE_A.copy()
    keep = np.zeros(1440, dtype=bool)
    if start < end:
        keep[start:end] = True
    else:
        keep[start:] = True
        keep[:end] = True
    day[~keep] = np.nan
    return make_recording(np.tile(day, 7), epoch_minutes=1)


@pytest.mark.parametrize("window, coverage, expected", [
    (600, 20 / 24, 500),   # M10 at 1-minute epochs
    (300, 20 / 24, 250),   # L5 at 1-minute epochs
    (120, 20 / 24, 100),   # M10 at 5-minute epochs
    (60, 20 / 24, 50),     # L5 at 5-minute epochs
    (600, 1.0, 600),       # every bin required
    (600, 0.9, 540),
])
def test_min_samples_for_coverage(window, coverage, expected):
    """
    By hand: 20/24 of 600 is 500 exactly. ceil(0.8333... * 600) would give
    501, because the product is 500.00000000000006 in floating point.
    """
    assert lm.min_samples_for_coverage(window, coverage) == expected


def test_a_minute_masked_on_every_day_is_averaged_over_the_rest_of_the_window():
    """
    Clock fixture C2 under the decided rule (20/24, methods.md 6.5).

    The 08:00-18:00 window has 540 of its 600 minutes, coverage 0.90 >= 0.833,
    so it is eligible, and its mean over those 540 minutes is 1000. No window
    can beat 1000, and no other window averages exactly 1000 (07:00-17:00 is
    (60*100 + 480*1000)/540 = 900; 08:30-18:30 is (510*1000 + 30*100)/540
    = 950), so M10 = 1000 starting 08:00, as without the mask. L5 does not
    touch noon: 0 starting 22:30, RA = 1.
    """
    m10, l5, ra, m10_start, _, l5_start, _ = lm.relative_amplitude(
        clock_fixture_c2_recording(), min_window_coverage=COVERAGE
    )

    assert m10 == pytest.approx(1000.0)
    assert m10_start == 480.0   # 08:00
    assert l5 == pytest.approx(0.0)
    assert l5_start == 1350.0   # 22:30
    assert ra == pytest.approx(1.0)


def test_full_coverage_reproduces_the_previous_behaviour():
    """
    Clock fixture C2 with coverage 1.0: any window containing a NaN bin is
    skipped, which is what the function did before 2026-10-01. The best
    window clear of noon is 13:00-23:00, by hand:
        5 h at 1000 + 4.5 h at 100 + 0.5 h at 0 = 5450 lux-hours / 10 = 545
    against 445 for the best window ending by 12:00 (02:00-12:00).
    """
    result = lm.relative_amplitude(
        clock_fixture_c2_recording(), min_window_coverage=1.0
    )

    assert result[0] == pytest.approx(545.0)
    assert result[M10_START] == 780.0  # 13:00
    assert result[L5_START] == 1350.0


@pytest.mark.parametrize("coverage, m10, m10_start", [
    # 540 of 600 bins is exactly 0.90, so the 08:00 window is eligible
    (540 / 600, 1000.0, 480.0),
    # One bin more is needed: 08:00-18:00 is out. A window may now hold at
    # most 59 NaN minutes. Best is 12:01-22:01: 59 NaN, 300 min at 1000,
    # 241 min at 100, so (300000 + 24100) / 541. (02:59-12:59, the best
    # window ending inside the gap, is (27000 + 240000) / 541 = 493.5.)
    (541 / 600, 324100.0 / 541, 721.0),
])
def test_coverage_threshold_is_inclusive(coverage, m10, m10_start):
    """Clock fixture C2 either side of the 08:00 window's own coverage."""
    result = lm.relative_amplitude(
        clock_fixture_c2_recording(), min_window_coverage=coverage
    )

    assert result[0] == pytest.approx(m10)
    assert result[M10_START] == m10_start


def test_l5_without_an_eligible_m10_window():
    """
    Clock fixture A keeping only 22:00-04:00 (6 h). A 10 h window holds at
    most 360 non-NaN minutes, below 500, so M10 is NaN and RA with it. The
    5 h window 22:30-03:30 is complete and all 0, so L5 = 0 starting 22:30.
    Every other window with 250+ minutes of data includes some 100 lux.
    """
    m10, l5, ra, m10_start, m10_time, l5_start, l5_time = lm.relative_amplitude(
        clock_fixture_keep_only(hhmm(22), hhmm(4)), min_window_coverage=COVERAGE
    )

    assert np.isnan(m10) and np.isnan(m10_start) and m10_time is None
    assert np.isnan(ra)
    assert l5 == pytest.approx(0.0)
    assert l5_start == 1350.0


def test_no_eligible_window_gives_nan():
    """
    Clock fixture A keeping only 08:00-12:00 (240 min): fewer than the 250 a
    5 h window needs, so neither M10 nor L5 has an eligible window.
    """
    m10, l5, ra, m10_start, m10_time, l5_start, l5_time = lm.relative_amplitude(
        clock_fixture_keep_only(hhmm(8), hhmm(12)), min_window_coverage=COVERAGE
    )

    assert np.isnan(m10) and np.isnan(l5) and np.isnan(ra)
    assert np.isnan(m10_start) and np.isnan(l5_start)
    assert m10_time is None and l5_time is None


def test_a_partial_window_at_the_start_of_the_profile_is_never_eligible():
    """
    No NaN anywhere: 0 lux 00:00-04:10 (250 min), 1000 lux otherwise.

    Every full 5 h window holds at most those 250 zeros plus 50 minutes at
    1000, so by hand L5 = 50 * 1000 / 300 = 166.67. Windows starting 23:10
    to 00:00 all tie at that value, and the first one found is the window
    starting 00:00. The 250-minute fragment 00:00-04:10 at the very start of
    the doubled profile meets 20/24 coverage by count, and would give L5 = 0
    if it were allowed to count as a window.
    """
    day = minute_profile([(0, hhmm(4, 10), 0.0)], base=1000.0)
    df = make_recording(np.tile(day, 7), epoch_minutes=1)

    m10, l5, ra, m10_start, _, l5_start, _ = lm.relative_amplitude(
        df, min_window_coverage=COVERAGE
    )

    assert l5 == pytest.approx(50 * 1000 / 300)
    assert l5_start == 0.0
    assert m10 == pytest.approx(1000.0)


def test_window_coverage_has_no_default():
    with pytest.raises(TypeError):
        lm.relative_amplitude(clock_fixture_a_recording())


@pytest.mark.parametrize("coverage", [0, -0.5, 1.01])
def test_impossible_window_coverage_raises(coverage):
    with pytest.raises(ValueError):
        lm.relative_amplitude(clock_fixture_a_recording(),
                              min_window_coverage=coverage)


def test_sampling_interval_survives_a_missing_second_row():
    """
    Clock fixture D. Clock fixture A with its second row deleted. The interval between the
    first two rows is then 2 minutes, but the recording is minute-level, so
    the answers must be clock fixture A's.
    """
    df = clock_fixture_a_recording().drop(index=1).reset_index(drop=True)

    assert lm.get_sampling_interval_minutes(df) == 1.0

    result = lm.relative_amplitude(df, min_window_coverage=COVERAGE)

    assert result[0] == pytest.approx(1000.0)
    assert result[M10_START] == 480.0
    assert result[L5_START] == 1350.0


def test_a_clock_minute_absent_on_every_day_does_not_shift_later_times():
    """
    Clock fixture D2. Clock fixture A with the 05:00 row removed from every day, so the
    24 h profile has no entry at all for that minute. Neither window contains
    05:00, so the answers must still be clock fixture A's. A profile indexed by
    position would put every clock time after 05:00 one minute early.
    """
    df = clock_fixture_a_recording()
    at_0500 = (df["timestamp"].dt.hour == 5) & (df["timestamp"].dt.minute == 0)
    df = df[~at_0500].reset_index(drop=True)

    result = lm.relative_amplitude(df, min_window_coverage=COVERAGE)

    assert result[M10_START] == 480.0
    assert result[L5_START] == 1350.0


def test_sampling_interval_that_does_not_divide_an_hour_is_rejected():
    """A 7-minute epoch has no whole number of samples per hour."""
    df = make_recording(np.zeros(500), epoch_minutes=7)

    with pytest.raises(ValueError):
        lm.get_sampling_interval_minutes(df)


def test_clock_start_times_are_circular_not_linear():
    """
    Clock fixture E. Why the start columns must be read as circular (methods.md
    8.2), shown on two participants whose L5 starts at 23:00 and at 01:00.

    They come out as 1380 and 60 minutes past midnight. Their arithmetic mean
    is 720 -- noon, the opposite of the truth. The circular mean, the direction
    of the mean of their unit vectors, is midnight. The analysis does circular
    statistics in R; this test only pins what the column means.
    """
    starts = []
    for l5_from in (hhmm(23), hhmm(1)):
        day = minute_profile([(l5_from, (l5_from + 300) % 1440, 0.0)])
        df = make_recording(np.tile(day, 7), epoch_minutes=1)
        starts.append(lm.relative_amplitude(df, min_window_coverage=COVERAGE)[L5_START])

    assert starts == [1380.0, 60.0]

    assert np.mean(starts) == 720.0  # the error methods.md 8.2 describes

    angles = np.asarray(starts) / 1440 * 2 * np.pi
    circular_mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    circular_mean_minutes = (circular_mean / (2 * np.pi) * 1440) % 1440

    # Midnight, allowing for floating point on either side of the wrap
    distance_from_midnight = min(circular_mean_minutes,
                                 1440 - circular_mean_minutes)
    assert distance_from_midnight == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Interdaily stability and intradaily variability
# ---------------------------------------------------------------------------

def test_identical_days_give_interdaily_stability_of_one():
    """IS is 1 when every day repeats exactly."""
    df = square_wave(days=7)

    assert lm.interdaily_stability(df) == pytest.approx(1.0)


def test_interdaily_stability_falls_when_days_differ():
    """Randomising each day's timing must reduce IS below the regular case."""
    regular = square_wave(days=7)

    rng = np.random.default_rng(seed=1)
    per_day = 288
    one_day = square_wave(days=1)["mean_lux"].to_numpy()
    shifted = np.concatenate(
        [np.roll(one_day, rng.integers(-per_day // 4, per_day // 4)) for _ in range(7)]
    )
    irregular = make_recording(shifted)

    assert lm.interdaily_stability(irregular) < lm.interdaily_stability(regular)


def test_constant_recording_gives_undefined_intradaily_variability():
    """
    A perfectly constant signal makes IV 0/0, so the result is NaN rather than
    0. Mathematically correct, but worth pinning: a participant whose sensor
    failed and returned a constant value yields NaN, not a suspicious zero.
    """
    df = make_recording(np.full(7 * 288, 100.0))

    assert np.isnan(lm.intradaily_variability(df))


def test_square_wave_has_low_intradaily_variability():
    """
    A square wave changes only twice a day, so IV is near zero. Computed by
    hand: 14 transitions of 1000 lux over 2016 samples, against a variance of
    250000, gives 14e6 / 2015 / 250000.
    """
    df = square_wave(days=7, high=1000.0, low=0.0)

    expected = (14 * 1000.0 ** 2 / (7 * 288 - 1)) / 250000.0

    assert lm.intradaily_variability(df) == pytest.approx(expected)


def test_alternating_signal_hits_the_intradaily_variability_maximum():
    """
    A signal flipping between two extremes every epoch is maximally
    fragmented. Each squared difference is 4x the variance, so IV is exactly 4,
    the upper bound of this formula.
    """
    values = np.tile([0.0, 1000.0], 7 * 144)

    assert lm.intradaily_variability(make_recording(values)) == pytest.approx(4.0)


def test_white_noise_gives_intradaily_variability_near_two():
    """
    For uncorrelated noise the expected squared successive difference is twice
    the variance, so IV tends to 2. This is the reference point against which
    real values are usually read.
    """
    rng = np.random.default_rng(seed=0)
    df = make_recording(rng.normal(500, 100, 7 * 288))

    assert lm.intradaily_variability(df) == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Sampling interval detection
# ---------------------------------------------------------------------------

def test_sampling_interval_detected_from_timestamps():
    assert lm.get_sampling_interval_minutes(make_recording(np.zeros(10))) == 5.0
    assert lm.get_sampling_interval_minutes(
        make_recording(np.zeros(10), epoch_minutes=60)
    ) == 60.0


def test_sampling_interval_ignores_a_gap_at_the_start():
    """
    A recording that starts with a gap still reports its true sampling rate.

    This test used to pin the opposite, as a known limitation: the interval
    was read from the first gap only, so this recording reported 65 minutes
    and every metric that scales by the epoch was wrong. It is now the most
    common gap (fixed 2026-10-01).
    """
    timestamps = list(pd.date_range("2013-06-01", periods=10, freq="5min"))
    # A one hour gap between the first and second sample
    timestamps[1:] = [t + pd.Timedelta(hours=1) for t in timestamps[1:]]
    df = pd.DataFrame({"timestamp": timestamps, "mean_lux": np.zeros(10)})

    assert lm.get_sampling_interval_minutes(df) == 5.0  # not 65.0


# ---------------------------------------------------------------------------
# Resolution dependence
# ---------------------------------------------------------------------------

def test_interdaily_stability_is_independent_of_input_resolution(sinusoid_recording):
    """
    The same underlying signal must score the same IS whether it arrives at
    5 min or hourly resolution, because IS resamples before computing.

    This is the property the earlier implementation lacked, and it is what
    makes the 5 minute and 1 Hz analyses comparable with each other.
    """
    five_min = sinusoid_recording
    hourly = (
        five_min.set_index("timestamp")["mean_lux"].resample("1h").mean().reset_index()
    )

    assert lm.interdaily_stability(five_min) == pytest.approx(
        lm.interdaily_stability(hourly), rel=1e-9
    )


def test_interdaily_stability_is_bounded_by_zero_and_one(noisy_recording):
    """IS is a variance ratio, so it cannot leave [0, 1]."""
    assert 0.0 <= lm.interdaily_stability(noisy_recording) <= 1.0


def test_interdaily_stability_respects_the_bin_size_argument(noisy_recording):
    """
    Computing at a finer epoch is possible and gives a lower number, because
    noise that averages out within an hourly bin survives in a 5 minute one.

    A perfectly repeating signal scores 1 at any bin size, so an irregular
    recording is needed to show the argument has an effect at all.
    """
    hourly = lm.interdaily_stability(noisy_recording, bin_size="1h")
    fine = lm.interdaily_stability(noisy_recording, bin_size="5min")

    assert fine < hourly


def test_interdaily_stability_ignores_gaps_rather_than_zero_filling(square_recording):
    """
    A recording with a missing stretch must not have the gap counted as dark.
    Dropping two hours should leave IS essentially unchanged for a signal whose
    days are otherwise identical.
    """
    with_gap = square_recording.drop(
        square_recording.index[100:124]  # two hours at 5 min epochs
    )

    assert lm.interdaily_stability(with_gap) == pytest.approx(1.0, abs=0.05)


def test_intradaily_variability_differs_between_5min_and_hourly_input(sinusoid_recording):
    """
    IV compares successive samples, so it is inherently resolution dependent.
    Recorded here so the constraint is explicit: IV from the 5 min analysis
    cannot be compared with IV from the 1 Hz analysis.
    """
    five_min = sinusoid_recording
    hourly = (
        five_min.set_index("timestamp")["mean_lux"].resample("1h").mean().reset_index()
    )

    assert lm.intradaily_variability(five_min) != pytest.approx(
        lm.intradaily_variability(hourly), rel=1e-3
    )
