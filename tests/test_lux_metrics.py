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
# M10, L5 and relative amplitude
# ---------------------------------------------------------------------------

def test_square_wave_m10_l5_and_ra():
    """
    With 12 h at 1000 lux and 12 h at 0, a 10 h window fits entirely inside
    the light period and a 5 h window entirely inside the dark period.
    """
    df = square_wave(days=7, high=1000.0, low=0.0, on_hour=6, off_hour=18)

    m10, l5, ra, *_ = lm.relative_amplitude(df)

    assert m10 == pytest.approx(1000.0)
    assert l5 == pytest.approx(0.0)
    assert ra == pytest.approx(1.0)


def test_constant_recording_has_zero_relative_amplitude():
    """No day-night difference means M10 == L5 and RA == 0."""
    df = make_recording(np.full(7 * 288, 100.0))

    m10, l5, ra, *_ = lm.relative_amplitude(df)

    assert m10 == pytest.approx(l5)
    assert ra == pytest.approx(0.0)


def test_m10_midpoint_falls_at_the_peak(sinusoid_recording):
    """
    A sinusoid peaking at midday puts the brightest 10 h window around 12:00,
    i.e. 720 minutes after midnight, to within one epoch.
    """
    *_, m10_midpoint_minutes, _, _, _ = lm.relative_amplitude(sinusoid_recording)

    assert m10_midpoint_minutes == pytest.approx(720.0, abs=10.0)


def test_l5_midpoint_falls_at_the_trough(sinusoid_recording):
    """The same sinusoid troughs at midnight, so the L5 midpoint is near 00:00."""
    *_, l5_midpoint_minutes, _ = lm.relative_amplitude(sinusoid_recording)

    assert l5_midpoint_minutes == pytest.approx(0.0, abs=10.0)


def test_m10_midpoint_tracks_a_shifted_peak(sinusoid_recording):
    """Shifting the whole profile 3 h later must shift the M10 midpoint with it."""
    values = sinusoid_recording["mean_lux"].to_numpy()
    shifted = make_recording(np.roll(values, 3 * 12))  # 3 h at 5 min epochs

    baseline_mid = lm.relative_amplitude(sinusoid_recording)[3]
    shifted_mid = lm.relative_amplitude(shifted)[3]

    assert shifted_mid - baseline_mid == pytest.approx(180.0, abs=10.0)


def test_m10_midpoint_on_a_plateau_picks_the_earliest_window():
    """
    Documents a tie-breaking behaviour rather than asserting correctness.

    A square wave with 12 h of light contains many 10 h windows of identical
    mean, so the M10 position is genuinely ambiguous. idxmax returns the first,
    which puts the window at light onset and the midpoint at 11:00 rather than
    the centre of the light period at 12:00.

    Real recordings rarely tie exactly, so this mostly matters when
    interpreting synthetic or heavily rounded data.
    """
    df = square_wave(days=7, on_hour=6, off_hour=18)

    m10_midpoint_minutes = lm.relative_amplitude(df)[3]

    assert m10_midpoint_minutes == pytest.approx(660.0)  # 11:00, not 12:00


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


def test_sampling_interval_is_inferred_from_the_first_two_samples_only():
    """
    Documents a real limitation: the interval is read from the first gap, so a
    recording that starts with a gap reports the wrong sampling rate, and every
    metric that scales by it is then wrong.
    """
    timestamps = list(pd.date_range("2013-06-01", periods=10, freq="5min"))
    # A one hour gap between the first and second sample
    timestamps[1:] = [t + pd.Timedelta(hours=1) for t in timestamps[1:]]
    df = pd.DataFrame({"timestamp": timestamps, "mean_lux": np.zeros(10)})

    assert lm.get_sampling_interval_minutes(df) == 65.0  # not 5.0


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
