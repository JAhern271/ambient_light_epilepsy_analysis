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

from conftest import make_recording, square_wave


# ---------------------------------------------------------------------------
# Day and night windows
# ---------------------------------------------------------------------------

def test_daytime_window_is_07_to_19():
    """Daytime is hours 07:00-18:59 inclusive; 19:00 is excluded."""
    # One value per hour, equal to the hour number
    values = np.arange(24, dtype=float)
    df = make_recording(values, epoch_minutes=60)

    result = lm.compute_mean_daytime_lux(df, day_start=7, day_end=19)

    assert result == pytest.approx(np.mean(np.arange(7, 19)))


def test_nighttime_window_wraps_midnight():
    """Night spans 20:00-04:59, i.e. it must cross midnight."""
    values = np.arange(24, dtype=float)
    df = make_recording(values, epoch_minutes=60)

    result = lm.compute_mean_nighttime_lux(df, night_start=20, night_end=5)

    expected_hours = list(range(20, 24)) + list(range(0, 5))
    assert result == pytest.approx(np.mean(expected_hours))


def test_daytime_returns_nan_when_no_samples_in_window():
    """A recording with no daytime samples gives NaN, not an error or zero."""
    df = make_recording(np.ones(4), start="2013-06-01 00:00:00", epoch_minutes=60)

    assert np.isnan(lm.compute_mean_daytime_lux(df, day_start=7, day_end=19))


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

def test_interdaily_stability_differs_between_5min_and_hourly_input(sinusoid_recording):
    """
    IS is computed with hourly bins in the numerator but raw epochs in the
    denominator, so its value depends on the sampling resolution of the input.

    The same underlying signal therefore scores differently at 5 min and at
    1 hour. This matters because the project has results computed at both 5 min
    and 1 Hz: IS values are not comparable across them.
    """
    five_min = sinusoid_recording

    hourly = (
        five_min.set_index("timestamp")["mean_lux"]
        .resample("1h")
        .mean()
        .reset_index()
        .rename(columns={"mean_lux": "mean_lux"})
    )

    is_5min = lm.interdaily_stability(five_min)
    is_hourly = lm.interdaily_stability(hourly)

    assert is_5min != pytest.approx(is_hourly, rel=1e-3)


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
