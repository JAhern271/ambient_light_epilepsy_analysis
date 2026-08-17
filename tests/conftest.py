"""
Synthetic light recordings with analytically known circadian metrics.

Every fixture returns a DataFrame in the shape the metric functions expect:
a 'timestamp' column and a 'mean_lux' column. Building signals whose correct
answers can be derived by hand is what lets the tests check correctness rather
than merely pinning current behaviour.
"""

import numpy as np
import pandas as pd
import pytest


def make_recording(values, start="2013-06-01 00:00:00", epoch_minutes=5):
    """Wrap an array of lux values in the DataFrame shape the metrics expect."""
    timestamps = pd.date_range(
        start=start, periods=len(values), freq=f"{epoch_minutes}min"
    )
    return pd.DataFrame({"timestamp": timestamps, "mean_lux": np.asarray(values, dtype=float)})


def square_wave(days=7, epoch_minutes=5, high=1000.0, low=0.0,
                on_hour=6, off_hour=18):
    """
    A perfectly regular recording: `high` lux from on_hour to off_hour, `low`
    otherwise, repeated identically every day.

    With the default 06:00-18:00 window the light period is 12 h, so:
      M10 = high  (a 10 h window fits entirely inside the light period)
      L5  = low   (a 5 h window fits entirely inside the dark period)
      RA  = (high - low) / (high + low) = 1 when low is 0
      IS  = 1     (every day is identical, and the pattern is hour-aligned)
    """
    per_hour = 60 // epoch_minutes
    per_day = 24 * per_hour

    hours = (np.arange(per_day) // per_hour)
    one_day = np.where((hours >= on_hour) & (hours < off_hour), high, low)

    return make_recording(np.tile(one_day, days), epoch_minutes=epoch_minutes)


@pytest.fixture
def constant_recording():
    """Seven days of unchanging light. IV is 0; RA and IS are degenerate."""
    return make_recording(np.full(7 * 288, 100.0))


@pytest.fixture
def square_recording():
    """Seven days of a perfect 12 h on / 12 h off square wave at 5 min epochs."""
    return square_wave()


@pytest.fixture
def sinusoid_recording():
    """
    Seven days of a smooth 24 h sinusoid, peaking at midday, offset to stay
    non-negative. Unlike the square wave this varies *within* each hour, which
    is what distinguishes metrics computed at different time resolutions.
    """
    per_day = 288
    t = np.arange(per_day * 7)
    # One full cycle per day, minimum at midnight, maximum at midday
    values = 500.0 * (1 - np.cos(2 * np.pi * t / per_day))
    return make_recording(values)


@pytest.fixture
def example_data_root(tmp_path):
    """
    A complete miniature data root, built on the fly rather than committed as
    binary fixtures.

    Contains four participants of synthetic LUX data in cycle 'X', laid out
    exactly as the real data is, so the loading path can be tested end to end
    without the W: drive.
    """
    lux_dir = tmp_path / "PAXLUX_X" / "parquet_5min"
    lux_dir.mkdir(parents=True)

    processed = tmp_path / "processed"
    processed.mkdir()

    pwe_seqns = [1001, 1002]
    control_seqns = [2001, 2002]

    # Give the PWE a dimmer daytime than the controls, mirroring the real
    # finding, so an integration test can check the direction comes out right.
    for seqn in pwe_seqns:
        recording = square_wave(days=7, high=200.0)
        recording.to_parquet(lux_dir / f"SEQN_{seqn}_5min.parquet", index=False)

    for seqn in control_seqns:
        recording = square_wave(days=7, high=1000.0)
        recording.to_parquet(lux_dir / f"SEQN_{seqn}_5min.parquet", index=False)

    pd.Series(pwe_seqns, name="SEQN").to_csv(processed / "freq_match_pwe_X.csv")
    pd.Series(control_seqns, name="SEQN").to_csv(
        processed / "freq_match_control_X.csv"
    )
    pd.Series(pwe_seqns, name="SEQN").to_csv(
        processed / "people_with_epilepsy_X.csv"
    )

    return tmp_path


@pytest.fixture
def noisy_recording():
    """A reproducible irregular recording, for tests that just need realism."""
    rng = np.random.default_rng(seed=20260817)
    per_day = 288
    t = np.arange(per_day * 7)
    daily = 400.0 * (1 - np.cos(2 * np.pi * t / per_day))
    noise = rng.gamma(shape=2.0, scale=50.0, size=t.size)
    return make_recording(np.clip(daily + noise, 0, None))
