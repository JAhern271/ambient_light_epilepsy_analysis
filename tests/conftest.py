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


# ---------------------------------------------------------------------------
# Synthetic PAXMIN records, for the wear and valid-day rules (methods.md 5)
# ---------------------------------------------------------------------------
#
# PAXMIN carries no clock time. The only time information is PAXSSNMP, an 80 Hz
# sample counter running from the first minute of the recording, and PAXDAYM,
# the calendar day of wear (1-9). The clock time of a minute is therefore
#
#     PAXFTIME (from PAXHD) + PAXSSNMP / (60 * 80)  minutes
#
# and PAXFTIME varies from 09:11 to 21:30 across cycle H, because the device
# was started when the participant left the exam centre. The builders below
# reproduce that structure so the day-splitting rules can be tested on
# recordings whose correct answers are derivable on paper.

MINUTE_SAMPLES = 60 * 80        # PAXSSNMP ticks in one minute
MINUTES_PER_DAY = 24 * 60


def clock_to_minutes(clock):
    """'16:30:00' -> 990, minutes past midnight. Mirrors PAXHD's HH:MM:SS."""
    hours, minutes = clock.split(":")[:2]
    return int(hours) * 60 + int(minutes)


def make_paxmin(seqn=1001, first_time="16:30:00", last_day=9,
                last_day_minutes=None, lux=100.0, activity=1.0):
    """
    One participant's PAXMIN records, in the shape the real table arrives in.

    The recording starts at `first_time` on calendar day 1 and runs to the end
    of calendar day `last_day`, which reproduces the NHANES protocol: day 1 and
    the last day are partial, every day between them is a full 1,440 minutes.

    Parameters
    ----------
    first_time : str
        PAXHD's PAXFTIME for this participant, HH:MM:SS.
    last_day : int
        PAXHD's PAXLDAY. 9 for a complete recording.
    last_day_minutes : int, optional
        Minute records on the final calendar day. Defaults to
        `first_time + 9`, which is the pattern the real data shows (PAXETLDY
        sits a few minutes after PAXFTIME on the last day). Pass an explicit
        value to build a recording that ends on a chosen clock time.
    lux, activity : float
        Constant PAXLXMM and PAXMTSM. Constant values make a masking failure
        show up as an obviously wrong mean rather than a plausible one.

    Returns
    -------
    DataFrame
        Columns and dtypes as released: PAXDAYM, PAXDAYWM, PAXPREDM and
        PAXFLGSM are strings, everything else float. Rows in time order,
        indexed 0..n-1 so a minute index is also a row position.
    """
    first_minute = clock_to_minutes(first_time)

    if last_day_minutes is None:
        last_day_minutes = first_minute + 9

    # Minutes on each calendar day: partial, then full, then partial
    per_day = ([MINUTES_PER_DAY - first_minute]
               + [MINUTES_PER_DAY] * (last_day - 2)
               + [last_day_minutes])

    day_labels = np.repeat(
        [str(d) for d in range(1, last_day + 1)], per_day
    )
    n = len(day_labels)

    # PAXDAYWM: day of week, arbitrary but consistent. Starts on Tuesday (3).
    weekday = ((np.repeat(np.arange(last_day), per_day) + 2) % 7) + 1

    return pd.DataFrame({
        "SEQN": float(seqn),
        "PAXDAYM": day_labels,
        "PAXDAYWM": [str(w) for w in weekday],
        "PAXSSNMP": np.arange(n, dtype=float) * MINUTE_SAMPLES,
        "PAXTSM": 60.0,
        "PAXPREDM": "1",            # wake wear
        "PAXMTSM": float(activity),
        "PAXLXMM": float(lux),
        "PAXQFM": 0.0,
        "PAXFLGSM": "",
    })


def set_minutes(df, start_minute, n_minutes, **values):
    """
    Overwrite columns on `n_minutes` consecutive rows, from minute
    `start_minute` of the recording. Returns a copy.

    Minute index equals row position for a frame from `make_paxmin`, so the
    caller can work out which analytic day it is hitting by arithmetic on
    PAXFTIME rather than by asking the code under test.
    """
    df = df.copy()
    rows = df.index[start_minute:start_minute + n_minutes]

    for column, value in values.items():
        df.loc[rows, column] = value

    return df


def mark_nonwear(df, start_minute, n_minutes):
    """Mark minutes as PAXPREDM non-wear, the commonest exclusion by far."""
    return set_minutes(df, start_minute, n_minutes, PAXPREDM="3")


def make_header(seqn=1001, first_time="16:30:00", last_day=9, status=1):
    """A PAXHD row or rows, indexed by SEQN as nhanes.load_PAXHD returns it."""
    if not isinstance(seqn, (list, tuple)):
        seqn, first_time, last_day, status = [seqn], [first_time], [last_day], [status]

    return pd.DataFrame(
        {
            "PAXSTS": [float(s) for s in status],
            "PAXFTIME": list(first_time),
            "PAXLDAY": [str(d) for d in last_day],
        },
        index=pd.Index([float(s) for s in seqn], name="SEQN"),
    )
