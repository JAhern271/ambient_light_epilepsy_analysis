# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:44:16 2026

@author: ahernj
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from . import params, paths


def compute_lux_summary(seqn_array, year, base_path=None, downsample="5min"):
    """
    Computes:
        - mean lux across recording
        - recording length (hours)
    
    Returns DataFrame indexed by SEQN
    """
    
    results = []

    # Day and night windows, read once for the whole cohort
    light = params.section("light")

    if downsample == "5min":
        cols = ["timestamp", "mean_lux"]
    elif downsample is None:
        cols = ["HEADER_TIMESTAMP", "LUX"]
    else:
        raise ValueError(
            f"Unknown downsample {downsample!r}; expected '5min' or None"
        )

    for seqn in seqn_array:

        file_path = paths.lux_file(seqn, year, downsample, base_path)

        if not file_path.exists():
            print(f"ERROR: path does not exist: {file_path}")
            continue
    
        try:
            pf = pq.ParquetFile(file_path)
    
            # Only read necessary columns
            table = pf.read(columns=cols)
            df = table.to_pandas()
            
            # Rename the columns
            df.columns = ['timestamp', 'mean_lux']
    
            if df.empty:
                print(f"ERROR: table is empty: {file_path}")
                continue
            
    # here would be a natural place to break up the function into a load and an
    # analysis part. analysis would deal with df with time/lux cols. 
    
            # Print a statemennt indicating that the analysis for SEQN is happening
            # \r moves cursor to start of line, end="" prevents a new line
            print(f"\rCohort {year} analysis happening for SEQN: {int(seqn)}", end="", flush=True)

    
            # Determine the timezone for all data
            tz = df["timestamp"].dt.tz
    
            # Calculate the total duration of the recording in hours
            t_min = df["timestamp"].min()
            t_max = df["timestamp"].max()
            duration_hours = (t_max - t_min).total_seconds() / 3600
    
                            
            # Calculate mean light exposure (not actually useful, may remove)
            mean_lux = df["mean_lux"].mean()
            
            # Calculate mean daytime light exposure. The windows come from
            # analysis_params.toml [light] (methods.md 6.1), never a literal.
            day_start, day_end = light["day_window"]
            daytime_lux = compute_mean_daytime_lux(
                df, day_start=day_start, day_end=day_end
            )

            # Calculate mean nighttime light exposure
            night_start, night_end = light["night_window"]
            nighttime_lux = compute_mean_nighttime_lux(
                df, night_start=night_start, night_end=night_end
            )
    
            # Calculate the time above threshold LUX level. This is the
            # superseded whole-recording version, kept for the frozen PAXLUX
            # route; see minutes_above_thresholds for the methods.md 6.2 one.
            threshold = light["primary_day_threshold"]
            mins_per_day_above = time_above_threshold_normalized(df, threshold=threshold)
    
            # Calculate m10, l5, the relative amplitude, and the clock times
            # at which M10 and L5 start (circular minutes past midnight)
            m10, l5, ra, m10_start_minutes, m10_start_time, l5_start_minutes, l5_start_time = relative_amplitude(df)
    
            # Calculate IS and IV
            IS = interdaily_stability(df)
            IV = intradaily_variability(df)
    
            results.append({
                "timezone": tz,
                "SEQN": seqn,
                "duration_hours": duration_hours,
                "mean_lux": mean_lux,
                "mean_daytime_lux": daytime_lux,
                "mean_nighttime_lux": nighttime_lux,
                "time_above_threshold": mins_per_day_above,
                "M10": m10,
                "L5": l5, 
                "RA": ra,
                # Circular: never average these arithmetically (methods.md 8.2)
                "m10_start_clock_min": m10_start_minutes,
                "l5_start_clock_min": l5_start_minutes,
                "IS": IS,
                "IV": IV
            })
    
        except Exception as e:
            print(f"Error processing {seqn}: {e}")
    
    
    return pd.DataFrame(results)




def in_clock_window(hours, start, end):
    """
    True for each hour of the clock that falls inside [start, end).

    `start` is inclusive and `end` exclusive, so (7, 19) covers 07:00-18:59.
    A window whose start is later than its end wraps midnight: (23, 6) covers
    23:00-05:59.

    Raises for a window that cannot be what was meant -- an hour outside
    0-23, or start == end, which would be either an empty window or the whole
    day depending on how it is read.
    """
    for hour in (start, end):
        if not 0 <= hour <= 23:
            raise ValueError(f"Window hour {hour!r} is outside 0-23")
    if start == end:
        raise ValueError(f"Window ({start}, {end}) has the same start and end")

    if start < end:
        # Does not cross midnight
        return (hours >= start) & (hours < end)

    # Crosses midnight
    return (hours >= start) | (hours < end)


def compute_mean_daytime_lux(df, *, day_start, day_end):
    """
    Computes mean daytime lux.

    The window has no default on purpose: the value belongs in
    analysis_params.toml [light] day_window (methods.md 6.1), and a default
    here is how a second, contradicting window once crept in.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns:
            - 'timestamp' (datetime)
            - 'mean_lux'   NaN for a masked minute, which is skipped
    day_start : int
        Start hour (inclusive)
    day_end : int
        End hour (exclusive)

    Returns
    -------
    float
        Mean daytime lux
    """

    hours = df["timestamp"].dt.hour

    mask = in_clock_window(hours, day_start, day_end)

    if mask.sum() == 0:
        return np.nan

    return df.loc[mask, "mean_lux"].mean()


def compute_mean_nighttime_lux(df, *, night_start, night_end):
    """
    Computes mean nighttime lux.

    Handles windows that cross midnight. As for the daytime window, there is
    no default: read analysis_params.toml [light] night_window.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns:
            - 'timestamp' (datetime)
            - 'mean_lux'   NaN for a masked minute, which is skipped
    night_start : int
        Start hour (inclusive)
    night_end : int
        End hour (exclusive)

    Returns
    -------
    float
        Mean nighttime lux
    """

    hours = df["timestamp"].dt.hour

    mask = in_clock_window(hours, night_start, night_end)

    if mask.sum() == 0:
        return np.nan

    return df.loc[mask, "mean_lux"].mean()



def get_sampling_interval_minutes(df):
    """
    The recording's epoch length in minutes: 1 for PAXMIN, 5 or 1/60 for the
    PAXLUX downsamples.

    Taken as the most common gap between consecutive timestamps, so a single
    missing or duplicated row cannot change it. (An earlier version read only
    the first two rows, so one missing minute at the start of a PAXMIN
    recording made it a 2-minute epoch and halved every window length.)

    Raises if the epoch is not positive, or does not fit a whole number of
    times into an hour, since the M10 and L5 windows are counted in samples.
    """
    timestamps = df["timestamp"].sort_values()
    gaps_seconds = timestamps.diff().dropna().dt.total_seconds()

    if gaps_seconds.empty:
        raise ValueError("Need at least two timestamps to find the epoch")

    # mode() returns every tied value in ascending order; take the smallest
    epoch_seconds = float(gaps_seconds.mode().iloc[0])

    if epoch_seconds <= 0:
        raise ValueError(f"Most common gap between timestamps is {epoch_seconds} s")
    if 3600 % epoch_seconds != 0:
        raise ValueError(
            f"Epoch of {epoch_seconds} s does not divide an hour evenly"
        )

    return epoch_seconds / 60



def time_above_threshold_normalized(df, threshold):
    """
    Minutes per day above `threshold`, over the whole recording.

    SUPERSEDED for the analysis; kept so the frozen PAXLUX route and its
    regression fixture stay reproducible. It does not implement methods.md
    6.2: it counts all 24 hours rather than the day window, has no notion of
    valid days, and divides by every row, so a masked (NaN) minute counts as
    "not above". Use minutes_above_thresholds on PAXMIN instead.
    """
    df = df.copy()
    df = df.sort_values("timestamp")
    
    # Detect sampling rate
    epoch_minutes = get_sampling_interval_minutes(df)
    
    # Compute time above threshold
    epochs_above = (df["mean_lux"] > threshold).sum()
    
    # Convert to percentage of recording
    percent_above = epochs_above / len(df)
    
    # Convert to an average mins per day above threshold
    mins_per_day_above = percent_above * 60 * 24
        
    return mins_per_day_above


# The two ways a masked minute inside the day window can be treated. The
# analysis uses "raw" (light.masked_minutes, decided 2026-10-01); it stays a
# required argument with no default, so callers read it from [light].
MASKED_MINUTE_RULES = ("raw", "rescale")


def minutes_above_thresholds(prepared, days, *, thresholds, day_window,
                             masked_minutes):
    """
    Daytime minutes per day above each light threshold (methods.md 6.2).

    For each VALID day, count the retained minutes inside the day window
    whose lux is strictly above the threshold, then average across valid
    days. Minutes outside the day window, and days that are not valid, play
    no part.

    Parameters
    ----------
    prepared : DataFrame
        From wear.prepare_minutes: needs 'timestamp', 'mean_lux' (NaN where
        masked), 'retained' and 'analytic_day'.
    days : DataFrame
        From wear.summarise_days, indexed by analytic_day, with 'is_valid'.
    thresholds : sequence of numbers
        Lux thresholds; light.day_thresholds.
    day_window : (int, int)
        Start and end hour; light.day_window.
    masked_minutes : "raw" or "rescale"
        What a masked minute inside the day window counts as.
          raw      it is simply not counted, so a day with less wear can
                   score lower for that reason alone.
          rescale  the count is scaled up by (window minutes / retained
                   window minutes), which assumes the masked minutes looked
                   like the retained ones.
        No default: read light.masked_minutes ("raw", decided 2026-10-01).

    Returns
    -------
    dict
        {threshold: mean minutes per valid day}. NaN for every threshold if
        the participant has no valid day.
    """
    if masked_minutes not in MASKED_MINUTE_RULES:
        raise ValueError(
            f"masked_minutes must be one of {MASKED_MINUTE_RULES}, "
            f"got {masked_minutes!r}"
        )

    start, end = day_window

    # Length of the window in minutes: 720 for 07:00-19:00
    window_minutes = 60 * int(in_clock_window(np.arange(24), start, end).sum())

    valid_days = days.index[days["is_valid"]]

    # Day-window minutes of valid days only
    hours = prepared["timestamp"].dt.hour
    keep = in_clock_window(hours, start, end) & prepared["analytic_day"].isin(valid_days)
    window = prepared.loc[keep]

    if len(valid_days) == 0:
        return {threshold: np.nan for threshold in thresholds}

    grouped = window.groupby("analytic_day")
    retained_per_day = grouped["retained"].sum()

    result = {}
    for threshold in thresholds:
        # A masked minute's lux is NaN, and NaN > threshold is False, so it is
        # never counted as above. That is the "raw" rule.
        above_per_day = (window["mean_lux"] > threshold).groupby(
            window["analytic_day"]
        ).sum()

        if masked_minutes == "rescale":
            above_per_day = above_per_day * window_minutes / retained_per_day

        result[threshold] = float(above_per_day.mean())

    return result



def window_start(window_end_index, window_samples, samples_per_day, epoch_minutes):
    """
    Clock time at which a rolling window starts, from the index of its last
    sample in the two-day extended profile used by relative_amplitude.

    Returns (minutes past midnight, datetime.time). The minutes are circular,
    0-1439: a window starting at 23:30 is 1410, one starting at 00:30 is 30,
    and the two are an hour apart, not 23 hours.
    """
    start_index = window_end_index - window_samples + 1

    # A window can start late on the first copy of the day and run into the
    # second; taking the index modulo one day returns its clock position.
    start_minutes = (start_index % samples_per_day) * epoch_minutes

    start_time = (pd.Timestamp("2000-01-01") + pd.Timedelta(minutes=start_minutes)).time()

    return start_minutes, start_time


def relative_amplitude(df):
    """
    M10, L5, relative amplitude, and the clock times at which M10 and L5 start
    (methods.md 6.5).

    The recording is averaged into one 24 h profile (each clock time's mean
    across days). M10 is the highest mean over any 10 h stretch of that
    profile and L5 the lowest over any 5 h stretch, with windows allowed to
    run across midnight. RA = (M10 - L5) / (M10 + L5).

    The start times are minutes past midnight, 0-1439, and they are CIRCULAR
    (methods.md 8.2). L5 typically starts either side of midnight, so two
    participants at 23:00 and 01:00 come out as 1380 and 60, whose arithmetic
    mean is noon. Never average, difference or regress these values as plain
    numbers; read them in R as circular, e.g.
        circular(x / 60, units = "hours", template = "clock24")
    Starts, not midpoints, because methods.md 6.5 and 8.2 name start times,
    as do nparACT and [Tang_2024] (decided 2026-10-01).

    Known behaviours, each an open item in doc/implementation-status.md rather
    than a settled definition:
      - Ties go to the first window found scanning from 00:00, so a tied
        stretch that crosses midnight starts at 00:00, not at its own start.
        Common for L5 on lux, where a dark room reads exactly 0.
      - A clock time that is masked (NaN) on every day leaves that profile
        point NaN, and no window containing it is eligible.
      - Every row passed in is used, valid day or not; restricting to valid
        days is the caller's job for now.

    Parameters
    ----------
    df : pandas DataFrame
        'timestamp' and 'mean_lux' (or any signal under that name), NaN where
        a sample is masked.

    Returns
    -------
    tuple
        (m10, l5, ra, m10_start_minutes, m10_start_time,
         l5_start_minutes, l5_start_time)
    """

    df = df.copy()
    df = df.sort_values("timestamp")

    epoch_minutes = get_sampling_interval_minutes(df)
    samples_per_day = int(round(24 * 60 / epoch_minutes))

    # Average 24 h profile, on an explicit clock grid: bin 0 is the epoch
    # starting at 00:00, bin 1 the next, and so on. Every bin is present even
    # if no row falls in it (it is then NaN), so a bin's position in `values`
    # is always its clock time. Grouping on the clock times that happen to be
    # present, as an earlier version did, shifts every later time earlier by
    # one epoch for each bin that is absent.
    seconds_into_day = (
        df["timestamp"].dt.hour * 3600
        + df["timestamp"].dt.minute * 60
        + df["timestamp"].dt.second
    )
    df["clock_bin"] = (seconds_into_day // (epoch_minutes * 60)).astype(int)
    mean_24h = df.groupby("clock_bin")["mean_lux"].mean()
    mean_24h = mean_24h.reindex(range(samples_per_day))

    values = mean_24h.values

    samples_per_hour = int(round(60 / epoch_minutes))
    m10_window = 10 * samples_per_hour
    l5_window = 5 * samples_per_hour

    # Two copies of the day end to end, so that a window can run across
    # midnight. Each rolling mean is labelled by the window's LAST sample.
    extended = np.concatenate([values, values])

    m10_roll = pd.Series(extended).rolling(m10_window).mean()
    l5_roll = pd.Series(extended).rolling(l5_window).mean()

    m10 = m10_roll.max()
    l5 = l5_roll.min()

    ra = (m10 - l5) / (m10 + l5)

    # idxmax / idxmin return the first window on a tie (see docstring)
    m10_start_minutes, m10_start_time = window_start(
        m10_roll.idxmax(), m10_window, samples_per_day, epoch_minutes
    )
    l5_start_minutes, l5_start_time = window_start(
        l5_roll.idxmin(), l5_window, samples_per_day, epoch_minutes
    )

    return (
        m10,
        l5,
        ra,
        m10_start_minutes,
        m10_start_time,
        l5_start_minutes,
        l5_start_time
    )


def interdaily_stability(df, bin_size="1h"):
    """
    Interdaily stability: how reliably the same pattern repeats day to day.

    Witting et al. (1990):

        IS = (n * sum_h (x_h - x)^2) / (p * sum_i (x_i - x)^2)

    where x_i are the epochs, x is the grand mean, p is the number of epochs
    per day, and x_h is the mean of time-of-day bin h across days. It is the
    variance of the average 24 h profile as a fraction of the total variance,
    so it runs from 0 (no day-to-day reproducibility) to 1 (identical days).

    The two halves of that ratio must be at the SAME time resolution. An
    earlier version of this function binned the numerator hourly while leaving
    the denominator at the raw epoch, so the denominator carried within-hour
    variance the numerator could not capture. That pushed IS down by a factor
    that varied per participant, and made values from the 5 minute and 1 Hz
    analyses incomparable with each other and with published figures.

    The recording is therefore resampled to `bin_size` before anything is
    computed. Hourly is the default because it is the usual convention in the
    nonparametric circadian literature, and because it makes results from
    different source resolutions directly comparable.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain 'timestamp' and 'mean_lux'.
    bin_size : str
        Pandas offset alias for the epoch to compute at. Default '1h'.

    Returns
    -------
    float
        IS, or NaN for a recording with no variance.
    """

    series = df.set_index("timestamp")["mean_lux"].sort_index()

    # Resample so that profile bins and epochs are the same thing.
    # Gaps resample to NaN and are dropped rather than counted as zero.
    binned = series.resample(bin_size).mean().dropna()

    if binned.empty:
        return np.nan

    bin_minutes = pd.Timedelta(bin_size).total_seconds() / 60
    epochs_per_day = int(round(24 * 60 / bin_minutes))

    values = binned.to_numpy()
    grand_mean = values.mean()

    # Average 24 h profile: mean of each time-of-day bin across days
    minutes_into_day = binned.index.hour * 60 + binned.index.minute
    time_of_day_bin = (minutes_into_day // bin_minutes).astype(int)
    profile = binned.groupby(time_of_day_bin).mean().to_numpy()

    numerator = len(values) * np.sum((profile - grand_mean) ** 2)
    denominator = epochs_per_day * np.sum((values - grand_mean) ** 2)

    if denominator == 0:
        return np.nan

    return numerator / denominator



def intradaily_variability(df):

    df = df.copy()
    df = df.sort_values("timestamp")

    X = df["mean_lux"].values
    N = len(X)

    mean_lux = np.mean(X)

    # numerator
    diff = np.diff(X)
    num = np.sum(diff ** 2) / (N - 1)

    # denominator
    denom = np.sum((X - mean_lux) ** 2) / N

    IV = num / denom

    return IV









