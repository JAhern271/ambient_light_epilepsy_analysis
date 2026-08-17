# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:44:16 2026

@author: ahernj
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from . import paths


def compute_lux_summary(seqn_array, year, base_path=None, downsample="5min"):
    """
    Computes:
        - mean lux across recording
        - recording length (hours)
    
    Returns DataFrame indexed by SEQN
    """
    
    results = []
    
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
            
            # Calculate mean daytime light exposure
            daytime_lux = compute_mean_daytime_lux(df, day_start=7, day_end=19)
            
            # Calculate mean nightitme light exposure
            nighttime_lux = compute_mean_nighttime_lux(df, night_start=20, night_end=5)
    
            # Calculate the time above threshold LUX level
            threshold = 1000
            mins_per_day_above = time_above_threshold_normalized(df, threshold=threshold)
    
            # Calculate m10, l5, theri midpoints and the relative amplitude
            m10, l5, ra, m10_midpoint_minutes, m10_midpoint_time, l5_midpoint_minutes, l5_midpoint_time = relative_amplitude(df)
    
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
                "m10_midpoint": m10_midpoint_minutes,
                "l5_midpoint": l5_midpoint_minutes,
                "IS": IS,
                "IV": IV
            })
    
        except Exception as e:
            print(f"Error processing {seqn}: {e}")
    
    
    return pd.DataFrame(results)




def compute_mean_daytime_lux(df, day_start=7, day_end=19):
    """
    Computes mean daytime lux.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns:
            - 'timestamp' (datetime)
            - 'mean_lux'
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

    mask = (hours >= day_start) & (hours < day_end)

    if mask.sum() == 0:
        return np.nan

    return df.loc[mask, "mean_lux"].mean()


def compute_mean_nighttime_lux(df, night_start=22, night_end=5):
    """
    Computes mean nighttime lux.

    Handles windows that cross midnight.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns:
            - 'timestamp' (datetime)
            - 'mean_lux'
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

    if night_start < night_end:
        # Does NOT cross midnight
        mask = (hours >= night_start) & (hours < night_end)
    else:
        # Crosses midnight (e.g., 22–05)
        mask = (hours >= night_start) | (hours < night_end)

    if mask.sum() == 0:
        return np.nan

    return df.loc[mask, "mean_lux"].mean()



def get_sampling_interval_minutes(df):
    df = df.sort_values("timestamp")
    delta = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
    return delta / 60



def time_above_threshold_normalized(df, threshold=1000):
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



def relative_amplitude(df):

    df = df.copy()
    df = df.sort_values("timestamp")
    
    epoch_minutes = get_sampling_interval_minutes(df)
    
    # Average 24h profile
    df["time_of_day"] = df["timestamp"].dt.time
    mean_24h = df.groupby("time_of_day")["mean_lux"].mean()
    
    values = mean_24h.values
    
    samples_per_hour = int(60 / epoch_minutes)
    m10_window = 10 * samples_per_hour
    l5_window = 5 * samples_per_hour
    
    # Circular extension
    extended = np.concatenate([values, values])
    
    # Rolling means
    m10_roll = pd.Series(extended).rolling(m10_window).mean()
    l5_roll = pd.Series(extended).rolling(l5_window).mean()
    
    m10 = m10_roll.max()
    l5 = l5_roll.min()
    
    ra = (m10 - l5) / (m10 + l5)
    
    minutes_per_sample = epoch_minutes
    
    # =========================
    # M10 midpoint
    # =========================
    
    m10_idx = m10_roll.idxmax()
    
    m10_start = m10_idx - m10_window + 1
    m10_midpoint_idx = m10_start + m10_window // 2
    
    m10_midpoint_idx = m10_midpoint_idx % len(values)
    
    m10_midpoint_minutes = m10_midpoint_idx * minutes_per_sample
    
    m10_hours = int(m10_midpoint_minutes // 60)
    m10_minutes = int(m10_midpoint_minutes % 60)
    
    m10_midpoint_time = pd.Timestamp(
        f"{m10_hours:02d}:{m10_minutes:02d}"
    ).time()
    
    # =========================
    # L5 midpoint
    # =========================
    
    l5_idx = l5_roll.idxmin()
    
    l5_start = l5_idx - l5_window + 1
    l5_midpoint_idx = l5_start + l5_window // 2
    
    l5_midpoint_idx = l5_midpoint_idx % len(values)
    
    l5_midpoint_minutes = l5_midpoint_idx * minutes_per_sample
    
    l5_hours = int(l5_midpoint_minutes // 60)
    l5_minutes = int(l5_midpoint_minutes % 60)
    
    l5_midpoint_time = pd.Timestamp(
        f"{l5_hours:02d}:{l5_minutes:02d}"
    ).time()
    
    return (
        m10,
        l5,
        ra,
        m10_midpoint_minutes,
        m10_midpoint_time,
        l5_midpoint_minutes,
        l5_midpoint_time
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









