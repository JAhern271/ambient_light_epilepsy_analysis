# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:44:16 2026

@author: ahernj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq


def compute_lux_summary(seqn_array, year, base_path):
    """
    Computes:
        - mean lux across recording
        - recording length (hours)
    
    Returns DataFrame indexed by SEQN
    """
    
    results = []
    
    for seqn in seqn_array:
        file_path = Path(base_path) / f"PAXLUX_{year}" / "parquet_5min" / f"SEQN_{int(seqn)}_5min.parquet"
    
        if not file_path.exists():
            print(f"ERROR: path does not exist: {file_path}")
            continue
    
        try:
            pf = pq.ParquetFile(file_path)
    
            # Only read necessary columns
            table = pf.read(columns=["timestamp", "mean_lux"])
            df = table.to_pandas()
    
            if df.empty:
                print(f"ERROR: table is empty: {file_path}")
                continue
    
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


def interdaily_stability(df):

    df = df.copy()
    df = df.sort_values("timestamp")

    # mean lux
    mean_lux = df["mean_lux"].mean()

    # extract hour of day
    df["hour"] = df["timestamp"].dt.hour

    # mean lux for each hour
    hourly_mean = df.groupby("hour")["mean_lux"].mean()

    # number of samples
    N = len(df)

    # numerator
    num = N * np.sum((hourly_mean - mean_lux) ** 2)

    # denominator
    denom = 24 * np.sum((df["mean_lux"] - mean_lux) ** 2)

    IS = num / denom

    return IS



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









