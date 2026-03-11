# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:44:16 2026

@author: ahernj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
