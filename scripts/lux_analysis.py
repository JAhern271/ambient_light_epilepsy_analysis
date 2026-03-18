# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:55:08 2026

@author: ahernj
"""

from pathlib import Path
import pandas as pd
import numpy as np
#import pyarrow.parquet as pq



import ambient_light_epilepsy.nhanes as nhn
import ambient_light_epilepsy.cohort as ch
import ambient_light_epilepsy.lux_metrics as lm


# Define your directory and filename for the final analysis
dir_path = Path("C:/Users/ahernj/Documents/Projects/ambient_light_epilepsy_analysis/analysis/lux")
save_name = "lux_5min_fmatch_analysis.csv"


# Define the directory for the data
base_path = "W:/projects/ambient_light_epilepsy_analysis/data"

# Run analysis for G cohort 
control_seqn, pwe_seqn = ch.load_freq_matched_control_groups("G")
df_control_summary_G = lm.compute_lux_summary(control_seqn, "G", base_path)
df_pwe_summary_G     = lm.compute_lux_summary(pwe_seqn, "G", base_path)

# Run analysis for H cohort 
control_seqn, pwe_seqn = ch.load_freq_matched_control_groups("H")
df_control_summary_H = lm.compute_lux_summary(control_seqn, "H", base_path)
df_pwe_summary_H     = lm.compute_lux_summary(pwe_seqn, "H", base_path)



# Add cohort and epilepsy labels before merging
def add_labels(df, cohort_label, epilepsy_status):
    df = df.copy()
    df["cohort"] = cohort_label
    df["epilepsy"] = epilepsy_status  # 1 = PWE, 0 = control
    return df
    
df_G_pwe = add_labels(df_pwe_summary_G, "G", 1)
df_G_ctrl = add_labels(df_control_summary_G, "G", 0)

df_H_pwe = add_labels(df_pwe_summary_H, "H", 1)
df_H_ctrl = add_labels(df_control_summary_H, "H", 0)

# Merge all 
df_all = pd.concat(
    [df_G_pwe, df_G_ctrl, df_H_pwe, df_H_ctrl],
    ignore_index=True
)


# Add employment and depression status to summary table 
df_all = nhn.add_employment_and_depression_status(df_all)

# Add demographic data 
df_all = nhn.add_demographic_data(df_all)

# Add outdoor time data
df_all = nhn.add_outdoor_time(df_all)

# Save the analysis results as a CSV file
save_path = dir_path / save_name
df_all.to_csv(save_path)
