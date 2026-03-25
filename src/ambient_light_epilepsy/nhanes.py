# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:08:20 2026

@author: ahernj
"""

from pathlib import Path
import pyreadstat
import pandas as pd
import numpy as np






def xpt_to_parquet(
    xpt_path,
    parquet_dir=None,
    overwrite=False,
):
    """
    Convert a SAS XPT file to Parquet format.

    Parameters
    ----------
    xpt_path : str or Path
        Path to the .xpt file.
    parquet_dir : str or Path, optional
        Directory to save the parquet file.
        Defaults to the same directory as xpt_path.
    overwrite : bool
        If False (default), do not overwrite existing parquet file.

    Returns
    -------
    parquet_path : Path
        Path to the saved parquet file.
    """
    xpt_path = Path(xpt_path)

    if not xpt_path.exists():
        raise FileNotFoundError(f"XPT file not found: {xpt_path}")

    if parquet_dir is None:
        parquet_dir = xpt_path.parent
    else:
        parquet_dir = Path(parquet_dir)
        parquet_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = parquet_dir / (xpt_path.stem + ".parquet")

    # This ensures that the conversion is not repeated if a file already exits 
    if parquet_path.exists() and not overwrite:
        return parquet_path

    # Read XPT (slow step)
    df, _ = pyreadstat.read_xport(xpt_path)

    # Write Parquet (fast)
    df.to_parquet(parquet_path, index=False)

    return parquet_path




def load_partial_demo(year, base_path):
    # Load the demographics table
    p = base_path / f"{year}" / f"DEMO_{year}.parquet"
    #p = Path(f"W:/projects/ambient_light_epilepsy_analysis/data/{year}/DEMO_{year}.parquet")
    
    cols_to_load = {"SEQN": "ID",
                    "RIDAGEYR": "age", 
                    "RIAGENDR": "sex", 
                    "RIDRETH3": "race", 
                    "DMDEDUC3": "p_ed",  # 6-19 yoa education level
                    "DMDEDUC2": "a_ed",  # 20+ yoa education level 
                    "INDFMPIR": "PIR",   # ratio of family income to poverty
                    "DMDHHSIZ": "NIH",   # number of people in household
                    "RIDEXMON": "6_month"  # 6 month time period when exam was performed
                   }
    df = pd.read_parquet(p, columns=list(cols_to_load.keys()))
    
    # Set SEQN as index and rename columns
    df = df.set_index("SEQN")
    df.columns = list(cols_to_load.values())[1:]

    return df


def load_PAXHD(year):
    
    p = Path(f"W:/projects/ambient_light_epilepsy_analysis/data/{year}/PAXHD_{year}.parquet")
    df = pd.read_parquet(p)
    df = df.set_index("SEQN")
    
    return df



def add_demo_labels(df_input):
    
    df = df_input.copy()
    
    
    # -----------------------------
    # Sex
    # -----------------------------
    df["sex_label"] = df["sex"].map({
        1: "Male",
        2: "Female"
    })
    
    
    df["season"] = df["6_month"].map({
        1: "Winter",
        2: "Summer"
    })
    
    # -----------------------------
    # Race / ethnicity (RIDRETH3)
    # -----------------------------
    race_map = {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian",
        7: "Other / Multiracial"
    }
    df["race_label"] = df["race"].map(race_map)
    
    # -----------------------------
    # Education
    # Use adult education. Child education may not be relevent factor to consider. Age amtching is more important for children.
    # -----------------------------
    edu_map = {
        1: "<9th grade",
        2: "9–11th grade",
        3: "High school / GED",
        4: "Some college / AA",
        5: "College graduate",
        7: "Refused",
        9: "Don't know"
    }
    
    df["education_label"] = df["a_ed"].map(edu_map)
    
    # -----------------------------
    # Income-to-poverty ratio (PIR)
    # -----------------------------
    df["PIR_cat"] = pd.cut(
        df["PIR"],
        bins=[0, 1, 4, 5],
        labels=[
            "<1 (Low)",
            "1–4 (Middle)",
            ">4 (High)",
        ]
    )
    
    return df



def load_employment(year, base_path):

    p = base_path / f"{year}" / "OCQ_{year}.parquet"
    #p = Path(f"W:/projects/ambient_light_epilepsy_analysis/data/{year}/OCQ_{year}.parquet")

    cols = ["SEQN", "OCD150"]

    df = pd.read_parquet(p, columns=cols)
    df = df.set_index("SEQN")

    df["employed"] = df["OCD150"].isin([1,2])
    df['employed'] = df['employed'].astype(int)


    return df




def load_dpq(year, base_path, dropna=True):
    
    p = base_path / f"{year}" / "dpq_{year}.parquet"
    #p = Path(f"W:/projects/ambient_light_epilepsy_analysis/data/{year}/dpq_{year}.parquet")

    phq_cols = [
        "DPQ010","DPQ020","DPQ030",
        "DPQ040","DPQ050","DPQ060",
        "DPQ070","DPQ080","DPQ090"
    ]

    df = pd.read_parquet(p, columns=["SEQN"] + phq_cols)
    df = df.set_index("SEQN")

    if dropna == True:
        df = df.dropna()


    df["phq9_total"] = df.sum(axis=1)
    df["depressed"] = df["phq9_total"] >= 10
    df['depressed'] = df['depressed'].astype(int)


    return df





def add_employment_and_depression_status(df_all, base_path):
    # Load the OCD150 and employment status 
    ocq_G = load_employment("G", base_path)
    ocq_H = load_employment("H", base_path)
    
    # Load the depression status
    dpq_G = load_dpq("G", base_path)
    dpq_H = load_dpq("H", base_path)
    
    # Ensure SEQN types match (NHANES tables use float)
    df_all["SEQN"] = df_all["SEQN"].astype(float)
    
    # Extract only the columns we need
    ocq_G = ocq_G[["employed"]]
    ocq_H = ocq_H[["employed"]]
    
    dpq_G = dpq_G[["depressed"]]
    dpq_H = dpq_H[["depressed"]]
    
    # Split df_all by cohort
    df_G = df_all[df_all["cohort"] == "G"].copy()
    df_H = df_all[df_all["cohort"] == "H"].copy()
    
    # Merge employment
    df_G = df_G.merge(ocq_G, left_on="SEQN", right_index=True, how="left")
    df_H = df_H.merge(ocq_H, left_on="SEQN", right_index=True, how="left")
    
    # Merge depression
    df_G = df_G.merge(dpq_G, left_on="SEQN", right_index=True, how="left")
    df_H = df_H.merge(dpq_H, left_on="SEQN", right_index=True, how="left")
    
    # Combine cohorts back together
    df_all = pd.concat([df_G, df_H], ignore_index=True)

    return df_all


def add_demographic_data(df_all, base_path):
    
    # Load demographics
    demo_G = load_partial_demo("G", base_path)
    demo_H = load_partial_demo("H", base_path)

    demo_G = demo_G.drop("p_ed", axis=1)
    demo_H = demo_H.drop("p_ed", axis=1)
    
    # Ensure SEQN type matches (NHANES index is float)
    df_all["SEQN"] = df_all["SEQN"].astype(float)
    
    # Split df_all by cohort
    df_G = df_all[df_all["cohort"] == "G"].copy()
    df_H = df_all[df_all["cohort"] == "H"].copy()
    
    # Merge demographics
    df_G = df_G.merge(demo_G, left_on="SEQN", right_index=True, how="left")
    df_H = df_H.merge(demo_H, left_on="SEQN", right_index=True, how="left")
    
    # Combine back together
    df_all = pd.concat([df_G, df_H], ignore_index=True)

    # Rename 6_month to season 
    df_all = df_all.rename(columns={'6_month': 'season'})


    return df_all


def add_outdoor_time(df_all, base_path):

    # Load DEQ tables
    p = base_path / "G" / "DEQ_G.parquet"
    #p = Path("W:/projects/ambient_light_epilepsy_analysis/data/G/DEQ_G.parquet")
    deq_G = pd.read_parquet(p, columns=["SEQN","DED120","DED125"])

    p = base_path / "H" / "DEQ_H.parquet"
    #p = Path("W:/projects/ambient_light_epilepsy_analysis/data/H/DEQ_H.parquet")
    deq_H = pd.read_parquet(p, columns=["SEQN","DED120","DED125"])


    # --- Clean special values ---
    special_codes = [3333, 7777, 9999]

    for df in [deq_G, deq_H]:
        df["DED120"] = df["DED120"].replace(special_codes, np.nan)
        df["DED125"] = df["DED125"].replace(special_codes, np.nan)

        # Collapse to single metric
        df["minutes_outdoors"] = df[["DED120","DED125"]].mean(axis=1)


    # Ensure SEQN type matches
    df_all["SEQN"] = df_all["SEQN"].astype(float)


    # --- Split main dataframe by cohort ---
    df_G = df_all[df_all["cohort"] == "G"].copy()
    df_H = df_all[df_all["cohort"] == "H"].copy()


    # --- Merge outdoor data ---
    df_G = df_G.merge(
        deq_G[["SEQN","minutes_outdoors"]],
        on="SEQN",
        how="left"
    )

    df_H = df_H.merge(
        deq_H[["SEQN","minutes_outdoors"]],
        on="SEQN",
        how="left"
    )


    # --- Combine back together ---
    df_all = pd.concat([df_G, df_H], ignore_index=True)


    return df_all
