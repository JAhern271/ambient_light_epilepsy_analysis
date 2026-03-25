# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:12:05 2026

@author: ahernj
"""

from pathlib import Path
import pyreadstat
import pandas as pd
import pyarrow.parquet as pq




def find_people_on_asm(year, base_path, overwrite=False):

    p = base_path / f"{year}" / f"RXQ_RX_{year}.parquet"
    #p = Path(f"C:/Users/ahernj/Documents/Projects/ambient_light_epilepsy_analysis/data/{year}/raw_parquet/RXQ_RX_{year}.parquet")

    if year == "G":
        # Load the prescription meds data
        rx = pd.read_parquet(p)
        
    else:
        table = pq.read_table(p)

        # Get all column names except the bad one
        good_cols = [c for c in table.column_names if c != "RXDRSD1"]
        
        # Select only good columns
        table_subset = table.select(good_cols)
        
        # Convert to pandas
        rx = table_subset.to_pandas()


    # Define epilepsy-specific meds to look for in the RX table
    epilepsy_specific_asms = [
        "phenytoin",
        "carbamazepine",
        "valproic acid",
        "divalproex sodium",
        "phenobarbital",
        "primidone",
        "ethosuximide",
        "levetiracetam",
        "lamotrigine",
        "topiramate",
        "oxcarbazepine",
        "zonisamide",
        #"gabapentin",      # borderline
        #"pregabalin"       # borderline
    ]

    # Convert drug names in table into lower case
    rx["drug_lower"] = rx["RXDDRUG"].str.lower()
    
    # Find the people who use ASM
    rx["is_asm"] = rx["drug_lower"].isin(epilepsy_specific_asms)
    
    # Check that medication was taken in the past 30 days
    rx["current_use"] = rx["RXDUSE"] == 1  
    
    # Define current ASM users
    asm_users = rx.loc[rx["is_asm"] & rx["current_use"], "SEQN"].unique()
    
    pwe = pd.Series(asm_users, name="SEQN")

    # Only save and return if the file does not exist already
    save_path = base_path / f"{year}" / "processed" / f"people_with_epilepsy_{year}.csv"
    if save_path.exists():

        if overwrite==True:
            pwe.to_csv(save_path)

        else:  
            print(f"CSV file already exists in {save_path}")
            pwe = pd.read_csv(save_path)
            
    else:
        pwe.to_csv(save_path)
        print(f"CSV saved in {save_path}")

    return pwe



def load_pwe_seqn(year, base_path):
    
    # Load SEQN numbers for people with epilepsy
    pwe_path = base_path / "processed" / f"people_with_epilepsy_{year}.csv"
    
    #pwe_path = Path(f"C:\\Users\\ahernj\\Documents\\Projects\\ambient_light_epilepsy_analysis\\data\\{year}\\processed\\people_with_epilepsy_{year}.csv")
    
    return pd.read_csv(pwe_path, index_col=0)
    


def load_freq_matched_control_groups(year, base_path):
    
    control_path = base_path / "processed" / f"freq_match_control_{year}.csv"
    pwe_path     = base_path / "processed" / f"freq_match_pwe_{year}.csv"
    
    control_s = pd.read_csv(control_path, index_col=0)
    pwe_s = pd.read_csv(pwe_path, index_col=0)
    
    return control_s.values.reshape(-1).astype(int), pwe_s.values.reshape(-1).astype(int)
    
    
    