# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:08:20 2026

@author: ahernj
"""

from pathlib import Path
import pyreadstat
import pandas as pd





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