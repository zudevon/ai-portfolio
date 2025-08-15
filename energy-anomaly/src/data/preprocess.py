#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c].replace({"?": np.nan, "": np.nan}), errors="coerce")
    return df

def preprocess(config_path: str = "config.yaml") -> Path:
    """
    Read raw UCI text, parse datetimes, handle missing, resample and save a processed parquet.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    txt_path = raw_dir / cfg["data"]["filename_txt"]
    if not txt_path.exists():
        raise FileNotFoundError(f"{txt_path} not found. Run src/data/download.py first.")
    # Read with semicolon separator
    df = pd.read_csv(txt_path, sep=";", low_memory=False)
    # Parse datetime
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df.index = dt
    df = df.drop(columns=["Date", "Time"]).sort_index()
    # Coerce numerics
    features = cfg["data"]["numeric_features"]
    df = _coerce_numeric(df, features)
    # Handle missing
    method = cfg["preprocess"]["fillna_method"]
    if method == "ffill":
        df = df.ffill()
    elif method == "bfill":
        df = df.bfill()
    elif method == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown fill method: {method}")
    if cfg["preprocess"]["dropna_after_fill"]:
        df = df.dropna()
    # Resample
    rule = cfg["data"]["resample_rule"]
    if rule and rule != "1T":
        df = df.resample(rule).mean().dropna()
    # Save
    out_path = processed_dir / f"household_power_{rule or '1T'}.parquet"
    df.to_parquet(out_path)
    print(f"Saved processed data: {out_path}, shape={df.shape}")
    return out_path

if __name__ == "__main__":
    preprocess()
