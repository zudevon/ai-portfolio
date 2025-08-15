#!/usr/bin/env python3
from __future__ import annotations
import os
import json
import math
import random
from typing import Tuple, Iterable, Dict, Any, Optional

import numpy as np
import pandas as pd

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def sliding_windows(arr: np.ndarray, seq_len: int, stride: int = 1) -> np.ndarray:
    """
    Create sliding window views over the first dimension of arr.
    Returns shape: (num_windows, seq_len, num_features)
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (time, features), got shape {arr.shape}")
    n = arr.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len, arr.shape[1]))
    idx = np.arange(n - seq_len + 1, step=stride)[:, None] + np.arange(seq_len)[None]
    return arr[idx]

def train_val_split_by_time(df: pd.DataFrame, boundary: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe by time index (<= boundary goes to train)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by DatetimeIndex.")
    train = df.loc[:boundary].copy()
    val = df.loc[boundary:].copy()
    # Drop the first row in val to avoid leakage at the split point
    if not val.empty:
        val = val.iloc[1:]
    return train, val

def compute_threshold(errors: np.ndarray, method: str = "quantile", quantile: float = 0.995, mad_scale: float = 3.5) -> float:
    """Compute anomaly threshold from training reconstruction errors."""
    if method == "quantile":
        return float(np.quantile(errors, quantile))
    elif method == "mad":
        med = np.median(errors)
        mad = np.median(np.abs(errors - med)) + 1e-8
        return float(med + mad_scale * 1.4826 * mad)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
