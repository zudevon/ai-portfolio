#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import yaml
import matplotlib.pyplot as plt

from src.utils import sliding_windows, load_json

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_path = Path("models/autoencoder.keras")
    th_path = Path("models/threshold.json")
    if not (model_path.exists() and th_path.exists()):
        raise FileNotFoundError("Model or threshold not found. Run training first.")

    # Load model & threshold/scaler params
    model = tf.keras.models.load_model(model_path)
    th_data = load_json(str(th_path))
    threshold = th_data["threshold"]
    features = th_data["features"]
    seq_len = th_data["seq_len"]
    rule = th_data.get("resample_rule", cfg["data"]["resample_rule"])

    # Load processed data
    data_path = Path(cfg["data"]["processed_dir"]) / f"household_power_{rule or '1T'}.parquet"
    df = pd.read_parquet(data_path)
    X = df[features].values.astype(np.float32)

    # Rebuild scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(th_data["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(th_data["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = len(features)
    Xs = scaler.transform(X)

    # Window
    X_seq = sliding_windows(Xs, seq_len=seq_len, stride=1)
    recon = model.predict(X_seq, verbose=0)
    errors = np.mean((X_seq - recon)**2, axis=(1,2))
    is_anom = errors > threshold

    # Align scores back to timestamps (use end of each window)
    ts = df.index[seq_len-1:]
    result = pd.DataFrame({"timestamp": ts, "recon_error": errors, "is_anomaly": is_anom})
    out_csv = Path("reports") / "anomaly_scores.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    print(f"Saved anomaly scores to {out_csv}")

    # Quick plot
    fig_path = Path("reports") / "anomalies.png"
    plt.figure(figsize=(12,4))
    plt.plot(result["timestamp"], result["recon_error"], label="reconstruction error")
    plt.axhline(threshold, linestyle="--", label="threshold")
    plt.scatter(result.loc[result["is_anomaly"], "timestamp"],
                result.loc[result["is_anomaly"], "recon_error"],
                s=10, label="anomaly")
    plt.legend()
    plt.title("Anomaly scores over time")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
