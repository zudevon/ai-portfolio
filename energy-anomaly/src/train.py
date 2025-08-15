#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import yaml

from src.utils import set_seed, sliding_windows, train_val_split_by_time, compute_threshold, save_json
from src.models.lstm_autoencoder import build_lstm_autoencoder

def prepare_sequences(df: pd.DataFrame, features: list, seq_len: int, scaler: StandardScaler | None = None, stride: int = 1) -> tuple[np.ndarray, StandardScaler]:
    X = df[features].values.astype(np.float32)
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    X_seq = sliding_windows(X, seq_len=seq_len, stride=stride)
    return X_seq, scaler

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(42)
    processed_dir = Path(cfg["data"]["processed_dir"])
    rule = cfg["data"]["resample_rule"]
    data_path = processed_dir / f"household_power_{rule or '1T'}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run preprocessing first.")
    df = pd.read_parquet(data_path)
    features = cfg["data"]["numeric_features"]

    # Split by time
    train_df, val_df = train_val_split_by_time(df, cfg["preprocess"]["train_end"])

    seq_len = int(cfg["model"]["seq_len"])
    X_train, scaler = prepare_sequences(train_df, features, seq_len=seq_len)
    X_val, _ = prepare_sequences(val_df, features, seq_len=seq_len, scaler=scaler)

    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    model = build_lstm_autoencoder(seq_len=seq_len, n_features=X_train.shape[-1],
                                   lstm_units=cfg["model"]["lstm_units"],
                                   latent_dim=cfg["model"]["latent_dim"],
                                   dropout=cfg["model"]["dropout"],
                                   l2=cfg["model"]["l2"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg["train"]["patience"], restore_best_weights=True),
        ModelCheckpoint("models/autoencoder.keras", monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        verbose=1,
        callbacks=callbacks
    )

    # Compute reconstruction error on the training set to set threshold
    train_recon = model.predict(X_train, verbose=0)
    train_err = np.mean((X_train - train_recon)**2, axis=(1,2))

    th_cfg = cfg["threshold"]
    threshold = compute_threshold(train_err, method=th_cfg["method"], quantile=th_cfg["quantile"], mad_scale=th_cfg["mad_scale"])
    os.makedirs("models", exist_ok=True)
    save_json({"threshold": float(threshold), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
               "features": features, "seq_len": seq_len, "resample_rule": rule}, "models/threshold.json")

    print(f"Saved model to models/autoencoder.keras and threshold={threshold:.6f} to models/threshold.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
