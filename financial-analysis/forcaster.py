# forcaster.py
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import amp

# bring in your feature pipeline (from the earlier script)
from data_preprocess import execute as build_features  # adjust path/module if needed

# Remove Warnings
import warnings
warnings.filterwarnings("ignore", message=".*NCCL.*")


# ---------------------------
# Data & feature utilities
# ---------------------------
def load_feature_data(
    ticker: str,
    amount_of_days: int,
    frequency: str = "1d",
    add_day_of_week: bool = True
) -> pd.DataFrame:
    df = build_features(ticker, amount_of_days, frequency, save_features=True)
    if add_day_of_week and "Date" in df.columns:
        df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek
    return df


def select_features_and_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Last 8 boolean columns are targets
    bool_cols = [c for c in df.columns if df[c].dtype == "bool"]
    if len(bool_cols) < 8:
        raise ValueError(f"Need at least 8 boolean columns, found {len(bool_cols)}.")
    target_cols = bool_cols[-8:]
    Y_df = df[target_cols].copy()

    # Features: numeric only, drop Date & target cols
    X_df = df.drop(columns=[c for c in ["Date"] if c in df.columns]).select_dtypes(include=["number"]).copy()
    X_df = X_df.drop(columns=[c for c in target_cols if c in X_df.columns], errors="ignore")

    # Align / drop NaNs created by shifts
    keep_idx = X_df.dropna().index.intersection(Y_df.dropna().index)
    X_df = X_df.loc[keep_idx].reset_index(drop=True)
    Y_df = Y_df.loc[keep_idx].reset_index(drop=True)

    return X_df, Y_df


def time_series_split(
    X: np.ndarray,
    Y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def make_loaders(
    X_train, Y_train, X_val, Y_val, X_test, Y_test,
    batch_size: int = 1024, num_workers: int = 0
):
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    val_ds   = TensorDataset(torch.from_numpy(X_val).float(),   torch.from_numpy(Y_val).float())
    test_ds  = TensorDataset(torch.from_numpy(X_test).float(),  torch.from_numpy(Y_test).float())

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl


# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 8, hidden: List[int] = [1024, 512, 256, 128], dropout: float = 0.25):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(h))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # BCEWithLogitsLoss expects raw logits


# ---------------------------
# Training / Eval
# ---------------------------
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_probs = []
    all_true = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_true.append(yb.detach().cpu().numpy())

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

    avg_loss = total_loss / max(n, 1)
    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    # AUROC (macro over labels); handle rare all-constant targets
    try:
        auc = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        auc = float("nan")

    return avg_loss, auc, y_true, y_prob


def early_stopping(best_score, current_score, patience_left):
    if np.isnan(current_score):
        return best_score, patience_left - 1, False
    if current_score > best_score:
        return current_score, patience_left, True
    return best_score, patience_left - 1, False


# ---------------------------
# Orchestrator
# ---------------------------
def run_forecast(
    ticker: str,
    amount_of_days: int,
    frequency: str = "1d",
    add_day_of_week: bool = True,
    epochs: int = 250,
    per_gpu_batch_size: int = 1024,
    hidden_units: Optional[List[int]] = None,
    dropout: float = 0.25,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    num_workers: int = 0,
):
    if hidden_units is None:
        hidden_units = [1024, 512, 256, 128]

    os.makedirs("cache", exist_ok=True)

    # Devices / multi-GPU
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if has_cuda else "cpu")
    num_gpus = torch.cuda.device_count() if has_cuda else 0
    print(f"GPUs detected: {num_gpus}")
    global_batch = per_gpu_batch_size * max(1, num_gpus)
    print(f"Global batch size: {global_batch} (per-GPU: {per_gpu_batch_size})")

    # Data
    df = load_feature_data(ticker, amount_of_days, frequency, add_day_of_week=add_day_of_week)
    X_df, Y_df = select_features_and_targets(df)
    feature_names = list(X_df.columns)
    target_names = list(Y_df.columns)

    X_np = X_df.values.astype(np.float32)
    Y_np = Y_df.values.astype(np.float32)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = time_series_split(X_np, Y_np, train_ratio=0.7, val_ratio=0.15)

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler
    scaler_path = os.path.join("cache", f"{ticker}_{amount_of_days}_{frequency}_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Dataloaders
    train_dl, val_dl, test_dl = make_loaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        batch_size=global_batch, num_workers=num_workers
    )

    # Model
    model = MLP(input_dim=X_train.shape[1], output_dim=Y_train.shape[1], hidden=hidden_units, dropout=dropout)
    if num_gpus > 1:
        model = nn.DataParallel(model)  # simple multi-GPU data parallel
    model = model.to(device)

    # Loss / Optim
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP
    grad_scaler = amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # Training loop with early stopping on val AUROC
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc = -np.inf
    best_state = None
    patience = 15

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_dl, criterion, optimizer, device, grad_scaler)
        val_loss, val_auc, _, _ = eval_epoch(model, val_dl, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  val_auc={val_auc:.5f}")

        best_auc, patience, improved = early_stopping(best_auc, val_auc, patience)
        if improved:
            best_state = {k: v.cpu() if hasattr(v, "device") else v for k, v in model.state_dict().items()}
        if patience <= 0:
            print("Early stopping.")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save history
    pd.DataFrame(history).to_csv(os.path.join("cache", "training_history.csv"), index=False)

    # Evaluate on test
    _, _, y_true, y_prob = eval_epoch(model, test_dl, criterion, device)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n=== Test Classification Report (threshold=0.5) ===")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # Save model
    model_path = os.path.join("cache", f"{ticker}_{amount_of_days}_{frequency}_multilabel.pt")
    torch.save(model.state_dict(), model_path)

    print("\nArtifacts saved in ./cache:")
    print(f"- Model: {model_path}")
    print(f"- Scaler: {scaler_path}")
    print(f"- History: training_history.csv")
    print(f"- Inputs: {len(feature_names)} features, Targets: {len(target_names)} labels")


import re

def _strip_module_prefix(state_dict):
    """Remove 'module.' prefix from DataParallel checkpoints if present."""
    return {re.sub(r'^module\.', '', k): v for k, v in state_dict.items()}

@torch.no_grad()
def predict_latest(
    ticker: str,
    amount_of_days: int,
    frequency: str = "1d",
    add_day_of_week: bool = True,
    per_gpu_batch_size: int = 1024,
    hidden_units: Optional[List[int]] = None,
    dropout: float = 0.25,
    threshold: float = 0.5
):
    """
    Loads artifacts (scaler + model) and predicts the 8-label outcomes
    for the LAST available row in the engineered dataframe.
    Returns: (probs_dict, preds_dict, feature_row_index)
    """
    if hidden_units is None:
        hidden_units = [1024, 512, 256, 128]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) Rebuild engineered data to know the correct columns
    df = load_feature_data(ticker, amount_of_days, frequency, add_day_of_week=add_day_of_week)
    X_df, Y_df = select_features_and_targets(df)  # Y_df just for label names
    feature_names = list(X_df.columns)
    target_names = list(Y_df.columns)

    # Keep only rows with valid features (as training did)
    X_valid = X_df.dropna().copy()
    if X_valid.empty:
        raise RuntimeError("No valid (non-NaN) feature rows available for prediction.")
    last_idx = X_valid.index[-1]
    x_row = X_valid.loc[last_idx:last_idx].values.astype(np.float32)  # shape (1, D)

    # 2) Load scaler
    scaler_path = os.path.join("cache", f"{ticker}_{amount_of_days}_{frequency}_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Train first.")
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    x_row_std = scaler.transform(x_row)

    # 3) Load model
    model_path = os.path.join("cache", f"{ticker}_{amount_of_days}_{frequency}_multilabel.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")

    model = MLP(
        input_dim=x_row_std.shape[1],
        output_dim=len(target_names),
        hidden=hidden_units,
        dropout=dropout
    ).to(device)

    state = torch.load(model_path, map_location=device)
    state = _strip_module_prefix(state)
    model.load_state_dict(state)
    model.eval()

    # 4) Predict (AMP, new API)
    xb = torch.from_numpy(x_row_std).to(device)
    with amp.autocast("cuda", enabled=torch.cuda.is_available()):
        logits = model(xb)
        probs = torch.sigmoid(logits).float().cpu().numpy()[0]  # shape (8,)

    preds = (probs >= threshold).astype(int)

    probs_dict = {name: float(p) for name, p in zip(target_names, probs)}
    preds_dict = {name: int(v) for name, v in zip(target_names, preds)}

    return probs_dict, preds_dict, int(last_idx)


def execute_prediction(ticker, amount_of_days: int, frequency: str = "1d"):
    # forcast models
    run_forecast(
        ticker=ticker,
        amount_of_days=amount_of_days,
        frequency=frequency,
        add_day_of_week=True,
        epochs=120,
        per_gpu_batch_size=512,     # scaled by number of GPUs detected
        hidden_units=[1024, 512, 256, 128],
        dropout=0.05,
        lr=1e-3,
        weight_decay=1e-5,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 0
    )
    # Predict
    probs, preds, idx = predict_latest(
        ticker=ticker,
        amount_of_days=amount_of_days,
        frequency=frequency,
        add_day_of_week=True,
        hidden_units=[1024, 512, 256, 128],
        dropout=0.25,
        threshold=0.80
    )

    print(f"\nPredictions for last row index {idx}:")
    print("Probabilities:")
    for k, v in probs.items():
        print(f"  {k}: {v:.4f}")
    print("Binary (>=0.5):")
    for k, v in preds.items():
        print(f"  {k}: {v}")


# -----------------------------
# CLI entrypoint example
# -----------------------------
if __name__ == "__main__":
    ticker = "SPY"
    amount_of_days = 50000
    frequency = '1d'

    execute_prediction(ticker, amount_of_days, frequency)


