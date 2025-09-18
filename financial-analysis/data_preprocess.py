import os
import pandas as pd

from data_extract import update_or_load_cache


# -----------------------
# Helpers
# -----------------------

def _cache_path(ticker: str, amount_of_days: int, frequency: str) -> str:
    os.makedirs("cache", exist_ok=True)
    return os.path.join("cache", f"{ticker}_{amount_of_days}_{frequency}.csv")


def _feature_cache_path(ticker: str, amount_of_days: int, frequency: str) -> str:
    os.makedirs("processed", exist_ok=True)
    return os.path.join("processed", f"{ticker}_{amount_of_days}_{frequency}_features.csv")


def _normalize_columns_for_single_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    If columns look like 'AAPL_Open', 'AAPL_Close', etc., strip the '<ticker>_' prefix.
    Leaves 'Date' intact. Operates case-insensitively.
    """
    out = df.copy()
    prefix = f"{ticker}_"
    # Handle both exact and case-insensitive variants
    def strip_prefix(col: str) -> str:
        if col == "Date":
            return col
        if col.startswith(prefix):
            return col[len(prefix):]
        # Case-insensitive check
        if col.lower().startswith(prefix.lower()):
            return col[len(prefix):]
        return col

    out.columns = [strip_prefix(str(c)) for c in out.columns]
    return out


# -----------------------
# Feature engineering
# -----------------------

def add_next_day_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds percent deltas to next trading day for: close, high, low, open, volume.
    Example column names: delta_next_close_pct, delta_next_volume_pct
    """
    out = df.copy()
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        next_col = f"delta_next_{col.lower()}_pct"
        out[next_col] = (out[col].shift(-1) - out[col]) / out[col]
    return out


def add_one_percent_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each delta_next_*_pct, add a boolean if absolute delta >= 1% (0.01).
    Example: delta_next_close_ge_1pct
    """
    out = df.copy()
    delta_cols = [c for c in out.columns if c.startswith("delta_next_") and c.endswith("_pct")]
    for c in delta_cols:
        flag = c.replace("_pct", "_ge_1pct")
        out[flag] = (out[c].abs() >= 0.01)
    return out


def add_future_close_flags(df: pd.DataFrame, horizons=(1, 3, 5, 10)) -> pd.DataFrame:
    """
    For each horizon, add:
      - close_above_{n}d: True if Close at t+n > Close at t
      - close_below_{n}d: True if Close at t+n < Close at t
    """
    out = df.copy()
    for n in horizons:
        fwd = out["Close"].shift(-n)
        out[f"close_above_{n}d"] = fwd > out["Close"]
        out[f"close_below_{n}d"] = fwd < out["Close"]
    return out


# -----------------------
# Orchestration
# -----------------------

def execute(ticker: str, amount_of_days: int, frequency: str = "1d", save_features: bool = True) -> pd.DataFrame:
    """
    Runs the full pipeline:
      1) If the raw cache file doesn't exist, create it via update_or_load_cache; otherwise load it.
      2) Add engineered features (next-day deltas, 1% flags, and future close flags for 1/3/5/10 days).
      3) Save engineered result to 'cache/{ticker}_{days}_{freq}_features.csv' if save_features=True.
      4) Return the engineered DataFrame.
    """
    raw_path = _cache_path(ticker, amount_of_days, frequency)

    if not os.path.exists(raw_path):
        base_df = update_or_load_cache(ticker, amount_of_days, frequency)
    else:
        base_df = pd.read_csv(raw_path, parse_dates=["Date"])
        base_df = base_df.sort_values("Date").reset_index(drop=True)

    # Build features
    feat_df = add_next_day_deltas(base_df)
    feat_df = add_one_percent_flags(feat_df)
    feat_df = add_future_close_flags(feat_df, horizons=(1, 3, 5, 10))

    # Save engineered dataset separately
    if save_features:
        feat_path = _feature_cache_path(ticker, amount_of_days, frequency)
        feat_df.to_csv(feat_path, index=False)

    return feat_df

if __name__ == "__main__":
    # Example
    ticker = "SPY"
    amount_of_days = 5000
    frequency = '1d'
    engineered = execute(ticker=ticker, amount_of_days=amount_of_days, frequency=frequency, save_features=True)
    print(engineered.tail(8))