# Energy Consumption Anomaly Detection (LSTM Autoencoder)

End-to-end project that detects anomalies in **Individual Household Electric Power Consumption** time series using an **LSTM autoencoder**.

## Dataset

We use the UCI **Individual Household Electric Power Consumption** dataset (minute-level data from a French household, Dec 2006–Nov 2010; 2,075,259 rows, 9 variables).

This project includes a downloader that fetches the official zip from UCI and extracts `household_power_consumption.txt` into `data/raw/`.

## Project Layout

```
energy-anomaly/
├── config.yaml
├── requirements.txt
├── data/
│   ├── raw/          # downloaded .zip and extracted .txt
│   ├── processed/    # parquet files after preprocessing (resampling, imputation)
│   └── interim/
├── models/           # trained Keras model & threshold.json
├── reports/          # anomaly_scores.csv and plots
└── src/
    ├── data/
    │   ├── download.py
    │   └── preprocess.py
    ├── models/
    │   └── lstm_autoencoder.py
    ├── train.py
    ├── detect.py
    ├── utils.py
    └── main.py
```

## Quickstart

```bash
# 1) Create & activate a virtual environment (example using Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Download raw data (from UCI) and extract
python -m src.main download

# 4) Preprocess (parse datetime, coerce numerics, impute, resample to 5-min)
python -m src.main preprocess

# 5) Train LSTM autoencoder (unsupervised)
python -m src.main train

# 6) Score anomalies and produce a CSV + PNG plot
python -m src.main detect
```

## How it works

- **Preprocessing**: Parses `Date` + `Time`, coerces numeric columns (handles `?` or blanks), forward-fills then drops remaining NaNs, and **resamples to 5-minute** means by default (configurable in `config.yaml`).  
- **Model**: Multivariate **LSTM autoencoder** trained to reconstruct normal sequences.  
- **Thresholding**: Anomaly score = MSE reconstruction error per window. Threshold is set from the **99.5th percentile** of training errors (configurable: quantile or MAD).  
- **Outputs**: `reports/anomaly_scores.csv` with timestamps, errors, and boolean `is_anomaly`; plus `reports/anomalies.png`.

## Configuration

Key knobs in `config.yaml`:
- `data.resample_rule`: e.g., `1T` (1-min), `5T`, `15T`  
- `model.seq_len`: sequence length (e.g., 60 steps × 5-min = 5 hours per sample)  
- `train.epochs`, `train.batch_size`  
- `threshold.method`: `quantile` or `mad`  

## Notes

- The dataset has ≈1.25% missing rows in the raw file; we handle this with fill-forward + drop.  
- There are **no ground-truth anomaly labels** in this dataset; the approach is **unsupervised** and flags unusual behavior via reconstruction error.

## License

Code in this repo is MIT. The dataset is provided by the UCI Machine Learning Repository (see their license/terms).
