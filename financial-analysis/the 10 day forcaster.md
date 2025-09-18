# 📈 Stock Forecasting with PyTorch  

This project builds a **multi-label classifier** to forecast short-term outcomes in stock prices using engineered features derived from Yahoo Finance data.  

---

## 🔍 What the Analysis Does
1. **Data Collection**  
   - Downloads historical OHLCV data (Open, High, Low, Close, Volume) for a given ticker from Yahoo Finance.  
   - Caches the data locally in a `cache/` folder and auto-updates with new rows on future runs.  

2. **Feature Engineering**  
   - Adds **next-day deltas** (percent changes for Open, High, Low, Close, Volume).  
   - Creates **boolean signals** for when these deltas exceed ±1%.  
   - Generates **future close flags**: whether the stock closes higher or lower after 1, 3, 5, and 10 days.  

3. **Model Training (PyTorch)**  
   - Builds a **deep feed-forward neural network (MLP)** with optional dropout and batch normalization.  
   - Trains with **mixed precision (AMP)** to use GPU Tensor Cores efficiently.  
   - Supports **multi-GPU training** (via `DataParallel`).  
   - Uses **binary cross-entropy loss** and tracks metrics like **AUROC** and **F1-score**.  

4. **Evaluation**  
   - Splits data into train, validation, and test sets **by time order** (no shuffling).  
   - Reports precision, recall, F1-score, and support for each boolean target.  
   - Saves artifacts:  
     - `*_multilabel.pt` → trained PyTorch model weights  
     - `*_scaler.pkl` → fitted scaler for preprocessing  
     - `training_history.csv` → logged training metrics  

5. **Prediction**  
   - After training, you can run `predict_latest()` to forecast the **last available trading day**.  
   - Outputs probabilities for each of the 8 boolean targets (e.g., `close_above_3d = 0.78`).  

---

## 📊 Metrics Explained
- **Precision** → Of the cases the model predicted as positive, how many were correct?  
- **Recall** → Of the actual positive cases, how many did the model find?  
- **F1-score** → Balance between precision and recall (harmonic mean).  
- **Support** → The number of actual examples of that class in the dataset.  

---

## ⚡ How to Run
You only need to change **three variables** at the bottom of `forcaster.py`:

```python
# -----------------------------
# CLI entrypoint example
# -----------------------------
if __name__ == "__main__":
    ticker = "SPY"         # Stock symbol
    amount_of_days = 50000 # Historical window (days)
    frequency = "1d"       # Interval ('1d', '1h', etc.)

    execute_prediction(ticker, amount_of_days, frequency)
