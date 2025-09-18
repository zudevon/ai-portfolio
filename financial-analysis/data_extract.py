import yfinance as yf
from datetime import datetime, timedelta
import os
import pandas as pd

def get_stock_data(ticker: str, amount_of_days: int, frequency: str = "1d"):
    """
    Download stock data from Yahoo Finance and save to CSV.

    Args:
        ticker (str): Stock ticker symbol (e.g. 'AAPL').
        amount_of_days (int): Number of past days of data to fetch.
        frequency (str): Data frequency (default '1d').
                         Options: '1m','2m','5m','15m','30m','60m','90m',
                                  '1h','1d','5d','1wk','1mo','3mo'
    Returns:
        pandas.DataFrame: Historical stock data.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=amount_of_days)

    data = yf.download(
        tickers=ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=frequency,
    )

    # Reset index to make Date a column
    data.reset_index(inplace=True)

    # Flatten MultiIndex columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            col[0] if col[1] else col[0] for col in data.columns.values
        ]

    return data


def update_or_load_cache(ticker: str, amount_of_days: int, frequency: str = "1d"):
    """
    Load cached file if exists, otherwise download.
    If file exists, append new data if available.
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    filename = f"{ticker}_{amount_of_days}_{frequency}.csv"
    filepath = os.path.join(cache_dir, filename)

    if os.path.exists(filepath):
        # Load existing cache
        cached_df = pd.read_csv(filepath, parse_dates=["Date"])

        # Find last date in cache
        last_date = cached_df["Date"].max()

        # Download new data from last_date + 1
        today = datetime.today().strftime("%Y-%m-%d")
        new_data = yf.download(
            tickers=ticker,
            start=(last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end=today,
            interval=frequency
        )

        if not new_data.empty:
            new_data.reset_index(inplace=True)

            # Flatten MultiIndex if present
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = [
                    col[0] if col[1] else col[0] for col in new_data.columns.values
                ]

            # Append and save updated file
            updated_df = pd.concat([cached_df, new_data], ignore_index=True)
            updated_df.to_csv(filepath, index=False)
            print(f"Cache updated with new data → {filepath}")
            return updated_df
        else:
            print(f"No new data. Using cached file → {filepath}")
            return cached_df

    else:
        # No cache, fetch fresh data
        df = get_stock_data(ticker, amount_of_days, frequency)
        df.to_csv(filepath, index=False)
        print(f"New cache created → {filepath}")
        return df


if __name__ == "__main__":
    ticker = "SPY"
    amount_of_days = 5000
    frequency = "1d"

    df = update_or_load_cache(ticker, amount_of_days, frequency)
    print(df.tail())