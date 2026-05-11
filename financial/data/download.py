"""Download and cache financial time series from Yahoo Finance."""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf


RAW_DIR = Path(__file__).parent / "raw"


def download_ticker(ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker and cache as parquet.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "NVDA"
    start_date, end_date : str
        ISO date strings, e.g. "2024-01-01"
    interval : str
        yfinance interval, e.g. "1h"

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex (timezone-aware) and columns including 'Close'.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{ticker}_{interval}_{start_date}_{end_date}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    print(f"Downloading {ticker} ({interval}) from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval,
                     auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({start_date} to {end_date}, {interval})")

    # Flatten MultiIndex columns if present (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_parquet(cache_path)
    print(f"  Cached {len(df)} rows to {cache_path}")
    return df


def load_aligned_prices(tickers: list, start_date: str, end_date: str, interval: str) -> tuple:
    """Load and align closing prices for multiple tickers on a common time grid.

    Parameters
    ----------
    tickers : list of str
    start_date, end_date : str
    interval : str

    Returns
    -------
    timestamps : pd.DatetimeIndex
        Aligned timestamps (timezone-aware).
    prices : np.ndarray
        Shape (T, D) where D = len(tickers). Column order matches tickers.
    """
    dfs = []
    for ticker in tickers:
        df = download_ticker(ticker, start_date, end_date, interval)
        dfs.append(df[['Close']].rename(columns={'Close': ticker}))

    # Inner join on timestamps — only keep times present in all tickers
    aligned = pd.concat(dfs, axis=1, join='inner')
    aligned = aligned.dropna()

    timestamps = aligned.index
    prices = aligned.values.astype(np.float64)  # (T, D)

    print(f"Aligned data: {len(timestamps)} timestamps, {len(tickers)} tickers")
    return timestamps, prices
