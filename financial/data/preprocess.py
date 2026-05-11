"""Preprocessing utilities: log-returns, calendar time conversion, normalisation."""

import numpy as np
import pandas as pd


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log-returns from price array.

    Parameters
    ----------
    prices : np.ndarray
        Shape (T, D) or (T,). Prices must be positive.

    Returns
    -------
    np.ndarray
        Shape (T-1, D) or (T-1,). log(p_t / p_{t-1}).
    """
    return np.log(prices[1:] / prices[:-1])


def timestamps_to_calendar_days(timestamps: pd.DatetimeIndex, origin=None) -> np.ndarray:
    """Convert timezone-aware timestamps to float days from an origin.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        Timezone-aware timestamps.
    origin : pd.Timestamp or None
        Reference point (default: first timestamp).

    Returns
    -------
    np.ndarray
        Float array of days elapsed since origin.
    """
    if origin is None:
        origin = timestamps[0]
    # Convert to UTC to avoid DST issues
    ts_utc = timestamps.tz_convert('UTC')
    origin_utc = origin.tz_convert('UTC')
    deltas = ts_utc - origin_utc
    return deltas.total_seconds().values / 86400.0


def normalise(returns: np.ndarray, mu: np.ndarray = None, sigma: np.ndarray = None):
    """Normalise returns to zero mean and unit variance.

    Parameters
    ----------
    returns : np.ndarray
        Shape (T, D).
    mu, sigma : np.ndarray or None
        If provided, apply existing normalisation. Otherwise compute from returns.

    Returns
    -------
    normalised : np.ndarray
        Shape (T, D).
    mu : np.ndarray
        Shape (D,).
    sigma : np.ndarray
        Shape (D,).
    """
    if mu is None:
        mu = returns.mean(axis=0)
    if sigma is None:
        sigma = returns.std(axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (returns - mu) / sigma, mu, sigma


def prepare_series(tickers, start_date, end_date, interval, data_dir=None):
    """Full pipeline: download → log-returns → calendar time.

    Returns
    -------
    times : np.ndarray
        Shape (T,). Calendar days from first return timestamp.
    returns : np.ndarray
        Shape (T, D). Raw (unnormalised) log-returns.
    return_timestamps : pd.DatetimeIndex
        Timezone-aware timestamps corresponding to each return.
    tickers : list
        Same as input, for reference.
    """
    from financial.data.download import load_aligned_prices
    timestamps, prices = load_aligned_prices(tickers, start_date, end_date, interval)

    # Log-returns: T-1 rows
    returns = compute_log_returns(prices)          # (T-1, D)
    return_timestamps = timestamps[1:]             # timestamps aligned with returns

    # Calendar time in days from first return
    times = timestamps_to_calendar_days(return_timestamps)  # (T-1,)

    return times, returns, return_timestamps, tickers
