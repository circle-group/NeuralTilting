"""Rolling window logic for the financial forecasting experiment."""

import pickle
import numpy as np
from pathlib import Path
from typing import Iterator


def generate_windows(
    times: np.ndarray,
    returns: np.ndarray,
    train_length: float,
    forecast_horizon: float,
    time_scale: float = 1.0,
    min_train_obs: int = 20,
) -> Iterator[dict]:
    """Generate rolling forecast windows over a financial time series.

    The window advances by forecast_horizon at each step, so forecast
    windows are non-overlapping.

    Parameters
    ----------
    times : np.ndarray
        Shape (T,). Calendar days from series start (monotonically increasing).
    returns : np.ndarray
        Shape (T, D). Raw (unnormalised) log-returns.
    train_length : float
        Training window length in days (calendar time).
    forecast_horizon : float
        Forecast horizon in days. Also the stride between windows.
    time_scale : float
        Multiplicative scale applied to relative times before storing.
        E.g. 1/30 maps a 30-day window onto [0, 1], matching the OU
        experiment time scale and keeping the jump-rate approximation
        (tau=0.01, max_jumps=10) well-calibrated.
        window_start_day is always stored in original calendar days.
    min_train_obs : int
        Skip windows with fewer training observations than this.

    Yields
    ------
    dict with keys:
        train_times      : np.ndarray (T_train,)  — scaled time from window start
        train_returns    : np.ndarray (T_train, D) — normalised returns
        forecast_times   : np.ndarray (T_fc,)     — scaled time from window start
        forecast_returns : np.ndarray (T_fc, D)   — normalised with train stats
        norm_mu          : np.ndarray (D,)
        norm_sigma       : np.ndarray (D,)
        window_start_day : float                  — in original calendar days
        time_scale       : float                  — scale factor used
        window_idx       : int
    """
    from financial.data.preprocess import normalise

    t_min = times[0]
    t_max = times[-1]

    window_start = t_min
    window_idx = 0

    while True:
        train_end = window_start + train_length
        forecast_end = train_end + forecast_horizon

        if forecast_end > t_max:
            break

        train_mask = (times >= window_start) & (times < train_end)
        forecast_mask = (times >= train_end) & (times < forecast_end)

        if train_mask.sum() < min_train_obs or forecast_mask.sum() < 1:
            window_start += forecast_horizon
            continue

        raw_train = returns[train_mask]              # (T_train, D)
        raw_forecast = returns[forecast_mask]        # (T_fc, D)

        # Normalise using training window statistics only
        norm_train, mu, sigma = normalise(raw_train)
        norm_forecast, _, _ = normalise(raw_forecast, mu=mu, sigma=sigma)

        # Times relative to window start, scaled for the SDE model.
        # window_start_day is kept in original calendar days for global positioning.
        t_train = (times[train_mask] - window_start) * time_scale   # (T_train,)
        t_fc    = (times[forecast_mask] - window_start) * time_scale # (T_fc,)

        yield {
            'train_times': t_train.astype(np.float32),
            'train_returns': norm_train.astype(np.float32),
            'forecast_times': t_fc.astype(np.float32),
            'forecast_returns': norm_forecast.astype(np.float32),
            'norm_mu': mu.astype(np.float32),
            'norm_sigma': sigma.astype(np.float32),
            'window_start_day': float(window_start),
            'time_scale': float(time_scale),
            'window_idx': window_idx,
        }

        window_start += forecast_horizon
        window_idx += 1


def count_windows(times: np.ndarray, train_length: float, forecast_horizon: float,
                  min_train_obs: int = 20) -> int:
    """Count number of windows without loading returns."""
    dummy = np.zeros((len(times), 1))
    return sum(1 for _ in generate_windows(times, dummy, train_length, forecast_horizon, min_train_obs))


def save_window(window: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(window, f)


def load_window(path: Path) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)
