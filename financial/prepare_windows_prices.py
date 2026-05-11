"""Pre-generate rolling window pkl files using cumulative log-prices as observations.

Instead of per-step log-returns, each observation is the cumulative log-return
from the window-start price: log(P_t / P_{window_start}).  The SDE starts at
x_0 = 0 (= log(P_0/P_0)) and observes the price PATH rather than increments.

Usage (run once locally before syncing):
    python financial/prepare_windows_prices.py \\
        --config financial/configs/nvda_tilted_stable_alpha1.9_sig4.0_obsstd0.10_d2w32.yaml

Output config name: same as prepare_windows.py output with '_prices' appended.
Output directory  : financial/data/prepared/{config_name}_prices/

The generated window pkl has the same keys as prepare_windows.py so run_window.py
and all model/training code work without any modification.  Only the content of
train_returns / forecast_returns differs (cumulative log-prices instead of
per-step returns).  An extra field 'price_at_window_start' (shape (D,)) is added
for use by stitch_prices.py to reconstruct absolute prices.

Normalisation:
    norm_mu  = 0  (cumulative series starts at 0 by construction)
    norm_sigma = std(cumulative_log_returns_in_training_window)
"""

import sys
import argparse
import json
import shutil
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from financial.data.download import load_aligned_prices
from financial.data.preprocess import compute_log_returns, timestamps_to_calendar_days
from financial.data.windows import save_window


def make_config_name(cfg: dict) -> str:
    """Same as prepare_windows.py but with '_prices' suffix."""
    tickers = '_'.join(t.lower() for t in sorted(cfg['data']['tickers']))
    model_type = cfg['model_type']
    obs_std = cfg['data']['obs_std']
    train_len = int(cfg['window']['train_length'])
    fc_len = int(cfg['window']['forecast_horizon'])
    sigma = cfg['model']['sigma']

    if 'tilted_stable' in model_type:
        alpha = cfg['model']['alpha']
        model_str = f"ts_a{alpha:.2f}_sig{sigma}"
    elif 'gaussian' in model_type:
        model_str = f"gaussian_sig{sigma}"
    else:
        model_str = model_type.replace('_financial', '')

    name = f"{tickers}_{model_str}_obsstd{obs_std:.2f}_train{train_len}d_fc{fc_len}d"
    variant = cfg.get('variant', '').strip()
    if variant:
        name = f"{name}_{variant}"
    return f"{name}_prices"


def generate_windows_prices(
    times: np.ndarray,
    returns: np.ndarray,
    prices: np.ndarray,
    train_length: float,
    forecast_horizon: float,
    time_scale: float = 1.0,
    min_train_obs: int = 20,
):
    """Generate rolling windows using cumulative log-prices as observations.

    Parameters
    ----------
    times : np.ndarray (T,)
        Calendar days of each return (return_timestamps).
    returns : np.ndarray (T, D)
        Per-step log-returns.
    prices : np.ndarray (T+1, D)
        Raw prices aligned with timestamps (one more row than returns).
    train_length, forecast_horizon : float
        In calendar days.
    time_scale : float
        Same as generate_windows().

    Yields
    ------
    dict  — same keys as generate_windows() plus 'price_at_window_start' (D,).
    """
    t_min = times[0]
    t_max = times[-1]
    D = returns.shape[1]

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

        # Base price: the price just before the first return in the training window.
        # returns[i] = log(prices[i+1] / prices[i]), so prices[i] is the base for returns[i].
        first_ret_idx = int(np.where(train_mask)[0][0])
        P_base = prices[first_ret_idx]  # (D,)

        # Cumulative log-returns from P_base
        raw_train = returns[train_mask]      # (T_train, D)
        raw_forecast = returns[forecast_mask]  # (T_fc, D)

        log_cumret_train = np.cumsum(raw_train, axis=0)  # log(P_t / P_base)
        log_cumret_forecast = (
            log_cumret_train[-1][None, :] + np.cumsum(raw_forecast, axis=0)
        )  # continuation from train end

        # Normalise: divide by std of training cumulative series (no mean subtraction)
        norm_sigma = log_cumret_train.std(axis=0)
        norm_sigma = np.where(norm_sigma < 1e-8, 1.0, norm_sigma)
        norm_mu = np.zeros(D, dtype=np.float32)

        norm_train = log_cumret_train / norm_sigma[None, :]
        norm_forecast = log_cumret_forecast / norm_sigma[None, :]

        # Scaled times relative to window start
        t_train = (times[train_mask] - window_start) * time_scale
        t_fc = (times[forecast_mask] - window_start) * time_scale

        yield {
            'train_times': t_train.astype(np.float32),
            'train_returns': norm_train.astype(np.float32),
            'forecast_times': t_fc.astype(np.float32),
            'forecast_returns': norm_forecast.astype(np.float32),
            'norm_mu': norm_mu,
            'norm_sigma': norm_sigma.astype(np.float32),
            'price_at_window_start': P_base.astype(np.float64),
            'window_start_day': float(window_start),
            'time_scale': float(time_scale),
            'window_idx': window_idx,
        }

        window_start += forecast_horizon
        window_idx += 1


def plot_price_overview(prices, timestamps, tickers, train_length, forecast_horizon, out_path):
    """Price-level overview plot (log scale) with rolling volatility of returns."""
    returns = compute_log_returns(prices)
    if prices.ndim == 1:
        prices = prices[:, None]
    if returns.ndim == 1:
        returns = returns[:, None]
    D = prices.shape[1]

    ts = pd.DatetimeIndex(timestamps).tz_convert('UTC').tz_localize(None)
    ts_arr = ts.to_pydatetime()
    ts_ret = ts_arr[1:]  # return timestamps

    date_fmt = mdates.DateFormatter('%b %Y')
    date_loc = mdates.MonthLocator(interval=2)

    fig, axes = plt.subplots(D + 1, 1, figsize=(16, 4 * (D + 1)))
    if D + 1 == 1:
        axes = [axes]

    for d in range(D):
        ax = axes[d]
        ax.semilogy(ts_arr, prices[:, d], lw=0.8, color='steelblue')
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(date_loc)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Price (log scale)', fontsize=9)
        ax.set_title(f'{tickers[d]} — hourly prices', fontsize=9)
        ax.grid(alpha=0.3)

    ax_vol = axes[D]
    roll_window = 35
    for d in range(D):
        rolling_std = pd.Series(returns[:, d]).rolling(roll_window, min_periods=5).std().values
        ax_vol.plot(ts_ret, rolling_std, lw=0.8, label=tickers[d])
    ax_vol.xaxis.set_major_formatter(date_fmt)
    ax_vol.xaxis.set_major_locator(date_loc)
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
    ax_vol.set_ylabel('σ (35-bar rolling)', fontsize=9)
    ax_vol.set_title('Rolling return volatility', fontsize=9)
    if D > 1:
        ax_vol.legend(fontsize=8)
    ax_vol.grid(alpha=0.3)

    fig.suptitle(
        f'Price overview: {", ".join(tickers)}  '
        f'(train={train_length:.0f}d  forecast={forecast_horizon:.0f}d)',
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Price overview  → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    config_name = make_config_name(cfg)
    print(f"Config name : {config_name}")

    prepared_dir = Path(__file__).parent / 'data' / 'prepared' / config_name
    prepared_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = prepared_dir / 'config.yaml'
    if not config_snapshot_path.exists():
        shutil.copy2(config_path, config_snapshot_path)
        print(f"Config snapshot → {config_snapshot_path}")

    data_cfg = cfg['data']
    tickers = data_cfg['tickers']
    D = len(tickers)

    # Download raw prices
    print(f"Downloading {tickers} ({data_cfg['interval']}) "
          f"from {data_cfg['start_date']} to {data_cfg['end_date']}...")
    timestamps, prices = load_aligned_prices(
        tickers, data_cfg['start_date'], data_cfg['end_date'], data_cfg['interval']
    )
    # prices: (T, D)

    returns = compute_log_returns(prices)           # (T-1, D)
    return_timestamps = timestamps[1:]
    times = timestamps_to_calendar_days(return_timestamps)

    print(f"Loaded {len(times)} return observations for {tickers}")
    print(f"Time range: {return_timestamps[0]} → {return_timestamps[-1]}")

    # Price overview plot
    data_key = '_'.join(t.lower() for t in sorted(tickers))
    data_key += (f"_{data_cfg['start_date']}_{data_cfg['end_date']}"
                 f"_{data_cfg['interval']}_prices")
    overview_path = (Path(__file__).parent / 'data' / 'prepared'
                     / f"{data_key}_overview.png")
    plot_price_overview(
        prices=prices,
        timestamps=timestamps,
        tickers=tickers,
        train_length=float(cfg['window']['train_length']),
        forecast_horizon=float(cfg['window']['forecast_horizon']),
        out_path=overview_path,
    )

    # Metadata
    metadata_path = prepared_dir / 'metadata.json'
    if not metadata_path.exists():
        metadata = {
            'series_start_iso': return_timestamps[0].isoformat(),
            'series_price_start_iso': timestamps[0].isoformat(),
            'n_observations': len(times),
            'tickers': tickers,
            'series_type': 'prices',
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Series metadata  → {metadata_path}")

    win_cfg = cfg['window']
    train_length = float(win_cfg['train_length'])
    forecast_horizon = float(win_cfg['forecast_horizon'])
    time_scale = float(cfg['data'].get('time_scale', 1.0))
    print(f"Time scale  : {time_scale}  (window spans [0, {train_length * time_scale:.3f}])")

    n_written = 0
    for window in generate_windows_prices(
        times, returns, prices, train_length, forecast_horizon, time_scale=time_scale
    ):
        idx = window['window_idx']
        path = prepared_dir / f"window_{idx:04d}.pkl"

        if path.exists():
            continue

        save_window(window, path)
        n_written += 1

        if n_written % 10 == 0 or n_written == 1:
            print(f"  Written window {idx:04d} "
                  f"(train obs: {len(window['train_times'])}, "
                  f"forecast obs: {len(window['forecast_times'])}, "
                  f"norm_sigma: {window['norm_sigma'][0]:.4f})")

    total = len(list(prepared_dir.glob('window_*.pkl')))
    print(f"\nDone. {total} window files in {prepared_dir}")
    print(f"\nNext step:")
    print(f"  bash sync_to_hpc.sh")
    print(f"  bash financial/submit_financial.sh --config-name {config_name}")


if __name__ == '__main__':
    main()
