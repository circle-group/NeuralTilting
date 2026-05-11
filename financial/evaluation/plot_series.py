"""Full time-series plot of model predictions vs actual returns.

Each rolling window produces a short forecast.  This script stitches them
into a continuous picture covering the whole evaluation period.

Usage (locally after sync):
    # Single model
    python financial/evaluation/plot_series.py \
        --ts-name nvda_ts_a1.50_obsstd0.05_train30d_fc2d

    # Two-model comparison
    python financial/evaluation/plot_series.py \
        --ts-name   nvda_ts_a1.50_obsstd0.05_train30d_fc2d \
        --gauss-name nvda_gaussian_obsstd0.05_train30d_fc2d

Output is saved to financial/results/analysis/.
"""

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from financial.evaluation.metrics import compute_path_weights, _weighted_quantile

RESULTS_DIR = Path(__file__).parent.parent / 'results'
PREPARED_DIR = Path(__file__).parent.parent / 'data' / 'prepared'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_series_origin(config_name: str) -> pd.Timestamp:
    """Read the series start timestamp saved by prepare_windows.py."""
    metadata_path = PREPARED_DIR / config_name / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {metadata_path}.\n"
            f"Re-run prepare_windows.py to regenerate it."
        )
    with open(metadata_path) as f:
        meta = json.load(f)
    return pd.Timestamp(meta['series_start_iso'])


def load_window_result(window_dir: Path) -> dict | None:
    """Load metrics and forecast samples for one window. Returns None if incomplete."""
    metrics_path = window_dir / 'metrics.json'
    samples_path = window_dir / 'forecast_samples.pkl'

    if not metrics_path.exists() or not samples_path.exists():
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    # forecast_obs must be present for plotting; skip windows from old runs
    if 'forecast_obs' not in metrics or 'forecast_times' not in metrics:
        return None

    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)  # (K, H, D)

    return {'metrics': metrics, 'samples': np.array(samples)}


def load_all_results(config_name: str) -> list[dict]:
    run_dir = RESULTS_DIR / config_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {run_dir}")

    results = []
    for window_dir in sorted(run_dir.glob('window_*')):
        r = load_window_result(window_dir)
        if r is not None:
            results.append(r)

    results.sort(key=lambda r: r['metrics']['window_idx'])
    print(f"  {config_name}: {len(results)} windows loaded")
    return results


# ---------------------------------------------------------------------------
# Series stitching
# ---------------------------------------------------------------------------

def stitch(results: list[dict], series_origin: pd.Timestamp) -> dict:
    """Combine per-window forecasts into arrays aligned on real calendar dates.

    All values are denormalized to original log-return scale:
        original = normalized * norm_sigma + norm_mu

    Returns a dict with keys:
        dates    : list of pd.Timestamp, length N
        obs      : np.ndarray (N,)   actual log-returns
        median   : np.ndarray (N,)   median forecast
        lo50     : np.ndarray (N,)   25th percentile
        hi50     : np.ndarray (N,)   75th percentile
        lo90     : np.ndarray (N,)   5th percentile
        hi90     : np.ndarray (N,)   95th percentile
        crps     : np.ndarray (N,)   per-step CRPS (normalized scale)
        win_boundaries : list of pd.Timestamp  start of each window (for gridlines)
    """
    dates, obs, median = [], [], []
    lo50, hi50, lo90, hi90, crps_vals = [], [], [], [], []
    win_boundaries = []

    for r in results:
        m = r['metrics']
        samples = r['samples']      # (K, H, D)

        norm_mu    = np.array(m['norm_mu'])    # (D,)
        norm_sigma = np.array(m['norm_sigma'])  # (D,)

        t_scaled = np.array(m['forecast_times'])   # (H,) in scaled model time
        obs_norm = np.array(m['forecast_obs'])     # (H, D)  normalized
        time_scale = m.get('time_scale', 1.0)

        # Convert scaled model time back to calendar days for global positioning
        global_days = m['window_start_day'] + t_scaled / time_scale  # (H,)
        step_dates  = [series_origin + pd.Timedelta(days=float(d)) for d in global_days]

        # Denormalize
        obs_orig = obs_norm * norm_sigma + norm_mu                      # (H, D)
        samp_orig = samples * norm_sigma[None, None, :] + norm_mu[None, None, :]  # (K, H, D)

        # For D=1 squeeze to 1-D
        if obs_orig.shape[1] == 1:
            obs_orig  = obs_orig[:, 0]          # (H,)
            samp_orig = samp_orig[:, :, 0]      # (K, H)
        else:
            # Multi-dimensional: plot first dimension only
            obs_orig  = obs_orig[:, 0]
            samp_orig = samp_orig[:, :, 0]

        # Per-step CRPS from metrics (stored as list, normalized scale)
        crps_step = np.array(m.get('crps_per_step', [np.nan] * len(global_days)))

        # Importance weights — downweight numerically diverged paths so that
        # a single blown-up Gaussian sample doesn't dominate the plot.
        M0 = m.get('path_weight_threshold', 20.0)
        weights = compute_path_weights(
            samp_orig[:, :, None] if samp_orig.ndim == 2 else samp_orig,
            M0=M0,
        )
        K, H = samp_orig.shape
        w_median = np.array([_weighted_quantile(samp_orig[:, h], weights, 0.50) for h in range(H)])
        w_lo50   = np.array([_weighted_quantile(samp_orig[:, h], weights, 0.25) for h in range(H)])
        w_hi50   = np.array([_weighted_quantile(samp_orig[:, h], weights, 0.75) for h in range(H)])
        w_lo90   = np.array([_weighted_quantile(samp_orig[:, h], weights, 0.05) for h in range(H)])
        w_hi90   = np.array([_weighted_quantile(samp_orig[:, h], weights, 0.95) for h in range(H)])

        win_boundaries.append(step_dates[0])
        dates.extend(step_dates)
        obs.extend(obs_orig.tolist())
        median.extend(w_median.tolist())
        lo50.extend(w_lo50.tolist())
        hi50.extend(w_hi50.tolist())
        lo90.extend(w_lo90.tolist())
        hi90.extend(w_hi90.tolist())
        crps_vals.extend(crps_step.tolist())

    return {
        'dates':          dates,
        'obs':            np.array(obs),
        'median':         np.array(median),
        'lo50':           np.array(lo50),
        'hi50':           np.array(hi50),
        'lo90':           np.array(lo90),
        'hi90':           np.array(hi90),
        'crps':           np.array(crps_vals),
        'win_boundaries': win_boundaries,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_panel(ax, series: dict, label: str, color: str):
    """Draw one model's predictions onto ax."""
    dates = series['dates']

    ax.fill_between(dates, series['lo90'], series['hi90'],
                    alpha=0.18, color=color, linewidth=0, label='90% interval')
    ax.fill_between(dates, series['lo50'], series['hi50'],
                    alpha=0.35, color=color, linewidth=0, label='50% interval')
    ax.plot(dates, series['median'],
            color=color, linewidth=0.8, alpha=0.9, label=f'{label} median')
    ax.plot(dates, series['obs'],
            color='black', linewidth=0.6, alpha=0.55, label='Actual returns')


def _format_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')


def plot_single(series: dict, config_name: str, output_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={'height_ratios': [3, 1]},
                             sharex=True)

    ax_ret, ax_crps = axes

    # --- Returns panel ---
    _draw_panel(ax_ret, series, label='Forecast', color='steelblue')
    ax_ret.set_ylabel('Log-return (original scale)')
    ax_ret.set_title(config_name)
    ax_ret.legend(loc='upper left', fontsize=8, framealpha=0.7)
    ax_ret.grid(axis='y', alpha=0.3)

    # --- CRPS panel ---
    ax_crps.plot(series['dates'], series['crps'],
                 color='steelblue', linewidth=0.8, alpha=0.8)
    ax_crps.set_ylabel('CRPS')
    ax_crps.set_xlabel('Date')
    ax_crps.grid(axis='y', alpha=0.3)

    _format_date_axis(ax_crps)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_comparison(ts_series: dict, g_series: dict,
                    ts_name: str, g_name: str, output_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [3, 3, 1.5]},
                             sharex=True)

    ax_ts, ax_g, ax_crps = axes

    _draw_panel(ax_ts, ts_series, label='Tilted Stable', color='steelblue')
    ax_ts.set_ylabel('Log-return')
    ax_ts.set_title(f'Tilted Stable  ({ts_name})')
    ax_ts.legend(loc='upper left', fontsize=8, framealpha=0.7)
    ax_ts.grid(axis='y', alpha=0.3)

    _draw_panel(ax_g, g_series, label='Gaussian', color='tomato')
    ax_g.set_ylabel('Log-return')
    ax_g.set_title(f'Gaussian  ({g_name})')
    ax_g.legend(loc='upper left', fontsize=8, framealpha=0.7)
    ax_g.grid(axis='y', alpha=0.3)

    # CRPS comparison — interpolate to shared dates for a fair line plot
    ax_crps.plot(ts_series['dates'], ts_series['crps'],
                 color='steelblue', linewidth=0.8, alpha=0.8, label='Tilted Stable')
    ax_crps.plot(g_series['dates'], g_series['crps'],
                 color='tomato', linewidth=0.8, alpha=0.8, label='Gaussian')
    ax_crps.set_ylabel('CRPS')
    ax_crps.set_xlabel('Date')
    ax_crps.legend(loc='upper left', fontsize=8, framealpha=0.7)
    ax_crps.grid(axis='y', alpha=0.3)

    _format_date_axis(ax_crps)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts-name', required=True,
                        help='Config name for the tilted stable run (used for metadata + output dir)')
    parser.add_argument('--ts-result-name', default=None,
                        help='Override results directory for TS (e.g. ts_name_v2). '
                             'Defaults to --ts-name.')
    parser.add_argument('--gauss-name', default=None,
                        help='Config name for the Gaussian run (optional)')
    args = parser.parse_args()

    ts_result_name = args.ts_result_name if args.ts_result_name else args.ts_name

    print("Loading tilted stable results...")
    ts_origin  = load_series_origin(args.ts_name)
    ts_results = load_all_results(ts_result_name)
    ts_series  = stitch(ts_results, ts_origin)

    if args.gauss_name:
        out_dir = RESULTS_DIR / 'analysis' / f'{ts_result_name}_vs_{args.gauss_name}'
        out_dir.mkdir(parents=True, exist_ok=True)

        print("Loading Gaussian results...")
        g_origin  = load_series_origin(args.gauss_name)
        g_results = load_all_results(args.gauss_name)
        g_series  = stitch(g_results, g_origin)

        plot_comparison(
            ts_series, g_series,
            ts_result_name, args.gauss_name,
            out_dir / 'series_comparison.png',
        )
    else:
        out_dir = RESULTS_DIR / 'analysis' / ts_result_name
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_single(ts_series, args.ts_name, out_dir / 'series_ts.png')


if __name__ == '__main__':
    main()
