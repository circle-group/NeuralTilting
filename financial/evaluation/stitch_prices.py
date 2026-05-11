"""Stitch per-window price forecasts into a full timeline with uncertainty bands.

Loads forecast_samples.pkl from each window, un-normalises from log-price space
back to actual prices, and plots the full price trajectory with predictive
intervals for an arbitrary number of models.

Because forecast windows are non-overlapping (stride = forecast horizon), the
forecast periods tile the full data range with no gaps or overlaps.

Un-normalisation:
    price[k, h] = price_at_window_start * exp(sample[k, h, 0] * norm_sigma)

Note: interval widths will step at window boundaries (each window has its own
norm_sigma).  This is expected — it reflects genuine volatility regime changes,
not an artefact.

Usage (multi-model):
    python financial/evaluation/stitch_prices.py \\
        --model nvda_ts_a1.90_sig1.0_obsstd0.10_train30d_fc2d_d2w32_prices:"Tilted Stable" \\
        --model nvda_gaussian_sig1.0_obsstd0.10_train30d_fc2d_d2w32_prices:Gaussian \\
        --model nvda_dlinear_obsstd0.10_train30d_fc2d_prices:DLinear:nvda_ts_a1.90_sig1.0_obsstd0.10_train30d_fc2d_d2w32_prices \\
        --model nvda_deepar_obsstd0.10_train30d_fc2d_prices:DeepAR:nvda_ts_a1.90_sig1.0_obsstd0.10_train30d_fc2d_d2w32_prices

Usage (legacy — two-model):
    python financial/evaluation/stitch_prices.py \\
        --ts-name   nvda_ts_a1.90_sig1.0_obsstd0.10_train30d_fc2d_d2w32_prices \\
        --gauss-name nvda_gaussian_sig1.0_obsstd0.10_train30d_fc2d_d2w32_prices

The third colon-separated field in --model is an optional prepared-windows
directory name.  Use it when a model reuses another config's prepared windows
(e.g. DLinear and DeepAR reuse the TS prepared directory).
"""

import sys
import argparse
import json
import hashlib
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from financial.data.windows import load_window
from financial.evaluation.metrics import compute_path_weights

RESULTS_DIR  = Path(__file__).parent.parent / 'results'
PREPARED_DIR = Path(__file__).parent.parent / 'data' / 'prepared'

COVERAGE_LEVELS = [0.50, 0.80, 0.95]
PALETTE = ['steelblue', 'tomato', 'forestgreen', 'darkorange', 'mediumpurple', 'saddlebrown']


def _analysis_dir(names: list, max_len: int = 200) -> Path:
    """Return analysis output directory, hashing the name if it would be too long."""
    joined = '_vs_'.join(names)
    if len(joined) <= max_len:
        return RESULTS_DIR / 'analysis' / joined
    h = hashlib.sha1(joined.encode()).hexdigest()[:8]
    short = f"{names[0][:60]}_vs_{len(names)-1}_others_{h}"
    return RESULTS_DIR / 'analysis' / short


def load_forecast_records(config_name: str, prepared_name: str = None) -> list:
    """Load forecast samples + metadata for every completed window.

    Parameters
    ----------
    config_name : str
        Results directory name (financial/results/{config_name}/).
    prepared_name : str, optional
        Prepared-windows directory name. Defaults to config_name.
        Pass explicitly when a model reuses another config's prepared windows
        (e.g. DLinear/DeepAR reuse the TS prepared directory).
    """
    run_dir  = RESULTS_DIR / config_name
    prep_dir = PREPARED_DIR / (prepared_name or config_name)
    if not run_dir.exists():
        raise FileNotFoundError(f"Results not found: {run_dir}")
    if not prep_dir.exists():
        raise FileNotFoundError(f"Prepared data not found: {prep_dir}")

    with open(prep_dir / 'metadata.json') as f:
        meta = json.load(f)
    series_start = pd.Timestamp(meta['series_start_iso'])

    records = []
    for window_dir in sorted(run_dir.glob('window_*')):
        metrics_path = window_dir / 'metrics.json'
        samples_path = window_dir / 'forecast_samples.pkl'
        if not metrics_path.exists() or not samples_path.exists():
            continue

        with open(metrics_path) as f:
            m = json.load(f)

        with open(samples_path, 'rb') as f:
            samples = np.array(pickle.load(f))  # (K, H) or (K, H, D)
        if samples.ndim == 2:
            samples = samples[:, :, None]       # (K, H, 1)

        window_idx = m['window_idx']
        pkl_path = prep_dir / f"window_{window_idx:04d}.pkl"
        if not pkl_path.exists():
            continue
        window = load_window(pkl_path)

        obs = np.array(m['forecast_obs'])
        if obs.ndim == 1:
            obs = obs[:, None]

        norm_sigma = np.array(window['norm_sigma'])
        P_base     = np.array(window['price_at_window_start'])
        window_start_day      = float(m['window_start_day'])
        time_scale            = float(m.get('time_scale', 1.0))
        forecast_times_scaled = np.array(m['forecast_times'])

        forecast_days = window_start_day + forecast_times_scaled / time_scale
        forecast_timestamps = [
            series_start + pd.Timedelta(days=float(d))
            for d in forecast_days
        ]

        path_weight_threshold = m.get('path_weight_threshold', 20.0)
        weights = compute_path_weights(samples, M0=path_weight_threshold)

        D = samples.shape[2]
        actual_prices = np.array([
            P_base[d] * np.exp(obs[:, d] * norm_sigma[d])
            for d in range(D)
        ]).T  # (H, D)

        sample_prices = np.array([
            P_base[d] * np.exp(samples[:, :, d] * norm_sigma[d])
            for d in range(D)
        ])  # (D, K, H)
        sample_prices = np.moveaxis(sample_prices, 0, -1)  # (K, H, D)

        records.append({
            'timestamps':    forecast_timestamps,
            'actual_prices': actual_prices,   # (H, D)
            'sample_prices': sample_prices,   # (K, H, D)
            'weights':       weights,         # (K,)
        })

    print(f"  {config_name}: {len(records)} windows loaded")
    return records, series_start


def weighted_quantile_at_steps(sample_prices, weights, q):
    """Compute weighted quantile q over samples at each (h, d) step."""
    K, H, D = sample_prices.shape
    result = np.zeros((H, D))
    w = weights / weights.sum()
    for h in range(H):
        for d in range(D):
            vals = sample_prices[:, h, d]
            idx  = np.argsort(vals)
            cdf  = np.cumsum(w[idx])
            result[h, d] = vals[idx[np.searchsorted(cdf, q)]]
    return result  # (H, D)


def stitch(records: list, d: int = 0):
    """Collect stitched timeline for dimension d."""
    all_ts, all_actual, all_q = [], [], {q: [] for q in COVERAGE_LEVELS}
    for r in records:
        all_ts.extend(r['timestamps'])
        all_actual.append(r['actual_prices'][:, d])
        for q in COVERAGE_LEVELS:
            lo = weighted_quantile_at_steps(r['sample_prices'], r['weights'], (1 - q) / 2)[:, d]
            hi = weighted_quantile_at_steps(r['sample_prices'], r['weights'], (1 + q) / 2)[:, d]
            all_q[q].append((lo, hi))
    return all_ts, np.concatenate(all_actual), {
        q: (np.concatenate([x[0] for x in v]),
            np.concatenate([x[1] for x in v]))
        for q, v in all_q.items()
    }


def _parse_models(args) -> list:
    """Build ordered list of (result_name, display_label, prepared_name) from parsed args."""
    models = []
    if args.ts_name:
        models.append((args.ts_name, 'Tilted Stable', args.ts_prepared_name))
    if args.gauss_name:
        models.append((args.gauss_name, 'Gaussian', args.gauss_prepared_name))
    for spec in (args.model or []):
        parts = spec.split(':', 2)
        name     = parts[0]
        label    = parts[1] if len(parts) > 1 else name
        prepared = parts[2] if len(parts) > 2 else None
        models.append((name, label, prepared))
    return models


def main():
    parser = argparse.ArgumentParser(
        description='Stitch per-window price forecasts for N models.')
    # Legacy args
    parser.add_argument('--ts-name',             default='',
                        help='(legacy) result name for Tilted Stable')
    parser.add_argument('--gauss-name',          default='',
                        help='(legacy) result name for Gaussian')
    parser.add_argument('--ts-prepared-name',    default=None)
    parser.add_argument('--gauss-prepared-name', default=None)
    # Multi-model
    parser.add_argument('--model', action='append', default=[],
                        metavar='NAME[:LABEL[:PREPARED]]',
                        help='Add a model. Third colon field overrides the '
                             'prepared-windows directory (for models that reuse '
                             'another config\'s prepared data). May be repeated.')
    parser.add_argument('--ticker-idx', type=int, default=0)
    args = parser.parse_args()

    models = _parse_models(args)
    if len(models) < 1:
        parser.error('At least one model is required.')

    d = args.ticker_idx

    print("Loading records...")
    all_data = []
    series_start = None
    for name, label, prepared in models:
        records, ss = load_forecast_records(name, prepared)
        if series_start is None:
            series_start = ss
        all_data.append((label, records))

    print("Stitching timelines...")
    stitched = []
    for label, records in all_data:
        ts, actual, q_intervals = stitch(records, d)
        stitched.append((label, ts, actual, q_intervals))

    # Y-axis bounds from the first model's actual prices
    _, _, first_actual, _ = stitched[0]
    y_min = float(np.nanmin(first_actual)) * 0.7
    y_max = float(np.nanmax(first_actual)) * 1.5

    n = len(stitched)
    fig, axes = plt.subplots(n, 1, figsize=(18, 5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    date_fmt = mdates.DateFormatter('%b %Y')
    date_loc = mdates.MonthLocator(interval=2)

    for i, (ax, (label, ts_list, actual, q_intervals)) in enumerate(zip(axes, stitched)):
        color = PALETTE[i % len(PALETTE)]
        alphas = [0.15, 0.25, 0.40]
        sorted_levels = sorted(COVERAGE_LEVELS, reverse=True)
        for lvl, alpha in zip(sorted_levels, alphas):
            lo, hi = q_intervals[lvl]
            lo_c = np.clip(lo, y_min, y_max)
            hi_c = np.clip(hi, y_min, y_max)
            ax.fill_between(ts_list, lo_c, hi_c, alpha=alpha, color=color,
                            label=f'{lvl:.0%} interval' if lvl == sorted_levels[0] else '')
        ax.plot(ts_list, actual, 'k-', lw=0.8, label='Actual price', alpha=0.9)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('Price ($)', fontsize=9)
        ax.set_title(f'{label} — price forecast with predictive intervals', fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(date_loc)

    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    labels_str = ' vs '.join(label for label, *_ in stitched)
    fig.suptitle(
        f'Price forecasts — {labels_str}  (30d train → 2d forecast)',
        fontsize=11,
    )
    fig.tight_layout()

    model_names = [name for name, _, _ in models]
    out_dir = _analysis_dir(model_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'price_forecast_stitched.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
