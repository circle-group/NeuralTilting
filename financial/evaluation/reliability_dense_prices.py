"""Dense reliability diagram for prices experiments.

Identical to reliability_dense.py — loads forecast_samples.pkl and evaluates
coverage in normalised cumulative log-price space.

Usage:
    python financial/evaluation/reliability_dense_prices.py \\
        --ts-name   nvda_ts_a1.90_sig4.0_obsstd0.10_train30d_fc2d_d2w32_prices \\
        --gauss-name nvda_gaussian_sig4.0_obsstd0.10_train30d_fc2d_prices
"""

import sys
import argparse
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from financial.evaluation.metrics import compute_path_weights, _weighted_quantile, DENSE_COVERAGE_LEVELS

RESULTS_DIR = Path(__file__).parent.parent / 'results'
DENSE_LEVELS = DENSE_COVERAGE_LEVELS


def load_windows(config_name: str, samples_suffix: str = '') -> list:
    run_dir = RESULTS_DIR / config_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Results not found: {run_dir}")
    metrics_filename = f'metrics{samples_suffix}.json' if samples_suffix else 'metrics.json'
    samples_filename = (f'forecast_samples{samples_suffix}.pkl'
                        if samples_suffix else 'forecast_samples.pkl')
    records = []
    for window_dir in sorted(run_dir.glob('window_*')):
        mp = window_dir / metrics_filename
        sp = window_dir / samples_filename
        if not mp.exists() or not sp.exists():
            continue
        with open(mp) as f:
            m = json.load(f)
        with open(sp, 'rb') as f:
            samples = np.array(pickle.load(f))
        obs = np.array(m['forecast_obs'])
        M0 = m.get('path_weight_threshold', 20.0)
        records.append({'samples': samples, 'obs': obs, 'M0': M0})
    print(f"  {config_name}: {len(records)} windows loaded")
    return records


def compute_dense_coverage(records: list) -> dict:
    inside = {q: [] for q in DENSE_LEVELS}
    for r in records:
        samples = r['samples']
        obs     = r['obs']
        if obs.ndim == 1:
            obs = obs[:, None]
        if samples.ndim == 2:
            samples = samples[:, :, None]
        _, H, D = samples.shape
        weights = compute_path_weights(samples, M0=r['M0'])
        for q in DENSE_LEVELS:
            lo_p = (1 - q) / 2
            hi_p = (1 + q) / 2
            for h in range(H):
                for d in range(D):
                    lo = _weighted_quantile(samples[:, h, d], weights, lo_p)
                    hi = _weighted_quantile(samples[:, h, d], weights, hi_p)
                    inside[q].append(float(lo <= obs[h, d] <= hi))
    return {q: float(np.mean(vals)) for q, vals in inside.items()}


def plot_reliability_dense(ts_cov: dict, gauss_cov: dict, output_path: Path):
    levels = DENSE_LEVELS
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.plot(levels, [ts_cov[q]    for q in levels], 'o-', color='steelblue',
            linewidth=2, markersize=4, label='Tilted Stable')
    ax.plot(levels, [gauss_cov[q] for q in levels], 's-', color='tomato',
            linewidth=2, markersize=4, label='Gaussian')
    ax.set_xlabel('Nominal coverage level')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Reliability diagram — dense (prices)')
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts-name',       required=True)
    parser.add_argument('--gauss-name',    required=True)
    parser.add_argument('--samples-suffix', default='')
    args = parser.parse_args()

    print("Loading windows...")
    ts_records    = load_windows(args.ts_name,    samples_suffix=args.samples_suffix)
    gauss_records = load_windows(args.gauss_name, samples_suffix=args.samples_suffix)

    print(f"Computing dense coverage at {len(DENSE_LEVELS)} levels...")
    ts_cov    = compute_dense_coverage(ts_records)
    gauss_cov = compute_dense_coverage(gauss_records)

    print(f"\n{'Nominal':>10}  {'TS':>8}  {'Gaussian':>10}")
    print('-' * 34)
    for q in DENSE_LEVELS:
        print(f"{q:10.4f}  {ts_cov[q]:8.4f}  {gauss_cov[q]:10.4f}")

    out_dir = RESULTS_DIR / 'analysis' / f'{args.ts_name}_vs_{args.gauss_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.samples_suffix or ''
    with open(out_dir / f'reliability_dense{suffix}.json', 'w') as f:
        json.dump({'tilted_stable': ts_cov, 'gaussian': gauss_cov,
                   'levels': DENSE_LEVELS}, f, indent=2)

    plot_reliability_dense(ts_cov, gauss_cov,
                           out_dir / f'reliability_diagram_dense{suffix}.png')


if __name__ == '__main__':
    main()
