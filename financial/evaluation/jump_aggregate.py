"""Jump-conditioned evaluation for financial forecasting experiments.

Computes CRPS (D=1) or Energy Score (D>1) restricted to forecast steps where
the realised return magnitude exceeds a percentile threshold, analogous to the
jump-conditioned metrics used in the synthetic OU/DW experiments.

All data is read from per-window metrics.json files — no forecast_samples.pkl
loading required.

Usage:
    python financial/evaluation/jump_aggregate.py \\
        --ts-name   nvda_ts_a1.30_obsstd0.30_train30d_fc2d \\
        --gauss-name nvda_gaussian_obsstd0.30_train30d_fc2d
"""

import sys
import pickle
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from financial.evaluation.metrics import crps as _crps, energy_score as _energy_score

RESULTS_DIR = Path(__file__).parent.parent / 'results'
PERCENTILES = [90, 95, 97.5, 99]
PERCENTILE_KEYS = ['p90', 'p95', 'p97_5', 'p99']


def load_records_unweighted(config_name: str) -> list:
    """Recompute per-step scores from forecast_samples.pkl with uniform weights."""
    run_dir = RESULTS_DIR / config_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Results not found: {run_dir}")

    records = []
    for window_dir in sorted(run_dir.glob('window_*')):
        pkl_path = window_dir / 'forecast_samples.pkl'
        metrics_path = window_dir / 'metrics.json'
        if not pkl_path.exists() or not metrics_path.exists():
            continue

        with open(pkl_path, 'rb') as f:
            samples = np.array(pickle.load(f))         # (K, H, D) or (K, H)
        with open(metrics_path) as f:
            meta = json.load(f)

        obs = np.array(meta['forecast_obs'])            # (H, D) or (H,)
        if obs.ndim == 1:
            obs = obs[:, None]
        if samples.ndim == 2:
            samples = samples[:, :, None]

        D = obs.shape[1]
        if D == 1:
            cs = _crps(samples, obs, weights=None).tolist()
            es = _energy_score(samples, obs, weights=None).tolist()
        else:
            cs = [float('nan')] * samples.shape[1]
            es = _energy_score(samples, obs, weights=None).tolist()

        records.append({
            'crps_per_step': cs,
            'energy_score_per_step': es,
            'forecast_obs': meta['forecast_obs'],
        })

    print(f"  {config_name}: {len(records)} windows (no_weights=True)")
    return records


def load_records(config_name: str, obs_noise: bool = False,
                 metrics_suffix: str = '') -> list:
    """Load all per-window metrics JSONs for a config."""
    run_dir = RESULTS_DIR / config_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Results not found: {run_dir}")
    if metrics_suffix:
        filename = f'metrics{metrics_suffix}.json'
    elif obs_noise:
        filename = 'metrics_obs_noise.json'
    else:
        filename = 'metrics.json'
    records = []
    for window_dir in sorted(run_dir.glob('window_*')):
        metrics_path = window_dir / filename
        if metrics_path.exists():
            with open(metrics_path) as f:
                records.append(json.load(f))
    print(f"  {config_name}: {len(records)} windows loaded (obs_noise={obs_noise})")
    return records


def _score_key(records: list) -> str:
    """Return 'crps_per_step' for D=1, 'energy_score_per_step' for D>1."""
    obs = np.array(records[0]['forecast_obs'])
    D = obs.shape[1] if obs.ndim == 2 else 1
    return 'crps_per_step' if D == 1 else 'energy_score_per_step'


def compute_thresholds(records: list) -> dict:
    """Compute global |return| percentile thresholds from all forecast steps.

    For D>1 uses max(|obs|) across dimensions per step so a step is flagged
    if any dimension has a large move.
    """
    all_magnitudes = []
    for r in records:
        obs = np.array(r['forecast_obs'])           # (H, D) or (H,)
        if obs.ndim == 1:
            obs = obs[:, None]
        all_magnitudes.append(np.max(np.abs(obs), axis=1))  # (H,)
    magnitudes = np.concatenate(all_magnitudes)     # (N_total_steps,)
    return {
        p_key: float(np.percentile(magnitudes, p))
        for p_key, p in zip(PERCENTILE_KEYS, PERCENTILES)
    }


def jump_conditioned_scores(records: list, thresholds: dict, score_key: str) -> dict:
    """For each threshold, compute mean score over flagged steps across all windows.

    Returns dict: percentile_key -> (mean, std, n_steps).
    """
    results = {p_key: [] for p_key in PERCENTILE_KEYS}

    for r in records:
        obs = np.array(r['forecast_obs'])           # (H, D) or (H,)
        if obs.ndim == 1:
            obs = obs[:, None]
        magnitudes = np.max(np.abs(obs), axis=1)    # (H,)
        scores = np.array(r[score_key])             # (H,)

        for p_key, threshold in thresholds.items():
            mask = magnitudes > threshold
            if mask.any():
                results[p_key].extend(scores[mask].tolist())

    summary = {}
    for p_key in PERCENTILE_KEYS:
        vals = results[p_key]
        summary[p_key] = {
            'mean': float(np.mean(vals)) if vals else float('nan'),
            'std':  float(np.std(vals))  if vals else float('nan'),
            'n_steps': len(vals),
        }
    return summary


def print_table(ts_jc: dict, gauss_jc: dict, score_label: str):
    print(f"\n{'='*72}")
    print(f"Jump-conditioned {score_label} (held-out forecast steps)")
    print(f"{'='*72}")
    print(f"{'Threshold':<12} {'TS mean':>12} {'Gauss mean':>12} {'Abs gap':>10} {'n steps (TS)':>14}")
    print(f"{'-'*72}")
    for p_key in PERCENTILE_KEYS:
        ts   = ts_jc[p_key]
        gauss = gauss_jc[p_key]
        gap  = gauss['mean'] - ts['mean']
        print(f"{p_key:<12} "
              f"{ts['mean']:>12.4f} "
              f"{gauss['mean']:>12.4f} "
              f"{gap:>10.4f} "
              f"{ts['n_steps']:>14d}")
    print(f"{'='*72}\n")


def plot_jump_crps(ts_jc: dict, gauss_jc: dict, score_label: str, output_path: Path):
    labels = PERCENTILE_KEYS
    x = np.arange(len(labels))
    width = 0.35

    ts_means   = [ts_jc[k]['mean']   for k in labels]
    ts_stds    = [ts_jc[k]['std']    for k in labels]
    gauss_means = [gauss_jc[k]['mean'] for k in labels]
    gauss_stds  = [gauss_jc[k]['std']  for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, ts_means,    width, yerr=ts_stds,    capsize=4,
           color='steelblue',  alpha=0.85, label='Tilted Stable')
    ax.bar(x + width/2, gauss_means, width, yerr=gauss_stds, capsize=4,
           color='tomato',     alpha=0.85, label='Gaussian')

    ax.set_xticks(x)
    ax.set_xticklabels(['p90', 'p95', 'p97.5', 'p99'])
    ax.set_xlabel('Jump threshold (percentile of |return|)')
    ax.set_ylabel(score_label)
    ax.set_title(f'Jump-conditioned {score_label}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ts-name',    required=True)
    parser.add_argument('--gauss-name', required=True)
    parser.add_argument('--obs-noise', action='store_true',
                        help='Load metrics_obs_noise.json instead of metrics.json')
    parser.add_argument('--no-weights', action='store_true',
                        help='Recompute per-step scores from forecast_samples.pkl with uniform weights')
    parser.add_argument('--metrics-suffix', default='',
                        help='Suffix for metrics filename applied to both models, e.g. _postfc.')
    args = parser.parse_args()

    if args.obs_noise and args.no_weights:
        parser.error('--obs-noise and --no-weights cannot be combined')
    if args.metrics_suffix and (args.obs_noise or args.no_weights):
        parser.error('--metrics-suffix cannot be combined with --obs-noise or --no-weights')

    if args.no_weights:
        print("Loading records (no_weights=True — recomputing from pkl)...")
        ts_records    = load_records_unweighted(args.ts_name)
        gauss_records = load_records_unweighted(args.gauss_name)
    elif args.metrics_suffix:
        print(f"Loading records (metrics_suffix={args.metrics_suffix!r})...")
        ts_records    = load_records(args.ts_name,    metrics_suffix=args.metrics_suffix)
        gauss_records = load_records(args.gauss_name, metrics_suffix=args.metrics_suffix)
    else:
        print(f"Loading records (obs_noise={args.obs_noise})...")
        ts_records    = load_records(args.ts_name,    obs_noise=args.obs_noise)
        gauss_records = load_records(args.gauss_name, obs_noise=args.obs_noise)

    score_key   = _score_key(ts_records)
    score_label = 'CRPS' if score_key == 'crps_per_step' else 'Energy Score'
    print(f"  Score metric: {score_label}")

    # Thresholds computed from TS records (model-agnostic — same observations)
    print("Computing jump thresholds from forecast observations...")
    thresholds = compute_thresholds(ts_records)
    for p_key, thresh in thresholds.items():
        print(f"  {p_key}: |return| > {thresh:.4f}")

    print("Computing jump-conditioned scores...")
    ts_jc    = jump_conditioned_scores(ts_records,    thresholds, score_key)
    gauss_jc = jump_conditioned_scores(gauss_records, thresholds, score_key)

    print_table(ts_jc, gauss_jc, score_label)

    out_dir = RESULTS_DIR / 'analysis' / f'{args.ts_name}_vs_{args.gauss_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = (args.metrics_suffix if args.metrics_suffix
              else '_obs_noise' if args.obs_noise
              else '_no_weights' if args.no_weights
              else '')

    result = {
        'thresholds': thresholds,
        'tilted_stable': ts_jc,
        'gaussian': gauss_jc,
        'score_metric': score_label,
    }
    out_json = out_dir / f'jump_crps{suffix}.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_json}")

    plot_jump_crps(ts_jc, gauss_jc, score_label,
                  out_dir / f'jump_crps_comparison{suffix}.png')


if __name__ == '__main__':
    main()
