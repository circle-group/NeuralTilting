"""Recompute metrics.json from stored forecast_samples.pkl (no retraining).

Used to fix metrics affected by bugs in compute_path_weights (e.g. nan
propagation from inf/nan forecast paths). Overwrites metrics.json in-place,
preserving all non-metric fields (window metadata, forecast_obs, etc.).

Usage:
    python financial/evaluation/regen_metrics_from_samples.py \
        --result-name nvda_njsde_fair_d3w256_obsstd0.10_train30d_fc2d_prices \
        [--only-bad]   # only reprocess windows with nan in crps_per_step
"""

import sys
import argparse
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from financial.evaluation.metrics import compute_all_metrics, DENSE_COVERAGE_LEVELS

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def has_bad_crps(metrics: dict) -> bool:
    crps = np.array(metrics.get('crps_per_step', []))
    return bool(np.any(~np.isfinite(crps)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-name', required=True)
    parser.add_argument('--only-bad', action='store_true',
                        help='Only reprocess windows with nan/inf in crps_per_step')
    args = parser.parse_args()

    result_dir = RESULTS_DIR / args.result_name
    window_dirs = sorted(result_dir.glob('window_*'))

    processed = 0
    skipped = 0
    for wd in window_dirs:
        metrics_path = wd / 'metrics.json'
        samples_path = wd / 'forecast_samples.pkl'

        if not metrics_path.exists() or not samples_path.exists():
            skipped += 1
            continue

        with open(metrics_path) as f:
            old_metrics = json.load(f)

        if args.only_bad and not has_bad_crps(old_metrics):
            skipped += 1
            continue

        with open(samples_path, 'rb') as f:
            forecast_samples = pickle.load(f)  # (K, H, D)

        forecast_obs = np.array(old_metrics['forecast_obs'])   # (H, D) or (H,)
        if forecast_obs.ndim == 1:
            forecast_obs = forecast_obs[:, None]
        if forecast_samples.ndim == 2:
            forecast_samples = forecast_samples[:, :, None]

        path_weight_threshold = old_metrics.get('path_weight_threshold', 20.0)
        quantile_levels = DENSE_COVERAGE_LEVELS

        new_metrics = compute_all_metrics(
            forecast_samples, forecast_obs, quantile_levels,
            path_weight_threshold=path_weight_threshold,
        )

        # Preserve window metadata from the original
        for key in ['window_idx', 'n_train_obs', 'n_forecast_obs', 'window_start_day',
                    'time_scale', 'forecast_times', 'forecast_obs', 'norm_mu',
                    'norm_sigma', 'train_loss']:
            if key in old_metrics:
                new_metrics[key] = old_metrics[key]

        with open(metrics_path, 'w') as f:
            json.dump(new_metrics, f, indent=2)

        crps_ok = np.all(np.isfinite(np.array(new_metrics['crps_per_step'])))
        print(f"{wd.name}: CRPS={new_metrics['mean_crps']:.4f}  "
              f"n_diverged={new_metrics['n_diverged_samples']}  "
              f"crps_clean={crps_ok}")
        processed += 1

    print(f"\nDone: {processed} reprocessed, {skipped} skipped.")


if __name__ == '__main__':
    main()
