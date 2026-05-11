"""Re-generate posterior-forecast metrics for a single window (called by sbatch).

This is the per-window worker used by run_regen_postfc_window.sh.
For local batch processing, use regen_posterior_forecast.py instead.

Usage
-----
python financial/evaluation/regen_single_window.py \\
    --config-name nvda_ts_a1.50_sig4.0_obsstd0.30_train30d_fc2d \\
    --result-name nvda_ts_a1.50_sig4.0_obsstd0.30_train30d_fc2d \\
    --window-idx 42
"""

import sys
import argparse
import json
import pickle
import yaml
import random as pyrandom
import numpy as np
import jax.random as jrandom
import equinox as eqx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from financial.data.windows import load_window
from financial.training.train_window import build_model_params
from financial.models.tilted_stable_sde import TiltedStableSDEFinancial
from financial.models.gaussian_sde import GaussianSDEFinancial
from financial.forecasting.predict import generate_forecast_samples
from financial.evaluation.metrics import compute_all_metrics, DENSE_COVERAGE_LEVELS

RESULTS_DIR = Path(__file__).parent.parent / 'results'
PREPARED_DIR = Path(__file__).parent.parent / 'data' / 'prepared'

MODEL_CLASSES = {
    'tilted_stable_sde_financial': TiltedStableSDEFinancial,
    'gaussian_sde_financial': GaussianSDEFinancial,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', required=True)
    parser.add_argument('--result-name', required=True,
                        help='Result directory name (may differ from config-name if versioned)')
    parser.add_argument('--window-idx', type=int, required=True)
    args = parser.parse_args()

    config_name = args.config_name
    result_name = args.result_name
    window_idx = args.window_idx

    # Skip if already done
    result_dir = RESULTS_DIR / result_name / f"window_{window_idx:04d}"
    out_path = result_dir / 'metrics_postfc.json'
    if out_path.exists():
        print(f"Window {window_idx:04d} already done (metrics_postfc.json exists), skipping.")
        return

    model_path = result_dir / 'model.eqx'
    if not model_path.exists():
        print(f"Window {window_idx:04d}: model.eqx not found at {model_path}, skipping.")
        return

    # Load config snapshot
    config_path = PREPARED_DIR / config_name / 'config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg['_config_name'] = config_name

    model_type = cfg['model_type']
    model_cls = MODEL_CLASSES[model_type]

    eval_cfg = cfg.get('evaluation', {})
    n_forecast_samples = eval_cfg.get('n_forecast_samples', 1000)
    quantile_levels = eval_cfg.get('quantile_levels', DENSE_COVERAGE_LEVELS)
    path_weight_threshold = eval_cfg.get('path_weight_threshold', 20.0)
    n_latent_steps = cfg['training']['n_latent_steps']

    # Load window data
    window_path = PREPARED_DIR / config_name / f"window_{window_idx:04d}.pkl"
    window = load_window(window_path)

    # Reconstruct model structure and load saved weights
    model_params = build_model_params(cfg, window)
    model_template = model_cls(**model_params)
    model = eqx.tree_deserialise_leaves(str(model_path), model_template)

    key = jrandom.key(pyrandom.randint(0, 2 ** 31))
    forecast_samples = generate_forecast_samples(
        model, window, n_forecast_samples, key,
        n_latent_steps=n_latent_steps,
        use_posterior_for_forecast=True,
    )

    # Save forecast samples
    with open(result_dir / 'forecast_samples_postfc.pkl', 'wb') as f:
        pickle.dump(forecast_samples, f)

    # Compute and save metrics
    forecast_obs = np.array(window['forecast_returns'])
    if forecast_obs.ndim == 1:
        forecast_obs = forecast_obs[:, None]
    if forecast_samples.ndim == 2:
        forecast_samples = forecast_samples[:, :, None]

    metrics = compute_all_metrics(
        forecast_samples, forecast_obs, quantile_levels,
        path_weight_threshold=path_weight_threshold,
    )
    metrics['window_idx'] = window_idx
    metrics['n_train_obs'] = int(len(window['train_times']))
    metrics['n_forecast_obs'] = int(len(window['forecast_times']))
    metrics['window_start_day'] = float(window['window_start_day'])
    metrics['time_scale'] = float(window.get('time_scale', 1.0))
    metrics['forecast_times'] = window['forecast_times'].tolist()
    metrics['forecast_obs'] = forecast_obs.tolist()
    metrics['norm_mu'] = window['norm_mu'].tolist()
    metrics['norm_sigma'] = window['norm_sigma'].tolist()

    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Window {window_idx:04d} done — CRPS={metrics['mean_crps']:.4f}")


if __name__ == '__main__':
    main()
