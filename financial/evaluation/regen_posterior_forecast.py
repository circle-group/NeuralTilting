"""Re-generate forecast samples using simulate_posterior instead of simulate_prior.

This script loads saved model weights for each window and regenerates forecast
samples using the posterior simulation (tilted/controlled process) rather than
the prior.  Metrics are saved alongside the originals with a ``_postfc`` suffix
so existing results are not overwritten.

Usage
-----
python financial/evaluation/regen_posterior_forecast.py \\
    --ts-name   nvda_ts_a1.50_sig4.0_obsstd0.30_train30d_fc2d \\
    --gauss-name nvda_gaussian_sig4.0_obsstd0.30_train30d_fc2d

Both models are re-evaluated.  Pass --ts-only or --gauss-only to run one model.

Outputs (per window)
--------------------
  financial/results/{name}/window_{idx:04d}/forecast_samples_postfc.pkl
  financial/results/{name}/window_{idx:04d}/metrics_postfc.json
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


def regen_one_model(config_name: str, result_name: str | None = None):
    """Re-generate posterior-forecast metrics for all windows of one model."""
    result_name = result_name or config_name

    config_path = PREPARED_DIR / config_name / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config snapshot not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg['_config_name'] = config_name

    model_type = cfg['model_type']
    model_cls = MODEL_CLASSES[model_type]

    results_root = RESULTS_DIR / result_name
    window_dirs = sorted(results_root.glob('window_*'))
    if not window_dirs:
        print(f"  No window directories found under {results_root}")
        return

    eval_cfg = cfg.get('evaluation', {})
    n_forecast_samples = eval_cfg.get('n_forecast_samples', 1000)
    quantile_levels = eval_cfg.get('quantile_levels', DENSE_COVERAGE_LEVELS)
    path_weight_threshold = eval_cfg.get('path_weight_threshold', 20.0)
    n_latent_steps = cfg['training']['n_latent_steps']

    done = skipped = 0
    for wd in window_dirs:
        model_path = wd / 'model.eqx'
        metrics_postfc_path = wd / 'metrics_postfc.json'

        if metrics_postfc_path.exists():
            skipped += 1
            continue

        if not model_path.exists():
            print(f"  {wd.name}: model.eqx missing, skipping")
            continue

        # Extract window index from directory name (window_NNNN)
        window_idx = int(wd.name.split('_')[1])
        window_path = PREPARED_DIR / config_name / f"window_{window_idx:04d}.pkl"
        if not window_path.exists():
            print(f"  {wd.name}: window pkl missing, skipping")
            continue

        window = load_window(window_path)

        # Reconstruct the model structure, then load saved weights
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
        samples_path = wd / 'forecast_samples_postfc.pkl'
        with open(samples_path, 'wb') as f:
            pickle.dump(forecast_samples, f)

        # Compute metrics
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

        with open(metrics_postfc_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        done += 1
        if done % 20 == 0:
            print(f"  {result_name}: {done} done, {skipped} skipped")

    print(f"  {result_name}: finished — {done} regenerated, {skipped} already done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ts-name', default=None)
    parser.add_argument('--gauss-name', default=None)
    parser.add_argument('--ts-result-name', default=None,
                        help='Versioned result dir for TS (e.g. config_v2); '
                             'defaults to --ts-name')
    parser.add_argument('--gauss-result-name', default=None)
    parser.add_argument('--ts-only', action='store_true')
    parser.add_argument('--gauss-only', action='store_true')
    args = parser.parse_args()

    if not args.ts_only and not args.gauss_only:
        run_ts = args.ts_name is not None
        run_gauss = args.gauss_name is not None
    else:
        run_ts = args.ts_only and args.ts_name is not None
        run_gauss = args.gauss_only and args.gauss_name is not None

    if run_ts:
        print(f"[TS] {args.ts_name}")
        regen_one_model(args.ts_name, args.ts_result_name)
    if run_gauss:
        print(f"[Gaussian] {args.gauss_name}")
        regen_one_model(args.gauss_name, args.gauss_result_name)


if __name__ == '__main__':
    main()
