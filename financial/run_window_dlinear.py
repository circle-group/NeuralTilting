"""Entry point for a single DLinear rolling-window job (train + forecast + metrics).

Called by run_financial_window_dlinear.sh:
    python financial/run_window_dlinear.py \\
        --prepared-name <name>  \\
        --result-name   <name>  \\
        --window-idx    <int>   \\
        [--config       <path>]

--prepared-name : the existing prepared-windows directory to read from
                  (financial/data/prepared/{prepared_name}/)
--result-name   : where to write results
                  (financial/results/{result_name}/window_{idx}/)
--config        : optional path to a DLinear config YAML.  When provided,
                  model/training/evaluation sections are read from this file
                  instead of the prepared-name config snapshot.

DLinear produces a deterministic point forecast. This script replicates it
K times to form degenerate forecast samples (shape K×H×D), so the downstream
evaluation pipeline (aggregate_prices.py, stitch_prices.py, …) runs unchanged.
CRPS of a degenerate distribution equals MAE — the standard convention for
comparing probabilistic and deterministic forecasters.
"""

import sys
import argparse
import json
import pickle
import yaml
import random as pyrandom
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from financial.data.windows import load_window
from financial.training.train_window_dlinear import train_dlinear
from financial.evaluation.metrics import compute_all_metrics, DENSE_COVERAGE_LEVELS

RESULTS_DIR  = Path(__file__).parent / 'results'
PREPARED_DIR = Path(__file__).parent / 'data' / 'prepared'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-name', required=True,
                        help='Prepared-windows directory name '
                             '(financial/data/prepared/{prepared_name}/)')
    parser.add_argument('--result-name', required=True,
                        help='Results directory name '
                             '(financial/results/{result_name}/)')
    parser.add_argument('--window-idx', type=int, required=True)
    parser.add_argument('--config', default=None,
                        help='Optional path to a DLinear config YAML.  When '
                             'provided, model/training/evaluation sections '
                             'are read from this file instead of the '
                             'prepared-name snapshot.')
    args = parser.parse_args()

    prepared_name = args.prepared_name
    result_name   = args.result_name
    window_idx    = args.window_idx

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        config_path = PREPARED_DIR / prepared_name / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config snapshot not found: {config_path}\n"
                f"Run prepare_windows_prices.py first for '{prepared_name}'."
            )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # DLinear-specific config lives under model_type: dlinear_financial
    # We expect cfg to have been prepared via a DLinear config YAML.
    # If not, we read DLinear params from cfg['model'] with sensible defaults.
    model_cfg    = cfg.get('model', {})
    training_cfg = cfg.get('training', {})
    eval_cfg     = cfg.get('evaluation', {})

    kernel_size     = model_cfg.get('kernel_size', 25)
    lookback_len    = model_cfg.get('lookback_len', 48)
    learning_rate   = training_cfg.get('learning_rate', 0.001)
    training_steps  = training_cfg.get('training_steps', 500)
    n_samples       = eval_cfg.get('n_forecast_samples', 1000)
    quantile_levels = eval_cfg.get('quantile_levels', DENSE_COVERAGE_LEVELS)

    # ------------------------------------------------------------------
    # Skip if already done
    # ------------------------------------------------------------------
    result_dir   = RESULTS_DIR / result_name / f"window_{window_idx:04d}"
    metrics_path = result_dir / 'metrics.json'
    if metrics_path.exists():
        print(f"Window {window_idx:04d} already evaluated, skipping.")
        return

    # ------------------------------------------------------------------
    # Load window
    # ------------------------------------------------------------------
    window_path = PREPARED_DIR / prepared_name / f"window_{window_idx:04d}.pkl"
    if not window_path.exists():
        raise FileNotFoundError(f"Window file not found: {window_path}")
    window = load_window(window_path)

    train_obs    = jnp.array(window['train_returns'])    # (T, D)
    forecast_obs = jnp.array(window['forecast_returns']) # (H, D)
    if forecast_obs.ndim == 1:
        forecast_obs = forecast_obs[:, None]

    print(f"Window {window_idx:04d}: "
          f"{train_obs.shape[0]} train obs, "
          f"{forecast_obs.shape[0]} forecast obs, "
          f"D={train_obs.shape[1]}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    seed = pyrandom.randint(0, 2 ** 31)
    key  = jrandom.key(seed)

    model, train_metrics = train_dlinear(
        train_obs=train_obs,
        forecast_obs=forecast_obs,
        kernel_size=kernel_size,
        lookback_len=lookback_len,
        learning_rate=learning_rate,
        training_steps=training_steps,
        key=key,
        verbose=True,
    )
    print(f"  initial_loss={train_metrics['initial_loss']:.6f}  "
          f"final_loss={train_metrics['final_loss']:.6f}  "
          f"n_sub_windows={train_metrics['n_sub_windows']}")

    # ------------------------------------------------------------------
    # Point forecast → degenerate samples (K identical copies)
    # Use the last lookback_len steps of training data as input.
    # ------------------------------------------------------------------
    lookback_input = train_obs[-lookback_len:]               # (lookback_len, D)
    point_forecast = model(lookback_input)                   # (H, D)
    point_np       = np.array(point_forecast)                # (H, D)
    samples        = np.repeat(point_np[None], n_samples, axis=0)  # (K, H, D)

    # ------------------------------------------------------------------
    # Save forecast samples
    # ------------------------------------------------------------------
    result_dir.mkdir(parents=True, exist_ok=True)
    samples_path = result_dir / 'forecast_samples.pkl'
    with open(samples_path, 'wb') as f:
        pickle.dump(samples, f)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    forecast_obs_np = np.array(forecast_obs)   # (H, D)

    metrics = compute_all_metrics(
        samples, forecast_obs_np, quantile_levels,
        path_weight_threshold=20.0,
    )

    metrics['window_idx']       = window_idx
    metrics['n_train_obs']      = int(train_obs.shape[0])
    metrics['n_forecast_obs']   = int(forecast_obs.shape[0])
    metrics['window_start_day'] = float(window['window_start_day'])
    metrics['time_scale']       = float(window.get('time_scale', 1.0))
    metrics['forecast_times']   = np.array(window['forecast_times']).tolist()
    metrics['forecast_obs']     = forecast_obs_np.tolist()
    metrics['norm_mu']          = np.array(window['norm_mu']).tolist()
    metrics['norm_sigma']       = np.array(window['norm_sigma']).tolist()
    metrics['point_forecast']   = point_np.tolist()
    metrics['train_loss']       = train_metrics['final_loss']

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Window {window_idx:04d} complete — "
          f"CRPS={metrics['mean_crps']:.4f} "
          f"(= MAE for degenerate samples)")


if __name__ == '__main__':
    main()
