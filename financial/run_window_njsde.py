"""Entry point for a single Neural Jump SDE rolling-window job.

Called by run_financial_window_njsde.sh:
    python financial/run_window_njsde.py \\
        --prepared-name <name>  \\
        --result-name   <name>  \\
        --window-idx    <int>

Reuses prepared windows from the TS prices experiment — no separate prepare
step needed.  Writes forecast_samples.pkl (K, H, obs_dim) and metrics.json
in the same format as all other financial models.
"""

import sys
import argparse
import json
import pickle
import yaml
import random as pyrandom
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from financial.data.windows import load_window
from financial.training.train_window_njsde import train_njsde
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
                        help='Optional path to an NJSDE config yaml. When provided, '
                             'model/training/evaluation sections are read from this '
                             'file instead of the prepared-name snapshot.')
    args = parser.parse_args()

    prepared_name = args.prepared_name
    result_name   = args.result_name
    window_idx    = args.window_idx

    # ------------------------------------------------------------------
    # Load config: model/training/eval from --config if given, else from
    # the prepared directory snapshot (legacy behaviour).
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

    model_cfg    = cfg.get('model', {})
    training_cfg = cfg.get('training', {})
    eval_cfg     = cfg.get('evaluation', {})

    # Model hyperparameters
    dim_c         = model_cfg.get('dim_c', 32)
    dim_h         = model_cfg.get('dim_h', 32)
    n_mixtures    = model_cfg.get('n_mixtures', 5)
    hidden_width  = model_cfg.get('hidden_width', 64)
    hidden_depth  = model_cfg.get('hidden_depth', 2)
    ortho         = model_cfg.get('ortho', True)
    n_ode_steps   = model_cfg.get('n_ode_steps', 4)

    # Training hyperparameters
    learning_rate  = training_cfg.get('learning_rate', 0.001)
    training_steps = training_cfg.get('training_steps', 1000)
    weight_decay   = training_cfg.get('weight_decay', 0.0)

    # Evaluation
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

    train_obs    = jnp.array(window['train_returns'])    # (T, obs_dim)
    forecast_obs = jnp.array(window['forecast_returns']) # (H, obs_dim)
    train_times  = np.array(window['train_times'])       # (T,) normalised time

    if forecast_obs.ndim == 1:
        forecast_obs = forecast_obs[:, None]
    if train_obs.ndim == 1:
        train_obs = train_obs[:, None]

    T = train_obs.shape[0]
    H = forecast_obs.shape[0]
    dt = float(train_times[1] - train_times[0])

    print(f"Window {window_idx:04d}: T={T} train obs, H={H} forecast obs, dt={dt:.6f}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    seed = pyrandom.randint(0, 2 ** 31)
    key  = jrandom.key(seed)
    key, model_key = jrandom.split(key)

    model, train_metrics = train_njsde(
        train_obs=train_obs,
        dt=dt,
        dim_c=dim_c,
        dim_h=dim_h,
        n_mixtures=n_mixtures,
        hidden_width=hidden_width,
        hidden_depth=hidden_depth,
        ortho=ortho,
        n_ode_steps=n_ode_steps,
        learning_rate=learning_rate,
        training_steps=training_steps,
        key=model_key,
        weight_decay=weight_decay,
        verbose=True,
    )
    print(f"  initial_nll={train_metrics['initial_loss']:.6f}  "
          f"final_nll={train_metrics['final_loss']:.6f}")

    # ------------------------------------------------------------------
    # Forecast: K stochastic sample paths
    # ------------------------------------------------------------------
    key, forecast_key = jrandom.split(key)
    forecast_samples = model.forecast(
        obs=train_obs,
        dt=dt,
        n_ode_steps=n_ode_steps,
        forecast_steps=H,
        n_samples=n_samples,
        key=forecast_key,
    )
    # forecast_samples: (K, H, obs_dim)
    forecast_samples_np = np.array(forecast_samples)

    # ------------------------------------------------------------------
    # Save forecast samples
    # ------------------------------------------------------------------
    result_dir.mkdir(parents=True, exist_ok=True)
    samples_path = result_dir / 'forecast_samples.pkl'
    with open(samples_path, 'wb') as f:
        pickle.dump(forecast_samples_np, f)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    forecast_obs_np = np.array(forecast_obs)   # (H, 1)

    metrics = compute_all_metrics(
        forecast_samples_np, forecast_obs_np, quantile_levels,
        path_weight_threshold=20.0,
    )

    metrics['window_idx']       = window_idx
    metrics['n_train_obs']      = T
    metrics['n_forecast_obs']   = H
    metrics['window_start_day'] = float(window['window_start_day'])
    metrics['time_scale']       = float(window.get('time_scale', 1.0))
    metrics['forecast_times']   = np.array(window['forecast_times']).tolist()
    metrics['forecast_obs']     = forecast_obs_np.tolist()
    metrics['norm_mu']          = np.array(window['norm_mu']).tolist()
    metrics['norm_sigma']       = np.array(window['norm_sigma']).tolist()
    metrics['train_loss']       = train_metrics['final_loss']

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Window {window_idx:04d} complete — CRPS={metrics['mean_crps']:.4f}")


if __name__ == '__main__':
    main()
