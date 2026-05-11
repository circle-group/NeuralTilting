"""Entry point for a single N-HiTS rolling-window job (train + forecast + metrics).

Called by run_financial_window_nhits.sh:
    python financial/run_window_nhits.py \\
        --prepared-name <name>  \\
        --result-name   <name>  \\
        --window-idx    <int>   \\
        [--config       <path>]

--prepared-name : existing prepared-windows directory
                  (financial/data/prepared/{prepared_name}/)
--result-name   : where to write results
                  (financial/results/{result_name}/window_{idx}/)
--config        : optional path to an N-HiTS config YAML.  When provided,
                  model/training/evaluation sections are read from this file
                  instead of the prepared-name config snapshot.

N-HiTS is a deterministic point-forecast model.  The point forecast is
replicated K times to form degenerate samples (K, H, D) so the downstream
pipeline runs unchanged.  CRPS of a degenerate distribution equals MAE.
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
from financial.training.train_window_nhits import train_nhits
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
                        help='Optional path to an N-HiTS config YAML.  When '
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

    model_cfg    = cfg.get('model', {})
    training_cfg = cfg.get('training', {})
    eval_cfg     = cfg.get('evaluation', {})

    # Model hyperparameters
    lookback_len        = model_cfg.get('lookback_len', 48)
    n_stacks            = model_cfg.get('n_stacks', 3)
    n_blocks_per_stack  = model_cfg.get('n_blocks_per_stack', 1)
    mlp_width           = model_cfg.get('mlp_width', 512)
    mlp_depth           = model_cfg.get('mlp_depth', 2)
    n_pool_kernel_size  = model_cfg.get('n_pool_kernel_size', [2, 2, 1])
    n_freq_downsample   = model_cfg.get('n_freq_downsample', [4, 2, 1])

    # Training hyperparameters
    learning_rate  = training_cfg.get('learning_rate', 0.001)
    training_steps = training_cfg.get('training_steps', 1000)

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

    train_obs    = jnp.array(window['train_returns'])    # (T, D) or (T,)
    forecast_obs = jnp.array(window['forecast_returns']) # (H, D) or (H,)

    # Ensure 2-D
    if train_obs.ndim == 1:
        train_obs = train_obs[:, None]
    if forecast_obs.ndim == 1:
        forecast_obs = forecast_obs[:, None]

    T = train_obs.shape[0]
    H = forecast_obs.shape[0]
    D = train_obs.shape[1]

    print(f"Window {window_idx:04d}: T={T} train obs, H={H} forecast obs, D={D}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    seed = pyrandom.randint(0, 2 ** 31)
    key  = jrandom.key(seed)

    model, train_metrics = train_nhits(
        train_obs=train_obs,
        forecast_obs=forecast_obs,
        lookback_len=lookback_len,
        n_stacks=n_stacks,
        n_blocks_per_stack=n_blocks_per_stack,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        n_pool_kernel_size=n_pool_kernel_size,
        n_freq_downsample=n_freq_downsample,
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
    # ------------------------------------------------------------------
    lookback_input = train_obs[-lookback_len:]             # (lookback_len, D)
    point_forecast = model(lookback_input)                 # (H, D)
    point_np       = np.array(point_forecast)              # (H, D)
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
    forecast_obs_np = np.array(forecast_obs)  # (H, D)

    metrics = compute_all_metrics(
        samples, forecast_obs_np, quantile_levels,
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
    metrics['point_forecast']   = point_np.tolist()
    metrics['train_loss']       = train_metrics['final_loss']

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Window {window_idx:04d} complete — "
          f"CRPS={metrics['mean_crps']:.4f} "
          f"(= MAE for degenerate samples)")


if __name__ == '__main__':
    main()
