"""
Evaluation Orchestrator for Trained SDE Models

Main script for evaluating training runs with comprehensive metrics.

Core path metrics (mse, mae) are always computed separately for the training,
heldout, and full observation subsets, reported as mse_training / mse_heldout /
mse_all (and mae_*). drift_error is computed once at the model level.
Additional metrics can be requested via --metrics (computed on --eval-obs subset).
Posterior samples are always saved.

Posterior samples are loaded from pre-generated files in the run directory
(posteriors_seed_*.pkl produced by generation/generate_posteriors.py) if they
exist, avoiding redundant simulation. Falls back to on-the-fly generation if no
pre-generated files are found.

Defaults:
    --eval-obs   all        evaluate on the full observation sequence
    --sim-grid   training   simulate on the training discretization, then
                            linearly interpolate to held-out times
    --ground-truth observations

Usage:
    # Evaluate single run with all defaults
    python evaluation/evaluate_runs.py \\
        --run-path training_runs/tilted_stable_sde/alpha_1.20_obsstd_0.10/data_16241891_train_45678625

    # Evaluate only on held-out times to measure generalisation
    python evaluation/evaluate_runs.py \\
        --run-path training_runs/gaussian_sde/alpha_1.20_obsstd_0.10/data_16241891_train_45678625 \\
        --eval-obs heldout

    # Simulate on the dense obs grid (no interpolation) and evaluate everywhere
    python evaluation/evaluate_runs.py \\
        --run-path training_runs/tilted_stable_sde/alpha_1.20_obsstd_0.10/data_16241891_train_45678625 \\
        --sim-grid obs --eval-obs all

    # Compare against the true latent path instead of noisy observations
    python evaluation/evaluate_runs.py \\
        --run-path training_runs/tilted_stable_sde/alpha_1.20_obsstd_0.10/data_16241891_train_45678625 \\
        --ground-truth latent

    # Add extra metrics on top of the core ones
    python evaluation/evaluate_runs.py \\
        --run-path training_runs/tilted_stable_sde/alpha_1.20_obsstd_0.10/data_16241891_train_45678625 \\
        --metrics nll coverage_95 \\
        --n-posterior-samples 200
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from evaluation.run_utils import (
    find_run_path,
    load_run_config,
    load_run_metadata,
    load_model,
    load_dataset_for_run,
    get_evaluation_output_path_full,
    extract_drift_parameters,
    recreate_observation_subsampling,
    EvaluationResults,
    DatasetError
)
from evaluation.loss_functions import (
    get_metric_function, list_available_metrics, compute_jump_mask,
    continuous_ranked_probability_score, mean_absolute_error, mean_squared_error,
)
from utils.training_utils import build_simulation_grid


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained SDE models with comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--run-path',
        type=str,
        required=True,
        help='Full path to the training run directory (e.g., training_runs/tilted_stable_sde/alpha_1.20_obsstd_0.10/data_16241891_train_45678625)'
    )

    # Metric selection
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=[],
        help=f'Extra metrics to compute on the --eval-obs subset (mse/mae per subset and drift_error are always computed). Available: {", ".join(list_available_metrics())}'
    )

    # Ground truth selection
    parser.add_argument(
        '--ground-truth',
        choices=['observations', 'latent'],
        default='observations',
        help='Ground truth to use for error metrics (default: observations)'
    )

    # Observation subset for evaluation
    parser.add_argument(
        '--eval-obs',
        choices=['training', 'heldout', 'all'],
        default='all',
        help=(
            'Which observation times to evaluate at (default: training). '
            '"training": only times seen during training; '
            '"heldout": only times withheld during training; '
            '"all": the full observation sequence.'
        )
    )

    # Simulation grid
    parser.add_argument(
        '--sim-grid',
        choices=['obs', 'training'],
        default='training',
        help=(
            'Time grid used to simulate the posterior (default: obs). '
            '"obs": simulate on all observation times — dense, no interpolation needed. '
            '"training": simulate on the training discretization (training obs + latent steps), '
            'then linearly interpolate to reach held-out times. '
            'Matches the dt regime the model was trained under.'
        )
    )

    parser.add_argument(
        '--n-latent-steps',
        type=int,
        default=None,
        help='Number of latent steps for the training grid (only used with --sim-grid training; default: value from config)'
    )

    # Posterior generation parameters
    parser.add_argument(
        '--n-posterior-samples',
        type=int,
        default=100,
        help='Number of posterior samples to generate (default: 100)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for posterior generation (default: 42)'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory (default: auto-generated hierarchical path)'
    )

    return parser.parse_args()


# =============================================================================
# POSTERIOR GENERATION
# =============================================================================

def generate_posterior_samples(
    model,
    state_init: jnp.ndarray,
    time_sequence: jnp.ndarray,
    n_samples: int,
    seed: int
) -> jnp.ndarray:
    """
    Generate posterior samples using vmap for parallel generation.

    Parameters
    ----------
    model : eqx.Module
        Trained model
    state_init : Array
        Initial state, shape (state_dim,)
    time_sequence : Array
        Time points for simulation, shape (n_times,)
    n_samples : int
        Number of posterior samples to generate
    seed : int
        Random seed

    Returns
    -------
    samples : Array
        Posterior samples, shape (n_samples, n_times, state_dim)
    """
    print(f"  Generating {n_samples} posterior samples...", flush=True)

    # Generate random keys for each sample
    key = jrandom.PRNGKey(seed)
    keys = jrandom.split(key, n_samples)

    # Vectorize model.simulate_posterior over keys
    vmap_simulate = jax.vmap(
        lambda k: model.simulate_posterior(state_init, time_sequence, k),
        in_axes=0
    )

    samples = vmap_simulate(keys)  # Shape: (n_samples, n_times, state_dim)

    return samples


def compute_posterior_statistics(samples: jnp.ndarray) -> Dict:
    """
    Compute mean, std, and quantiles for uncertainty bands.

    Parameters
    ----------
    samples : Array
        Posterior samples, shape (n_samples, n_times, state_dim)

    Returns
    -------
    stats : Dict
        Dictionary with keys: 'mean', 'std', 'quantiles'
    """
    mean = jnp.mean(samples, axis=0)  # Shape: (n_times, state_dim)
    std = jnp.std(samples, axis=0)    # Shape: (n_times, state_dim)

    quantiles = {
        '0.15%': jnp.percentile(samples, 0.15, axis=0),   # 99.7% CI lower
        '2.5%': jnp.percentile(samples, 2.5, axis=0),     # 95% CI lower
        '16%': jnp.percentile(samples, 16, axis=0),       # 68% CI lower (1σ)
        '50%': jnp.percentile(samples, 50, axis=0),       # Median
        '84%': jnp.percentile(samples, 84, axis=0),       # 68% CI upper (1σ)
        '97.5%': jnp.percentile(samples, 97.5, axis=0),   # 95% CI upper
        '99.85%': jnp.percentile(samples, 99.85, axis=0)  # 99.7% CI upper
    }

    return {'mean': mean, 'std': std, 'quantiles': quantiles}


def find_existing_posteriors(run_dir: Path) -> Optional[Dict]:
    """
    Find pre-generated posterior samples in a run directory.

    Looks for posteriors_metadata_seed_*.json written by
    generation/generate_posteriors.py and returns paths to the matching
    samples and stats pickles if they all exist.

    Returns dict with 'samples_path', 'stats_path', 'seed', 'n_samples',
    or None if no complete set is found.
    """
    metadata_files = sorted(run_dir.glob('posteriors_metadata_seed_*.json'))
    if not metadata_files:
        return None

    # Use the most recently modified set
    metadata_path = max(metadata_files, key=lambda p: p.stat().st_mtime)
    seed_str = metadata_path.stem.replace('posteriors_metadata_seed_', '')

    samples_path = run_dir / f'posteriors_seed_{seed_str}.pkl'
    stats_path = run_dir / f'posteriors_stats_seed_{seed_str}.pkl'

    if not samples_path.exists() or not stats_path.exists():
        return None

    import json as _json
    with open(metadata_path) as f:
        meta = _json.load(f)

    return {
        'samples_path': samples_path,
        'stats_path': stats_path,
        'seed': meta.get('generation_seed'),
        'n_samples': meta.get('n_posterior_samples'),
    }


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_metrics(
    metric_names: List[str],
    posterior_samples: jnp.ndarray,
    posterior_stats: Dict,
    ground_truth: jnp.ndarray,
    observations: jnp.ndarray,
    obs_std: float,
    trained_drift_params: Optional[Dict],
    true_drift_params: Optional[Dict],
    config: Dict
) -> Dict:
    """
    Compute all requested metrics.

    Parameters
    ----------
    metric_names : List[str]
        Names of metrics to compute
    posterior_samples : Array
        Posterior samples, shape (n_samples, n_times, state_dim)
    posterior_stats : Dict
        Posterior statistics (mean, std, quantiles)
    ground_truth : Array
        Ground truth for error metrics (observations or latent)
    observations : Array
        Observations (for NLL)
    obs_std : float
        Observation noise std
    trained_drift_params : Dict, optional
        Learned drift parameters
    true_drift_params : Dict, optional
        True drift parameters
    config : Dict
        Run configuration

    Returns
    -------
    metric_results : Dict
        Dictionary mapping metric names to computed values
    """
    metric_results = {}

    for metric_name in metric_names:
        print(f"  Computing metric: {metric_name}", flush=True)

        try:
            metric_fn = get_metric_function(metric_name)

            # Determine which arguments the metric needs
            if metric_name in ['mse', 'mae', 'rmse']:
                # Path error metrics: use posterior mean vs ground truth
                value = metric_fn(
                    posterior_mean=posterior_stats['mean'],
                    ground_truth=ground_truth,
                    config=config
                )

            elif metric_name in ['time_se', 'time_ae']:
                # Time-varying error metrics
                value = metric_fn(
                    posterior_mean=posterior_stats['mean'],
                    ground_truth=ground_truth,
                    config=config
                )

            elif metric_name == 'nll':
                # Negative log-likelihood uses observations
                value = metric_fn(
                    posterior_samples=posterior_samples,
                    observations=observations,
                    obs_std=obs_std,
                    config=config
                )

            elif metric_name == 'crps':
                value = metric_fn(
                    posterior_samples=posterior_samples,
                    ground_truth=ground_truth,
                    config=config
                )

            elif 'coverage' in metric_name or 'width' in metric_name:
                # Probabilistic metrics
                if 'coverage' in metric_name:
                    value = metric_fn(
                        posterior_samples=posterior_samples,
                        ground_truth=ground_truth,
                        config=config
                    )
                else:  # width
                    value = metric_fn(
                        posterior_samples=posterior_samples,
                        config=config
                    )

            elif metric_name == 'drift_error':
                # Drift parameter comparison
                if trained_drift_params is None or true_drift_params is None:
                    print(f"    Warning: Skipping drift_error (drift parameters not available)")
                    continue

                value = metric_fn(
                    trained_drift_params=trained_drift_params,
                    true_drift_params=true_drift_params,
                    config=config
                )

            else:
                print(f"    Warning: Unknown metric type '{metric_name}', skipping")
                continue

            metric_results[metric_name] = value

        except Exception as e:
            print(f"    Error computing {metric_name}: {e}")
            metric_results[metric_name] = None

    return metric_results


# =============================================================================
# MAIN EVALUATION WORKFLOW
# =============================================================================

def evaluate_run(args: argparse.Namespace) -> EvaluationResults:
    """
    Main evaluation workflow for a single training run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments

    Returns
    -------
    results : EvaluationResults
        Evaluation results

    Raises
    ------
    Various exceptions for loading failures
    """
    # Step 1: Load run artifacts
    run_path = Path(args.run_path)
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    run_id = run_path.name

    print(f"\n{'='*80}")
    print(f"EVALUATING RUN: {run_id}")
    print(f"{'='*80}\n")

    print("Step 1: Loading run artifacts...")
    print(f"  Run path: {run_path}")

    config = load_run_config(run_path)
    training_metadata = load_run_metadata(run_path)
    print(f"  Model type: {config['model_type']}")
    print(f"  Status: {training_metadata['status']}")

    # Step 2: Load dataset
    print("\nStep 2: Loading dataset...")
    dataset = load_dataset_for_run(config)
    print(f"  Dataset: alpha={dataset['alpha']}, obs_std={dataset['obs_std']}, seed={dataset['data_seed']}")
    print(f"  Observations: {dataset['observations'].shape}")
    print(f"  Latent path: {dataset['latent_path'].shape}")

    # Step 3: Recreate training conditions
    print("\nStep 3: Recreating training conditions...")
    all_obs_times = dataset['time_sequence']
    training_indices = recreate_observation_subsampling(all_obs_times, config)
    all_indices = jnp.arange(len(all_obs_times))
    heldout_indices = jnp.setdiff1d(all_indices, training_indices)
    print(f"  Training observations: {len(training_indices)}/{len(all_obs_times)}")
    print(f"  Held-out observations: {len(heldout_indices)}/{len(all_obs_times)}")

    # Determine which indices to evaluate at
    eval_obs = args.eval_obs
    if eval_obs == 'training':
        eval_indices = training_indices
    elif eval_obs == 'heldout':
        eval_indices = heldout_indices
    else:  # 'all'
        eval_indices = all_indices
    print(f"  Evaluating at '{eval_obs}' observation times: {len(eval_indices)} points")

    # Build simulation grid
    sim_grid = args.sim_grid
    if sim_grid == 'obs':
        # Simulate directly on all observation times — every obs time is a grid point,
        # so no interpolation is ever needed regardless of --eval-obs.
        time_sequence = jnp.array(all_obs_times)
        print(f"  Simulation grid: 'obs' — {len(time_sequence)} points (all observation times)")
    else:  # 'training'
        # Reproduce the training discretization: training obs times + uniform latent steps.
        # Held-out times are NOT grid points; posterior values there will be interpolated.
        n_latent_steps = args.n_latent_steps or config['training']['n_latent_steps']
        time_sequence = build_simulation_grid(
            T_start=config['data']['time_start'],
            T_end=config['data']['time_end'],
            n_steps=n_latent_steps,
            obs_times=all_obs_times[training_indices]
        )
        print(f"  Simulation grid: 'training' — {len(time_sequence)} points "
              f"({n_latent_steps} latent steps + {len(training_indices)} training obs times)")

    # Step 4: Load model
    print("\nStep 4: Loading trained model...")
    model = load_model(run_path, config)
    print(f"  Model loaded successfully")

    # Step 5: Load pre-generated or generate posterior samples
    print("\nStep 5: Loading/generating posterior samples...")
    existing = find_existing_posteriors(run_path)
    if existing is not None:
        print(f"  Found pre-generated samples (seed={existing['seed']}, n={existing['n_samples']})")
        print(f"  Loading {existing['samples_path'].name}...")
        with open(existing['samples_path'], 'rb') as f:
            posterior_samples = pickle.load(f)
        # Validate the loaded samples match the expected simulation grid.
        # Pre-generated posteriors are on the training grid; a mismatch (e.g. when
        # --sim-grid obs or --n-latent-steps override is used) would otherwise cause
        # silent out-of-range indexing errors.
        _n_grid = np.array(posterior_samples).shape[1]
        if _n_grid != len(time_sequence):
            print(f"  Warning: pre-generated samples have {_n_grid} time points but "
                  f"expected {len(time_sequence)} (sim_grid='{sim_grid}'). Regenerating...")
            existing = None
    if existing is None:
        print("  Generating posterior samples...")
        state_init = jnp.array(config['data'].get('state_init_vector') or [config['data']['state_init']])
        posterior_samples = generate_posterior_samples(
            model=model,
            state_init=state_init,
            time_sequence=time_sequence,
            n_samples=args.n_posterior_samples,
            seed=args.seed
        )
        generation_seed = args.seed
    else:
        generation_seed = existing['seed']
    print(f"  Posterior samples shape: {np.array(posterior_samples).shape}")

    # Step 6: Load pre-computed or compute posterior statistics
    print("\nStep 6: Loading/computing posterior statistics...")
    if existing is not None:
        print(f"  Loading {existing['stats_path'].name}...")
        with open(existing['stats_path'], 'rb') as f:
            posterior_stats = pickle.load(f)
    else:
        posterior_stats = compute_posterior_statistics(posterior_samples)
    print(f"  Posterior mean shape: {np.array(posterior_stats['mean']).shape}")

    # Step 7: Interpolate posterior samples to all obs times, then slice per subset.
    # Simulation stays on the training grid (~n_latent_steps points); we use cheap
    # numpy linear interpolation to reach obs times we didn't simulate at directly.
    # This gives training/heldout/all metrics in one pass at negligible extra cost.
    print(f"\nStep 7: Interpolating posterior to obs times (sim_grid='{sim_grid}')...")

    all_obs_times_np = np.array(all_obs_times)
    time_sequence_np = np.array(time_sequence)
    posterior_samples_np = np.array(posterior_samples)  # (n_samples, n_grid, state_dim)
    state_dim = posterior_samples_np.shape[2]
    n_samples_gen = posterior_samples_np.shape[0]

    if sim_grid == 'obs':
        # Grid IS all obs times — every obs time is a grid point, no interpolation needed.
        posterior_samples_at_all_obs = posterior_samples_np
    else:
        # Linearly interpolate each sample path to all obs times.
        # Shape: (n_samples, n_all_obs, state_dim)
        posterior_samples_at_all_obs = np.stack([
            np.stack([
                np.interp(all_obs_times_np, time_sequence_np, posterior_samples_np[s, :, d])
                for d in range(state_dim)
            ], axis=1)
            for s in range(n_samples_gen)
        ], axis=0)

    # Recompute stats from the interpolated samples (consistent with the interpolated paths)
    quantile_levels = [0.15, 2.5, 16.0, 50.0, 84.0, 97.5, 99.85]
    quantile_keys = ['0.15%', '2.5%', '16%', '50%', '84%', '97.5%', '99.85%']
    mean_at_all_obs = posterior_samples_at_all_obs.mean(axis=0)
    std_at_all_obs = posterior_samples_at_all_obs.std(axis=0)
    quantiles_at_all_obs = {
        k: np.percentile(posterior_samples_at_all_obs, q, axis=0)
        for k, q in zip(quantile_keys, quantile_levels)
    }

    # Ground truth at all obs times
    if args.ground_truth == 'observations':
        ground_truth_all_obs = np.array(dataset['observations'])
    else:  # latent
        if 'latent_path' not in dataset:
            raise DatasetError(
                "Dataset does not contain a 'latent_path' key. "
                "Use --ground-truth observations or regenerate the dataset with latent paths saved."
            )
        ground_truth_all_obs = np.array(dataset['latent_path'])

    print(f"  Posterior at all obs times: {posterior_samples_at_all_obs.shape}")
    print(f"  Ground truth shape: {ground_truth_all_obs.shape}")

    # Step 8: Extract drift parameters (if available)
    print("\nStep 8: Extracting drift parameters...")
    try:
        trained_drift_params = extract_drift_parameters(model.drift)
        true_drift_params = extract_drift_parameters(dataset['drift_component'])
        print(f"  Trained drift parameters: {list(trained_drift_params.keys())}")
    except Exception as e:
        print(f"  Warning: Could not extract drift parameters: {e}")
        trained_drift_params = None
        true_drift_params = None

    # Step 9: Compute metrics.
    # Core path metrics (mse, mae) are computed separately for training, heldout, and all
    # observation subsets — reported as mse_training, mse_heldout, mse_all, etc.
    # drift_error is model-level (not subset-dependent) and computed once.
    # Any extra --metrics are computed on the --eval-obs subset.
    CORE_PATH_METRICS = ['mse', 'mae', 'crps']
    subset_map = {
        'training': np.array(training_indices),
        'heldout': np.array(heldout_indices),
        'all': np.array(all_indices),
    }
    print(f"\nStep 9: Computing core metrics per subset + drift_error"
          f"{f' + extra: {list(args.metrics)}' if args.metrics else ''}...")

    metric_results = {}

    # Core path metrics for each subset
    for subset_name, indices in subset_map.items():
        if len(indices) == 0:
            print(f"  Skipping '{subset_name}' subset (empty)")
            continue
        subset_results = compute_metrics(
            metric_names=CORE_PATH_METRICS,
            posterior_samples=posterior_samples_at_all_obs[:, indices, :],
            posterior_stats={
                'mean': mean_at_all_obs[indices],
                'std': std_at_all_obs[indices],
                'quantiles': {k: v[indices] for k, v in quantiles_at_all_obs.items()}
            },
            ground_truth=ground_truth_all_obs[indices],
            observations=np.array(dataset['observations'])[indices],
            obs_std=dataset['obs_std'],
            trained_drift_params=None,
            true_drift_params=None,
            config=config
        )
        for metric_name, value in subset_results.items():
            metric_results[f'{metric_name}_{subset_name}'] = value

    # Drift error: model-level, computed once
    drift_results = compute_metrics(
        metric_names=['drift_error'],
        posterior_samples=posterior_samples_at_all_obs,
        posterior_stats={'mean': mean_at_all_obs, 'std': std_at_all_obs, 'quantiles': quantiles_at_all_obs},
        ground_truth=ground_truth_all_obs,
        observations=np.array(dataset['observations']),
        obs_std=dataset['obs_std'],
        trained_drift_params=trained_drift_params,
        true_drift_params=true_drift_params,
        config=config
    )
    metric_results.update(drift_results)

    # Jump-conditioned metrics: MSE, MAE, CRPS computed only at time steps where
    # the observed increment |y_t - y_{t-1}| exceeds a percentile threshold.
    # Thresholds are computed from the full observation sequence so they are
    # consistent across subsets. heldout and all subsets are reported.
    JUMP_PERCENTILES = [90, 95, 97.5, 99]
    JUMP_SUBSETS = ['heldout', 'all']

    def _pct_str(p: float) -> str:
        return f"p{int(p)}" if p == int(p) else f"p{str(p).replace('.', '_')}"

    print(f"\nStep 9b: Computing jump-conditioned metrics "
          f"(percentiles: {JUMP_PERCENTILES}, subsets: {JUMP_SUBSETS})...")

    for pct in JUMP_PERCENTILES:
        pct_str = _pct_str(pct)
        jump_mask_full = compute_jump_mask(ground_truth_all_obs, pct)
        n_jumps = int(jump_mask_full.sum())
        print(f"  {pct_str}: {n_jumps}/{len(jump_mask_full)} jump steps "
              f"({100*n_jumps/len(jump_mask_full):.1f}%)")
        if n_jumps == 0:
            continue

        for subset_name in JUMP_SUBSETS:
            indices = np.array(subset_map[subset_name])
            jump_mask_subset = jump_mask_full[indices]
            jump_within = np.where(jump_mask_subset)[0]
            if len(jump_within) == 0:
                continue

            jump_samples = posterior_samples_at_all_obs[:, indices, :][:, jump_within, :]
            jump_gt = ground_truth_all_obs[indices][jump_within]
            jump_mean = mean_at_all_obs[indices][jump_within]

            metric_results[f'jump_crps_{pct_str}_{subset_name}'] = float(
                continuous_ranked_probability_score(jump_samples, jump_gt)
            )
            metric_results[f'jump_mae_{pct_str}_{subset_name}'] = float(
                mean_absolute_error(jump_mean, jump_gt)
            )
            metric_results[f'jump_mse_{pct_str}_{subset_name}'] = float(
                mean_squared_error(jump_mean, jump_gt)
            )

    # Extra metrics on the --eval-obs subset
    if args.metrics:
        extra_metrics = [m for m in args.metrics if m not in CORE_PATH_METRICS + ['drift_error']]
        if extra_metrics:
            print(f"  Computing extra metrics on '{eval_obs}' subset...")
            extra_results = compute_metrics(
                metric_names=extra_metrics,
                posterior_samples=posterior_samples_at_all_obs[:, eval_indices, :],
                posterior_stats={
                    'mean': mean_at_all_obs[eval_indices],
                    'std': std_at_all_obs[eval_indices],
                    'quantiles': {k: v[eval_indices] for k, v in quantiles_at_all_obs.items()}
                },
                ground_truth=ground_truth_all_obs[eval_indices],
                observations=np.array(dataset['observations'])[eval_indices],
                obs_std=dataset['obs_std'],
                trained_drift_params=None,
                true_drift_params=None,
                config=config
            )
            metric_results.update(extra_results)

    # Step 10: Package results
    print("\nStep 10: Packaging results...")
    posterior_samples_np = np.array(posterior_samples)
    results = EvaluationResults(
        run_id=run_id,
        model_type=config['model_type'],
        run_config=config,
        metrics=metric_results,
        posterior_stats={
            'mean': np.array(posterior_stats['mean']),
            'std': np.array(posterior_stats['std']),
            'quantiles': {k: np.array(v) for k, v in posterior_stats['quantiles'].items()}
        },
        time_sequence=np.array(time_sequence),
        ground_truth_type=args.ground_truth,
        metadata={
            'evaluation_time': datetime.now().isoformat(),
            'n_posterior_samples': posterior_samples_np.shape[0],
            'sim_grid': sim_grid,
            'n_grid_points': int(len(time_sequence)),
            'seed': generation_seed,
            'training_seed': training_metadata.get('training_seed'),
            'eval_obs': eval_obs,
            'n_training_obs': int(len(training_indices)),
            'n_heldout_obs': int(len(heldout_indices)),
            'n_eval_obs': int(len(eval_indices)),
            'metrics_computed': list(metric_results.keys())
        },
        posterior_samples=posterior_samples_np
    )

    print("\nEvaluation complete!")
    return results


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_evaluation_results(results: EvaluationResults, output_dir: Path):
    """
    Save evaluation results to disk.

    Creates:
    - evaluation_summary.json: Human-readable metrics summary
    - evaluation_details.pkl: Full EvaluationResults object
    - posterior_samples.pkl: Samples (if included)

    Parameters
    ----------
    results : EvaluationResults
        Evaluation results to save
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    # 1. JSON summary (human-readable)
    summary = {
        'run_id': results.run_id,
        'model_type': results.model_type,
        'ground_truth_type': results.ground_truth_type,
        'eval_obs': results.metadata['eval_obs'],
        'sim_grid': results.metadata['sim_grid'],
        'evaluation_metadata': results.metadata,
        'dataset_info': {
            'alpha': results.run_config['model'].get('alpha') or results.run_config['data'].get('alpha'),
            'obs_std': results.run_config['data']['obs_std'],
            'data_seed': results.run_config['data']['data_seed'],
        },
        'training_info': {
            'training_seed': results.metadata.get('training_seed'),
            'training_steps': results.run_config['training']['training_steps'],
            'obs_subsample_count': results.run_config['training']['obs_subsample_count']
        },
        'metrics': {}
    }

    # Convert metrics to JSON-serializable format
    for name, value in results.metrics.items():
        if value is None:
            summary['metrics'][name] = None
        elif isinstance(value, dict):
            # Drift errors (nested dict)
            summary['metrics'][name] = {
                k: float(v) if isinstance(v, (jnp.ndarray, np.ndarray)) else v
                for k, v in value.items()
            }
        elif isinstance(value, (jnp.ndarray, np.ndarray)):
            if value.size == 1:
                summary['metrics'][name] = float(value)
            else:
                summary['metrics'][name] = value.tolist()
        else:
            summary['metrics'][name] = value

    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    # 2. Pickle details (full data)
    details_path = output_dir / 'evaluation_details.pkl'
    with open(details_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved details: {details_path}")

    # 3. Samples (optional)
    if results.posterior_samples is not None:
        samples_path = output_dir / 'posterior_samples.pkl'
        with open(samples_path, 'wb') as f:
            pickle.dump(results.posterior_samples, f)
        print(f"  Saved samples: {samples_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()

    # Validate extra metrics
    available_metrics = list_available_metrics()
    for metric in args.metrics:
        if metric not in available_metrics:
            print(f"Error: Unknown metric '{metric}'")
            print(f"Available metrics: {', '.join(available_metrics)}")
            sys.exit(1)

    # Evaluate run
    try:
        results = evaluate_run(args)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Auto-generate hierarchical path
            output_dir = get_evaluation_output_path_full(
                run_config=results.run_config,
                run_id=results.run_id,
                ground_truth=args.ground_truth,
                eval_obs=args.eval_obs,
                sim_grid=args.sim_grid
            )

        # Save results
        save_evaluation_results(results, output_dir)

        print(f"\n{'='*80}")
        print("SUCCESS")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
