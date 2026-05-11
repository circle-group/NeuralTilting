"""
Posterior Generation and Visualization Script

Generate posterior samples from trained Tilted Stable SDE and Gaussian SDE models,
with support for single-model visualization and two-model comparison.

Usage:
    # Single model
    python generation/generate_posteriors.py training_runs/tilted_stable_sde/alpha_1.20_obsstd_0.10/data_17713753_train_26679765

    # Comparison mode
    python generation/generate_posteriors.py \
        training_runs/tilted_stable_sde/alpha_1.20_obsstd_0.10/data_16241891_train_70023520 \
        training_runs/tilted_stable_sde/alpha_1.20_obsstd_0.10/data_16241891_train_84267516

    # With parameter overrides
    python generation/generate_posteriors.py training_runs/tilted_stable_sde/... \
        --n-posterior-samples 200 --n-latent-steps 2000 --seed 42
"""

# Standard library imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import argparse
import yaml
import json
import pickle
import random
import math
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from copy import deepcopy

# Third-party imports
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

# Library-specific imports
from models.tilted_stable_sde import TiltedStableDrivenSDE
from models.gaussian_sde import GaussianDrivenSDE
from models.tilted_stable_sde_double_well import TiltedStableDrivenSDEDoubleWell
from models.gaussian_sde_double_well import GaussianDrivenSDEDoubleWell
from utils.dataset_utils import get_dataset_path
from utils.training_utils import subsample_observations, build_simulation_grid

_DOUBLE_WELL_DATASET_BASE = Path("datasets/tilted_stable_sde_double_well")

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class PosteriorGenerationError(Exception):
    """Base exception for posterior generation errors."""
    pass

class ModelLoadError(PosteriorGenerationError):
    """Failed to load trained model."""
    pass

class CompatibilityError(PosteriorGenerationError):
    """Models are incompatible for comparison."""
    pass

class DatasetError(PosteriorGenerationError):
    """Dataset not found or invalid."""
    pass

class ParameterOverrideError(PosteriorGenerationError):
    """Invalid parameter override for model type."""
    pass

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and visualize posterior samples from trained SDE models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Positional arguments
    parser.add_argument(
        'model_paths',
        nargs='+',
        type=str,
        help='Path(s) to trained model run directories (1-2 paths)'
    )

    # Generation parameters
    parser.add_argument(
        '--n-posterior-samples',
        type=int,
        default=50,
        help='Number of posterior trajectory samples to generate (default: 20)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for generation (default: auto-generate)'
    )

    # Parameter overrides (both models)
    parser.add_argument(
        '--n-latent-steps',
        type=int,
        default=None,
        help='Override number of time steps for simulation'
    )

    # Output control
    parser.add_argument(
        '--plot-dpi',
        type=int,
        default=200,
        help='DPI for output plots (default: 200)'
    )

    parser.add_argument(
        '--save-samples',
        action='store_true',
        default=True,
        help='Save posterior samples (default: True)'
    )

    parser.add_argument(
        '--no-save-samples',
        action='store_false',
        dest='save_samples',
        help='Do not save posterior samples'
    )

    args = parser.parse_args()

    # Validation
    if len(args.model_paths) < 1 or len(args.model_paths) > 2:
        parser.error("Must provide 1 or 2 model paths")

    # Auto-generate seed if not provided
    if args.seed is None:
        args.seed = random.randint(10000000, 99999999)

    return args

# =============================================================================
# PATH VALIDATION AND CONFIG LOADING
# =============================================================================

def validate_model_paths(paths: List[str]) -> List[Path]:
    """Validate model paths exist with required files."""
    validated_paths = []

    for path_str in paths:
        path = Path(path_str)

        # Check directory exists
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")

        if not path.is_dir():
            raise NotADirectoryError(f"Model path is not a directory: {path}")

        # Check required files
        required_files = ['model.eqx', 'config.yaml', 'metadata.json']
        missing_files = [f for f in required_files if not (path / f).exists()]

        if missing_files:
            raise FileNotFoundError(
                f"Model directory is incomplete: {path}\n"
                f"Missing files: {missing_files}\n"
                f"Required files: {required_files}"
            )

        validated_paths.append(path)

    return validated_paths

def load_run_config(run_dir: Path) -> Dict:
    """Load and parse config.yaml from run directory."""
    config_path = run_dir / "config.yaml"

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ModelLoadError(f"Failed to load config from {config_path}: {e}")

def load_run_metadata(run_dir: Path) -> Dict:
    """Load and parse metadata.json from run directory."""
    metadata_path = run_dir / "metadata.json"

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        raise ModelLoadError(f"Failed to load metadata from {metadata_path}: {e}")

# =============================================================================
# MODEL COMPATIBILITY VALIDATION
# =============================================================================

def validate_model_compatibility(configs: List[Dict]) -> None:
    """
    Validate models are compatible for comparison.

    Critical checks:
    - Same dataset (alpha, obs_std, data_seed)
    - Same state_dim
    - Same observation subsampling

    Note: Different model types (tilted_stable_sde vs gaussian_sde) are allowed!
    """
    if len(configs) != 2:
        return  # Only validate in comparison mode

    config1, config2 = configs
    errors = []

    # Dataset parameters
    if config1['model'].get('alpha') is not None:
        alpha1 = config1['model'].get('alpha')
    else:
        alpha1 = config1['data'].get('alpha')

    if config2['model'].get('alpha') is not None:
        alpha2 = config2['model'].get('alpha')
    else:
        alpha2 = config2['data'].get('alpha')
    if alpha1 != alpha2:
        errors.append(
            f"alpha mismatch: model1={alpha1}, "
            f"model2={alpha2}"
        )

    if config1['data']['obs_std'] != config2['data']['obs_std']:
        errors.append(
            f"obs_std mismatch: model1={config1['data']['obs_std']}, "
            f"model2={config2['data']['obs_std']}"
        )

    if config1['data']['data_seed'] != config2['data']['data_seed']:
        errors.append(
            f"data_seed mismatch: model1={config1['data']['data_seed']}, "
            f"model2={config2['data']['data_seed']}"
        )

    # Model architecture
    if config1['model']['state_dim'] != config2['model']['state_dim']:
        errors.append(
            f"state_dim mismatch: model1={config1['model']['state_dim']}, "
            f"model2={config2['model']['state_dim']}"
        )

    # Observation subsampling (critical for fair comparison)
    if config1['training']['obs_subsample_method'] != config2['training']['obs_subsample_method']:
        errors.append(
            f"obs_subsample_method mismatch: model1={config1['training']['obs_subsample_method']}, "
            f"model2={config2['training']['obs_subsample_method']}"
        )

    if config1['training']['obs_subsample_count'] != config2['training']['obs_subsample_count']:
        errors.append(
            f"obs_subsample_count mismatch: model1={config1['training']['obs_subsample_count']}, "
            f"model2={config2['training']['obs_subsample_count']}"
        )

    if config1['training']['obs_subsample_seed'] != config2['training']['obs_subsample_seed']:
        errors.append(
            f"obs_subsample_seed mismatch: model1={config1['training']['obs_subsample_seed']}, "
            f"model2={config2['training']['obs_subsample_seed']}"
        )

    if errors:
        raise CompatibilityError(
            "Models are incompatible for comparison:\n  " +
            "\n  ".join(errors) +
            "\n\nFor fair comparison, models must have:\n" +
            "  - Same dataset (alpha, obs_std, data_seed)\n" +
            "  - Same state_dim\n" +
            "  - Same observation subsampling (method, count, seed)\n" +
            "\nNote: Different model types (tilted_stable_sde vs gaussian_sde) ARE allowed!"
        )

# =============================================================================
# PARAMETER OVERRIDES
# =============================================================================

def apply_parameter_overrides(config: Dict, args: argparse.Namespace) -> Tuple[Dict, List[str]]:
    """
    Apply CLI parameter overrides and return modified config + log of changes.
    """
    # Create deep copy to avoid mutation
    config = deepcopy(config)
    override_log = []
    model_type = config['model_type']

    # Common overrides (both model types)
    if args.n_latent_steps is not None:
        old = config['training']['n_latent_steps']
        config['training']['n_latent_steps'] = args.n_latent_steps
        override_log.append(f"n_latent_steps: {old} -> {args.n_latent_steps}")

    # Number of posterior samples (overrides n_loss_samples from training)
    old = config['training']['n_loss_samples']
    config['training']['n_loss_samples'] = args.n_posterior_samples
    override_log.append(f"n_posterior_samples (n_loss_samples in config): {old} -> {args.n_posterior_samples}")

    return config, override_log

# =============================================================================
# MODEL AND DATASET LOADING
# =============================================================================

def build_model_params(config: Dict) -> Dict:
    """
    Extract model initialization parameters from config.

    Follows the pattern from train_tilted_stable_sde.py:prepare_model_params()
    """
    model_type = config['model_type']
    model_config = config['model']

    # Generate random seeds for any None values (needed for model initialization)
    drift_seed = model_config.get('drift_seed')
    if drift_seed is None:
        drift_seed = random.randint(10000000, 99999999)

    diffusion_seed = model_config.get('diffusion_seed')
    if diffusion_seed is None:
        diffusion_seed = random.randint(10000000, 99999999)

    # Common parameters
    model_params = {
        'state_dim': model_config['state_dim'],
        'sigma': model_config['sigma'],
        'period': config['data']['time_end'],
        'drift_seed': drift_seed,
        'diffusion_seed': diffusion_seed,
        'trainable_drift': model_config['trainable_drift'],
        'initial_diffusion_weight': model_config.get('initial_diffusion_weight'),
    }

    # Model-specific parameters
    if model_type == 'tilted_stable_sde':
        phi_seed = model_config.get('phi_seed')
        if phi_seed is None:
            phi_seed = random.randint(10000000, 99999999)

        model_params.update({
            'alpha': model_config['alpha'],
            'tau': model_config['tau'],
            'loss_sample_size': model_config['loss_sample_size'],
            'max_rejection_attempts': model_config.get('max_rejection_attempts', 100),
            'max_jumps': model_config.get('max_jumps', 10000),
            'tilting_width': model_config.get('tilting_width', 64),
            'tilting_depth': model_config.get('tilting_depth', 3),
            'phi_seed': phi_seed,
            # QuadraticNeuralPotential parameters
            'n_time_features': model_config.get('n_time_features', 0),
            'a_min': model_config.get('a_min', 1e-3),
            'use_adaptive_scaling': model_config.get('use_adaptive_scaling', False),
            'n_attention_references': model_config.get('n_attention_references', 0),
            'attention_embed_dim': model_config.get('attention_embed_dim', 16),
            'attention_sharpness': model_config.get('attention_sharpness', 100.0),
            # Linear drift init
            'initial_drift_weight': model_config.get('initial_drift_weight'),
            'initial_drift_bias': model_config.get('initial_drift_bias'),
        })
    elif model_type == 'tilted_stable_sde_double_well':
        phi_seed = model_config.get('phi_seed')
        if phi_seed is None:
            phi_seed = random.randint(10000000, 99999999)

        model_params.update({
            'alpha': model_config['alpha'],
            'tau': model_config['tau'],
            'loss_sample_size': model_config['loss_sample_size'],
            'max_rejection_attempts': model_config.get('max_rejection_attempts', 100),
            'max_jumps': model_config.get('max_jumps', 10000),
            'tilting_width': model_config.get('tilting_width', 64),
            'tilting_depth': model_config.get('tilting_depth', 3),
            'phi_seed': phi_seed,
            # QuadraticNeuralPotential parameters
            'n_time_features': model_config.get('n_time_features', 0),
            'a_min': model_config.get('a_min', 1e-3),
            'use_adaptive_scaling': model_config.get('use_adaptive_scaling', False),
            'n_attention_references': model_config.get('n_attention_references', 0),
            'attention_embed_dim': model_config.get('attention_embed_dim', 16),
            'attention_sharpness': model_config.get('attention_sharpness', 100.0),
            # Double well drift init
            'initial_drift_theta1': model_config.get('initial_drift_theta1'),
            'initial_drift_theta2': model_config.get('initial_drift_theta2'),
        })
    elif model_type == 'gaussian_sde':
        control_seed = model_config.get('control_seed')
        if control_seed is None:
            control_seed = random.randint(10000000, 99999999)

        model_params.update({
            'control_width': model_config['control_width'],
            'control_depth': model_config['control_depth'],
            'n_time_features': model_config['n_time_features'],
            'control_seed': control_seed,
            # Linear drift init
            'initial_drift_weight': model_config.get('initial_drift_weight'),
            'initial_drift_bias': model_config.get('initial_drift_bias'),
        })
    elif model_type == 'gaussian_sde_double_well':
        control_seed = model_config.get('control_seed')
        if control_seed is None:
            control_seed = random.randint(10000000, 99999999)

        model_params.update({
            'control_width': model_config['control_width'],
            'control_depth': model_config['control_depth'],
            'n_time_features': model_config['n_time_features'],
            'control_seed': control_seed,
            # Double well drift init
            'initial_drift_theta1': model_config.get('initial_drift_theta1'),
            'initial_drift_theta2': model_config.get('initial_drift_theta2'),
        })
    else:
        raise ModelLoadError(f"Unsupported model_type: {model_type}")

    return model_params

def load_trained_model(run_dir: Path, model_params: Dict, model_type: str) -> eqx.Module:
    """
    Load trained model using equinox deserialization.

    Follows the pattern from train_tilted_stable_sde.py
    """
    model_path = run_dir / "model.eqx"

    # Import appropriate model class
    if model_type == 'tilted_stable_sde':
        model_class = TiltedStableDrivenSDE
    elif model_type == 'tilted_stable_sde_double_well':
        model_class = TiltedStableDrivenSDEDoubleWell
    elif model_type == 'gaussian_sde':
        model_class = GaussianDrivenSDE
    elif model_type == 'gaussian_sde_double_well':
        model_class = GaussianDrivenSDEDoubleWell
    else:
        raise ModelLoadError(f"Unsupported model_type: {model_type}")

    try:
        # Create template model for deserialization
        template_model = model_class(**model_params)

        # Load trained weights
        model = eqx.tree_deserialise_leaves(model_path, template_model)

        return model
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {model_path}: {e}")

def load_dataset_for_run(config: Dict) -> Dict:
    """
    Load dataset using existing utilities.
    """
    model_type = config['model_type']
    if model_type in ('tilted_stable_sde', 'tilted_stable_sde_double_well'):
        alpha = config['model'].get('alpha')
    else:
        alpha = config['data'].get('alpha')
    obs_std = config['data']['obs_std']
    data_seed = config['data']['data_seed']

    # Construct dataset path (double well experiments use a separate dataset folder)
    if model_type in ('tilted_stable_sde_double_well', 'gaussian_sde_double_well'):
        dataset_path = get_dataset_path(alpha, obs_std, data_seed, base_path=_DOUBLE_WELL_DATASET_BASE)
    else:
        dataset_path = get_dataset_path(alpha, obs_std, data_seed)

    if not dataset_path.exists():
        raise DatasetError(
            f"Dataset not found: {dataset_path}\n"
            f"Expected dataset with:\n"
            f"  - alpha: {alpha}\n"
            f"  - obs_std: {obs_std}\n"
            f"  - data_seed: {data_seed}"
        )

    try:
        with open(dataset_path, 'rb') as f:
            data_dict = pickle.load(f)

        # Validate parameters match (sanity check)
        if alpha is not None and data_dict.get('alpha') != alpha:
            raise DatasetError(f"Dataset alpha mismatch: file={data_dict.get('alpha')}, config={alpha}")

        if data_dict['obs_std'] != obs_std:
            raise DatasetError(f"Dataset obs_std mismatch: file={data_dict['obs_std']}, config={obs_std}")

        if data_dict['data_seed'] != data_seed:
            raise DatasetError(f"Dataset data_seed mismatch: file={data_dict['data_seed']}, config={data_seed}")

        return data_dict
    except Exception as e:
        raise DatasetError(f"Failed to load dataset from {dataset_path}: {e}")

def prepare_initial_state(config: Dict) -> jnp.ndarray:
    """
    Prepare initial state from configuration.
    """
    state_dim = config['model']['state_dim']

    if config['data'].get('state_init_vector') is not None:
        state_init = jnp.array(config['data']['state_init_vector'])

        if len(state_init) != state_dim:
            raise ValueError(
                f"state_init_vector length ({len(state_init)}) must match "
                f"state_dim ({state_dim})"
            )
    else:
        # Use scalar value for all dimensions
        state_init = jnp.full((state_dim,), config['data']['state_init'])

    return state_init

# =============================================================================
# OBSERVATION SUBSAMPLING RECREATION
# =============================================================================

def recreate_observation_subsampling(obs_times: jnp.ndarray, config: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Recreate training/held-out observation split using existing utilities.

    Uses utils.training_utils.subsample_observations() to ensure exact match.

    Parameters
    ----------
    obs_times : array
        Full observation time sequence from dataset
    config : dict
        Configuration dictionary with training subsample settings

    Returns
    -------
    training_indices : array
        Indices of training observations in obs_times
    heldout_indices : array
        Indices of held-out observations in obs_times
    """
    # Create config-like object for subsample_observations()
    class ConfigMock:
        class Training:
            def __init__(self, method, count, seed):
                self.obs_subsample_method = method
                self.obs_subsample_count = count
                self.obs_subsample_seed = seed

        def __init__(self, config_dict):
            self.training = self.Training(
                config_dict['training']['obs_subsample_method'],
                config_dict['training']['obs_subsample_count'],
                config_dict['training']['obs_subsample_seed']
            )

    mock_config = ConfigMock(config)
    training_indices = subsample_observations(obs_times, mock_config)

    # Compute held-out indices as complement
    all_indices = jnp.arange(len(obs_times))
    heldout_indices = jnp.setdiff1d(all_indices, training_indices)

    return training_indices, heldout_indices

# =============================================================================
# POSTERIOR GENERATION
# =============================================================================

def generate_posterior_samples(
    model: eqx.Module,
    state_init: jnp.ndarray,
    time_sequence: jnp.ndarray,
    n_samples: int,
    seed: int
) -> jnp.ndarray:
    """
    Generate posterior samples using vmap for parallel generation.

    Returns:
        samples: Array of shape (n_samples, n_times, state_dim)
    """
    # Generate random keys for each sample
    key = jrandom.PRNGKey(seed)
    keys = jrandom.split(key, n_samples)

    # Vectorize model.simulate_posterior over keys
    vmap_simulate = jax.vmap(
        lambda k: model.simulate_posterior(state_init, time_sequence, k),
        in_axes=0
    )

    print(f"  Generating {n_samples} posterior samples...", flush=True)
    samples = vmap_simulate(keys)  # Shape: (n_samples, n_times, state_dim)

    return samples

def compute_posterior_statistics(samples: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute mean, std, and quantiles for uncertainty bands.

    Args:
        samples: Array of shape (n_samples, n_times, state_dim)

    Returns:
        Dict with keys: 'mean', 'std', 'quantiles'
    """
    mean = jnp.mean(samples, axis=0)  # Shape: (n_times, state_dim)
    std = jnp.std(samples, axis=0)    # Shape: (n_times, state_dim)

    quantiles = {
        '0.15%': jnp.percentile(samples, 0.15, axis=0),   # 99.7% CI lower
        '2.5%': jnp.percentile(samples, 2.5, axis=0),     # 95% CI lower
        '16%': jnp.percentile(samples, 16, axis=0),       # 68% CI lower
        '50%': jnp.percentile(samples, 50, axis=0),       # Median
        '84%': jnp.percentile(samples, 84, axis=0),       # 68% CI upper
        '97.5%': jnp.percentile(samples, 97.5, axis=0),   # 95% CI upper
        '99.85%': jnp.percentile(samples, 99.85, axis=0)  # 99.7% CI upper
    }

    return {'mean': mean, 'std': std, 'quantiles': quantiles}

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_posterior_with_uncertainty(
    ax: plt.Axes,
    time_sequence: jnp.ndarray,
    obs_times: jnp.ndarray,
    samples: np.ndarray,
    stats: Dict,
    observations: jnp.ndarray,
    training_indices: jnp.ndarray,
    heldout_indices: jnp.ndarray,
    latent_path: Optional[jnp.ndarray],
    dim_idx: int,
    model_label: str,
    show_legend: bool = True
):
    """
    Plot posterior samples with uncertainty bands on a single axis.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on
    time_sequence : array
        Simulation time grid (used for trajectories and uncertainty bands)
    obs_times : array
        Observation time points (used for plotting observations)
    samples : array
        Posterior samples at simulation grid, shape (n_samples, len(time_sequence), state_dim)
    stats : dict
        Statistics at simulation grid (mean, std, quantiles)
    observations : array
        Observed values, shape (len(obs_times), state_dim)
    training_indices : array
        Indices of training observations in obs_times
    heldout_indices : array
        Indices of held-out observations in obs_times
    latent_path : array, optional
        True latent path at simulation grid, shape (len(time_sequence), state_dim)
    dim_idx : int
        State dimension index to plot
    model_label : str
        Label for the model
    show_legend : bool
        Whether to show legend

    Layering order (bottom to top):
    1. Individual posterior samples
    2. 3σ uncertainty band (99.7% CI)
    3. 2σ uncertainty band (95% CI)
    4. 1σ uncertainty band (68% CI)
    5. Held-out observations
    6. Posterior mean
    7. Training observations
    8. True latent path (top layer - most visible)
    """
    mean = stats['mean'][:, dim_idx]
    quantiles = stats['quantiles']

    q_lower_997 = quantiles['0.15%'][:, dim_idx]
    q_upper_997 = quantiles['99.85%'][:, dim_idx]
    q_lower_95 = quantiles['2.5%'][:, dim_idx]
    q_upper_95 = quantiles['97.5%'][:, dim_idx]
    q_lower_68 = quantiles['16%'][:, dim_idx]
    q_upper_68 = quantiles['84%'][:, dim_idx]

    # Layer 1: Individual sample paths
    n_samples_to_plot = min(30, samples.shape[0])
    sample_indices = np.linspace(0, samples.shape[0]-1, n_samples_to_plot, dtype=int)

    for i in sample_indices:
        ax.plot(
            time_sequence,
            samples[i, :, dim_idx],
            color='#daa520',
            alpha=0.6,
            linewidth=1.5,
            zorder=1
        )

    if show_legend and n_samples_to_plot > 0:
        ax.plot([], [], color='#daa520', alpha=0.6, linewidth=1.5, label=f'Posterior Samples (n={n_samples_to_plot})')

    # Layer 2: 99.7% CI band (quantile-based)
    ax.fill_between(
        time_sequence,
        q_lower_997,
        q_upper_997,
        color='#d0d0d0',
        alpha=0.6,
        label='99.7% CI' if show_legend else None,
        zorder=2
    )

    # Layer 3: 95% CI band (quantile-based)
    ax.fill_between(
        time_sequence,
        q_lower_95,
        q_upper_95,
        color='#b0b0b0',
        alpha=0.7,
        label='95% CI' if show_legend else None,
        zorder=3
    )

    # Layer 4: 68% CI band (quantile-based)
    ax.fill_between(
        time_sequence,
        q_lower_68,
        q_upper_68,
        color='#808080',
        alpha=0.8,
        label='68% CI' if show_legend else None,
        zorder=4
    )

    # Layer 5: Held-out observations
    if len(heldout_indices) > 0:
        ax.scatter(
            obs_times[heldout_indices],
            observations[heldout_indices, dim_idx],
            marker='o',
            color='#d62728',
            s=8,
            alpha=0.4,
            label='Held-out Obs' if show_legend else None,
            zorder=5
        )

    # Layer 6: True latent path
    if latent_path is not None:
        ax.plot(
            time_sequence,
            latent_path[:, dim_idx],
            '--',
            color='#1f77b4',
            linewidth=2.0,
            alpha=1.0,
            label='True Latent' if show_legend else None,
            zorder=6
        )

    # Layer 7: Posterior mean
    ax.plot(
        time_sequence,
        mean,
        color='black',
        linewidth=2.0,
        label='Posterior Mean' if show_legend else None,
        zorder=7
    )

    # Layer 8: Training observations
    if len(training_indices) > 0:
        ax.scatter(
            obs_times[training_indices],
            observations[training_indices, dim_idx],
            marker='o',
            color='#ff7f0e',
            s=20,
            label='Training Obs' if show_legend else None,
            zorder=8
        )

    # Styling
    ax.set_title(f"Dimension {dim_idx} - {model_label}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("State", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#edebeb')

    if show_legend:
        ax.legend(loc='best', fontsize=8, framealpha=0.9)

def plot_single_model_posterior(
    samples: np.ndarray,
    stats: Dict,
    time_sequence: jnp.ndarray,
    obs_times: jnp.ndarray,
    observations: jnp.ndarray,
    training_indices: jnp.ndarray,
    heldout_indices: jnp.ndarray,
    latent_path: Optional[jnp.ndarray],
    model_label: str,
    output_path: Path,
    dpi: int
):
    """
    Create visualization for single model.

    Parameters
    ----------
    samples : array
        Posterior samples at simulation grid, shape (n_samples, len(time_sequence), state_dim)
    stats : dict
        Statistics at simulation grid (mean, std, quantiles)
    time_sequence : array
        Simulation time grid (for trajectories)
    obs_times : array
        Observation time points (for plotting observations)
    observations : array
        Observed values, shape (len(obs_times), state_dim)
    training_indices : array
        Indices of training observations in obs_times
    heldout_indices : array
        Indices of held-out observations in obs_times
    latent_path : array, optional
        True latent path at simulation grid
    model_label : str
        Label for the model
    output_path : Path
        Path to save the plot
    dpi : int
        DPI for the output plot
    """
    state_dim = samples.shape[2]

    # Calculate grid layout
    n_cols = min(state_dim, 2)
    n_rows = math.ceil(state_dim / n_cols)
    figsize = (10 * n_cols, 6 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor('white')

    # Plot each dimension
    for dim_idx in range(state_dim):
        row = dim_idx // n_cols
        col = dim_idx % n_cols
        ax = axes[row, col]

        show_legend = (dim_idx == 0)  # Only show legend on first subplot

        plot_posterior_with_uncertainty(
            ax=ax,
            time_sequence=time_sequence,
            obs_times=obs_times,
            samples=samples,
            stats=stats,
            observations=observations,
            training_indices=training_indices,
            heldout_indices=heldout_indices,
            latent_path=latent_path,
            dim_idx=dim_idx,
            model_label=model_label,
            show_legend=show_legend
        )

    # Hide unused subplots
    for dim_idx in range(state_dim, n_rows * n_cols):
        row = dim_idx // n_cols
        col = dim_idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved visualization: {output_path}", flush=True)

def plot_comparison_posterior(
    samples_list: List[np.ndarray],
    stats_list: List[Dict],
    time_sequence: jnp.ndarray,
    obs_times: jnp.ndarray,
    observations: jnp.ndarray,
    training_indices: jnp.ndarray,
    heldout_indices: jnp.ndarray,
    latent_path: Optional[jnp.ndarray],
    model_labels: List[str],
    output_path: Path,
    dpi: int
):
    """
    Create side-by-side comparison visualization.

    Parameters
    ----------
    samples_list : list of arrays
        List of posterior samples for each model
    stats_list : list of dicts
        List of statistics for each model
    time_sequence : array
        Simulation time grid (for trajectories)
    obs_times : array
        Observation time points (for plotting observations)
    observations : array
        Observed values, shape (len(obs_times), state_dim)
    training_indices : array
        Indices of training observations in obs_times
    heldout_indices : array
        Indices of held-out observations in obs_times
    latent_path : array, optional
        True latent path at simulation grid
    model_labels : list of str
        Labels for each model
    output_path : Path
        Path to save the plot
    dpi : int
        DPI for the output plot
    """
    state_dim = samples_list[0].shape[2]

    # Calculate grid layout: one row per dimension, two columns for models
    n_rows = state_dim
    n_cols = 2
    figsize = (20, 6 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor('white')

    # Plot each dimension
    for dim_idx in range(state_dim):
        # Compute global y-axis limits for this dimension across both models
        y_min = float('inf')
        y_max = float('-inf')

        for i in range(2):
            q = stats_list[i]['quantiles']
            q_lower = q['0.15%'][:, dim_idx]
            q_upper = q['99.85%'][:, dim_idx]

            data_min = min(
                np.min(q_lower),
                np.min(observations[:, dim_idx]) if observations is not None else float('inf')
            )
            data_max = max(
                np.max(q_upper),
                np.max(observations[:, dim_idx]) if observations is not None else float('-inf')
            )

            if latent_path is not None:
                data_min = min(data_min, np.min(latent_path[:, dim_idx]))
                data_max = max(data_max, np.max(latent_path[:, dim_idx]))

            y_min = min(y_min, data_min)
            y_max = max(y_max, data_max)

        # Add 5% margin
        y_range = y_max - y_min
        y_min -= 0.05 * y_range
        y_max += 0.05 * y_range

        # Plot both models
        for col in range(2):
            ax = axes[dim_idx, col]
            show_legend = (dim_idx == 0 and col == 0)  # Only show legend on top-left

            plot_posterior_with_uncertainty(
                ax=ax,
                time_sequence=time_sequence,
                obs_times=obs_times,
                samples=samples_list[col],
                stats=stats_list[col],
                observations=observations,
                training_indices=training_indices,
                heldout_indices=heldout_indices,
                latent_path=latent_path,
                dim_idx=dim_idx,
                model_label=model_labels[col],
                show_legend=show_legend
            )

            # Apply shared y-axis limits
            ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved comparison visualization: {output_path}", flush=True)

# =============================================================================
# OUTPUT MANAGEMENT
# =============================================================================

def determine_output_directory(
    run_dirs: List[Path],
    configs: List[Dict],
    metadatas: List[Dict],
    comparison_mode: bool
) -> Tuple[Path, Path]:
    """
    Determine output directories based on mode.

    Returns:
        (plot_dir, data_dir): Tuple of plot directory and data directory
    """
    if not comparison_mode:
        # Single model mode
        plot_dir = run_dirs[0] / "plots"
        data_dir = run_dirs[0]  # Save data files to main run directory
    else:
        # Comparison mode: create centralized comparison directory
        config = configs[0]
        alpha = config['model'].get('alpha') or config['data'].get('alpha')
        obs_std = config['data']['obs_std']
        data_seed = config['data']['data_seed']

        run_id_1 = metadatas[0]['run_id']
        run_id_2 = metadatas[1]['run_id']

        comparison_dir = (
            Path("training_runs") / "comparisons" /
            f"alpha_{alpha:.2f}_obsstd_{obs_std:.2f}" /
            f"data_{data_seed}" /
            f"{run_id_1}__vs__{run_id_2}"
        )

        plot_dir = comparison_dir
        data_dir = comparison_dir

    # Create directories if they don't exist
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return plot_dir, data_dir

def save_posterior_samples(
    samples: np.ndarray,
    output_dir: Path,
    model_name: str,
    seed: int,
    comparison_mode: bool
) -> Path:
    """
    Save posterior samples to pickle file.
    """
    if comparison_mode:
        filename = f"posteriors_{model_name}_seed_{seed}.pkl"
    else:
        filename = f"posteriors_seed_{seed}.pkl"

    filepath = output_dir / filename

    with open(filepath, 'wb') as f:
        pickle.dump(samples, f)

    return filepath

def save_posterior_statistics(
    stats: Dict,
    output_dir: Path,
    model_name: str,
    seed: int,
    comparison_mode: bool
) -> Path:
    """
    Save posterior statistics to pickle file.
    """
    if comparison_mode:
        filename = f"posteriors_stats_{model_name}_seed_{seed}.pkl"
    else:
        filename = f"posteriors_stats_seed_{seed}.pkl"

    filepath = output_dir / filename

    with open(filepath, 'wb') as f:
        pickle.dump(stats, f)

    return filepath

def save_generation_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    configs: List[Dict],
    metadatas: List[Dict],
    override_logs: List[List[str]],
    stats_list: List[Dict],
    generation_time: float,
    seed: int,
    comparison_mode: bool
) -> Path:
    """
    Save generation metadata to JSON file.
    """
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'generation_seed': seed,
        'n_posterior_samples': args.n_posterior_samples,
        'models': [
            {
                'run_id': metadatas[i]['run_id'],
                'model_type': configs[i]['model_type'],
                'data_seed': configs[i]['data']['data_seed'],
                'training_seed': metadatas[i]['training_seed'],
                'parameter_overrides': override_logs[i],
                'posterior_mean_summary': {
                    'mean_value': float(np.mean(stats_list[i]['mean'])),
                    'std_value': float(np.mean(stats_list[i]['std']))
                }
            }
            for i in range(len(configs))
        ],
        'comparison_mode': comparison_mode,
        'generation_time_seconds': generation_time,
        'cli_args': vars(args)
    }

    if comparison_mode:
        filename = "posteriors_metadata.json"
    else:
        filename = f"posteriors_metadata_seed_{seed}.json"

    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    return filepath

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main orchestration function."""
    print("=" * 80, flush=True)
    print("POSTERIOR GENERATION AND VISUALIZATION", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    # Parse arguments
    args = parse_arguments()
    print(f"Random seed: {args.seed}", flush=True)
    print(f"Posterior samples: {args.n_posterior_samples}", flush=True)
    print(flush=True)

    # Validate paths
    print("Validating model paths...", flush=True)
    run_dirs = validate_model_paths(args.model_paths)
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"  Model {i}: {run_dir}", flush=True)
    print("✓ All paths validated", flush=True)
    print(flush=True)

    # Load configs
    print("Loading configurations...", flush=True)
    configs = [load_run_config(run_dir) for run_dir in run_dirs]
    metadatas = [load_run_metadata(run_dir) for run_dir in run_dirs]
    for i, (config, metadata) in enumerate(zip(configs, metadatas), 1):
        print(f"  Model {i}: {config['model_type']}, run_id={metadata['run_id']}", flush=True)
    print("✓ Configurations loaded", flush=True)
    print(flush=True)

    # Validate compatibility (if comparison mode)
    if len(configs) == 2:
        print("Validating model compatibility...", flush=True)
        validate_model_compatibility(configs)
        print("✓ Models are compatible for comparison", flush=True)
        print(flush=True)

    # Apply parameter overrides
    print("Applying parameter overrides...", flush=True)
    modified_configs = []
    override_logs = []
    for i, config in enumerate(configs, 1):
        modified_config, override_log = apply_parameter_overrides(config, args)
        modified_configs.append(modified_config)
        override_logs.append(override_log)

        if override_log:
            print(f"  Model {i} overrides:", flush=True)
            for override in override_log:
                print(f"    - {override}", flush=True)
        else:
            print(f"  Model {i}: No overrides", flush=True)
    print(flush=True)

    # Load datasets
    print("Loading datasets...", flush=True)
    datasets = [load_dataset_for_run(config) for config in modified_configs]
    for i, dataset in enumerate(datasets, 1):
        print(f"  Model {i}: {len(dataset['time_sequence'])} time points", flush=True)
    print("✓ Datasets loaded", flush=True)
    print(flush=True)

    # Recreate observation subsampling
    print("Recreating observation subsampling...", flush=True)
    obs_times = datasets[0]['time_sequence']  # Full observation time sequence
    training_indices, heldout_indices = recreate_observation_subsampling(obs_times, modified_configs[0])
    print(f"  Training observations: {len(training_indices)}", flush=True)
    print(f"  Held-out observations: {len(heldout_indices)}", flush=True)
    print(flush=True)

    # Build simulation grid (matching training approach)
    # This merges latent simulation steps with TRAINING observation times
    print("Building simulation time grid...", flush=True)
    T_end = modified_configs[0]['data']['time_end']
    n_latent_steps = modified_configs[0]['training']['n_latent_steps']
    training_obs_times = obs_times[training_indices]

    time_sequence = build_simulation_grid(
        T_start=0.0,
        T_end=T_end,
        n_steps=n_latent_steps,
        obs_times=training_obs_times
    )
    print(f"  Simulation grid: {len(time_sequence)} time points", flush=True)
    print(f"  Latent steps: {n_latent_steps}, Training obs: {len(training_obs_times)}", flush=True)
    print(flush=True)

    # Prepare initial states
    print("Preparing initial states...", flush=True)
    state_inits = [prepare_initial_state(config) for config in modified_configs]
    print(flush=True)

    # Build model parameters and load models
    print("Loading trained models...", flush=True)
    start_load_time = time.time()
    models = []
    for i, (run_dir, config) in enumerate(zip(run_dirs, modified_configs), 1):
        print(f"  Model {i}: {config['model_type']}", flush=True)
        model_params = build_model_params(config)
        model = load_trained_model(run_dir, model_params, config['model_type'])
        models.append(model)
    load_time = time.time() - start_load_time
    print(f"✓ Models loaded in {load_time:.2f}s", flush=True)
    print(flush=True)

    # Generate posterior samples
    print("=" * 80, flush=True)
    print("GENERATING POSTERIOR SAMPLES", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    start_gen_time = time.time()
    samples_list = []
    stats_list = []

    for i, (model, state_init, config) in enumerate(zip(models, state_inits, modified_configs), 1):
        print(f"Model {i} ({config['model_type']}):", flush=True)

        # Generate samples
        samples = generate_posterior_samples(
            model=model,
            state_init=state_init,
            time_sequence=time_sequence,
            n_samples=args.n_posterior_samples,
            seed=args.seed  # Same seed for all models for reproducibility
        )
        samples_list.append(np.array(samples))  # Convert to numpy for plotting

        # Compute statistics
        print(f"  Computing statistics...", flush=True)
        stats = compute_posterior_statistics(samples)
        stats_numpy = {k: np.array(v) for k, v in stats.items() if k != 'quantiles'}
        stats_numpy['quantiles'] = {qk: np.array(qv) for qk, qv in stats['quantiles'].items()}
        stats_list.append(stats_numpy)

        print(f"  ✓ Model {i} complete", flush=True)
        print(flush=True)

    generation_time = time.time() - start_gen_time
    print(f"✓ Posterior generation complete in {generation_time:.1f}s", flush=True)
    print(flush=True)

    # Determine output directories
    comparison_mode = (len(models) == 2)
    plot_dir, data_dir = determine_output_directory(run_dirs, modified_configs, metadatas, comparison_mode)
    print(f"Plot directory: {plot_dir}", flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    print(flush=True)

    # Create visualizations
    print("=" * 80, flush=True)
    print("CREATING VISUALIZATIONS", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    # Get observation and latent data
    observations = datasets[0]['observations']

    # Interpolate latent path onto simulation grid if available
    if 'latent_path' in datasets[0]:
        latent_path_dataset = datasets[0]['latent_path']
        obs_times_dataset = datasets[0]['time_sequence']

        # Interpolate each dimension onto simulation grid
        state_dim = latent_path_dataset.shape[1]
        latent_path = jnp.stack([
            jnp.interp(time_sequence, obs_times_dataset, latent_path_dataset[:, dim])
            for dim in range(state_dim)
        ], axis=1)
    else:
        latent_path = None

    if not comparison_mode:
        # Single model visualization
        model_label = f"{configs[0]['model_type']} ({metadatas[0]['run_id']})"
        plot_path = plot_dir / "posteriors.png"

        plot_single_model_posterior(
            samples=samples_list[0],
            stats=stats_list[0],
            time_sequence=time_sequence,
            obs_times=obs_times,
            observations=observations,
            training_indices=training_indices,
            heldout_indices=heldout_indices,
            latent_path=latent_path,
            model_label=model_label,
            output_path=plot_path,
            dpi=args.plot_dpi
        )
    else:
        # Comparison visualization
        model_labels = [
            f"{configs[i]['model_type']} ({metadatas[i]['run_id']})"
            for i in range(2)
        ]
        plot_path = plot_dir / "comparison.png"

        plot_comparison_posterior(
            samples_list=samples_list,
            stats_list=stats_list,
            time_sequence=time_sequence,
            obs_times=obs_times,
            observations=observations,
            training_indices=training_indices,
            heldout_indices=heldout_indices,
            latent_path=latent_path,
            model_labels=model_labels,
            output_path=plot_path,
            dpi=args.plot_dpi
        )

    print(flush=True)

    # Save outputs
    if args.save_samples:
        print("Saving posterior samples and statistics...", flush=True)

        model_names = ['model1', 'model2'] if comparison_mode else ['']

        for i, (samples, stats, model_name) in enumerate(zip(samples_list, stats_list, model_names)):
            # Save samples to data directory
            samples_path = save_posterior_samples(
                samples=samples,
                output_dir=data_dir,
                model_name=model_name,
                seed=args.seed,
                comparison_mode=comparison_mode
            )
            print(f"  ✓ Saved samples: {samples_path.name}", flush=True)

            # Save statistics to data directory
            stats_path = save_posterior_statistics(
                stats=stats,
                output_dir=data_dir,
                model_name=model_name,
                seed=args.seed,
                comparison_mode=comparison_mode
            )
            print(f"  ✓ Saved statistics: {stats_path.name}", flush=True)

        print(flush=True)

    # Save metadata to data directory
    print("Saving generation metadata...", flush=True)
    metadata_path = save_generation_metadata(
        output_dir=data_dir,
        args=args,
        configs=modified_configs,
        metadatas=metadatas,
        override_logs=override_logs,
        stats_list=stats_list,
        generation_time=generation_time,
        seed=args.seed,
        comparison_mode=comparison_mode
    )
    print(f"  ✓ Saved metadata: {metadata_path.name}", flush=True)
    print(flush=True)

    # Print summary
    print("=" * 80, flush=True)
    print("GENERATION COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    if comparison_mode:
        print(f"Output directory: {data_dir}", flush=True)
    else:
        print(f"Plot directory: {plot_dir}", flush=True)
        print(f"Data directory: {data_dir}", flush=True)
    print(f"Visualization: {plot_path.name}", flush=True)
    if args.save_samples:
        print(f"Samples and statistics saved to: {data_dir}", flush=True)
    print(f"Generation time: {generation_time:.1f}s", flush=True)
    print(flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
