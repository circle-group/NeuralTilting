"""
GPU-enabled training orchestration script for Gaussian SDE models.

This is the GPU-aware counterpart to train_gaussian_sde.py.  The original
CPU-only script is left completely untouched so you can revert to it at any time.

Hardware support
----------------
Apple Metal (macOS, M-series or AMD GPU):
    pip install jax-metal

NVIDIA CUDA (Linux / Windows):
    pip install -U "jax[cuda12_pip]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    # CUDA 11 users: use jax[cuda11_pip] instead

CPU fallback:
    Automatic — if no GPU backend is detected the script falls back to CPU and
    prints instructions for enabling GPU on the current platform.

Why this works without changing the simulation code
---------------------------------------------------
JAX selects its compute backend (Metal / CUDA / CPU) once, at import time.
All jax.numpy operations then dispatch automatically to that backend.  The
existing models, loss functions, and training loops do not need any device-aware
changes — they are already pure JAX and will run on whichever backend is active.

Usage (same CLI as the original script)
----------------------------------------
  # New training run
  python training/train_gaussian_sde_gpu.py \
      --config training_configs/gaussian/alpha_1.20_obsstd_0.10_seed_16241891.yaml

  # Continue from parent run
  python training/train_gaussian_sde_gpu.py \
      --config training_configs/gaussian/alpha_1.20_obsstd_0.10_seed_16241891.yaml \
      --parent-run "alpha_1.20_obsstd_0.10/data_16241891_train_12345678"
"""

# =============================================================================
# DEVICE CONFIGURATION
# This block MUST run before JAX is imported.  Once JAX is imported its backend
# is fixed for the lifetime of the process.
# =============================================================================

import os
import sys
import platform
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _detect_gpu_backend() -> dict:
    """
    Detect the best available GPU backend for JAX on the current machine.

    Runs entirely before JAX is imported, so that any necessary plugin packages
    can be loaded (or environment variables set) before JAX locks its backend.

    Returns
    -------
    info : dict
        Keys:
          system          - 'Darwin', 'Linux', 'Windows', …
          machine         - 'arm64', 'x86_64', …
          requested       - backend we are asking JAX to use, or None (auto)
          gpu_likely      - True if we expect a GPU to be available
          install_hint    - human-readable install instructions (or None)
    """
    system  = platform.system()
    machine = platform.machine()

    info = dict(
        system       = system,
        machine      = machine,
        requested    = None,   # backend name or None → let JAX auto-detect
        gpu_likely   = False,
        install_hint = None,
    )

    # ------------------------------------------------------------------
    # macOS — Apple Metal
    # jax-metal registers itself as a JAX plugin via Python entry-points.
    # Importing the package before JAX ensures the plugin is registered.
    # ------------------------------------------------------------------
    if system == 'Darwin':
        metal_spec = importlib.util.find_spec('jax_metal')

        if metal_spec is not None:
            # Trigger plugin registration before JAX is imported.
            try:
                import jax_metal  # noqa: F401
            except Exception:
                pass  # If it fails, JAX will fall back to CPU gracefully.

            info['requested']  = 'metal'
            info['gpu_likely'] = True

        else:
            info['install_hint'] = (
                "To enable Apple GPU (Metal) acceleration:\n"
                "    pip install jax-metal\n"
                "Requires macOS 13.0+ and Apple Silicon or an AMD GPU.\n"
                "See: https://pypi.org/project/jax-metal/"
            )

    # ------------------------------------------------------------------
    # Linux / Windows — NVIDIA CUDA
    # jaxlib built with CUDA support auto-enables GPU; no special import
    # is needed.  We just try to detect whether CUDA jaxlib is installed
    # by checking for known CUDA-specific attributes in jaxlib.
    # ------------------------------------------------------------------
    else:
        jaxlib_spec = importlib.util.find_spec('jaxlib')
        cuda_available = False

        if jaxlib_spec is not None:
            try:
                import jaxlib
                cuda_available = hasattr(jaxlib, 'cuda_versions') or hasattr(jaxlib, 'version')
            except Exception:
                pass

        info['gpu_likely']   = cuda_available
        info['install_hint'] = (
            "To enable NVIDIA GPU (CUDA) acceleration on Linux:\n"
            "    pip install -U \"jax[cuda12_pip]\" \\\n"
            "        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\n"
            "    # For CUDA 11: use jax[cuda11_pip] instead\n"
            "See: https://jax.readthedocs.io/en/latest/installation.html"
        ) if not cuda_available else None

    return info


# Run device detection BEFORE importing JAX.
_pre_import_info = _detect_gpu_backend()


# =============================================================================
# STANDARD IMPORTS  (after device detection)
# =============================================================================

import argparse
import yaml
import json
import random
import pickle
from datetime import datetime
import jax.numpy as jnp
import equinox as eqx

# Library-specific imports
from models.gaussian_sde import GaussianDrivenSDE
from training.train_model import train_gaussian_model
from utils.visualization_utils import plot_training_results
from utils.dataset_utils import load_dataset, get_dataset_path


# =============================================================================
# DEVICE VERIFICATION  (after JAX is imported)
# =============================================================================

import jax

def _build_device_info(pre_import_info: dict) -> dict:
    """
    Confirm which backend JAX actually selected and build a full device info dict.

    Called immediately after JAX is imported.
    """
    info = dict(pre_import_info)  # copy pre-import fields

    try:
        confirmed_backend = jax.default_backend()
        devices           = jax.devices()
        info['confirmed_backend'] = confirmed_backend.lower()
        info['n_devices']         = len(devices)
        info['device_list']       = [str(d) for d in devices]
        info['gpu_active']        = confirmed_backend.lower() not in ('cpu',)
    except Exception as exc:
        info['confirmed_backend'] = 'unknown'
        info['gpu_active']        = False
        info['error']             = str(exc)

    return info


def print_device_summary(device_info: dict) -> None:
    """Print a concise device configuration block to stdout."""
    backend  = device_info.get('confirmed_backend', 'unknown')
    n_dev    = device_info.get('n_devices', '?')
    gpu_on   = device_info.get('gpu_active', False)
    sys_name = device_info.get('system', '')
    machine  = device_info.get('machine', '')

    print("=" * 80, flush=True)
    print("DEVICE CONFIGURATION", flush=True)
    print("=" * 80, flush=True)
    print(f"  Platform : {sys_name} ({machine})", flush=True)
    print(f"  Backend  : {backend}", flush=True)
    print(f"  Devices  : {n_dev}  →  {device_info.get('device_list', [])}", flush=True)

    if gpu_on:
        print("  Status   : GPU acceleration ENABLED", flush=True)
    else:
        print("  Status   : Running on CPU  (no GPU backend detected)", flush=True)
        hint = device_info.get('install_hint')
        if hint:
            print(flush=True)
            print("  How to enable GPU:", flush=True)
            for line in hint.splitlines():
                print(f"    {line}", flush=True)

    if 'error' in device_info:
        print(f"  Warning  : {device_info['error']}", flush=True)

    print(flush=True)


# Build confirmed device info now that JAX has been imported.
_device_info = _build_device_info(_pre_import_info)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Gaussian SDE model with hierarchical run management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training configuration YAML file'
    )

    parser.add_argument(
        '--parent-run',
        type=str,
        default=None,
        help='Parent run path to continue training from (e.g., "alpha_1.30_obsstd_0.10/data_10324553_train_12345678")'
    )

    return parser.parse_args()

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path):
    """Load and validate training configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required top-level keys
    required_keys = ['model_type', 'model', 'data', 'training']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Config missing required keys: {missing_keys}")

    # Validate model type
    if config['model_type'] != 'gaussian_sde':
        raise ValueError(f"Unsupported model_type: {config['model_type']}. Expected 'gaussian_sde'")

    return config

def extract_dataset_params(config):
    """Extract alpha and obs_std from config for dataset identification.

    Note: For Gaussian SDE, alpha identifies which stable dataset to train on,
    but is NOT a parameter of the Gaussian model itself.
    """
    alpha = config['data']['alpha']
    obs_std = config['data']['obs_std']
    data_seed = config['data']['data_seed']

    return alpha, obs_std, data_seed

def validate_parent_run_compatibility(parent_config, current_config):
    """
    Validate that parent run config is compatible with current config.

    Critical parameters that MUST match:
    - Dataset: alpha, obs_std, data_seed (identifies which stable dataset)
    - Model architecture: state_dim, control_width, control_depth, n_time_features
    - Model parameters: sigma
    - Observation subsampling: method, count, seed (must train on same observations)

    Parameters that CAN differ:
    - Training hyperparameters: learning_rate, training_steps, regularization, etc.
    """
    errors = []

    # Dataset parameters (alpha identifies which stable dataset)
    if parent_config['data']['alpha'] != current_config['data']['alpha']:
        errors.append(f"alpha mismatch: parent={parent_config['data']['alpha']}, current={current_config['data']['alpha']}")

    if parent_config['data']['obs_std'] != current_config['data']['obs_std']:
        errors.append(f"obs_std mismatch: parent={parent_config['data']['obs_std']}, current={current_config['data']['obs_std']}")

    if parent_config['data']['data_seed'] != current_config['data']['data_seed']:
        errors.append(f"data_seed mismatch: parent={parent_config['data']['data_seed']}, current={current_config['data']['data_seed']}")

    # Model architecture (critical for loading)
    if parent_config['model']['state_dim'] != current_config['model']['state_dim']:
        errors.append(f"state_dim mismatch: parent={parent_config['model']['state_dim']}, current={current_config['model']['state_dim']}")

    if parent_config['model']['control_width'] != current_config['model']['control_width']:
        errors.append(f"control_width mismatch: parent={parent_config['model']['control_width']}, current={current_config['model']['control_width']}")

    if parent_config['model']['control_depth'] != current_config['model']['control_depth']:
        errors.append(f"control_depth mismatch: parent={parent_config['model']['control_depth']}, current={current_config['model']['control_depth']}")

    if parent_config['model']['n_time_features'] != current_config['model']['n_time_features']:
        errors.append(f"n_time_features mismatch: parent={parent_config['model']['n_time_features']}, current={current_config['model']['n_time_features']}")

    # Model parameters
    if parent_config['model']['sigma'] != current_config['model']['sigma']:
        errors.append(f"sigma mismatch: parent={parent_config['model']['sigma']}, current={current_config['model']['sigma']}")

    # Observation subsampling (must train on same observations)
    if parent_config['training']['obs_subsample_method'] != current_config['training']['obs_subsample_method']:
        errors.append(f"obs_subsample_method mismatch: parent={parent_config['training']['obs_subsample_method']}, current={current_config['training']['obs_subsample_method']}")

    if parent_config['training']['obs_subsample_count'] != current_config['training']['obs_subsample_count']:
        errors.append(f"obs_subsample_count mismatch: parent={parent_config['training']['obs_subsample_count']}, current={current_config['training']['obs_subsample_count']}")

    if parent_config['training']['obs_subsample_seed'] != current_config['training']['obs_subsample_seed']:
        errors.append(f"obs_subsample_seed mismatch: parent={parent_config['training']['obs_subsample_seed']}, current={current_config['training']['obs_subsample_seed']}")

    if errors:
        raise ValueError(
            "Parent run configuration is incompatible with current config:\n  " +
            "\n  ".join(errors) +
            "\n\nWhen continuing training, the following must match:\n" +
            "  - Dataset: alpha, obs_std, data_seed\n" +
            "  - Model: state_dim, control_width, control_depth, n_time_features, sigma\n" +
            "  - Observations: obs_subsample_method, obs_subsample_count, obs_subsample_seed"
        )

    print("✓ Parent run configuration validated successfully", flush=True)

# =============================================================================
# RUN DIRECTORY MANAGEMENT
# =============================================================================

def generate_training_seed():
    """Generate a random 8-digit training seed."""
    return random.randint(10000000, 99999999)


def create_run_directory(alpha, obs_std, data_seed, training_seed, base_dir="training_runs"):
    """
    Create hierarchical run directory structure.

    Structure:
        training_runs/gaussian_sde/alpha_{alpha:.2f}_obsstd_{obs_std:.2f}/data_{data_seed}_train_{training_seed}/

    Returns:
        run_dir (Path): The created run directory
    """
    base_path = Path(base_dir) / "gaussian_sde"

    # Dataset parameter directory
    dataset_dir = base_path / f"alpha_{alpha:.2f}_obsstd_{obs_std:.2f}"

    # Individual run directory
    run_dir = dataset_dir / f"data_{data_seed}_train_{training_seed}"

    # Create all subdirectories
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    return run_dir


def get_parent_run_directory(parent_run_path, base_dir="training_runs"):
    """
    Resolve parent run directory from relative path.

    Args:
        parent_run_path: e.g., "alpha_1.30_obsstd_0.10/data_10324553_train_12345678"
        base_dir: Base directory for training runs

    Returns:
        parent_run_dir (Path): Absolute path to parent run directory
    """
    parent_run_dir = Path(base_dir) / "gaussian_sde" / parent_run_path

    if not parent_run_dir.exists():
        raise FileNotFoundError(f"Parent run directory not found: {parent_run_dir}")

    # Validate required files exist
    required_files = ['model.eqx', 'config.yaml', 'metadata.json']
    missing_files = [f for f in required_files if not (parent_run_dir / f).exists()]

    if missing_files:
        raise FileNotFoundError(
            f"Parent run directory is incomplete. Missing files: {missing_files}\n"
            f"Directory: {parent_run_dir}"
        )

    return parent_run_dir

# =============================================================================
# METADATA AND REGISTRY MANAGEMENT
# =============================================================================

def create_metadata(config, training_seed, device_info, parent_run_path=None):
    """
    Create metadata dictionary for the training run.

    Adds device_backend/device_gpu_active fields so runs can be filtered
    by compute environment.
    """
    alpha, obs_std, data_seed = extract_dataset_params(config)

    metadata = {
        'run_id': f"data_{data_seed}_train_{training_seed}",
        'data_seed': data_seed,
        'training_seed': training_seed,
        'alpha': alpha,
        'obs_std': obs_std,
        'status': 'running',
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'parent_run': parent_run_path,
        'training_steps_completed': 0,
        'final_loss': None,
        # GPU-specific fields
        'device_backend':    device_info.get('confirmed_backend', 'unknown'),
        'device_gpu_active': device_info.get('gpu_active', False),
        'device_n_devices':  device_info.get('n_devices', 1),
    }

    return metadata


def save_metadata(metadata, run_dir):
    """Save metadata to JSON file."""
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def update_metadata(run_dir, updates):
    """Update metadata file with new values."""
    metadata_path = run_dir / "metadata.json"

    # Load existing metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Apply updates
    metadata.update(updates)

    # Save back
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def finalize_run_metadata(run_dir, metrics, success=True):
    """
    Update metadata when run completes or fails.

    Args:
        run_dir: Run directory path
        metrics: Training metrics dictionary
        success: Whether training completed successfully
    """
    updates = {
        'status': 'completed' if success else 'failed',
        'completed_at': datetime.now().isoformat(),
        'training_steps_completed': len(metrics.get('loss_history', [])),
        'final_loss': float(metrics.get('final_loss', -1)),
    }

    update_metadata(run_dir, updates)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(config):
    """
    Load dataset from the pre-generated datasets folder.

    NOTE: Gaussian SDE models are trained on tilted stable SDE datasets
    for cross-model performance comparison.

    Args:
        config: Training configuration dictionary

    Returns:
        data_dict: Dictionary containing observations, latent_path, time_sequence, etc.
    """
    alpha, obs_std, data_seed = extract_dataset_params(config)

    # Construct dataset path using existing utility
    # This loads from datasets/tilted_stable_sde/
    dataset_path = get_dataset_path(alpha, obs_std, data_seed)

    print(f"Loading dataset from: {dataset_path}", flush=True)

    # Load dataset
    with open(dataset_path, 'rb') as f:
        data_dict = pickle.load(f)

    # Validate dataset parameters match config (sanity check)
    if data_dict['alpha'] != alpha:
        raise ValueError(
            f"Dataset alpha mismatch: file={data_dict['alpha']}, config={alpha}"
        )

    if data_dict['obs_std'] != obs_std:
        raise ValueError(
            f"Dataset obs_std mismatch: file={data_dict['obs_std']}, config={obs_std}"
        )

    if data_dict['data_seed'] != data_seed:
        raise ValueError(
            f"Dataset data_seed mismatch: file={data_dict['data_seed']}, config={data_seed}"
        )

    print(f"✓ Loaded dataset with {len(data_dict['time_sequence'])} time points", flush=True)
    print(f"  Alpha (of stable dataset): {data_dict['alpha']}, Sigma: {data_dict['sigma']}, Tau: {data_dict['tau']}", flush=True)
    print(f"  Observation noise std: {data_dict['obs_std']}", flush=True)

    return data_dict


def prepare_initial_state(config):
    """
    Prepare initial state from configuration.

    Args:
        config: Training configuration dictionary

    Returns:
        state_init: JAX array of initial state
    """
    state_dim = config['model']['state_dim']

    if config['data']['state_init_vector'] is not None:
        state_init = jnp.array(config['data']['state_init_vector'])

        if len(state_init) != state_dim:
            raise ValueError(
                f"state_init_vector length ({len(state_init)}) must match "
                f"state_dim ({state_dim})"
            )
    else:
        # Use scalar value for all dimensions
        state_init = jnp.full((state_dim,), config['data']['state_init'])

    print(f"Initial state: {state_init}", flush=True)

    return state_init

# =============================================================================
# MODEL INITIALIZATION AND LOADING
# =============================================================================

def prepare_model_params(config, data_dict):
    """
    Prepare model parameters dictionary for GaussianDrivenSDE.

    Args:
        config: Training configuration dictionary
        data_dict: Loaded dataset dictionary

    Returns:
        model_params: Dictionary of model parameters
    """
    # Generate random seeds for any None values
    drift_seed = config['model']['drift_seed']
    if drift_seed is None:
        drift_seed = random.randint(10000000, 99999999)
        print(f"  Generated drift_seed: {drift_seed}", flush=True)

    diffusion_seed = config['model']['diffusion_seed']
    if diffusion_seed is None:
        diffusion_seed = random.randint(10000000, 99999999)
        print(f"  Generated diffusion_seed: {diffusion_seed}", flush=True)

    control_seed = config['model']['control_seed']
    if control_seed is None:
        control_seed = random.randint(10000000, 99999999)
        print(f"  Generated control_seed: {control_seed}", flush=True)

    model_params = {
        'state_dim': config['model']['state_dim'],
        'sigma': config['model']['sigma'],
        'drift_seed': drift_seed,
        'diffusion_seed': diffusion_seed,
        'control_seed': control_seed,
        'control_width': config['model']['control_width'],
        'control_depth': config['model']['control_depth'],
        'n_time_features': config['model']['n_time_features'],
        'trainable_drift': config['model']['trainable_drift'],
        'initial_drift_weight': config['model']['initial_drift_weight'],
        'initial_drift_bias': config['model']['initial_drift_bias'],
        'initial_diffusion_weight': config['model']['initial_diffusion_weight'],
        'period': config['data']['time_end'],
    }

    return model_params


def initialize_or_load_model(config, model_params, parent_run_dir=None):
    """
    Initialize new model or load from parent run.

    Note: Parent run compatibility should already be validated before calling this function.

    Args:
        config: Training configuration dictionary
        model_params: Model parameters dictionary
        parent_run_dir: Optional parent run directory to load model from

    Returns:
        model: GaussianDrivenSDE model instance
        model_source: String indicating model source ('new' or parent run path)
    """
    if parent_run_dir is not None:
        # Load model from parent run
        parent_model_path = parent_run_dir / "model.eqx"

        print(f"Loading model from parent run: {parent_model_path}", flush=True)

        # Create template model for deserialization
        template_model = GaussianDrivenSDE(**model_params)

        # Load trained model
        model = eqx.tree_deserialise_leaves(parent_model_path, template_model)

        print("✓ Model loaded from parent run", flush=True)

        model_source = str(parent_run_dir.relative_to(Path("training_runs") / "gaussian_sde"))

    else:
        # Initialize new model
        print("Initializing new model...", flush=True)

        model = GaussianDrivenSDE(**model_params)

        print("✓ New model initialized", flush=True)

        model_source = 'new'

    return model, model_source

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

def prepare_training_params(config):
    """
    Prepare training parameters dictionary.

    Args:
        config: Training configuration dictionary

    Returns:
        training_params: Dictionary of training parameters
    """
    training_config = config['training']

    training_params = {
        # -------------------------------------------------------------------------
        # Optimizer configuration
        # -------------------------------------------------------------------------
        'learning_rate': training_config['learning_rate'],
        'transition_steps': training_config['transition_steps'],
        'decay_rate': training_config['decay_rate'],

        # -------------------------------------------------------------------------
        # Training loop
        # -------------------------------------------------------------------------
        'training_steps': training_config['training_steps'],
        'n_loss_samples': training_config['n_loss_samples'],
        'n_latent_steps': training_config['n_latent_steps'],

        # -------------------------------------------------------------------------
        # Regularization
        # -------------------------------------------------------------------------
        'control_regularisation': training_config['control_regularisation'],
        'drift_regularization': training_config['drift_regularization'],

        # -------------------------------------------------------------------------
        # Memory management (for HPC environments)
        # -------------------------------------------------------------------------
        'gc_interval': training_config.get('gc_interval', 50),
        'clear_cache_interval': training_config.get('clear_cache_interval', 150),

        # -------------------------------------------------------------------------
        # Checkpointing
        # -------------------------------------------------------------------------
        'checkpoint_interval': training_config['checkpoint_interval'],
    }

    return training_params

# =============================================================================
# MAIN TRAINING ORCHESTRATION
# =============================================================================

def main():
    """Main training orchestration function."""

    # Parse arguments
    args = parse_arguments()

    # -------------------------------------------------------------------------
    # Print device configuration first so the user knows what hardware is used
    # -------------------------------------------------------------------------
    print_device_summary(_device_info)

    print("="*80, flush=True)
    print("GAUSSIAN SDE TRAINING  (GPU-enabled script)", flush=True)
    print("="*80, flush=True)
    print(flush=True)

    # Load configuration
    print("Loading configuration...", flush=True)
    config = load_config(args.config)
    alpha, obs_std, data_seed = extract_dataset_params(config)
    print(f"✓ Configuration loaded from: {args.config}", flush=True)
    print(f"  Dataset (stable SDE): alpha={alpha}, obs_std={obs_std}, data_seed={data_seed}", flush=True)
    print(flush=True)

    # Validate dataset exists BEFORE creating any directories
    print("Validating dataset availability...", flush=True)
    dataset_path = get_dataset_path(alpha, obs_std, data_seed)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Expected dataset with:\n"
            f"  - alpha: {alpha}\n"
            f"  - obs_std: {obs_std}\n"
            f"  - data_seed: {data_seed}\n"
            f"Please generate the dataset first before starting training."
        )
    print(f"✓ Dataset found: {dataset_path}", flush=True)
    print(flush=True)

    # Handle parent run if specified and validate compatibility EARLY
    parent_run_dir = None
    if args.parent_run is not None:
        print("Resolving parent run...", flush=True)
        parent_run_dir = get_parent_run_directory(args.parent_run)
        print(f"✓ Parent run found: {parent_run_dir}", flush=True)

        # Validate compatibility BEFORE creating any directories
        print("Validating parent run compatibility...", flush=True)
        parent_config_path = parent_run_dir / "config.yaml"
        with open(parent_config_path, 'r') as f:
            parent_config = yaml.safe_load(f)

        validate_parent_run_compatibility(parent_config, config)
        print(flush=True)

    # Generate training seed and create run directory
    training_seed = generate_training_seed()
    run_dir = create_run_directory(alpha, obs_std, data_seed, training_seed)

    print(f"Created run directory: {run_dir}", flush=True)
    print(f"Training seed: {training_seed}", flush=True)
    print(flush=True)

    # Save configuration to run directory
    config_save_path = run_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Configuration saved to: {config_save_path}", flush=True)

    # Create and save initial metadata (includes device info)
    metadata = create_metadata(config, training_seed, _device_info, parent_run_path=args.parent_run)
    save_metadata(metadata, run_dir)
    print(f"✓ Metadata initialized  (backend: {_device_info.get('confirmed_backend', 'unknown')})", flush=True)

    # Load dataset
    print("Loading dataset...", flush=True)
    data_dict = load_training_data(config)
    observations = data_dict['observations']
    time_sequence = data_dict['time_sequence']
    true_drift_component = data_dict.get('drift_component', None)
    print(flush=True)

    # Prepare initial state
    print("Preparing initial state...", flush=True)
    state_init = prepare_initial_state(config)
    print(flush=True)

    # Prepare model parameters
    print("Preparing model parameters...", flush=True)
    model_params = prepare_model_params(config, data_dict)

    # Initialize or load model
    model, model_source = initialize_or_load_model(config, model_params, parent_run_dir)
    print(flush=True)

    # Prepare training parameters
    training_params = prepare_training_params(config)

    # Setup output paths
    output_paths = {
        'run': run_dir,
        'checkpoints': run_dir / "checkpoints",
        'plots': run_dir / "plots",
        'model': run_dir / "model.eqx",
        'metrics': run_dir / "metrics.pkl",
        'metadata': run_dir / "metadata.json",
    }

    # Train model
    print("="*80, flush=True)
    print("STARTING TRAINING", flush=True)
    print("="*80, flush=True)
    print(flush=True)

    try:
        trained_model, metrics = train_gaussian_model(
            # Data
            observations=observations,
            time_sequence=time_sequence,
            state_init=state_init,
            obs_std=config['data']['obs_std'],

            # Observation subsampling
            obs_subsample_method=config['training']['obs_subsample_method'],
            obs_subsample_count=config['training']['obs_subsample_count'],
            obs_subsample_seed=config['training']['obs_subsample_seed'],

            # Model parameters
            model_params=model_params,

            # Training parameters
            training_params=training_params,

            # Seeds
            training_seed=training_seed,

            # Output paths
            output_paths=output_paths,

            # Optional: pass pre-loaded model
            model=model,
            verbose=True,
        )

        # If continuing from parent run, merge metrics histories
        if parent_run_dir is not None:
            print("\nMerging training history with parent run...", flush=True)
            parent_metrics_path = parent_run_dir / "metrics.pkl"

            if parent_metrics_path.exists():
                with open(parent_metrics_path, 'rb') as f:
                    parent_metrics = pickle.load(f)

                # Prepend parent's loss history
                if 'loss_history' in parent_metrics and 'loss_history' in metrics:
                    metrics['loss_history'] = parent_metrics['loss_history'] + metrics['loss_history']
                    print(f"  Combined loss history: {len(parent_metrics['loss_history'])} (parent) + {len(metrics['loss_history']) - len(parent_metrics['loss_history'])} (current) = {len(metrics['loss_history'])} steps", flush=True)

                # Prepend parent's parameter history if it exists
                if 'parameter_history' in parent_metrics and 'parameter_history' in metrics:
                    metrics['parameter_history'] = parent_metrics['parameter_history'] + metrics['parameter_history']

                print("✓ Training histories merged", flush=True)
            else:
                print(f"⚠ Warning: Parent metrics file not found at {parent_metrics_path}", flush=True)
                print("  Continuing with current run metrics only", flush=True)

        print(flush=True)
        print("="*80, flush=True)
        print("TRAINING COMPLETED SUCCESSFULLY", flush=True)
        print("="*80, flush=True)
        print(flush=True)

        # -----------------------------------------------------------------------
        # NaN GUARD: check model parameters and fall back to best checkpoint
        # -----------------------------------------------------------------------
        model_leaves = jax.tree_util.tree_leaves(eqx.filter(trained_model, eqx.is_inexact_array))
        model_has_nan = any(bool(jnp.any(jnp.isnan(leaf))) for leaf in model_leaves)

        model_fallback = None  # recorded in metadata
        if model_has_nan:
            print("⚠ Final model contains NaN parameters.", flush=True)
            # Try checkpoints in reverse order (latest to earliest)
            fallback_candidates = sorted(
                output_paths['checkpoints'].glob("step_*.eqx"),
                key=lambda p: int(p.stem.split("_")[1]),
                reverse=True
            )
            for ckpt_path in fallback_candidates:
                template = GaussianDrivenSDE(**model_params)
                candidate = eqx.tree_deserialise_leaves(ckpt_path, template)
                candidate_leaves = jax.tree_util.tree_leaves(eqx.filter(candidate, eqx.is_inexact_array))
                if not any(bool(jnp.any(jnp.isnan(leaf))) for leaf in candidate_leaves):
                    trained_model = candidate
                    model_fallback = ckpt_path.name
                    print(f"  → Fell back to checkpoint: {ckpt_path.name}", flush=True)
                    break
            else:
                print("  → All checkpoints also contain NaN. Saving NaN model as-is.", flush=True)
                model_fallback = "none_all_nan"
        # -----------------------------------------------------------------------

        # Save final model
        print(f"Saving trained model to: {output_paths['model']}", flush=True)
        eqx.tree_serialise_leaves(output_paths['model'], trained_model)
        print("✓ Model saved", flush=True)

        # Save metrics
        print(f"Saving metrics to: {output_paths['metrics']}", flush=True)
        with open(output_paths['metrics'], 'wb') as f:
            pickle.dump(metrics, f)
        print("✓ Metrics saved", flush=True)

        # Generate plots if requested
        if config.get('output', {}).get('save_plots', True):
            print("\nGenerating training plots...", flush=True)

            # Prepare config dicts for plotting
            model_config_dict = {
                'state_dim': config['model']['state_dim'],
                'sigma': config['model']['sigma'],
                'control_width': config['model']['control_width'],
                'control_depth': config['model']['control_depth'],
                'trainable_drift': config['model']['trainable_drift'],
            }

            training_config_dict = {
                'learning_rate': config['training']['learning_rate'],
                'n_loss_samples': config['training']['n_loss_samples'],
                'n_latent_steps': config['training']['n_latent_steps'],
            }

            plot_dpi = config.get('output', {}).get('plot_dpi', 200)

            plot_training_results(
                metrics=metrics,
                model_config=model_config_dict,
                training_config=training_config_dict,
                output_paths=output_paths,
                plot_dpi=plot_dpi,
                true_drift_component=true_drift_component
            )

            print("✓ Training plots generated", flush=True)

        # Update metadata with completion status (+ NaN fallback info if applicable)
        finalize_run_metadata(run_dir, metrics, success=True)
        if model_fallback is not None:
            update_metadata(run_dir, {'model_fallback': model_fallback})

        # Print summary
        print("\n" + "="*80, flush=True)
        print("TRAINING SUMMARY", flush=True)
        print("="*80, flush=True)
        print(f"Run directory: {run_dir}", flush=True)
        print(f"Run ID: {metadata['run_id']}", flush=True)
        print(f"Training seed: {training_seed}", flush=True)
        print(f"Device: {_device_info.get('confirmed_backend', 'unknown')}  "
              f"({'GPU' if _device_info.get('gpu_active') else 'CPU'})", flush=True)
        if args.parent_run:
            print(f"Parent run: {args.parent_run}", flush=True)
        print(f"\nFinal loss: {metrics['final_loss']:.6f}", flush=True)
        print(f"Total training steps: {len(metrics['loss_history'])}", flush=True)

        # Print drift parameters if trainable
        if 'trained_drift_component' in metrics:
            print(f"\nTrained drift component: {metrics['drift_type']}", flush=True)
            drift = metrics['trained_drift_component']

            if hasattr(drift, 'theta') and hasattr(drift, 'mu'):
                theta_val = float(drift.theta.item()) if drift.theta.ndim > 0 else float(drift.theta)
                mu_val = float(drift.mu.item()) if drift.mu.ndim > 0 else float(drift.mu)
                print(f"  θ (mean-reversion rate): {theta_val:.6f}", flush=True)
                print(f"  μ (mean-reversion level): {mu_val:.6f}", flush=True)

                if true_drift_component is not None and hasattr(true_drift_component, 'theta'):
                    true_theta = float(true_drift_component.theta.item()) if true_drift_component.theta.ndim > 0 else float(true_drift_component.theta)
                    true_mu = float(true_drift_component.mu.item()) if true_drift_component.mu.ndim > 0 else float(true_drift_component.mu)
                    print(f"\nTrue OU parameters:", flush=True)
                    print(f"  θ: {true_theta:.6f}", flush=True)
                    print(f"  μ: {true_mu:.6f}", flush=True)
                    print(f"\nErrors:", flush=True)
                    print(f"  θ error: {abs(theta_val - true_theta):.6f}", flush=True)
                    print(f"  μ error: {abs(mu_val - true_mu):.6f}", flush=True)

        print("\n" + "="*80, flush=True)
        print("OUTPUT FILES:", flush=True)
        print(f"  Model:    {output_paths['model']}", flush=True)
        print(f"  Metrics:  {output_paths['metrics']}", flush=True)
        print(f"  Metadata: {output_paths['metadata']}", flush=True)
        print(f"  Config:   {config_save_path}", flush=True)
        print(f"  Plots:    {output_paths['plots']}", flush=True)
        if config['training']['checkpoint_interval'] > 0:
            print(f"  Checkpoints: {output_paths['checkpoints']}", flush=True)
        print("="*80, flush=True)
        print(flush=True)

    except Exception as e:
        print("\n" + "="*80, flush=True)
        print("TRAINING FAILED", flush=True)
        print("="*80, flush=True)
        print(f"Error: {e}", flush=True)
        print(flush=True)

        # Update metadata with failure status
        finalize_run_metadata(run_dir, {}, success=False)

        raise


if __name__ == "__main__":
    main()
