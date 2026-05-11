"""
GPU-enabled training orchestration script for tilted stable SDE models with double well drift.

Identical to train_tilted_stable_sde_gpu.py except:
- Model class: TiltedStableDrivenSDEDoubleWell (drift f(x) = θ₁x - θ₂x³)
- model_type: 'tilted_stable_sde_double_well'
- Run directory: training_runs/tilted_stable_sde_double_well/
- Dataset path: datasets/tilted_stable_sde_double_well/
- Drift params: initial_drift_theta1 / initial_drift_theta2 (instead of weight/bias)

Usage
-----
  # New training run
  python training/train_tilted_stable_sde_double_well_gpu.py \
      --config training_configs/tilted_stable_double_well/alpha_1.20_obsstd_0.10_seed_<seed>.yaml

  # Continue from parent run
  python training/train_tilted_stable_sde_double_well_gpu.py \
      --config training_configs/tilted_stable_double_well/alpha_1.20_obsstd_0.10_seed_<seed>.yaml \
      --parent-run "alpha_1.20_obsstd_0.10/data_<data_seed>_train_<train_seed>"
"""

# =============================================================================
# DEVICE CONFIGURATION  (must run before JAX is imported)
# =============================================================================

import os
import sys
import platform
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _detect_gpu_backend() -> dict:
    system  = platform.system()
    machine = platform.machine()

    info = dict(
        system       = system,
        machine      = machine,
        requested    = None,
        gpu_likely   = False,
        install_hint = None,
    )

    if system == 'Darwin':
        metal_spec = importlib.util.find_spec('jax_metal')
        if metal_spec is not None:
            try:
                import jax_metal  # noqa: F401
            except Exception:
                pass
            info['requested']  = 'metal'
            info['gpu_likely'] = True
        else:
            info['install_hint'] = (
                "To enable Apple GPU (Metal) acceleration:\n"
                "    pip install jax-metal\n"
                "Requires macOS 13.0+ and Apple Silicon or an AMD GPU.\n"
                "See: https://pypi.org/project/jax-metal/"
            )
    else:
        jaxlib_spec    = importlib.util.find_spec('jaxlib')
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
            "See: https://jax.readthedocs.io/en/latest/installation.html"
        ) if not cuda_available else None

    return info


_pre_import_info = _detect_gpu_backend()


# =============================================================================
# STANDARD IMPORTS
# =============================================================================

import argparse
import yaml
import json
import random
import pickle
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import equinox as eqx

from models.tilted_stable_sde_double_well import TiltedStableDrivenSDEDoubleWell
from training.train_model import train_tilted_stable_model
from utils.visualization_utils import plot_training_results
from utils.dataset_utils import load_dataset, get_dataset_path


# =============================================================================
# DEVICE VERIFICATION
# =============================================================================

def _build_device_info(pre_import_info: dict) -> dict:
    info = dict(pre_import_info)
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


_device_info = _build_device_info(_pre_import_info)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train tilted stable SDE model with double well drift (GPU support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration YAML file')
    parser.add_argument('--parent-run', type=str, default=None,
                        help='Parent run path to continue training from')
    return parser.parse_args()


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    required_keys = ['model_type', 'model', 'data', 'training']
    missing_keys  = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Config missing required keys: {missing_keys}")

    if config['model_type'] != 'tilted_stable_sde_double_well':
        raise ValueError(f"Unsupported model_type: {config['model_type']}. "
                         f"Expected 'tilted_stable_sde_double_well'")

    return config


def extract_dataset_params(config):
    alpha     = config['model']['alpha']
    obs_std   = config['data']['obs_std']
    data_seed = config['data']['data_seed']
    return alpha, obs_std, data_seed


def validate_parent_run_compatibility(parent_config, current_config):
    errors = []

    parent_model    = parent_config['model']
    current_model   = current_config['model']
    parent_training = parent_config['training']
    current_training = current_config['training']

    if parent_model['alpha'] != current_model['alpha']:
        errors.append(f"alpha mismatch: parent={parent_model['alpha']}, current={current_model['alpha']}")

    if parent_config['data']['obs_std'] != current_config['data']['obs_std']:
        errors.append(f"obs_std mismatch: parent={parent_config['data']['obs_std']}, current={current_config['data']['obs_std']}")

    if parent_config['data']['data_seed'] != current_config['data']['data_seed']:
        errors.append(f"data_seed mismatch: parent={parent_config['data']['data_seed']}, current={current_config['data']['data_seed']}")

    if parent_model['state_dim'] != current_model['state_dim']:
        errors.append(f"state_dim mismatch: parent={parent_model['state_dim']}, current={current_model['state_dim']}")

    parent_width  = parent_model.get('tilting_width', 64)
    current_width = current_model.get('tilting_width', 64)
    if parent_width != current_width:
        errors.append(f"tilting_width mismatch: parent={parent_width}, current={current_width}")

    parent_depth  = parent_model.get('tilting_depth', 3)
    current_depth = current_model.get('tilting_depth', 3)
    if parent_depth != current_depth:
        errors.append(f"tilting_depth mismatch: parent={parent_depth}, current={current_depth}")

    parent_n_time  = parent_model.get('n_time_features', 0)
    current_n_time = current_model.get('n_time_features', 0)
    if parent_n_time != current_n_time:
        errors.append(f"n_time_features mismatch: parent={parent_n_time}, current={current_n_time}")

    parent_n_attn_refs  = parent_model.get('n_attention_references', 0)
    current_n_attn_refs = current_model.get('n_attention_references', 0)
    if parent_n_attn_refs != current_n_attn_refs:
        errors.append(f"n_attention_references mismatch: parent={parent_n_attn_refs}, current={current_n_attn_refs}")

    parent_attn_embed_dim  = parent_model.get('attention_embed_dim', 8)
    current_attn_embed_dim = current_model.get('attention_embed_dim', 8)
    if parent_attn_embed_dim != current_attn_embed_dim:
        errors.append(f"attention_embed_dim mismatch: parent={parent_attn_embed_dim}, current={current_attn_embed_dim}")

    if parent_model['sigma'] != current_model['sigma']:
        errors.append(f"sigma mismatch: parent={parent_model['sigma']}, current={current_model['sigma']}")

    if parent_model['tau'] != current_model['tau']:
        errors.append(f"tau mismatch: parent={parent_model['tau']}, current={current_model['tau']}")

    if parent_training['obs_subsample_method'] != current_training['obs_subsample_method']:
        errors.append(f"obs_subsample_method mismatch: parent={parent_training['obs_subsample_method']}, current={current_training['obs_subsample_method']}")

    if parent_training['obs_subsample_count'] != current_training['obs_subsample_count']:
        errors.append(f"obs_subsample_count mismatch: parent={parent_training['obs_subsample_count']}, current={current_training['obs_subsample_count']}")

    if parent_training['obs_subsample_seed'] != current_training['obs_subsample_seed']:
        errors.append(f"obs_subsample_seed mismatch: parent={parent_training['obs_subsample_seed']}, current={current_training['obs_subsample_seed']}")

    if errors:
        raise ValueError(
            "Parent run configuration is incompatible with current config:\n  " +
            "\n  ".join(errors) +
            "\n\nWhen continuing training, the following must match:\n"
            "  - Dataset: alpha, obs_std, data_seed\n"
            "  - Model: state_dim, tilting_width/depth, n_time_features, sigma, tau\n"
            "  - Attention: n_attention_references, attention_embed_dim\n"
            "  - Observations: obs_subsample_method, obs_subsample_count, obs_subsample_seed\n"
            "\nParameters that CAN differ:\n"
            "  - Training: learning_rate, training_steps, regularization, etc."
        )

    print("✓ Parent run configuration validated successfully", flush=True)


# =============================================================================
# RUN DIRECTORY MANAGEMENT
# =============================================================================

def generate_training_seed():
    return random.randint(10000000, 99999999)


def create_run_directory(alpha, obs_std, data_seed, training_seed, base_dir="training_runs"):
    base_path   = Path(base_dir) / "tilted_stable_sde_double_well"
    dataset_dir = base_path / f"alpha_{alpha:.2f}_obsstd_{obs_std:.2f}"
    run_dir     = dataset_dir / f"data_{data_seed}_train_{training_seed}"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    return run_dir


def get_parent_run_directory(parent_run_path, base_dir="training_runs"):
    parent_run_dir = Path(base_dir) / "tilted_stable_sde_double_well" / parent_run_path

    if not parent_run_dir.exists():
        raise FileNotFoundError(f"Parent run directory not found: {parent_run_dir}")

    required_files = ['config.yaml', 'metadata.json']
    missing_files  = [f for f in required_files if not (parent_run_dir / f).exists()]

    if missing_files:
        raise FileNotFoundError(
            f"Parent run directory is incomplete. Missing files: {missing_files}\n"
            f"Directory: {parent_run_dir}\n"
            f"Note: model.eqx is optional (can continue from checkpoints)"
        )

    return parent_run_dir


# =============================================================================
# METADATA MANAGEMENT
# =============================================================================

def create_metadata(config, training_seed, device_info, parent_run_path=None):
    alpha, obs_std, data_seed = extract_dataset_params(config)

    metadata = {
        'run_id':                   f"data_{data_seed}_train_{training_seed}",
        'data_seed':                data_seed,
        'training_seed':            training_seed,
        'alpha':                    alpha,
        'obs_std':                  obs_std,
        'status':                   'running',
        'created_at':               datetime.now().isoformat(),
        'completed_at':             None,
        'parent_run':               parent_run_path,
        'training_steps_completed': 0,
        'final_loss':               None,
        'device_backend':           device_info.get('confirmed_backend', 'unknown'),
        'device_gpu_active':        device_info.get('gpu_active', False),
        'device_n_devices':         device_info.get('n_devices', 1),
    }

    return metadata


def save_metadata(metadata, run_dir):
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def update_metadata(run_dir, updates):
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    metadata.update(updates)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def finalize_run_metadata(run_dir, metrics, success=True):
    updates = {
        'status':                   'completed' if success else 'failed',
        'completed_at':             datetime.now().isoformat(),
        'training_steps_completed': len(metrics.get('loss_history', [])),
        'final_loss':               float(metrics.get('final_loss', -1)),
    }
    update_metadata(run_dir, updates)


# =============================================================================
# DATA LOADING
# =============================================================================

_DOUBLE_WELL_DATASET_BASE = Path("datasets/tilted_stable_sde_double_well")


def load_training_data(config):
    alpha, obs_std, data_seed = extract_dataset_params(config)

    dataset_path = get_dataset_path(alpha, obs_std, data_seed,
                                    base_path=_DOUBLE_WELL_DATASET_BASE)

    print(f"Loading dataset from: {dataset_path}", flush=True)

    with open(dataset_path, 'rb') as f:
        data_dict = pickle.load(f)

    if data_dict['alpha'] != alpha:
        raise ValueError(f"Dataset alpha mismatch: file={data_dict['alpha']}, config={alpha}")

    if data_dict['obs_std'] != obs_std:
        raise ValueError(f"Dataset obs_std mismatch: file={data_dict['obs_std']}, config={obs_std}")

    if data_dict['data_seed'] != data_seed:
        raise ValueError(f"Dataset data_seed mismatch: file={data_dict['data_seed']}, config={data_seed}")

    print(f"✓ Loaded dataset with {len(data_dict['time_sequence'])} time points", flush=True)
    print(f"  Alpha: {data_dict['alpha']}, Sigma: {data_dict['sigma']}, Tau: {data_dict['tau']}", flush=True)
    print(f"  Observation noise std: {data_dict['obs_std']}", flush=True)

    if 'gt_theta1' in data_dict and 'gt_theta2' in data_dict:
        print(f"  Ground truth θ₁: {data_dict['gt_theta1']}", flush=True)
        print(f"  Ground truth θ₂: {data_dict['gt_theta2']}", flush=True)
        if 'gt_well_locations' in data_dict:
            print(f"  Well locations ±√(θ₁/θ₂): ±{data_dict['gt_well_locations']}", flush=True)

    return data_dict


def prepare_initial_state(config):
    state_dim = config['model']['state_dim']

    if config['data']['state_init_vector'] is not None:
        state_init = jnp.array(config['data']['state_init_vector'])
        if len(state_init) != state_dim:
            raise ValueError(
                f"state_init_vector length ({len(state_init)}) must match "
                f"state_dim ({state_dim})"
            )
    else:
        state_init = jnp.full((state_dim,), config['data']['state_init'])

    print(f"Initial state: {state_init}", flush=True)
    return state_init


# =============================================================================
# MODEL INITIALIZATION AND LOADING
# =============================================================================

def prepare_model_params(config, data_dict):
    model_config = config['model']

    drift_seed = model_config.get('drift_seed')
    if drift_seed is None:
        drift_seed = random.randint(10000000, 99999999)
        print(f"  Generated drift_seed: {drift_seed}", flush=True)

    diffusion_seed = model_config.get('diffusion_seed')
    if diffusion_seed is None:
        diffusion_seed = random.randint(10000000, 99999999)
        print(f"  Generated diffusion_seed: {diffusion_seed}", flush=True)

    phi_seed = model_config.get('phi_seed')
    if phi_seed is None:
        phi_seed = random.randint(10000000, 99999999)
        print(f"  Generated phi_seed: {phi_seed}", flush=True)

    model_params = {
        'state_dim':              model_config['state_dim'],
        'alpha':                  model_config['alpha'],
        'tau':                    model_config['tau'],
        'sigma':                  model_config['sigma'],
        'loss_sample_size':       model_config['loss_sample_size'],
        'max_rejection_attempts': model_config.get('max_rejection_attempts', 100),
        'max_jumps':              model_config.get('max_jumps', 10000),
        'tilting_width':          model_config.get('tilting_width', 64),
        'tilting_depth':          model_config.get('tilting_depth', 3),
        'drift_seed':             drift_seed,
        'diffusion_seed':         diffusion_seed,
        'phi_seed':               phi_seed,
        'trainable_drift':        model_config['trainable_drift'],
        'initial_drift_theta1':   model_config.get('initial_drift_theta1'),
        'initial_drift_theta2':   model_config.get('initial_drift_theta2'),
        'initial_diffusion_weight': model_config.get('initial_diffusion_weight'),
        'period':                 config['data']['time_end'],
        'n_time_features':        model_config.get('n_time_features', 0),
        'a_min':                  model_config.get('a_min', 1e-3),
        'use_adaptive_scaling':   model_config.get('use_adaptive_scaling', False),
        'n_attention_references': model_config.get('n_attention_references', 0),
        'attention_embed_dim':    model_config.get('attention_embed_dim', 16),
        'attention_sharpness':    model_config.get('attention_sharpness', 100.0),
    }

    return model_params


def find_latest_checkpoint(checkpoints_dir):
    if not checkpoints_dir.exists():
        return None
    checkpoint_files = list(checkpoints_dir.glob("step_*.eqx"))
    if not checkpoint_files:
        return None
    def get_step_number(path):
        return int(path.stem.split('_')[1])
    latest_checkpoint = max(checkpoint_files, key=get_step_number)
    return latest_checkpoint, get_step_number(latest_checkpoint)


def initialize_or_load_model(config, model_params, parent_run_dir=None):
    checkpoint_step = None

    if parent_run_dir is not None:
        parent_model_path = parent_run_dir / "model.eqx"

        if parent_model_path.exists():
            print(f"Loading model from parent run: {parent_model_path}", flush=True)
            model_path   = parent_model_path
            model_source = str(parent_run_dir.relative_to(
                Path("training_runs") / "tilted_stable_sde_double_well"
            ))
        else:
            print(f"⚠ model.eqx not found in parent run (training may have failed)", flush=True)
            print(f"  Attempting to recover from checkpoints...", flush=True)

            checkpoints_dir   = parent_run_dir / "checkpoints"
            checkpoint_result = find_latest_checkpoint(checkpoints_dir)

            if checkpoint_result is None:
                raise FileNotFoundError(
                    f"Cannot load model from parent run: {parent_run_dir}\n"
                    f"  - model.eqx not found\n"
                    f"  - No checkpoints found in {checkpoints_dir}\n"
                    f"Unable to continue training."
                )

            model_path, checkpoint_step = checkpoint_result
            print(f"✓ Found checkpoint at step {checkpoint_step}: {model_path.name}", flush=True)
            model_source = (
                f"{parent_run_dir.relative_to(Path('training_runs') / 'tilted_stable_sde_double_well')} "
                f"(checkpoint step {checkpoint_step})"
            )

        template_model = TiltedStableDrivenSDEDoubleWell(**model_params)
        model          = eqx.tree_deserialise_leaves(model_path, template_model)
        print("✓ Model loaded successfully", flush=True)

    else:
        print("Initializing new model...", flush=True)
        model        = TiltedStableDrivenSDEDoubleWell(**model_params)
        model_source = 'new'
        print("✓ New model initialized", flush=True)

    return model, model_source, checkpoint_step


# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

def prepare_training_params(config):
    training_config = config['training']

    training_params = {
        'learning_rate':           training_config['learning_rate'],
        'lr_multiplier_mlp':       training_config.get('lr_multiplier_mlp', 1.0),
        'lr_multiplier_attention': training_config.get('lr_multiplier_attention', 10.0),
        'lr_multiplier_drift':     training_config.get('lr_multiplier_drift', 1000.0),
        'frozen_params':           training_config.get('frozen_params', ['diffusion.raw_weight']),
        'training_steps':          training_config['training_steps'],
        'n_loss_samples':          training_config['n_loss_samples'],
        'n_latent_steps':          training_config['n_latent_steps'],
        'tilting_regularisation':  training_config.get('tilting_regularisation', 0.05),
        'drift_regularization':    training_config.get('drift_regularization', 0.01),
        'coeff_A_regularization':  training_config.get('coeff_A_regularization', 1e-4),
        'coeff_B_regularization':  training_config.get('coeff_B_regularization', 5e-5),
        'gc_interval':             training_config.get('gc_interval', 50),
        'clear_cache_interval':    training_config.get('clear_cache_interval', 150),
        'checkpoint_interval':     training_config.get('checkpoint_interval', 0),
    }

    return training_params


# =============================================================================
# MAIN TRAINING ORCHESTRATION
# =============================================================================

def main():
    args = parse_arguments()

    print_device_summary(_device_info)

    print("=" * 80, flush=True)
    print("TILTED STABLE SDE (DOUBLE WELL) TRAINING  (GPU-enabled script)", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    # Load configuration
    print("Loading configuration...", flush=True)
    config = load_config(args.config)
    alpha, obs_std, data_seed = extract_dataset_params(config)
    print(f"✓ Configuration loaded from: {args.config}", flush=True)
    print(f"  Dataset: alpha={alpha}, obs_std={obs_std}, data_seed={data_seed}", flush=True)
    print(flush=True)

    # Validate dataset exists BEFORE creating any directories
    print("Validating dataset availability...", flush=True)
    dataset_path = get_dataset_path(alpha, obs_std, data_seed,
                                    base_path=_DOUBLE_WELL_DATASET_BASE)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Expected double well dataset with:\n"
            f"  - alpha: {alpha}\n"
            f"  - obs_std: {obs_std}\n"
            f"  - data_seed: {data_seed}\n"
            f"Please generate the dataset first:\n"
            f"  python generation/generate_batch_prior_tilted_stable_sde_double_well.py"
        )
    print(f"✓ Dataset found: {dataset_path}", flush=True)
    print(flush=True)

    # Handle parent run
    parent_run_dir = None
    if args.parent_run is not None:
        print("Resolving parent run...", flush=True)
        parent_run_dir = get_parent_run_directory(args.parent_run)
        print(f"✓ Parent run found: {parent_run_dir}", flush=True)

        print("Validating parent run compatibility...", flush=True)
        parent_config_path = parent_run_dir / "config.yaml"
        with open(parent_config_path, 'r') as f:
            parent_config = yaml.safe_load(f)
        validate_parent_run_compatibility(parent_config, config)
        print(flush=True)

    # Create run directory
    training_seed = generate_training_seed()
    run_dir       = create_run_directory(alpha, obs_std, data_seed, training_seed)

    print(f"Created run directory: {run_dir}", flush=True)
    print(f"Training seed: {training_seed}", flush=True)
    print(flush=True)

    # Save config
    config_save_path = run_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Configuration saved to: {config_save_path}", flush=True)

    # Create and save initial metadata
    metadata = create_metadata(config, training_seed, _device_info, parent_run_path=args.parent_run)
    save_metadata(metadata, run_dir)
    print(f"✓ Metadata initialized  (backend: {_device_info.get('confirmed_backend', 'unknown')})", flush=True)

    # Load dataset
    print("Loading dataset...", flush=True)
    data_dict = load_training_data(config)
    observations         = data_dict['observations']
    time_sequence        = data_dict['time_sequence']
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
    model, model_source, checkpoint_step = initialize_or_load_model(
        config, model_params, parent_run_dir
    )
    print(flush=True)

    # Prepare training parameters
    training_params = prepare_training_params(config)

    # Output paths
    output_paths = {
        'run':         run_dir,
        'checkpoints': run_dir / "checkpoints",
        'plots':       run_dir / "plots",
        'model':       run_dir / "model.eqx",
        'metrics':     run_dir / "metrics.pkl",
        'metadata':    run_dir / "metadata.json",
    }

    print("=" * 80, flush=True)
    print("STARTING TRAINING", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    try:
        trained_model, metrics = train_tilted_stable_model(
            observations=observations,
            time_sequence=time_sequence,
            state_init=state_init,
            obs_std=config['data']['obs_std'],

            obs_subsample_method=config['training']['obs_subsample_method'],
            obs_subsample_count=config['training']['obs_subsample_count'],
            obs_subsample_seed=config['training']['obs_subsample_seed'],

            model_params=model_params,
            training_params=training_params,
            training_seed=training_seed,
            output_paths=output_paths,

            model_class=TiltedStableDrivenSDEDoubleWell,
            model=model,
            verbose=True,
        )

        # Merge metrics with parent run if continuing
        if parent_run_dir is not None:
            print("\nMerging training history with parent run...", flush=True)
            parent_metrics_path = parent_run_dir / "metrics.pkl"
            parent_metrics = None

            if parent_metrics_path.exists():
                with open(parent_metrics_path, 'rb') as f:
                    parent_metrics = pickle.load(f)
                print(f"  Loaded metrics from completed parent run", flush=True)
            elif checkpoint_step is not None:
                checkpoint_metrics_path = (
                    parent_run_dir / "checkpoints" / f"step_{checkpoint_step}_metrics.pkl"
                )
                if checkpoint_metrics_path.exists():
                    with open(checkpoint_metrics_path, 'rb') as f:
                        parent_metrics = pickle.load(f)
                    print(f"  Loaded metrics from checkpoint at step {checkpoint_step}", flush=True)
                else:
                    print(f"⚠ Warning: Checkpoint metrics file not found at {checkpoint_metrics_path}", flush=True)
                    print("  Continuing with current run metrics only", flush=True)
            else:
                print(f"⚠ Warning: Parent metrics file not found at {parent_metrics_path}", flush=True)
                print("  Continuing with current run metrics only", flush=True)

            if parent_metrics is not None:
                if 'loss_history' in parent_metrics and 'loss_history' in metrics:
                    metrics['loss_history'] = parent_metrics['loss_history'] + metrics['loss_history']
                    print(
                        f"  Combined loss history: "
                        f"{len(parent_metrics['loss_history'])} (parent) + "
                        f"{len(metrics['loss_history']) - len(parent_metrics['loss_history'])} (current) = "
                        f"{len(metrics['loss_history'])} steps",
                        flush=True,
                    )
                if 'parameter_history' in parent_metrics and 'parameter_history' in metrics:
                    metrics['parameter_history'] = (
                        parent_metrics['parameter_history'] + metrics['parameter_history']
                    )
                print("✓ Training histories merged", flush=True)

        print(flush=True)
        print("=" * 80, flush=True)
        print("TRAINING COMPLETED SUCCESSFULLY", flush=True)
        print("=" * 80, flush=True)
        print(flush=True)

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

            model_config_dict = {
                'state_dim':       config['model']['state_dim'],
                'alpha':           config['model']['alpha'],
                'tau':             config['model']['tau'],
                'sigma':           config['model']['sigma'],
                'tilting_width':   config['model'].get('tilting_width', 64),
                'tilting_depth':   config['model'].get('tilting_depth', 3),
                'trainable_drift': config['model']['trainable_drift'],
            }

            training_config_dict = {
                'learning_rate':  config['training']['learning_rate'],
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
                true_drift_component=true_drift_component,
            )

            print("✓ Training plots generated", flush=True)

        # Update metadata with completion status
        finalize_run_metadata(run_dir, metrics, success=True)

        # Print summary
        print("\n" + "=" * 80, flush=True)
        print("TRAINING SUMMARY", flush=True)
        print("=" * 80, flush=True)
        print(f"Run directory: {run_dir}", flush=True)
        print(f"Run ID: {metadata['run_id']}", flush=True)
        print(f"Training seed: {training_seed}", flush=True)
        print(f"Device: {_device_info.get('confirmed_backend', 'unknown')}  "
              f"({'GPU' if _device_info.get('gpu_active') else 'CPU'})", flush=True)
        if args.parent_run:
            print(f"Parent run: {args.parent_run}", flush=True)
        print(f"\nFinal loss: {metrics['final_loss']:.6f}", flush=True)
        print(f"Total training steps: {len(metrics['loss_history'])}", flush=True)

        if 'trained_drift_component' in metrics:
            print(f"\nTrained drift component: {metrics['drift_type']}", flush=True)
            drift = metrics['trained_drift_component']

            if hasattr(drift, 'theta1') and hasattr(drift, 'theta2'):
                def _fmt(x):
                    return float(x.item()) if x.ndim > 0 else float(x)
                theta1_val = _fmt(drift.theta1)
                theta2_val = _fmt(drift.theta2)
                print(f"  θ₁ (linear coefficient):  {theta1_val:.6f}", flush=True)
                print(f"  θ₂ (cubic coefficient):   {theta2_val:.6f}", flush=True)
                print(f"  Well locations ±√(θ₁/θ₂): ±{(theta1_val / theta2_val) ** 0.5:.6f}", flush=True)

                if true_drift_component is not None and hasattr(true_drift_component, 'theta1'):
                    true_theta1 = _fmt(true_drift_component.theta1)
                    true_theta2 = _fmt(true_drift_component.theta2)
                    print(f"\nTrue double well parameters:", flush=True)
                    print(f"  θ₁: {true_theta1:.6f}", flush=True)
                    print(f"  θ₂: {true_theta2:.6f}", flush=True)
                    print(f"\nErrors:", flush=True)
                    print(f"  θ₁ error: {abs(theta1_val - true_theta1):.6f}", flush=True)
                    print(f"  θ₂ error: {abs(theta2_val - true_theta2):.6f}", flush=True)

        print("\n" + "=" * 80, flush=True)
        print("OUTPUT FILES:", flush=True)
        print(f"  Model:    {output_paths['model']}", flush=True)
        print(f"  Metrics:  {output_paths['metrics']}", flush=True)
        print(f"  Metadata: {output_paths['metadata']}", flush=True)
        print(f"  Config:   {config_save_path}", flush=True)
        print(f"  Plots:    {output_paths['plots']}", flush=True)
        if config['training']['checkpoint_interval'] > 0:
            print(f"  Checkpoints: {output_paths['checkpoints']}", flush=True)
        print("=" * 80, flush=True)
        print(flush=True)

    except Exception as e:
        print("\n" + "=" * 80, flush=True)
        print("TRAINING FAILED", flush=True)
        print("=" * 80, flush=True)
        print(f"Error: {e}", flush=True)
        print(flush=True)

        finalize_run_metadata(run_dir, {}, success=False)

        raise


if __name__ == "__main__":
    main()
