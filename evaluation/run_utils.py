"""
Utility functions for evaluation system.

Handles:
- Loading training runs (configs, metadata, models)
- Finding run directories
- Loading datasets
- Constructing output paths
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import yaml
import pickle
import random
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import equinox as eqx
from jax import Array
import numpy as np

from models.tilted_stable_sde import TiltedStableDrivenSDE
from models.gaussian_sde import GaussianDrivenSDE
from models.tilted_stable_sde_double_well import TiltedStableDrivenSDEDoubleWell
from models.gaussian_sde_double_well import GaussianDrivenSDEDoubleWell
from utils.dataset_utils import get_dataset_path
from utils.training_utils import subsample_observations

_DOUBLE_WELL_DATASET_BASE = Path("datasets/tilted_stable_sde_double_well")


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class EvaluationError(Exception):
    """Base exception for evaluation errors."""
    pass


class RunNotFoundError(EvaluationError):
    """Training run not found."""
    pass


class ModelLoadError(EvaluationError):
    """Failed to load trained model."""
    pass


class DatasetError(EvaluationError):
    """Dataset not found or invalid."""
    pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    run_id: str
    model_type: str
    run_config: Dict
    metrics: Dict
    posterior_stats: Dict
    time_sequence: np.ndarray
    ground_truth_type: str
    metadata: Dict
    posterior_samples: Optional[np.ndarray] = None  # Optional to save space


# =============================================================================
# RUN DISCOVERY
# =============================================================================

def find_run_path(run_id: str, base_path: Optional[Path] = None) -> Path:
    """
    Find training run directory from run_id.

    Searches through all model_type and parameter directories to find
    the run matching the given run_id.

    Parameters
    ----------
    run_id : str
        Run identifier (e.g., 'data_16241891_train_29699958')
    base_path : Path, optional
        Base training runs directory (default: training_runs/)

    Returns
    -------
    run_path : Path
        Full path to the training run directory

    Raises
    ------
    RunNotFoundError
        If run_id is not found in any directory
    """
    if base_path is None:
        base_path = Path.cwd() / 'training_runs'

    # Search through all subdirectories
    for model_type_dir in base_path.iterdir():
        if not model_type_dir.is_dir():
            continue

        for param_dir in model_type_dir.iterdir():
            if not param_dir.is_dir():
                continue

            run_dir = param_dir / run_id
            if run_dir.exists() and run_dir.is_dir():
                return run_dir

    raise RunNotFoundError(
        f"Training run '{run_id}' not found in {base_path}\n"
        f"Searched through all model_type/parameter directories"
    )


def get_evaluation_output_path_full(
    run_config: Dict,
    run_id: str,
    ground_truth: str = 'observations',
    eval_obs: str = 'training',
    sim_grid: str = 'obs',
    base_path: Optional[Path] = None
) -> Path:
    """
    Construct hierarchical output path encoding ground truth type, evaluation
    observation subset, and simulation grid choice.

    Structure:
    evaluation/results/{model_type}/alpha_{alpha:.2f}_obsstd_{obs_std:.2f}/{run_id}_gt_{ground_truth}_obs_{eval_obs}_sim_{sim_grid}/
    """
    if base_path is None:
        base_path = Path.cwd() / 'evaluation' / 'results'

    model_type = run_config['model_type']

    if model_type in ('tilted_stable_sde', 'tilted_stable_sde_double_well'):
        alpha = run_config['model']['alpha']
    else:
        alpha = run_config['data']['alpha']

    obs_std = run_config['data']['obs_std']

    param_dir = f"alpha_{alpha:.2f}_obsstd_{obs_std:.2f}"
    run_dir = f"{run_id}_gt_{ground_truth}_obs_{eval_obs}_sim_{sim_grid}"
    output_path = base_path / model_type / param_dir / run_dir

    return output_path


# =============================================================================
# CONFIG AND METADATA LOADING
# =============================================================================

def load_run_config(run_path: Path) -> Dict:
    """
    Load configuration from training run directory.

    Parameters
    ----------
    run_path : Path
        Path to training run directory

    Returns
    -------
    config : Dict
        Configuration dictionary

    Raises
    ------
    RunNotFoundError
        If config.yaml not found
    """
    config_path = run_path / 'config.yaml'

    if not config_path.exists():
        raise RunNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RunNotFoundError(f"Failed to load config from {config_path}: {e}")


def load_run_metadata(run_path: Path) -> Dict:
    """
    Load metadata from training run directory.

    Parameters
    ----------
    run_path : Path
        Path to training run directory

    Returns
    -------
    metadata : Dict
        Metadata dictionary

    Raises
    ------
    RunNotFoundError
        If metadata.json not found
    """
    metadata_path = run_path / 'metadata.json'

    if not metadata_path.exists():
        raise RunNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        raise RunNotFoundError(f"Failed to load metadata from {metadata_path}: {e}")


# =============================================================================
# MODEL LOADING
# =============================================================================

def build_model_params(config: Dict) -> Dict:
    """
    Extract model initialization parameters from config.

    Follows the pattern from generation/generate_posteriors.py

    Parameters
    ----------
    config : Dict
        Run configuration dictionary

    Returns
    -------
    model_params : Dict
        Parameters for model initialization

    Raises
    ------
    ModelLoadError
        If model_type is unsupported
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


def load_model(run_path: Path, config: Dict) -> eqx.Module:
    """
    Load trained model from run directory.

    Creates template model from config, then deserializes trained weights.

    Parameters
    ----------
    run_path : Path
        Path to training run directory
    config : Dict
        Run configuration dictionary

    Returns
    -------
    model : eqx.Module
        Loaded model with trained weights

    Raises
    ------
    ModelLoadError
        If model loading fails
    """
    model_path = run_path / "model.eqx"
    model_type = config['model_type']

    if not model_path.exists():
        raise ModelLoadError(f"Model file not found: {model_path}")

    # Build model parameters
    model_params = build_model_params(config)

    # Select model class
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


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset_for_run(config: Dict) -> Dict:
    """
    Load dataset corresponding to a training run.

    Parameters
    ----------
    config : Dict
        Run configuration dictionary

    Returns
    -------
    data_dict : Dict
        Dataset dictionary containing observations, latent_path, etc.

    Raises
    ------
    DatasetError
        If dataset not found or parameters mismatch
    """
    # Extract dataset parameters from config
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
            raise DatasetError(
                f"Dataset alpha mismatch: file={data_dict.get('alpha')}, config={alpha}"
            )

        if data_dict.get('obs_std') != obs_std:
            raise DatasetError(
                f"Dataset obs_std mismatch: file={data_dict.get('obs_std')}, config={obs_std}"
            )

        if data_dict.get('data_seed') != data_seed:
            raise DatasetError(
                f"Dataset data_seed mismatch: file={data_dict.get('data_seed')}, config={data_seed}"
            )

        return data_dict

    except DatasetError:
        raise
    except Exception as e:
        raise DatasetError(f"Failed to load dataset from {dataset_path}: {e}")


# =============================================================================
# OBSERVATION SUBSAMPLING
# =============================================================================

def recreate_observation_subsampling(full_time_sequence, config: Dict):
    """
    Recreate training observation indices from config.

    Parameters
    ----------
    full_time_sequence : Array
        Full observation time sequence from dataset
    config : Dict
        Run configuration

    Returns
    -------
    training_indices : Array
        Indices of training observations
    """
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
    return subsample_observations(full_time_sequence, mock_config)


# =============================================================================
# DRIFT PARAMETER EXTRACTION
# =============================================================================

def extract_drift_parameters(drift_component) -> Dict:
    """
    Extract drift parameters from a drift component.

    Handles both OU drift and linear drift.

    Parameters
    ----------
    drift_component : Module
        Drift component (OUDiagonalLinearFunction or similar)

    Returns
    -------
    drift_params : Dict
        Dictionary with drift parameters
        For OU: {'theta': Array, 'mu': Array}
        For linear: {'weight': Array, 'bias': Array}
    """
    # Try to extract OU parameters (theta, mu)
    if hasattr(drift_component, 'theta') and hasattr(drift_component, 'mu'):
        return {
            'theta': drift_component.theta,
            'mu': drift_component.mu
        }
    # Try double well parameters (theta1, theta2)
    elif hasattr(drift_component, 'theta1') and hasattr(drift_component, 'theta2'):
        return {
            'theta1': drift_component.theta1,
            'theta2': drift_component.theta2
        }
    # Try linear parameters (weight, bias)
    elif hasattr(drift_component, 'weight') and hasattr(drift_component, 'bias'):
        return {
            'weight': drift_component.weight,
            'bias': drift_component.bias
        }
    else:
        raise ValueError(
            f"Unknown drift component type: {type(drift_component)}\n"
            f"Expected OU (theta, mu), double well (theta1, theta2), or linear (weight, bias)"
        )
