"""
Utility functions for dataset management in systematic experiments.

This module provides functions for:
- Path construction and folder management
- Dataset registry operations (load, save, query)
- Random seed generation for datasets
- Dataset validation
"""

import json
import pickle
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


def get_dataset_path(alpha: float, obs_std: float, data_seed: int, base_path: Optional[Path] = None) -> Path:
    """
    Get the path where a dataset should be saved.

    Parameters
    ----------
    alpha : float
        Stability parameter of the Lévy process
    obs_std : float
        Observation noise standard deviation
    data_seed : int
        8-digit random seed for the dataset
    base_path : Path, optional
        Base directory for datasets. Defaults to experiments/datasets/tilted_stable_sde

    Returns
    -------
    Path
        Full path to the dataset pickle file

    Examples
    --------
    >>> path = get_dataset_path(1.5, 0.05, 27482070)
    >>> print(path)
    datasets/tilted_stable_sde/alpha_1.50_obsstd_0.05/seed_27482070.pkl
    """
    if base_path is None:
        base_path = Path("datasets/tilted_stable_sde")

    folder_name = f"alpha_{alpha:.2f}_obsstd_{obs_std:.2f}"
    filename = f"seed_{data_seed}.pkl"

    return base_path / folder_name / filename


def get_visualization_path(alpha: float, obs_std: float, data_seed: int, base_path: Optional[Path] = None) -> Path:
    """
    Get the path where a dataset visualization should be saved.

    Parameters
    ----------
    alpha : float
        Stability parameter of the Lévy process
    obs_std : float
        Observation noise standard deviation
    data_seed : int
        8-digit random seed for the dataset
    base_path : Path, optional
        Base directory for datasets. Defaults to experiments/datasets/tilted_stable_sde

    Returns
    -------
    Path
        Full path to the visualization PNG file
    """
    pkl_path = get_dataset_path(alpha, obs_std, data_seed, base_path)
    return pkl_path.with_suffix('.png')


def generate_random_seed() -> int:
    """
    Generate a random 8-digit integer for dataset seeding.

    Returns
    -------
    int
        Random integer in range [10,000,000, 99,999,999]

    Notes
    -----
    Seeds are 8 digits for consistent filename sorting and readability.
    Random (not sequential) to avoid correlations between datasets.
    """
    return random.randint(10_000_000, 99_999_999)


def create_dataset_folder(alpha: float, obs_std: float, base_path: Optional[Path] = None) -> Path:
    """
    Create folder for storing datasets with specific (alpha, obs_std) combination.

    Parameters
    ----------
    alpha : float
        Stability parameter of the Lévy process
    obs_std : float
        Observation noise standard deviation
    base_path : Path, optional
        Base directory for datasets. Defaults to datasets/tilted_stable_sde

    Returns
    -------
    Path
        Path to the created folder
    """
    if base_path is None:
        base_path = Path("datasets/tilted_stable_sde")

    folder_name = f"alpha_{alpha:.2f}_obsstd_{obs_std:.2f}"
    folder_path = base_path / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    return folder_path


def load_registry(base_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the dataset registry.

    Parameters
    ----------
    base_path : Path, optional
        Base directory for datasets. Defaults to experiments/datasets/tilted_stable_sde

    Returns
    -------
    dict
        Registry dictionary with 'model_type' and 'datasets' keys

    Raises
    ------
    FileNotFoundError
        If registry file does not exist
    """
    if base_path is None:
        base_path = Path("experiments/datasets/tilted_stable_sde")

    registry_path = base_path / "dataset_registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    with open(registry_path, 'r') as f:
        return json.load(f)


def save_registry(registry: Dict[str, Any], base_path: Optional[Path] = None):
    """
    Save the dataset registry.

    Parameters
    ----------
    registry : dict
        Registry dictionary with 'model_type' and 'datasets' keys
    base_path : Path, optional
        Base directory for datasets. Defaults to experiments/datasets/tilted_stable_sde
    """
    if base_path is None:
        base_path = Path("experiments/datasets/tilted_stable_sde")

    registry_path = base_path / "dataset_registry.json"

    # Create directory if it doesn't exist
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


def register_dataset(dataset_info: Dict[str, Any], base_path: Optional[Path] = None):
    """
    Add a dataset entry to the registry.

    Parameters
    ----------
    dataset_info : dict
        Dataset metadata dictionary with keys:
        - dataset_id: str (e.g., "alpha_1.50_obsstd_0.05/seed_27482070")
        - file_path: str (relative path to pickle file)
        - parameters: dict (all dataset parameters)
        - generated_timestamp: str (ISO format timestamp)
    base_path : Path, optional
        Base directory for datasets. Defaults to experiments/datasets/tilted_stable_sde

    Examples
    --------
    >>> dataset_info = {
    ...     'dataset_id': 'alpha_1.50_obsstd_0.05/seed_27482070',
    ...     'file_path': 'alpha_1.50_obsstd_0.05/seed_27482070.pkl',
    ...     'parameters': {
    ...         'alpha': 1.5,
    ...         'obs_std': 0.05,
    ...         'data_seed': 27482070,
    ...         # ... other parameters
    ...     },
    ...     'generated_timestamp': '2025-10-20T12:00:00'
    ... }
    >>> register_dataset(dataset_info)
    """
    if base_path is None:
        base_path = Path("experiments/datasets/tilted_stable_sde")

    registry_path = base_path / "dataset_registry.json"

    # Load existing registry or create new one
    if registry_path.exists():
        registry = load_registry(base_path)
    else:
        registry = {"model_type": "tilted_stable_sde", "datasets": []}

    # Append new dataset
    registry['datasets'].append(dataset_info)

    # Save registry
    save_registry(registry, base_path)


def query_datasets(
    alpha: Optional[float] = None,
    obs_std: Optional[float] = None,
    data_seed: Optional[int] = None,
    drift_seed: Optional[int] = None,
    state_dim: Optional[int] = None,
    base_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Find datasets matching criteria.

    Parameters
    ----------
    alpha : float, optional
        Filter by stability parameter
    obs_std : float, optional
        Filter by observation noise standard deviation
    data_seed : int, optional
        Filter by data generation seed
    drift_seed : int, optional
        Filter by drift initialization seed
    state_dim : int, optional
        Filter by state dimensionality
    base_path : Path, optional
        Base directory for datasets

    Returns
    -------
    list of dict
        List of dataset entries matching all specified criteria

    Examples
    --------
    >>> # Find all datasets with alpha=1.5
    >>> datasets = query_datasets(alpha=1.5)

    >>> # Find all datasets with alpha=1.5 and obs_std=0.05
    >>> datasets = query_datasets(alpha=1.5, obs_std=0.05)

    >>> # Find a specific dataset by seed
    >>> datasets = query_datasets(data_seed=27482070)
    """
    registry = load_registry(base_path)
    datasets = registry['datasets']

    # Filter by parameters
    if alpha is not None:
        datasets = [d for d in datasets if d['parameters']['alpha'] == alpha]
    if obs_std is not None:
        datasets = [d for d in datasets if d['parameters']['obs_std'] == obs_std]
    if data_seed is not None:
        datasets = [d for d in datasets if d['parameters']['data_seed'] == data_seed]
    if drift_seed is not None:
        datasets = [d for d in datasets if d['parameters']['drift_seed'] == drift_seed]
    if state_dim is not None:
        datasets = [d for d in datasets if d['parameters']['state_dim'] == state_dim]

    return datasets


def load_dataset(dataset_id: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load a dataset by its ID.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier in format "alpha_X.XX_obsstd_Y.YY/seed_XXXXXXXX"
    base_path : Path, optional
        Base directory for datasets

    Returns
    -------
    dict
        Dataset dictionary loaded from pickle file

    Examples
    --------
    >>> data = load_dataset("alpha_1.50_obsstd_0.05/seed_27482070")
    >>> print(data.keys())
    dict_keys(['observations', 'latent_path', 'time_sequence', ...])
    """
    if base_path is None:
        base_path = Path("datasets/tilted_stable_sde")

    dataset_path = base_path / f"{dataset_id}.pkl"

    with open(dataset_path, 'rb') as f:
        return pickle.load(f)


def validate_dataset(data_dict: Dict[str, Any]) -> bool:
    """
    Validate that a dataset dictionary contains all required fields.

    Parameters
    ----------
    data_dict : dict
        Dataset dictionary to validate

    Returns
    -------
    bool
        True if valid, False otherwise

    Raises
    ------
    ValueError
        If validation fails, with description of the issue
    """
    required_keys = [
        'observations', 'latent_path', 'time_sequence', 'state_init', 'obs_std',
        'alpha', 'sigma', 'tau', 'drift_component', 'drift_type',
        'data_seed', 'drift_seed'
    ]

    missing_keys = [key for key in required_keys if key not in data_dict]
    if missing_keys:
        raise ValueError(f"Dataset missing required keys: {missing_keys}")

    # Validate shapes are consistent
    n_steps = len(data_dict['time_sequence'])

    if data_dict['observations'].shape[0] != n_steps:
        raise ValueError(
            f"observations shape[0] ({data_dict['observations'].shape[0]}) "
            f"does not match time_sequence length ({n_steps})"
        )

    if data_dict['latent_path'].shape[0] != n_steps:
        raise ValueError(
            f"latent_path shape[0] ({data_dict['latent_path'].shape[0]}) "
            f"does not match time_sequence length ({n_steps})"
        )

    state_dim = data_dict['observations'].shape[1] if len(data_dict['observations'].shape) > 1 else 1
    if len(data_dict['state_init']) != state_dim:
        raise ValueError(
            f"state_init length ({len(data_dict['state_init'])}) "
            f"does not match inferred state_dim ({state_dim})"
        )

    return True


def get_registry_summary(base_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get a summary of the dataset registry.

    Parameters
    ----------
    base_path : Path, optional
        Base directory for datasets

    Returns
    -------
    dict
        Summary statistics including:
        - total_datasets: int
        - unique_alphas: list
        - unique_obs_stds: list
        - datasets_by_params: dict mapping (alpha, obs_std) to count
    """
    registry = load_registry(base_path)
    datasets = registry['datasets']

    alphas = set()
    obs_stds = set()
    params_count = {}

    for dataset in datasets:
        alpha = dataset['parameters']['alpha']
        obs_std = dataset['parameters']['obs_std']

        alphas.add(alpha)
        obs_stds.add(obs_std)

        key = (alpha, obs_std)
        params_count[key] = params_count.get(key, 0) + 1

    return {
        'total_datasets': len(datasets),
        'unique_alphas': sorted(list(alphas)),
        'unique_obs_stds': sorted(list(obs_stds)),
        'datasets_by_params': {f"alpha={a:.2f}, obs_std={o:.2f}": count
                               for (a, o), count in sorted(params_count.items())}
    }
