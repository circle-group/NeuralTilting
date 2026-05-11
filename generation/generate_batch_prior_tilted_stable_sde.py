# Third-party imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import jax.numpy as jnp
import jax.random as random
import itertools
import pickle
from datetime import datetime

# Library-specific imports
from models.tilted_stable_sde import TiltedStableDrivenSDE
from utils.dataset_utils import (
    generate_random_seed,
    create_dataset_folder,
    get_dataset_path,
    get_visualization_path,
    save_registry
)
from utils.visualization_utils import plot_dataset

print("=" * 80)
print("BATCH DATASET GENERATION FOR SYSTEMATIC EXPERIMENTS")
print("=" * 80)

# ============================================================================
# PARAMETER GRID DEFINITION
# ============================================================================
# Define the parameter combinations to generate

param_grid = {
    # Primary experimental variables
    'alpha': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],   # Stability parameter (1.1 added)
    'obs_std': [0.05, 0.10],           # Observation noise std (0.20 excluded — not used in experiments)
    'n_realizations': 40,              # Number of NEW datasets per (alpha, obs_std) combination

    # Fixed data parameters
    'state_dim': 1,
    'n_obs_steps': 10000,                 # Dense observation grid (1001 time points)
    'time_start': 0.0,
    'time_end': 10.0,
    'state_init': 0.0,

    # Fixed SDE parameters
    'tau': 0.01,                         # Truncation parameter
    'sigma': 1.0,                        # Scale parameter

    # Drift parameters
    'drift_seed': None,                  # None = generate random seed per dataset
    'trainable_drift': True,
    'initial_drift_weight': None,        # Randomly initialized
    'initial_drift_bias': None,          # Randomly initialized

    # Model parameters (needed for initialization, not used in generation)
    'tilting_width': 50,
    'tilting_depth': 2,
    'n_time_features': 16,
    'max_jumps': 10,
    'max_rejection_attempts': 50,
    'loss_sample_size': 500,
    'phi_seed': 321,
    'diffusion_seed': 1,

    # Visualization
    'plot_dpi': 200,
}

# Calculate total number of datasets
total_datasets = len(param_grid['alpha']) * len(param_grid['obs_std']) * param_grid['n_realizations']
print(f"\nParameter Grid:")
print(f"  Alpha values: {param_grid['alpha']}")
print(f"  Obs std values: {param_grid['obs_std']}")
print(f"  Realizations per combination: {param_grid['n_realizations']}")
print(f"  Total datasets to generate: {total_datasets}")
print()

# ============================================================================
# SETUP
# ============================================================================

# Base path for datasets
base_path = Path("datasets/tilted_stable_sde")
base_path.mkdir(parents=True, exist_ok=True)
print(f"Base path: {base_path}")

# Initialize registry
registry = {"model_type": "tilted_stable_sde", "datasets": []}

# ============================================================================
# BATCH GENERATION LOOP
# ============================================================================

dataset_counter = 0

for alpha, obs_std in itertools.product(param_grid['alpha'], param_grid['obs_std']):
    print(f"\n{'=' * 80}")
    print(f"Generating datasets for alpha={alpha:.2f}, obs_std={obs_std:.2f}")
    print(f"{'=' * 80}")

    # Create folder for this (alpha, obs_std) combination
    folder_path = create_dataset_folder(alpha, obs_std, base_path)
    print(f"Folder: {folder_path}")

    for realization_idx in range(param_grid['n_realizations']):
        dataset_counter += 1

        print(f"\n[{dataset_counter}/{total_datasets}] Generating realization {realization_idx + 1}/{param_grid['n_realizations']}")

        # ====================================================================
        # Generate random seeds (avoid collisions with existing datasets)
        # ====================================================================
        while True:
            data_seed = generate_random_seed()
            candidate_path = get_dataset_path(alpha, obs_std, data_seed, base_path)
            if not candidate_path.exists():
                break

        if param_grid['drift_seed'] is None:
            drift_seed = generate_random_seed()
        else:
            drift_seed = param_grid['drift_seed']

        print(f"  data_seed: {data_seed}")
        print(f"  drift_seed: {drift_seed}")

        # ====================================================================
        # Initialize model
        # ====================================================================
        model = TiltedStableDrivenSDE(
            state_dim=param_grid['state_dim'],
            alpha=alpha,
            tau=param_grid['tau'],
            sigma=param_grid['sigma'],
            loss_sample_size=param_grid['loss_sample_size'],
            max_jumps=param_grid['max_jumps'],
            max_rejection_attempts=param_grid['max_rejection_attempts'],
            tilting_width=param_grid['tilting_width'],
            tilting_depth=param_grid['tilting_depth'],
            drift_seed=drift_seed,
            diffusion_seed=param_grid['diffusion_seed'],
            phi_seed=param_grid['phi_seed'],
            trainable_drift=param_grid['trainable_drift'],
            initial_drift_weight=param_grid['initial_drift_weight'],
            initial_drift_bias=param_grid['initial_drift_bias'],
            n_time_features=param_grid['n_time_features'],
            period=param_grid['time_end'],
        )

        # ====================================================================
        # Setup randomness
        # ====================================================================
        key = random.key(data_seed)
        key, subkey1, subkey2 = random.split(key, num=3)

        # ====================================================================
        # Create time grid
        # ====================================================================
        time_sequence = jnp.linspace(
            param_grid['time_start'],
            param_grid['time_end'],
            param_grid['n_obs_steps'] + 1
        )

        # ====================================================================
        # Initial state
        # ====================================================================
        if param_grid['state_dim'] == 1:
            state_init = jnp.array([param_grid['state_init']])
        else:
            state_init = jnp.full((param_grid['state_dim'],), param_grid['state_init'])

        # ====================================================================
        # Generate prior paths
        # ====================================================================
        print(f"  Simulating prior SDE...")
        latent_path = model.simulate_prior(state_init, time_sequence, subkey1)

        # ====================================================================
        # Add observation noise
        # ====================================================================
        observations = latent_path + obs_std * random.normal(subkey2, latent_path.shape)

        # ====================================================================
        # Create dataset dictionary
        # ====================================================================
        data_dict = {
            # Core data
            'observations': observations,
            'latent_path': latent_path,
            'time_sequence': time_sequence,
            'state_init': state_init,
            'obs_std': obs_std,

            # Process parameters
            'alpha': alpha,
            'tau': param_grid['tau'],
            'sigma': param_grid['sigma'],

            # Drift component (full PyTree for exact reproducibility)
            'drift_component': model.drift,
            'drift_type': type(model.drift).__name__,

            # Seeds
            'data_seed': data_seed,
            'drift_seed': drift_seed,
        }

        # ====================================================================
        # Save pickle file
        # ====================================================================
        pkl_path = get_dataset_path(alpha, obs_std, data_seed, base_path)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pkl_path, 'wb') as f:
            pickle.dump(data_dict, f)

        print(f"  Saved dataset: {pkl_path}")

        # ====================================================================
        # Generate and save visualization
        # ====================================================================
        png_path = get_visualization_path(alpha, obs_std, data_seed, base_path)
        plot_dataset(data_dict, png_path, plot_dpi=param_grid['plot_dpi'])
        print(f"  Saved visualization: {png_path}")

        # ====================================================================
        # Register dataset
        # ====================================================================
        dataset_info = {
            'dataset_id': f"alpha_{alpha:.2f}_obsstd_{obs_std:.2f}/seed_{data_seed}",
            'file_path': f"alpha_{alpha:.2f}_obsstd_{obs_std:.2f}/seed_{data_seed}.pkl",
            'parameters': {
                'alpha': alpha,
                'obs_std': obs_std,
                'data_seed': data_seed,
                'drift_seed': drift_seed,
                'state_dim': param_grid['state_dim'],
                'n_obs_steps': param_grid['n_obs_steps'],
                'time_start': param_grid['time_start'],
                'time_end': param_grid['time_end'],
                'state_init': param_grid['state_init'],
                'tau': param_grid['tau'],
                'sigma': param_grid['sigma'],
            },
            'generated_timestamp': datetime.now().isoformat()
        }

        registry['datasets'].append(dataset_info)

# ============================================================================
# Save registry
# ============================================================================
print(f"\n{'=' * 80}")
print("Saving dataset registry...")
save_registry(registry, base_path)
print(f"Registry saved: {base_path / 'dataset_registry.json'}")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'=' * 80}")
print("BATCH GENERATION COMPLETE")
print(f"{'=' * 80}")
print(f"Total datasets generated: {len(registry['datasets'])}")
print(f"Registry location: {base_path / 'dataset_registry.json'}")
print(f"Dataset folders: {base_path}")
print()

# Print summary by (alpha, obs_std)
from collections import defaultdict
summary = defaultdict(int)
for dataset in registry['datasets']:
    alpha = dataset['parameters']['alpha']
    obs_std = dataset['parameters']['obs_std']
    summary[(alpha, obs_std)] += 1

print("Datasets by parameters:")
for (alpha, obs_std), count in sorted(summary.items()):
    print(f"  alpha={alpha:.2f}, obs_std={obs_std:.2f}: {count} datasets")

print()
print("Done!")
