"""
Generate training config YAML files for OU (Ornstein-Uhlenbeck) experiments.

Scans datasets/tilted_stable_sde/ for available seeds and writes one
config per (alpha, seed) combination for both model types.

Usage:
    source venv/bin/activate && python generate_ou_configs.py
    python generate_ou_configs.py --obs-std 0.10 --n-seeds 50 --dry-run
    python generate_ou_configs.py --obs-std 0.05 --n-seeds 50
"""

import argparse
from pathlib import Path

import yaml


# =============================================================================
# PARAMETERS
# =============================================================================

ALPHAS       = [1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90]
DATASET_BASE = Path("datasets/tilted_stable_sde")

TS_CONFIG_DIR = Path("training_configs/tilted_stable")
GS_CONFIG_DIR = Path("training_configs/gaussian")


# =============================================================================
# CONFIG TEMPLATES
# =============================================================================

def tilted_stable_config(alpha: float, obs_std: float, data_seed: int) -> dict:
    return {
        "model_type": "tilted_stable_sde",

        "model": {
            # Process parameters
            "alpha": round(alpha, 2),
            "tau": 0.01,
            "sigma": 1.0,
            "state_dim": 1,

            # Tilting network architecture
            "tilting_width": 256,
            "tilting_depth": 6,
            "n_time_features": 0,
            "period": 10.0,

            # Temporal attention
            "n_attention_references": 100,
            "attention_embed_dim": 64,
            "attention_sharpness": 100.0,

            # Adaptive scaling
            "use_adaptive_scaling": False,

            # Coefficient bounds
            "a_min": 0.001,

            # Drift configuration
            "trainable_drift": True,
            "drift_seed": None,
            "initial_drift_weight": None,
            "initial_drift_bias": None,

            # Diffusion configuration
            "diffusion_seed": None,
            "initial_diffusion_weight": [1.0],

            # Sampling configuration
            "loss_sample_size": 1000,
            "max_rejection_attempts": 50,
            "max_jumps": 10,

            # Network seeds
            "phi_seed": 713,
        },

        "data": {
            "data_seed": data_seed,
            "state_init": 0.0,
            "state_init_vector": None,
            "n_obs_steps": 10000,
            "time_start": 0.0,
            "time_end": 10.0,
            "obs_std": obs_std,
        },

        "training": {
            # Optimizer
            "learning_rate": 0.0001,

            # LR multipliers
            "lr_multiplier_mlp": 1.0,
            "lr_multiplier_attention": 10.0,
            "lr_multiplier_drift": 100,

            # Frozen params
            "frozen_params": ["diffusion.raw_weight"],

            # Training loop
            "training_steps": 3000,
            "n_loss_samples": 500,
            "n_latent_steps": 1000,

            # Observation subsampling
            "obs_subsample_method": "uniform",
            "obs_subsample_count": 1000,
            "obs_subsample_seed": 42,

            # Regularization
            "tilting_regularisation": 0.00001,
            "drift_regularization": 0.1,
            "coeff_A_regularization": 0.00001,
            "coeff_B_regularization": 0.00001,

            # Memory management
            "gc_interval": 50,
            "clear_cache_interval": 500,

            # Checkpointing
            "checkpoint_interval": 500,
        },

        "output": {
            "save_plots": True,
            "plot_dpi": 200,
        },
    }


def gaussian_config(alpha: float, obs_std: float, data_seed: int) -> dict:
    return {
        "model_type": "gaussian_sde",

        "model": {
            # Process parameters
            "sigma": 1.0,
            "state_dim": 1,

            # Control network architecture
            "control_width": 256,
            "control_depth": 6,
            "n_time_features": 0,
            "period": 10.0,
            "control_seed": 321,

            # Drift configuration
            "trainable_drift": True,
            "drift_seed": None,
            "initial_drift_weight": None,
            "initial_drift_bias": None,

            # Diffusion configuration
            "diffusion_seed": None,
            "initial_diffusion_weight": [1.0],
        },

        "data": {
            # alpha stored under data for Gaussian model (not a model parameter)
            "alpha": round(alpha, 1),   # minimal float: 1.1, 1.2 etc.
            "data_seed": data_seed,
            "obs_std": obs_std,
            "state_init": 0.0,
            "state_init_vector": None,
            "n_obs_steps": 10000,
            "time_start": 0.0,
            "time_end": 10.0,
        },

        "training": {
            # Optimizer
            "learning_rate": 0.0001,
            "transition_steps": 100,
            "decay_rate": 0.95,

            # Training loop
            "training_steps": 3000,
            "n_loss_samples": 500,
            "n_latent_steps": 1000,

            # Observation subsampling
            "obs_subsample_method": "uniform",
            "obs_subsample_count": 1000,
            "obs_subsample_seed": 42,

            # Regularization
            "control_regularisation": 0.5,
            "drift_regularization": 0.01,

            # Memory management
            "gc_interval": 50,
            "clear_cache_interval": 500,

            # Checkpointing
            "checkpoint_interval": 500,
        },

        "output": {
            "save_plots": True,
            "plot_dpi": 200,
        },
    }


# =============================================================================
# SEED DISCOVERY
# =============================================================================

def find_seeds(alpha: float, obs_std: float, n_seeds: int) -> list:
    """Return sorted list of up to n_seeds data seeds for this (alpha, obs_std)."""
    folder = DATASET_BASE / f"alpha_{alpha:.2f}_obsstd_{obs_std:.2f}"
    if not folder.exists():
        print(f"  [WARN] Dataset folder not found: {folder}")
        return []

    seeds = sorted(
        int(p.stem.split("_")[1])   # "seed_XXXXXXXX" -> int
        for p in folder.glob("seed_*.pkl")
    )

    if not seeds:
        print(f"  [WARN] No seed files found in {folder}")
        return []

    selected = seeds[:n_seeds]
    if len(seeds) < n_seeds:
        print(f"  [WARN] Only {len(seeds)} seeds available for alpha={alpha:.2f} "
              f"obs_std={obs_std:.2f} (requested {n_seeds})")

    return selected


# =============================================================================
# YAML WRITER
# =============================================================================

def write_yaml(config: dict, path: Path, dry_run: bool) -> bool:
    """Write config to path. Returns True if written, False if skipped."""
    if path.exists():
        return False

    if dry_run:
        print(f"  [dry-run] would write: {path}")
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return True


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obs-std",  type=float, default=0.10,
                   help="Observation noise std to use (default: 0.10)")
    p.add_argument("--n-seeds",  type=int,   default=50,
                   help="Number of seeds to pick per alpha (default: 50)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print file paths without writing anything")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Generating OU training configs")
    print(f"  obs_std : {args.obs_std}")
    print(f"  n_seeds : {args.n_seeds}")
    print(f"  alphas  : {[f'{a:.2f}' for a in ALPHAS]}")
    if args.dry_run:
        print("  [DRY-RUN mode — no files written]")
    print()

    ts_written = ts_skipped = 0
    gs_written = gs_skipped = 0

    for alpha in ALPHAS:
        seeds = find_seeds(alpha, args.obs_std, args.n_seeds)
        if not seeds:
            continue

        alpha_fmt = f"{alpha:.2f}"
        obs_fmt   = f"{args.obs_std:.2f}"
        print(f"alpha={alpha_fmt}  ({len(seeds)} seeds)")

        for seed in seeds:
            # ── tilted stable OU ──────────────────────────────────────────────
            ts_cfg  = tilted_stable_config(alpha, args.obs_std, seed)
            ts_path = TS_CONFIG_DIR / f"alpha_{alpha_fmt}_obsstd_{obs_fmt}_seed_{seed}.yaml"
            if write_yaml(ts_cfg, ts_path, args.dry_run):
                ts_written += 1
            else:
                ts_skipped += 1

            # ── gaussian OU ───────────────────────────────────────────────────
            gs_cfg  = gaussian_config(alpha, args.obs_std, seed)
            gs_path = GS_CONFIG_DIR / f"alpha_{alpha_fmt}_obsstd_{obs_fmt}_seed_{seed}.yaml"
            if write_yaml(gs_cfg, gs_path, args.dry_run):
                gs_written += 1
            else:
                gs_skipped += 1

    print()
    print(f"Tilted-stable: {ts_written} written, {ts_skipped} skipped (already exist) -> {TS_CONFIG_DIR}/")
    print(f"Gaussian:      {gs_written} written, {gs_skipped} skipped (already exist) -> {GS_CONFIG_DIR}/")


if __name__ == "__main__":
    main()
