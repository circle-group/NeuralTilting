"""Train a financial SDE model on a single rolling window."""

import random as pyrandom
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from financial.models.tilted_stable_sde import TiltedStableSDEFinancial
from financial.models.gaussian_sde import GaussianSDEFinancial, GaussianSDEFinancialMultivariate
from training.train_model import train_tilted_stable_model, train_gaussian_model


def build_model_params(cfg: dict, window: dict) -> dict:
    """Build model initialisation params from config and window data.

    The period is set to the actual training window time span so that
    the attention reference times are initialised in the right range.
    """
    model_cfg = cfg['model']
    model_type = cfg['model_type']

    train_times = window['train_times']
    period = float(train_times[-1])  # actual time span of this window
    state_dim = len(cfg['data']['tickers'])

    def _seed(key):
        v = model_cfg.get(key)
        return v if v is not None else pyrandom.randint(10_000_000, 99_999_999)

    common = dict(
        state_dim=state_dim,
        sigma=model_cfg['sigma'],
        period=period,
        n_time_features=model_cfg.get('n_time_features', 0),
        drift_width=model_cfg['drift_width'],
        drift_depth=model_cfg['drift_depth'],
        drift_seed=_seed('drift_seed'),
        diffusion_seed=_seed('diffusion_seed'),
        trainable_drift=model_cfg.get('trainable_drift', True),
        initial_diffusion_weight=model_cfg.get('initial_diffusion_weight'),
    )

    if model_type == 'tilted_stable_sde_financial':
        phi_seed = _seed('phi_seed')
        common.update(dict(
            alpha=model_cfg['alpha'],
            tau=model_cfg['tau'],
            loss_sample_size=model_cfg['loss_sample_size'],
            max_rejection_attempts=model_cfg.get('max_rejection_attempts', 50),
            max_jumps=model_cfg.get('max_jumps', 10),
            tilting_width=model_cfg['tilting_width'],
            tilting_depth=model_cfg['tilting_depth'],
            phi_seed=phi_seed,
            a_min=model_cfg.get('a_min', 0.001),
            use_adaptive_scaling=model_cfg.get('use_adaptive_scaling', False),
            n_attention_references=model_cfg.get('n_attention_references', 100),
            attention_embed_dim=model_cfg.get('attention_embed_dim', 64),
            attention_sharpness=model_cfg.get('attention_sharpness', 100.0),
            use_gradient_checkpointing=model_cfg.get('use_gradient_checkpointing', False),
        ))

    elif model_type == 'gaussian_sde_financial':
        common.update(dict(
            control_width=model_cfg['control_width'],
            control_depth=model_cfg['control_depth'],
            control_seed=_seed('control_seed'),
            use_gradient_checkpointing=model_cfg.get('use_gradient_checkpointing', False),
        ))

    elif model_type == 'gaussian_sde_financial_multivariate':
        common.update(dict(
            control_width=model_cfg['control_width'],
            control_depth=model_cfg['control_depth'],
            control_seed=_seed('control_seed'),
            cholesky_width=model_cfg.get('cholesky_width', 32),
            cholesky_depth=model_cfg.get('cholesky_depth', 2),
            use_gradient_checkpointing=model_cfg.get('use_gradient_checkpointing', False),
        ))

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return common


def build_output_paths(cfg: dict, window_idx: int, results_dir: Path) -> dict:
    config_name = cfg.get('_config_name', 'experiment')
    run_dir = results_dir / config_name / f"window_{window_idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    return {
        'run': run_dir,
        'checkpoints': checkpoints_dir,
        'plots': run_dir,
        'model': run_dir / 'model.eqx',
        'metrics': run_dir / 'training_metrics.pkl',
        'metadata': run_dir / 'metadata.json',
    }


def train_window(cfg: dict, window: dict, results_dir: Path, training_seed: int = None) -> tuple:
    """Train a model on one window. Returns (model, training_metrics).

    Parameters
    ----------
    cfg : dict
        Full experiment config.
    window : dict
        Window data as loaded from pkl (output of prepare_windows.py).
    results_dir : Path
        Root results directory (financial/results/).
    training_seed : int or None
        If None, auto-generated.

    Returns
    -------
    model : eqx.Module
    metrics : dict
    """
    model_type = cfg['model_type']
    training_cfg = cfg['training']
    window_idx = window['window_idx']

    if training_seed is None:
        training_seed = pyrandom.randint(0, 2 ** 31)

    # Data
    train_times = jnp.array(window['train_times'])          # (T,)
    train_returns = jnp.array(window['train_returns'])      # (T, D)
    state_dim = train_returns.shape[1]
    state_init = jnp.zeros(state_dim)

    model_params = build_model_params(cfg, window)
    output_paths = build_output_paths(cfg, window_idx, results_dir)

    train_params = dict(
        learning_rate=training_cfg['learning_rate'],
        training_steps=training_cfg['training_steps'],
        n_loss_samples=training_cfg['n_loss_samples'],
        n_latent_steps=training_cfg['n_latent_steps'],
        obs_subsample_method=training_cfg.get('obs_subsample_method', 'all'),
        obs_subsample_count=training_cfg.get('obs_subsample_count', len(train_times)),
        obs_subsample_seed=training_cfg.get('obs_subsample_seed', 42),
        gc_interval=training_cfg.get('gc_interval', 50),
        clear_cache_interval=training_cfg.get('clear_cache_interval', 500),
        checkpoint_interval=training_cfg.get('checkpoint_interval', 0),
    )

    if model_type == 'tilted_stable_sde_financial':
        train_params.update(dict(
            lr_multiplier_mlp=training_cfg.get('lr_multiplier_mlp', 1.0),
            lr_multiplier_attention=training_cfg.get('lr_multiplier_attention', 10.0),
            lr_multiplier_drift=training_cfg.get('lr_multiplier_drift', 100),
            frozen_params=training_cfg.get('frozen_params', ['diffusion.raw_weight']),
            tilting_regularisation=training_cfg.get('tilting_regularisation', 1e-5),
            drift_regularization=training_cfg.get('drift_regularization', 0.1),
            coeff_A_regularization=training_cfg.get('coeff_A_regularization', 1e-5),
            coeff_B_regularization=training_cfg.get('coeff_B_regularization', 1e-5),
        ))
        model, metrics = train_tilted_stable_model(
            observations=train_returns,
            time_sequence=train_times,
            state_init=state_init,
            obs_std=cfg['data']['obs_std'],
            obs_subsample_method=train_params.pop('obs_subsample_method'),
            obs_subsample_count=train_params.pop('obs_subsample_count'),
            obs_subsample_seed=train_params.pop('obs_subsample_seed'),
            model_params=model_params,
            training_params=train_params,
            training_seed=training_seed,
            output_paths=output_paths,
            model_class=TiltedStableSDEFinancial,
            verbose=True,
        )

    elif model_type in ('gaussian_sde_financial', 'gaussian_sde_financial_multivariate'):
        train_params.update(dict(
            transition_steps=training_cfg.get('transition_steps', 100),
            decay_rate=training_cfg.get('decay_rate', 0.95),
            lr_multiplier_drift=training_cfg.get('lr_multiplier_drift', 1000.0),
            control_regularisation=training_cfg.get('control_regularisation', 0.5),
            drift_regularization=training_cfg.get('drift_regularization', 0.01),
            cholesky_regularisation=training_cfg.get('cholesky_regularisation', 0.0),
            frozen_params=training_cfg.get('frozen_params', []),
        ))
        model_class = (
            GaussianSDEFinancialMultivariate
            if model_type == 'gaussian_sde_financial_multivariate'
            else GaussianSDEFinancial
        )
        model, metrics = train_gaussian_model(
            observations=train_returns,
            time_sequence=train_times,
            state_init=state_init,
            obs_std=cfg['data']['obs_std'],
            obs_subsample_method=train_params.pop('obs_subsample_method'),
            obs_subsample_count=train_params.pop('obs_subsample_count'),
            obs_subsample_seed=train_params.pop('obs_subsample_seed'),
            model_params=model_params,
            training_params=train_params,
            training_seed=training_seed,
            output_paths=output_paths,
            model_class=model_class,
            verbose=True,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, metrics
