"""
Core training loop for tilted stable SDE.

Orchestrates:
- Loss and gradient computation
- Gradient validation and sanitization
- Optimizer updates
- Checkpointing
- Memory management
"""

from typing import Tuple, Optional, Callable, Any, Dict
import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
import gc
import pickle

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.loss import loss_and_grad_for_tilted_stable_sde
from utils.training_utils import sanitize_gradients
from .training_monitor import TrainingMonitor


def run_training_loop(
    model: Any,
    # Data
    state_init: jnp.ndarray,
    training_times: jnp.ndarray,
    training_obs: jnp.ndarray,
    obs_std: float,
    # Config (from YAML)
    training_params: dict,
    # Components
    optimizer,
    opt_state,
    optax_params_fn: Callable,
    monitor: TrainingMonitor,
    # Output
    output_paths: dict,
    # Random
    key: jax.random.PRNGKey,
    # Callbacks
    step_callback: Optional[Callable] = None,
) -> Tuple[Any, Dict]:
    """
    Core training loop - transforms model -> trained model.

    Handles:
    - Loss and gradient computation (JIT'd)
    - Gradient validation and sanitization
    - Optimizer updates (with frozen parameter masking)
    - Memory management
    - Checkpointing
    - Metrics tracking

    Args:
        model: The TiltedStableDrivenSDE model to train
        state_init: Initial state for simulation
        training_times: Time points for training observations
        training_obs: Training observations
        obs_std: Observation noise std
        training_params: Dict with all training hyperparameters
        optimizer: optax optimizer (with masking for frozen params)
        opt_state: Optimizer state
        optax_params_fn: Function to extract gradient-updatable params
        monitor: TrainingMonitor for logging
        output_paths: Dict with paths for checkpoints
        key: JAX PRNG key
        step_callback: Optional callback(step, model, loss, metrics)

    Returns:
        (trained_model, metrics_dict)
    """
    # Extract training parameters
    n_steps = training_params['training_steps']
    n_loss_samples = training_params['n_loss_samples']
    n_latent_steps = training_params['n_latent_steps']

    # Regularization
    tilting_reg = training_params.get('tilting_regularisation', 0.05)
    drift_reg = training_params.get('drift_regularization', 0.01)
    coeff_A_reg = training_params.get('coeff_A_regularization', 1e-4)
    coeff_B_reg = training_params.get('coeff_B_regularization', 5e-5)

    # Checkpointing and memory management
    checkpoint_interval = training_params.get('checkpoint_interval', 0)
    gc_interval = training_params.get('gc_interval', 50)
    clear_cache_interval = training_params.get('clear_cache_interval', 150)

    # Log training start
    monitor.log_training_start(
        n_steps=n_steps,
        lr=training_params['learning_rate'],
        n_loss_samples=n_loss_samples,
        n_latent_steps=n_latent_steps,
        gc_interval=gc_interval,
        clear_cache_interval=clear_cache_interval,
    )

    # Tracking
    loss_history = []
    parameter_history = []
    grad_norm = None  # Track for callback

    for i in range(n_steps):
        key, subkey = random.split(key)

        # ================================================================
        # COMPUTE LOSS AND GRADIENTS (JIT'd - the bottleneck)
        # ================================================================
        (loss_value, aux_data), grad_value = loss_and_grad_for_tilted_stable_sde(
            model,
            n_loss_samples,
            state_init,
            training_times,
            training_obs,
            obs_std,
            subkey,
            n_latent_steps,
            regularisation_strength=tilting_reg,
            drift_regularization=drift_reg,
            coeff_A_regularization=coeff_A_reg,
            coeff_B_regularization=coeff_B_reg
        )

        # Log step
        monitor.log_step(i, loss_value.item(), aux_data)
        monitor.log_compilation_memory(i)

        # ================================================================
        # VALIDATION CHECKS
        # ================================================================

        # Check for NaN/Inf in loss
        loss_is_nan = jnp.isnan(loss_value)
        loss_is_inf = jnp.isinf(loss_value)

        if loss_is_nan or loss_is_inf:
            monitor.log_nan_inf_error(i, loss_is_nan, loss_is_inf)
            continue

        # Check for NaN/Inf in gradients
        grad_has_nan = jax.tree_util.tree_reduce(
            lambda x, y: x or y,
            jax.tree_util.tree_map(
                lambda g: jnp.any(jnp.isnan(g)) if g is not None else False,
                grad_value
            )
        )
        grad_has_inf = jax.tree_util.tree_reduce(
            lambda x, y: x or y,
            jax.tree_util.tree_map(
                lambda g: jnp.any(jnp.isinf(g)) if g is not None else False,
                grad_value
            )
        )

        if grad_has_nan or grad_has_inf:
            monitor.log_gradient_sanitization(i)
            grad_value = sanitize_gradients(grad_value)

        # Check for extremely high loss
        if loss_value > 1e10:
            monitor.log_high_loss_skip(i, loss_value.item())
            continue

        # ================================================================
        # MONITORING
        # ================================================================
        grad_norm = monitor.log_gradients(i, grad_value, model)
        monitor.log_coefficients(i, model, training_times)
        monitor.log_attention(i, model, grad_value)
        monitor.log_memory(i)

        # ================================================================
        # OPTIMIZER UPDATE
        # ================================================================
        # Note: Frozen parameters (diffusion, drift if not trainable) are
        # automatically masked by optax.masked() in the optimizer

        # Extract parameters for optimizer
        params = optax_params_fn(model)
        grads = optax_params_fn(grad_value)

        # Compute updates
        updates, opt_state = optimizer.update(grads, opt_state, params)

        monitor.log_update_norm(i, updates)

        # Apply updates to model
        model = eqx.apply_updates(model, updates)

        # Record loss
        loss_history.append(loss_value.item())

        # ================================================================
        # MEMORY MANAGEMENT
        # ================================================================
        if gc_interval > 0 and (i + 1) % gc_interval == 0:
            gc.collect()
            if (i + 1) % (gc_interval * 10) == 0:
                monitor.log_gc(i)

        if clear_cache_interval > 0 and (i + 1) % clear_cache_interval == 0:
            jax.clear_caches()
            monitor.log_cache_clear(i)

        # ================================================================
        # DRIFT PARAMETER TRACKING
        # ================================================================
        drift_params = monitor.log_drift_parameters(i, model)
        if drift_params is not None:
            parameter_history.append(drift_params)

        # ================================================================
        # DIFFUSION PARAMETER TRACKING (for monitoring frozen params)
        # ================================================================
        diffusion_params = monitor.log_diffusion_parameters(i, model)
        if diffusion_params is not None:
            parameter_history.append(diffusion_params)

        # ================================================================
        # CHECKPOINTING
        # ================================================================
        if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
            checkpoint_path = output_paths['checkpoints'] / f"step_{i+1}.eqx"
            eqx.tree_serialise_leaves(checkpoint_path, model)

            checkpoint_metrics_path = output_paths['checkpoints'] / f"step_{i+1}_metrics.pkl"
            checkpoint_metrics = {
                'loss_history': loss_history.copy(),
                'final_loss': loss_history[-1] if loss_history else None,
            }
            if parameter_history:
                checkpoint_metrics['parameter_history'] = parameter_history.copy()

            with open(checkpoint_metrics_path, 'wb') as f:
                pickle.dump(checkpoint_metrics, f)

            monitor.log_checkpoint(i, str(checkpoint_path), str(checkpoint_metrics_path))

        # ================================================================
        # CALLBACK
        # ================================================================
        if step_callback is not None:
            callback_metrics = {
                'loss': loss_value.item(),
                'grad_norm': grad_norm.item() if grad_norm is not None else None
            }
            step_callback(i, model, loss_value, callback_metrics)

    # ================================================================
    # FINALIZE
    # ================================================================
    monitor.log_training_complete(len(loss_history))

    # Prepare metrics
    metrics = {
        'loss_history': loss_history,
        'final_loss': loss_history[-1] if loss_history else None,
    }

    if parameter_history:
        metrics['parameter_history'] = parameter_history

    if model.trainable_drift:
        metrics['trained_drift_component'] = model.drift
        metrics['drift_type'] = type(model.drift).__name__

    return model, metrics
