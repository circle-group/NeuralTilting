"""
Reusable training function for tilted stable SDE models.

This module provides a core training function that can be used by:
- Ad-hoc training scripts
- Batch experiments
- Future: Other model types with same API

The training function handles:
- Observation subsampling (at training time)
- Model initialization
- Optimizer setup with multi-transform for different parameter groups
- Training loop with comprehensive debugging/monitoring
- Checkpointing
- Metrics tracking

Architecture:
- train_tilted_stable_model() is the main entry point
- Core logic is delegated to modular components in training/components/:
  - optimiser_config.py: Optimizer creation and parameter labeling
  - training_monitor.py: Logging and monitoring
  - training_loop.py: Core training loop
"""

# Third-party imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import jax
import jax.numpy as jnp
import jax.random as random
import jax.tree_util as jtu
import equinox as eqx
import optax
import gc
import pickle
import resource
import platform
from typing import Dict, Tuple, Any, Optional, Callable

# Library-specific imports
from models.tilted_stable_sde import TiltedStableDrivenSDE
from models.gaussian_sde import GaussianDrivenSDE
from training.loss import loss_and_grad_for_gaussian_sde
from utils.training_utils import (
    sanitize_gradients,
    subsample_observations,
)

# New modular components
from training.components import (
    create_tilted_stable_optimizer,
    create_frozen_param_mask,
    TrainingMonitor,
    run_training_loop,
)


def get_memory_usage_gb():
    """
    Get current memory usage in GB, handling platform differences.

    macOS: ru_maxrss is in bytes
    Linux: ru_maxrss is in kilobytes
    """
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == 'Darwin':  # macOS
        return mem_usage / 1024**3  # bytes to GB
    else:  # Linux and others
        return mem_usage / 1024**2  # KB to GB

def train_tilted_stable_model(
    # Data
    observations,
    time_sequence,
    state_init,
    obs_std,

    # Observation subsampling (happens at TRAINING time)
    obs_subsample_method: str,
    obs_subsample_count: int,
    obs_subsample_seed: int,

    # Model parameters (dict matching TiltedStableDrivenSDE constructor)
    model_params: Dict,

    # Training parameters
    training_params: Dict,

    # Seeds
    training_seed: int,

    # Output paths
    output_paths: Dict[str, Path],

    # Optional
    model_class: type = TiltedStableDrivenSDE,
    model: Optional[Any] = None,
    verbose: bool = True,
    step_callback: Optional[Callable] = None,
) -> Tuple[Any, Dict]:
    """
    Train a tilted stable SDE model.

    This is the core training function extracted from train_tilted_stable_sde.py,
    made reusable for both ad-hoc and batch experiments.

    Parameters
    ----------
    observations : array
        Full observation array from dataset, shape (n_time_points, state_dim)
    time_sequence : array
        Full time sequence from dataset, shape (n_time_points,)
    state_init : array
        Initial state, shape (state_dim,)
    obs_std : float
        Observation noise standard deviation
    obs_subsample_method : str
        Method for subsampling observations: 'uniform', 'random', 'endpoints', 'all'
    obs_subsample_count : int
        Number of observations to use for training
    obs_subsample_seed : int
        Random seed for reproducible subsampling
    model_params : dict
        Dictionary of parameters for TiltedStableDrivenSDE initialization.
        Required keys: state_dim, alpha, tau, sigma, tilting_width, tilting_depth, etc.
    training_params : dict
        Dictionary of training parameters.
        Required keys: learning_rate, training_steps, n_loss_samples, n_latent_steps
        Optional keys: tilting_regularisation, drift_regularization, coeff_A_regularization, etc.
    training_seed : int
        Random seed for training (optimizer randomness, etc.)
    output_paths : dict
        Dictionary with paths for saving outputs:
        - 'run': Main run directory
        - 'checkpoints': Checkpoints subdirectory
        - 'plots': Plots subdirectory
        - 'model': Path for final model file
        - 'metrics': Path for metrics pickle file
        - 'metadata': Path for metadata JSON file
    model : TiltedStableDrivenSDE, optional
        Pre-initialized model. If None, creates new model from model_params.
    verbose : bool, optional
        Whether to print progress information (default: True)
    step_callback : callable, optional
        Optional callback function called after each step: callback(step, model, loss, metrics)

    Returns
    -------
    model : TiltedStableDrivenSDE
        Trained model
    metrics : dict
        Training metrics including:
        - 'loss_history': List of loss values
        - 'final_loss': Final loss value
        - 'parameter_history': List of parameter dicts (if drift is trainable)
        - 'trained_drift_component': Final drift component (if trainable)
        - 'drift_type': Drift component type name

    Notes
    -----
    The training loop is NOT JIT-compiled to preserve all debugging functionality.
    Only loss_and_grad_for_tilted_stable_sde is JIT'd (which is the bottleneck).
    """

    # ========================================================================
    # 1. SUBSAMPLE OBSERVATIONS
    # ========================================================================

    class SubsampleConfig:
        class training:
            pass

    config = SubsampleConfig()
    config.training.obs_subsample_method = obs_subsample_method
    config.training.obs_subsample_count = obs_subsample_count
    config.training.obs_subsample_seed = obs_subsample_seed

    training_indices = subsample_observations(time_sequence, config)
    training_times = time_sequence[training_indices]
    training_obs = observations[training_indices]

    if verbose:
        print(f"Subsampled {len(training_indices)}/{len(time_sequence)} observations using '{obs_subsample_method}' method", flush=True)

    # ========================================================================
    # 2. INITIALIZE MODEL
    # ========================================================================

    if model is None:
        if verbose:
            print("Initializing new model...", flush=True)
        model = model_class(**model_params)
    else:
        if verbose:
            print("Using pre-initialized model", flush=True)

    # ========================================================================
    # 3. CREATE OPTIMIZER
    # ========================================================================

    optimizer, optax_params_fn = create_tilted_stable_optimizer(model, training_params)
    opt_state = optimizer.init(optax_params_fn(model))

    # ========================================================================
    # 4. CREATE MONITOR
    # ========================================================================

    monitor = TrainingMonitor(verbose=verbose)

    # ========================================================================
    # 5. RUN TRAINING LOOP
    # ========================================================================

    key = random.key(training_seed)

    model, metrics = run_training_loop(
        model=model,
        state_init=state_init,
        training_times=training_times,
        training_obs=training_obs,
        obs_std=obs_std,
        training_params=training_params,
        optimizer=optimizer,
        opt_state=opt_state,
        optax_params_fn=optax_params_fn,
        monitor=monitor,
        output_paths=output_paths,
        key=key,
        step_callback=step_callback,
    )

    return model, metrics

def train_gaussian_model(
    # Data
    observations,
    time_sequence,
    state_init,
    obs_std,

    # Observation subsampling (happens at TRAINING time)
    obs_subsample_method: str,
    obs_subsample_count: int,
    obs_subsample_seed: int,

    # Model parameters (dict matching GaussianDrivenSDE constructor)
    model_params: Dict,

    # Training parameters
    training_params: Dict,

    # Seeds
    training_seed: int,

    # Output paths
    output_paths: Dict[str, Path],

    # Optional
    model_class: type = GaussianDrivenSDE,
    model: Optional[Any] = None,
    verbose: bool = True,
    step_callback: Optional[Callable] = None,
) -> Tuple[Any, Dict]:
    """
    Train a Gaussian SDE model with drift control.

    This is the Gaussian counterpart to train_tilted_stable_model(),
    made reusable for both ad-hoc and batch experiments.

    Parameters
    ----------
    observations : array
        Full observation array from dataset, shape (n_time_points, state_dim)
    time_sequence : array
        Full time sequence from dataset, shape (n_time_points,)
    state_init : array
        Initial state, shape (state_dim,)
    obs_std : float
        Observation noise standard deviation
    obs_subsample_method : str
        Method for subsampling observations: 'uniform', 'random', 'endpoints', 'all'
    obs_subsample_count : int
        Number of observations to use for training
    obs_subsample_seed : int
        Random seed for reproducible subsampling
    model_params : dict
        Dictionary of parameters for GaussianDrivenSDE initialization.
        Required keys: state_dim, sigma, control_width, control_depth, etc.
    training_params : dict
        Dictionary of training parameters.
        Required keys: learning_rate, training_steps, n_loss_samples, n_latent_steps
        Optional keys: transition_steps, decay_rate, control_regularisation, etc.
    training_seed : int
        Random seed for training (optimizer randomness, etc.)
    output_paths : dict
        Dictionary with paths for saving outputs:
        - 'run': Main run directory
        - 'checkpoints': Checkpoints subdirectory
        - 'plots': Plots subdirectory
        - 'model': Path for final model file
        - 'metrics': Path for metrics pickle file
        - 'metadata': Path for metadata JSON file
    model : GaussianDrivenSDE, optional
        Pre-initialized model. If None, creates new model from model_params.
    verbose : bool, optional
        Whether to print progress information (default: True)
    step_callback : callable, optional
        Optional callback function called after each step: callback(step, model, loss, metrics)

    Returns
    -------
    model : GaussianDrivenSDE
        Trained model
    metrics : dict
        Training metrics including:
        - 'loss_history': List of loss values
        - 'final_loss': Final loss value
        - 'parameter_history': List of parameter dicts (if drift is trainable)
        - 'trained_drift_component': Final drift component (if trainable)
        - 'drift_type': Drift component type name

    Notes
    -----
    The training loop is NOT JIT-compiled to preserve all debugging functionality.
    Only loss_and_grad_for_gaussian_sde is JIT'd (which is the bottleneck).
    """

    # ========================================================================
    # 1. SUBSAMPLE OBSERVATIONS
    # ========================================================================
    # This happens at TRAINING time, not generation time!

    # Create minimal config object for subsample_observations utility
    class SubsampleConfig:
        class training:
            pass

    config = SubsampleConfig()
    config.training.obs_subsample_method = obs_subsample_method
    config.training.obs_subsample_count = obs_subsample_count
    config.training.obs_subsample_seed = obs_subsample_seed

    training_indices = subsample_observations(time_sequence, config)
    training_times = time_sequence[training_indices]
    training_obs = observations[training_indices]

    if verbose:
        print(f"Subsampled {len(training_indices)}/{len(time_sequence)} observations using '{obs_subsample_method}' method", flush=True)

    # ========================================================================
    # 2. INITIALIZE MODEL
    # ========================================================================

    if model is None:
        if verbose:
            print("Initializing new model...", flush=True)
        model = model_class(**model_params)
    else:
        if verbose:
            print("Using pre-initialized model", flush=True)

    # ========================================================================
    # 3. SETUP OPTIMIZER
    # ========================================================================
    # Multi-transform optimizer with separate learning rates for control and drift

    frozen_paths = training_params.get('frozen_params', [])
    if frozen_paths:
        print(f"  Frozen parameters: {frozen_paths}", flush=True)

    def param_labels(params):
        """Label each parameter for multi-transform optimizer."""
        def label_fn(path, value):
            path_str = jtu.keystr(path)
            if 'control' in path_str:
                return 'control'
            elif 'drift' in path_str:
                return 'drift'
            else:
                return 'other'
        return jtu.tree_map_with_path(label_fn, params)

    # Extract learning rate and schedule params
    lr = training_params['learning_rate']
    transition_steps = training_params.get('transition_steps', 100)
    decay_rate = training_params.get('decay_rate', 0.95)
    lr_multiplier_drift = training_params.get('lr_multiplier_drift', 1000.0)

    # Create schedulers
    scheduler_control = optax.exponential_decay(
        init_value=lr,
        transition_steps=transition_steps,
        decay_rate=decay_rate
    )

    scheduler_drift = optax.exponential_decay(
        init_value=lr * lr_multiplier_drift,
        transition_steps=transition_steps,
        decay_rate=decay_rate
    )

    # Multi-transform optimizer
    base_optimizer = optax.multi_transform(
        transforms={
            'control': optax.chain(
                optax.adaptive_grad_clip(10.0),
                optax.adam(scheduler_control)
            ),
            'drift': optax.chain(
                optax.adaptive_grad_clip(10.0),
                optax.adam(scheduler_drift)
            ),
            'other': optax.chain(
                optax.adaptive_grad_clip(10.0),
                optax.adam(scheduler_control)
            )
        },
        param_labels=param_labels
    )

    if frozen_paths:
        mask = create_frozen_param_mask(model, frozen_paths)
        optimizer = optax.chain(
            optax.masked(base_optimizer, mask),
            optax.masked(optax.set_to_zero(), jtu.tree_map(lambda m: not m, mask))
        )
    else:
        optimizer = base_optimizer

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # ========================================================================
    # 4. TRAINING LOOP
    # ========================================================================
    # NOT JIT-compiled to preserve debugging functionality

    key = random.key(training_seed)
    loss_history = []
    parameter_history = []

    n_steps = training_params['training_steps']
    n_loss_samples = training_params['n_loss_samples']
    n_latent_steps = training_params['n_latent_steps']

    # Regularization parameters
    control_reg = training_params.get('control_regularisation', 0.1)
    drift_reg = training_params.get('drift_regularization', 0.01)
    cholesky_reg = training_params.get('cholesky_regularisation', 0.0)

    # Checkpointing
    checkpoint_interval = training_params.get('checkpoint_interval', 0)

    # Memory management (for HPC environments with limited memory)
    gc_interval = training_params.get('gc_interval', 50)  # Run garbage collection every N steps (0 = disabled)
    clear_cache_interval = training_params.get('clear_cache_interval', 150)  # Clear JAX cache every N steps (0 = disabled)

    if verbose:
        print(f"Starting training for {n_steps} steps...", flush=True)
        print(f"  Learning rate: {lr}", flush=True)
        print(f"  Loss samples per step: {n_loss_samples}", flush=True)
        print(f"  Latent steps: {n_latent_steps}", flush=True)
        if gc_interval > 0:
            print(f"  Garbage collection interval: {gc_interval} steps", flush=True)
        if clear_cache_interval > 0:
            print(f"  JAX cache clear interval: {clear_cache_interval} steps", flush=True)

    for i in range(n_steps):
        key, subkey = random.split(key)

        # Compute loss and gradients (THIS IS JIT'D - the bottleneck)
        loss_value, grad_value = loss_and_grad_for_gaussian_sde(
            model,
            n_loss_samples,
            state_init,
            training_times,
            training_obs,
            obs_std,
            subkey,
            n_latent_steps,
            T=None,  # Infer from obs_times
            regularisation_strength=control_reg,
            drift_regularization=drift_reg,
            cholesky_regularisation=cholesky_reg,
        )

        if verbose:
            print(f"Step {i+1}: Loss: {loss_value.item():.6f}", flush=True)

        # ====================================================================
        # DEBUGGING CHECKS (NOT JIT'D - important for robustness)
        # ====================================================================

        # Check for NaN/Inf in loss
        loss_is_nan = jnp.isnan(loss_value)
        loss_is_inf = jnp.isinf(loss_value)

        if loss_is_nan or loss_is_inf:
            if verbose:
                print(f"        Error at step {i+1}", flush=True)
                print("          Loss is NaN:", loss_is_nan, flush=True)
                print("          Loss is Inf:", loss_is_inf, flush=True)
                print("          Skipping this step and continuing...", flush=True)
            continue

        # Check for NaN/Inf in gradients
        grad_has_nan = jax.tree_util.tree_reduce(
            lambda x, y: x or y,
            jax.tree_util.tree_map(lambda g: jnp.any(jnp.isnan(g)) if g is not None else False, grad_value)
        )
        grad_has_inf = jax.tree_util.tree_reduce(
            lambda x, y: x or y,
            jax.tree_util.tree_map(lambda g: jnp.any(jnp.isinf(g)) if g is not None else False, grad_value)
        )

        if grad_has_nan or grad_has_inf:
            if verbose:
                print(f"        Step {i+1}: Gradients contain NaN/Inf, sanitizing gradients and continuing update", flush=True)
            grad_value = sanitize_gradients(grad_value)

        # Monitor gradient norm
        if verbose and i % 10 == 0:
            grad_norm = jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(lambda g: jnp.sum(g**2) if g is not None else 0.0, grad_value)
            )
            grad_norm = jnp.sqrt(grad_norm)
            print(f"        Gradient norm: {grad_norm:.2e}", flush=True)

            # Monitor control and drift gradient norms separately
            control_grads = jax.tree_util.tree_leaves(eqx.filter(grad_value.control, eqx.is_inexact_array))
            control_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in control_grads if g is not None))
            print(f"          Control grad norm: {control_grad_norm:.2e}", flush=True)

            if hasattr(grad_value.drift, 'weight'):
                drift_grads = jax.tree_util.tree_leaves(eqx.filter(grad_value.drift, eqx.is_inexact_array))
                drift_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in drift_grads if g is not None))
                print(f"          Drift grad norm: {drift_grad_norm:.2e}", flush=True)

        # Check for extremely high loss
        if loss_value > 1e10:
            if verbose:
                print(f"        Step {i+1}: Loss is too high ({loss_value:.2f}), skipping update", flush=True)
            continue

        # ====================================================================
        # APPLY UPDATES
        # ====================================================================

        # Zero out gradients for parameters that should not be trained
        # This is JIT-compatible because we always apply tree_at, just with different targets

        # Zero out diffusion gradients for DiagonalScaleFactor (frozen by design).
        # CholeskyDiffusion and other learnable diffusions are intentionally skipped.
        if hasattr(model.diffusion, 'raw_weight'):
            grad_value_filtered = eqx.tree_at(
                lambda m: m.diffusion.raw_weight,
                grad_value,
                replace=jnp.zeros_like(model.diffusion.raw_weight)
            )
        else:
            grad_value_filtered = grad_value

        # Conditionally zero out drift gradients if drift is not trainable
        if not model.trainable_drift:
            grad_value_filtered = eqx.tree_at(
                lambda m: m.drift,
                grad_value_filtered,
                replace=jax.tree_util.tree_map(
                    lambda x: jnp.zeros_like(x) if eqx.is_inexact_array(x) else x,
                    model.drift
                )
            )

        # Update with filtered gradients
        updates, opt_state = optimizer.update(
            grad_value_filtered,
            opt_state,
            eqx.filter(model, eqx.is_inexact_array)
        )

        # Monitor update norm
        if verbose and i % 10 == 0:
            update_norm = jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(lambda u: jnp.sum(u**2) if u is not None else 0.0, updates)
            )
            update_norm = jnp.sqrt(update_norm)
            print(f"          Total grad update norm: {update_norm:.2e}", flush=True)

        # Apply updates to model
        model = eqx.apply_updates(model, updates)

        # Record loss
        loss_history.append(loss_value.item())

        # ====================================================================
        # MEMORY MANAGEMENT (for HPC with limited memory)
        # ====================================================================

        # Periodic garbage collection to free Python objects
        if gc_interval > 0 and (i + 1) % gc_interval == 0:
            gc.collect()
            if verbose and (i + 1) % (gc_interval * 10) == 0:  # Only print occasionally
                print(f"        Ran garbage collection at step {i+1}", flush=True)

        # Periodic JAX cache clearing (use sparingly - will slow down training)
        # Only use if you're hitting OOM errors on HPC
        if clear_cache_interval > 0 and (i + 1) % clear_cache_interval == 0:
            jax.clear_caches()
            if verbose:
                print(f"        Cleared JAX compilation cache at step {i+1}", flush=True)

        # ====================================================================
        # DRIFT PARAMETER TRACKING
        # ====================================================================

        if hasattr(model.drift, 'weight') and hasattr(model.drift, 'bias'):
            # For OU drift, track interpretable parameters θ and μ
            # For other linear drifts, track raw weight and bias
            if hasattr(model.drift, 'theta') and hasattr(model.drift, 'mu'):
                # OUDiagonalLinearFunction: track θ (mean-reversion rate) and μ (mean-reversion level)
                current_theta = float(model.drift.theta.item()) if model.drift.theta.ndim > 0 else float(model.drift.theta)
                current_mu = float(model.drift.mu.item()) if model.drift.mu.ndim > 0 else float(model.drift.mu)
                parameter_history.append({
                    'theta': current_theta,
                    'mu': current_mu,
                    'param_type': 'ou'
                })
            else:
                # Generic DiagonalLinearFunction: track raw weight and bias
                current_weight = float(model.drift.weight.item()) if model.drift.weight.ndim > 0 else float(model.drift.weight)
                current_bias = float(model.drift.bias.item()) if model.drift.bias.ndim > 0 else float(model.drift.bias)
                parameter_history.append({
                    'weight': current_weight,
                    'bias': current_bias,
                    'param_type': 'linear'
                })
        elif hasattr(model.drift, 'theta1') and hasattr(model.drift, 'theta2'):
            # DoubleWellDriftFunction: track θ₁ and θ₂
            def _s(x):
                return float(x.item()) if x.ndim > 0 else float(x)
            parameter_history.append({
                'theta1': _s(model.drift.theta1),
                'theta2': _s(model.drift.theta2),
                'param_type': 'double_well'
            })

        # ====================================================================
        # CHECKPOINTING
        # ====================================================================

        if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
            checkpoint_path = output_paths['checkpoints'] / f"step_{i+1}.eqx"
            eqx.tree_serialise_leaves(checkpoint_path, model)

            # Also save metrics history at this checkpoint
            checkpoint_metrics_path = output_paths['checkpoints'] / f"step_{i+1}_metrics.pkl"
            checkpoint_metrics = {
                'loss_history': loss_history.copy(),
                'final_loss': loss_history[-1] if loss_history else None,
            }
            if parameter_history:
                checkpoint_metrics['parameter_history'] = parameter_history.copy()

            with open(checkpoint_metrics_path, 'wb') as f:
                pickle.dump(checkpoint_metrics, f)

            if verbose:
                print(f"  → Saved checkpoint: {checkpoint_path}", flush=True)
                print(f"  → Saved metrics: {checkpoint_metrics_path}", flush=True)

        # ====================================================================
        # CALLBACK
        # ====================================================================

        if step_callback is not None:
            callback_metrics = {
                'loss': loss_value.item(),
                'grad_norm': grad_norm.item() if 'grad_norm' in locals() else None
            }
            step_callback(i, model, loss_value, callback_metrics)

    if verbose:
        print(f"Training completed: {len(loss_history)} steps", flush=True)

    # ========================================================================
    # 5. PREPARE METRICS
    # ========================================================================

    metrics = {
        'loss_history': loss_history,
        'final_loss': loss_history[-1] if loss_history else None,
    }

    # Add parameter history if tracked
    if parameter_history:
        metrics['parameter_history'] = parameter_history

    # Add drift component if trainable
    if model.trainable_drift:
        metrics['trained_drift_component'] = model.drift
        metrics['drift_type'] = type(model.drift).__name__

    return model, metrics