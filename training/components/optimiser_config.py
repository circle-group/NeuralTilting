"""
Optimizer configuration for attention-based tilted stable SDE training.

This module creates a specialized optimizer for training tilted stable SDEs with:
- Quantile-based gradient clipping for heavy-tailed Levy gradients
- Per-component constant learning rates (mlp_A, mlp_B, temporal_attention, drift)
- RMSProp for phi networks (robust to heavy-tailed gradients)
- Frozen parameter masking (e.g., diffusion.weight, attention_sharpness)

Architecture: QuadraticNeuralPotential with TemporalAttentionEncoding
-----------------------------------------------------------------
Parameter groups:
- 'mlp_A': phi.mlp_A parameters
- 'mlp_B': phi.mlp_B parameters
- 'temporal_attention': phi.temporal_attention parameters
- 'drift': drift parameters (non-phi parameters)

IMPORTANT: Label Consistency
-----------------------------
If you modify parameter labels in create_param_labels(), you must also update:
1. create_schedulers() - add learning rate schedule for the new label
2. create_tilted_stable_optimizer() - add transform for the new label

Frozen Parameters
-----------------
Specify frozen parameters via training_params['frozen_params'] as a list of path strings.
Default: ['diffusion.raw_weight'] (Levy scale parameter - must stay fixed)
Optional: Add 'phi.temporal_attention.attention_sharpness' if over-smoothing occurs

Example:
    frozen_params: ['diffusion.raw_weight', 'phi.temporal_attention.attention_sharpness']
"""

from typing import Callable, Tuple
import jax.tree_util as jtu
import equinox as eqx
import optax

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.training_utils import scale_by_layerwise_quantile_norm


def create_param_labels() -> Callable:
    """
    Create parameter labeling function for multi_transform optimizer.

    Labels:
    - 'mlp_A': phi.mlp_A parameters
    - 'mlp_B': phi.mlp_B parameters
    - 'temporal_attention': phi.temporal_attention parameters
    - 'drift': drift parameters (non-phi parameters)

    Returns:
        Function that labels a parameter pytree
    """
    def param_labels(params):
        def label_fn(path, value):
            if value is None:
                return None

            path_str = jtu.keystr(path)

            # Non-phi parameters (drift, diffusion, etc.)
            if 'phi' not in path_str:
                return 'drift'

            # Phi network components
            if 'temporal_attention' in path_str:
                return 'temporal_attention'
            elif 'mlp_A' in path_str:
                return 'mlp_A'
            elif 'mlp_B' in path_str:
                return 'mlp_B'
            else:
                return 'drift'

        return jtu.tree_map_with_path(label_fn, params)

    return param_labels


def create_schedulers(training_params: dict) -> dict:
    """
    Create constant learning rate schedulers for each parameter group.

    Reads from training_params:
        - learning_rate: Base learning rate
        - lr_multiplier_mlp: LR multiplier for mlp_A/B (default: 1.0)
        - lr_multiplier_attention: LR multiplier for temporal_attention (default: 10.0)
        - lr_multiplier_drift: LR multiplier for drift (default: 1000.0)

    Returns:
        Dict with constant schedule for each parameter group
    """
    lr = training_params['learning_rate']

    # Learning rate multipliers (configurable)
    lr_mult_mlp = training_params.get('lr_multiplier_mlp', 1.0)
    lr_mult_attention = training_params.get('lr_multiplier_attention', 10.0)
    lr_mult_drift = training_params.get('lr_multiplier_drift', 1000.0)

    return {
        'mlp_A': optax.constant_schedule(lr * lr_mult_mlp),
        'mlp_B': optax.constant_schedule(lr * lr_mult_mlp),
        'temporal_attention': optax.constant_schedule(lr * lr_mult_attention),
        'drift': optax.constant_schedule(lr * lr_mult_drift),
    }


def create_frozen_param_mask(model, frozen_paths: list) -> dict:
    """
    Create a boolean mask pytree for optax.masked().

    Parameters marked as frozen will have False in the mask,
    preventing gradient updates.

    Args:
        model: The model to create mask for
        frozen_paths: List of parameter paths to freeze, e.g.:
            ['diffusion.weight', 'phi.temporal_attention.attention_sharpness']

    Returns:
        Boolean pytree compatible with optax.masked()
    """
    params = eqx.filter(model, eqx.is_inexact_array)

    def should_update(path, value):
        """Return True if param should receive gradient updates."""
        if value is None:
            return False

        path_str = jtu.keystr(path)

        # Check if this parameter matches any frozen path
        for frozen_path in frozen_paths:
            if frozen_path in path_str:
                return False

        return True

    mask = jtu.tree_map_with_path(should_update, params)
    return mask


def create_tilted_stable_optimizer(
    model,
    training_params: dict
) -> Tuple[optax.GradientTransformation, Callable]:
    """
    Create multi-transform optimizer for tilted stable SDE training.

    Optimizer stack:
    1. Quantile-based gradient clipping (handles heavy-tailed Levy gradients)
    2. RMSProp for phi networks (robust to heavy-tailed gradients)
    3. Adam for drift (smoother gradients, benefits from momentum)
    4. Per-group constant learning rates
    5. Final gradient clipping
    6. Masking for frozen parameters

    Args:
        model: The TiltedStableDrivenSDE model
        training_params: Dict with learning_rate, frozen_params, etc.
            Required:
                - learning_rate: Base learning rate
            Optional:
                - lr_multiplier_mlp: Multiplier for mlp_A/B (default: 1.0)
                - lr_multiplier_attention: Multiplier for attention (default: 10.0)
                - lr_multiplier_drift: Multiplier for drift (default: 1000.0)
                - frozen_params: List of paths to freeze (default: ['diffusion.weight'])

    Returns:
        (optimizer, optax_params_fn) tuple
    """
    # Get frozen parameter paths
    frozen_paths = training_params.get('frozen_params', ['diffusion.raw_weight']).copy()

    # Automatically freeze drift if not trainable
    if hasattr(model, 'trainable_drift') and not model.trainable_drift:
        frozen_paths.append('drift')

    print(f"Optimizer: Attention-based architecture with constant learning rates", flush=True)
    print(f"  Using RMSProp for phi networks (robust to heavy-tailed gradients)", flush=True)
    if frozen_paths:
        print(f"  Frozen parameters: {frozen_paths}", flush=True)

    # Create constant learning rate schedulers
    schedulers = create_schedulers(training_params)

    # Create parameter labeling function
    param_labels = create_param_labels()

    # Multi-transform optimizer with per-component learning rates
    base_optimizer = optax.multi_transform(
        transforms={
            # Phi networks: quantile clipping + RMSProp (robust to heavy tails)
            'mlp_A': optax.chain(
                scale_by_layerwise_quantile_norm(quantile=0.99),
                #optax.sgd(learning_rate=1.0, momentum=0.9, nesterov=True),
                optax.rmsprop(learning_rate=1.0, decay=0.9, eps=1e-6),
                optax.scale_by_schedule(schedulers['mlp_A']),
                optax.clip(max_delta=20.0),
            ),
            'mlp_B': optax.chain(
                scale_by_layerwise_quantile_norm(quantile=0.99),
                #optax.sgd(learning_rate=1.0, momentum=0.9, nesterov=True),
                optax.rmsprop(learning_rate=1.0, decay=0.9, eps=1e-6),
                optax.scale_by_schedule(schedulers['mlp_B']),
                optax.clip(max_delta=20.0),
            ),
            'temporal_attention': optax.chain(
                scale_by_layerwise_quantile_norm(quantile=0.99),
                #optax.sgd(learning_rate=1.0, momentum=0.9, nesterov=True),
                optax.rmsprop(learning_rate=1.0, decay=0.9, eps=1e-6),
                optax.scale_by_schedule(schedulers['temporal_attention']),
                optax.clip(max_delta=20.0),
            ),
            # Drift: Adam (smoother gradients, benefits from momentum)
            'drift': optax.chain(
                optax.clip_by_global_norm(1e4),
                optax.adam(schedulers['drift'], b1=0.9, b2=0.9, eps=1e-7),
            )
        },
        param_labels=param_labels
    )

    # Create frozen parameter mask
    mask = create_frozen_param_mask(model, frozen_paths)

    # Wrap with masking to prevent updates to frozen parameters
    # Note: optax.masked passes through gradients for mask=False parameters,
    # so we need to explicitly zero them out
    optimizer = optax.chain(
        optax.masked(base_optimizer, mask),
        optax.masked(optax.set_to_zero(), jtu.tree_map(lambda m: not m, mask))  # Invert mask: zero where original was False
    )

    # Simple optax_params function (just extract inexact arrays)
    def optax_params(m):
        return eqx.filter(m, eqx.is_inexact_array)

    return optimizer, optax_params
