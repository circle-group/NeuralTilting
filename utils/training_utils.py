"""
Utility functions for training the tilted stable SDE model.
"""

import jax.tree_util as tree_util
import jax.numpy as jnp
import jax.random as random

def build_simulation_grid(T_start, T_end, n_steps, obs_times):
    """
    Build unified simulation grid for SDE with observation time inclusion.

    This function creates a time discretization grid for SDE simulation that includes
    both a uniform fine grid and observation times. This ensures observations
    are exactly hit during simulation (no interpolation needed).

    Parameters
    ----------
    T_start : float
        Start time of simulation
    T_end : float
        End time of simulation
    n_steps : int
        Number of time steps for uniform discretization
    obs_times : ndarray
        Observation times to include in the grid.
        Shape: (n_obs,). For no observations, pass empty array jnp.array([]).

    Returns
    -------
    time_sequence : ndarray
        Sorted, unique time points including both the uniform discretization
        and all observation times.

    Examples
    --------
    >>> # Grid with observations
    >>> obs = jnp.array([0.15, 0.37, 0.89])
    >>> grid = build_simulation_grid(0.0, 1.0, 10, obs_times=obs)
    >>> # Returns: [0.0, 0.1, 0.15, 0.2, 0.3, 0.37, ..., 0.89, 0.9, 1.0]

    >>> # Grid without observations (use empty array)
    >>> grid = build_simulation_grid(0.0, 1.0, 10, obs_times=jnp.array([]))
    >>> # Returns: [0.0, 0.1, 0.2, ..., 1.0]

    Notes
    -----
    This function is JIT-compatible. The obs_times parameter must always be
    a JAX array (use jnp.array([]) for no observations, not None).
    """
    fine_grid = jnp.linspace(T_start, T_end, n_steps + 1)
    # Use sort instead of unique for JIT compatibility
    # Duplicate time points (when obs_times coincide with fine_grid) are harmless: dt=0 means no state change
    time_sequence = jnp.sort(jnp.concatenate([fine_grid, obs_times]))
    return time_sequence


def subsample_observations(time_sequence, config):
    """
    Generate indices for subsampling observations based on configuration.

    This function determines which observations from the dense generation
    should be used for training, enabling sparse observation experiments.

    Parameters
    ----------
    time_sequence : ndarray
        The full time sequence at which observations were generated.
        Shape: (n_total_obs,)
    config : ExperimentConfig
        Configuration object containing training.obs_subsample_* parameters

    Returns
    -------
    training_indices : ndarray
        Array of indices (sorted) indicating which observations to use for training.
        Shape: (n_training_obs,) where n_training_obs <= len(time_sequence)

    Examples
    --------
    >>> time_sequence = jnp.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    >>> config.training.obs_subsample_method = 'uniform'
    >>> config.training.obs_subsample_count = 5
    >>> indices = subsample_observations(time_sequence, config)
    >>> # Returns: [0, 2, 5, 7, 10] or similar evenly spaced indices

    >>> config.training.obs_subsample_method = 'all'
    >>> indices = subsample_observations(time_sequence, config)
    >>> # Returns: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (all observations)
    """

    n_total_obs = len(time_sequence)
    method = config.training.obs_subsample_method
    count = config.training.obs_subsample_count
    seed = config.training.obs_subsample_seed

    if method == 'all':
        # Use all observations
        return jnp.arange(n_total_obs)

    # Validate count only when subsampling
    if count > n_total_obs:
        raise ValueError(
            f"obs_subsample_count ({count}) cannot exceed total observations ({n_total_obs})"
        )

    elif method == 'uniform':
        # Evenly spaced indices using linspace on the INDEX space
        # This ensures we pick actual indices from the time_sequence
        indices = jnp.linspace(0, n_total_obs - 1, count)
        indices = jnp.round(indices).astype(int)
        return jnp.unique(indices)  # Remove duplicates and ensure sorted

    elif method == 'random':
        # Random subset of indices
        key = random.key(seed)
        indices = random.choice(key, n_total_obs, shape=(count,), replace=False)
        return jnp.sort(indices)

    elif method == 'endpoints':
        # Always include first and last observation, random interior points
        if count < 2:
            raise ValueError("obs_subsample_count must be >= 2 for 'endpoints' method")

        n_interior = count - 2
        if n_interior > 0:
            key = random.key(seed)
            # Sample from interior indices [1, 2, ..., n_total_obs-2]
            interior_indices = random.choice(
                key, n_total_obs - 2, shape=(n_interior,), replace=False
            ) + 1
            indices = jnp.concatenate([
                jnp.array([0, n_total_obs - 1]),
                interior_indices
            ])
        else:
            indices = jnp.array([0, n_total_obs - 1])

        return jnp.sort(indices)

    else:
        raise ValueError(
            f"Unknown obs_subsample_method: '{method}'. "
            f"Valid options: 'uniform', 'random', 'endpoints', 'all'"
        )

# Function to sanitize gradients by setting NaN/Inf values to zero
def sanitize_gradients(grads):
    """Replace NaN/Inf values in gradients with zeros to allow continued training."""
    def replace_nan_inf(g):
        if g is None:
            return g
        return jnp.where(jnp.isnan(g) | jnp.isinf(g), 0.0, g)
    return tree_util.tree_map(replace_nan_inf, grads)


def clip_by_quantile(quantile=0.95):
    """
    Clip gradients based on quantile of their absolute values.

    Unlike clip_by_global_norm, this:
    - Uses the empirical distribution of gradient magnitudes (no Gaussian assumption)
    - Clips each parameter to the same absolute threshold (no norm-based scaling)
    - Adapts to the actual gradient scale each step (no arbitrary threshold like 5e5)

    When used with optax.multi_transform, the quantile is computed independently
    for each parameter group (e.g., A_coef, B_coef), preventing cross-network coupling.

    Parameters
    ----------
    quantile : float, default=0.95
        Quantile threshold for clipping. Gradient values above this percentile
        (in absolute value) get clipped.
        - 0.95: clips top 5% of gradients by magnitude
        - 0.99: clips top 1% (more permissive for heavy-tailed distributions)

    Returns
    -------
    optax.GradientTransformation
        Stateless transformation that clips gradients based on their empirical distribution.

    Notes
    -----
    For heavy-tailed gradient distributions (e.g., from Lévy processes):
    - Start with quantile=0.95 (conservative)
    - If updates are too small, increase to 0.97-0.99
    - If training is unstable, decrease to 0.90-0.93

    Examples
    --------
    >>> # Drop-in replacement for clip_by_global_norm
    >>> optimizer = optax.chain(
    ...     clip_by_quantile(quantile=0.95),
    ...     optax.adam(1e-3)
    ... )

    >>> # With multi_transform (clips independently per network)
    >>> optimizer = optax.multi_transform({
    ...     'A_coef': optax.chain(clip_by_quantile(0.95), optax.adam(1e-3)),
    ...     'B_coef': optax.chain(clip_by_quantile(0.95), optax.adam(1e-4)),
    ... }, param_labels=...)
    """
    # Import here to avoid circular dependency
    import optax
    from typing import NamedTuple

    class ClipState(NamedTuple):
        """Empty state - clipping is stateless."""
        pass

    def init_fn(params):
        """Initialize (stateless, so just return empty state)."""
        del params
        return ClipState()

    def update_fn(updates, state, params=None):
        """Clip gradients to quantile threshold computed from their distribution."""
        del params  # Not needed

        # Flatten all gradients in this parameter group into single array
        flat_grads = jnp.concatenate([
            jnp.abs(g).ravel()
            for g in tree_util.tree_leaves(updates)
        ])

        # Compute threshold from empirical distribution
        # Use maximum to prevent threshold from being zero (can happen if all gradients are tiny)
        threshold = jnp.maximum(jnp.quantile(flat_grads, quantile), 1e-12)

        # Clip each parameter independently to this threshold
        clipped_updates = tree_util.tree_map(
            lambda g: jnp.clip(g, -threshold, threshold),
            updates
        )

        return clipped_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def clip_by_quantile_layerwise(quantile=0.95):
    """
    Clip gradients based on quantile of absolute values, computed independently per layer.

    Unlike clip_by_quantile which computes a single threshold across all parameters,
    this function computes a separate threshold for each parameter tensor (layer).
    This is crucial for heavy-tailed gradient distributions where different layers
    have vastly different gradient scales.

    Benefits over network-wide quantile clipping:
    - Preserves gradient information in layers with small gradients
    - Prevents aggressive clipping in layers with naturally large gradients
    - Better suited for networks with heterogeneous layer behaviors (e.g., Lévy processes)

    Parameters
    ----------
    quantile : float, default=0.95
        Quantile threshold for clipping, applied independently to each layer.
        Gradient values above this percentile (in absolute value) get clipped.
        - 0.95: clips top 5% of gradients by magnitude per layer
        - 0.99: clips top 1% (more permissive for heavy-tailed distributions)

    Returns
    -------
    optax.GradientTransformation
        Stateless transformation that clips gradients layer-wise based on
        their empirical distribution.

    Notes
    -----
    For heavy-tailed gradient distributions (e.g., from Lévy processes):
    - Start with quantile=0.95 (conservative)
    - If updates are too small, increase to 0.97-0.99
    - If training is unstable, decrease to 0.90-0.93

    This method is particularly useful when:
    - Different layers have very different gradient scales
    - Gradients exhibit heavy tails (Lévy processes, multiplicative interactions)
    - Network-wide clipping removes too much information from some layers

    Examples
    --------
    >>> # Drop-in replacement for clip_by_quantile
    >>> optimizer = optax.chain(
    ...     clip_by_quantile_layerwise(quantile=0.95),
    ...     optax.adam(1e-3)
    ... )

    >>> # With multi_transform (clips independently per layer within each network)
    >>> optimizer = optax.multi_transform({
    ...     'A_coef': optax.chain(clip_by_quantile_layerwise(0.95), optax.adam(1e-3)),
    ...     'B_coef': optax.chain(clip_by_quantile_layerwise(0.95), optax.adam(1e-4)),
    ... }, param_labels=...)
    """
    # Import here to avoid circular dependency
    import optax
    from typing import NamedTuple

    class ClipState(NamedTuple):
        """Empty state - clipping is stateless."""
        pass

    def init_fn(params):
        """Initialize (stateless, so just return empty state)."""
        del params
        return ClipState()

    def update_fn(updates, state, params=None):
        """Clip gradients to quantile threshold computed per-layer."""
        del params  # Not needed

        def clip_single_param(g):
            """Compute quantile threshold and clip for a single parameter tensor."""
            if g is None:
                return g

            # Flatten this parameter's gradients
            flat_g = jnp.abs(g).ravel()

            # Compute threshold from this parameter's empirical distribution
            # Use maximum to prevent threshold from being zero
            threshold = jnp.maximum(jnp.quantile(flat_g, quantile), 1e-12)

            # Clip to this layer-specific threshold
            return jnp.clip(g, -threshold, threshold)

        # Apply clipping independently to each parameter tensor
        clipped_updates = tree_util.tree_map(clip_single_param, updates)

        return clipped_updates, state

    return optax.GradientTransformation(init_fn, update_fn)

def scale_by_layerwise_quantile_norm(quantile=0.95, eps=1e-8):
    """
    Scales the entire gradient tensor by a robust layerwise norm.
    Preserves relative gradient magnitudes (no clipping).
    """
    import optax
    import jax.numpy as jnp
    import jax.tree_util as jtu
    from typing import NamedTuple

    class ScaleState(NamedTuple):
        pass

    def init_fn(params):
        del params
        return ScaleState()

    def update_fn(updates, state, params=None):
        del params

        def rescale(g):
            if g is None:
                return g

            flat = jnp.abs(g).ravel()
            q = jnp.quantile(flat, quantile) + eps

            # robust layerwise norm
            norm = jnp.linalg.norm(g) / jnp.sqrt(g.size)

            scale = jnp.maximum(1.0, norm / q)
            return g / scale

        return jtu.tree_map(rescale, updates), state

    return optax.GradientTransformation(init_fn, update_fn)


def compute_jump_locations_from_paths(
    paths: jnp.ndarray,
    time_sequence: jnp.ndarray,
    observations: jnp.ndarray,
    obs_times: jnp.ndarray,
    obs_std: float,
    n_reference_times: int,
    detection_quantile: float = 0.90,
    min_spacing_fraction: float = 0.02,
    accumulated_increments: jnp.ndarray = None,
    accumulation_decay: float = 0.7
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Identify times with large increments, weighted by LOCAL fit quality.

    Uses PER-TIME likelihood weighting: a path that is wrong overall but good
    at a specific time still contributes at that time. This is more robust than
    per-path weighting against single bad jumps.

    Key features for stability:
    - Accumulates evidence across batches using exponential moving average
    - Uses softer temperature scaling to avoid extreme weight concentration
    - Applies stronger temporal smoothing

    Parameters
    ----------
    paths : array, shape (n_samples, n_times, state_dim)
        Simulated posterior paths (reused from loss computation)
    time_sequence : array, shape (n_times,)
        Time points corresponding to path indices
    observations : array, shape (n_obs, state_dim)
        Observed data values
    obs_times : array, shape (n_obs,)
        Times at which observations were made
    obs_std : float
        Observation noise standard deviation
    n_reference_times : int
        Number of reference times to return
    detection_quantile : float
        Increments above this quantile are considered potential jumps
    min_spacing_fraction : float
        Minimum spacing between reference times as fraction of total period
    accumulated_increments : array or None
        Previous accumulated increment statistics (for EMA across batches)
    accumulation_decay : float
        Decay factor for exponential moving average (0.7 = 70% old, 30% new)

    Returns
    -------
    new_reference_times : array, shape (n_reference_times,)
        Suggested reference time positions, sorted and bounded
    new_accumulated_increments : array, shape (n_times-1,)
        Updated accumulated statistics for next call
    """
    n_samples, n_times, state_dim = paths.shape
    period = time_sequence[-1] - time_sequence[0]

    # ==================================================================
    # Step 1: Compute per-time, per-path local fit quality
    # ==================================================================
    # For each time point, find closest observation and compute local error

    # Find which observation is closest to each time point
    obs_indices = jnp.searchsorted(obs_times, time_sequence)
    obs_indices = jnp.clip(obs_indices, 0, len(obs_times) - 1)

    # Also check previous observation (whichever is closer)
    obs_indices_prev = jnp.clip(obs_indices - 1, 0, len(obs_times) - 1)
    dist_to_current = jnp.abs(time_sequence - obs_times[obs_indices])
    dist_to_prev = jnp.abs(time_sequence - obs_times[obs_indices_prev])
    closest_obs_idx = jnp.where(dist_to_current < dist_to_prev, obs_indices, obs_indices_prev)

    # Get observations at closest times: (n_times, state_dim)
    closest_obs = observations[closest_obs_idx]

    # Compute per-time local error for each path
    local_errors = jnp.sum((paths - closest_obs[None, :, :]) ** 2, axis=2)
    # local_errors shape: (n_samples, n_times)

    # Convert to local weights using SOFTER temperature scaling
    # Use median error as reference scale instead of obs_std^2 (which can be tiny)
    median_error = jnp.median(local_errors)
    temperature = jnp.maximum(median_error, obs_std ** 2 * state_dim) * 2.0

    # Clip the exponent to prevent extreme weights
    scaled_errors = jnp.clip(-local_errors / temperature, -10.0, 0.0)
    local_weights = jnp.exp(scaled_errors)
    # local_weights shape: (n_samples, n_times)

    # Normalize weights per time point (so they sum to 1 across samples)
    # Add larger epsilon to prevent division issues
    local_weights = local_weights / (jnp.sum(local_weights, axis=0, keepdims=True) + 1e-6)

    # ==================================================================
    # Step 2: Compute increments and normalize per path
    # ==================================================================
    increments = jnp.abs(jnp.diff(paths, axis=1))  # (n_samples, n_times-1, state_dim)
    times_mid = (time_sequence[1:] + time_sequence[:-1]) / 2.0

    # Robust normalization per path using MAD (Median Absolute Deviation)
    median_inc = jnp.median(increments, axis=(1, 2), keepdims=True)
    mad_inc = jnp.median(jnp.abs(increments - median_inc), axis=(1, 2), keepdims=True)
    normalized_inc = increments / (mad_inc + 1e-8)

    # Sum over state dimensions
    normalized_inc_sum = jnp.sum(normalized_inc, axis=2)  # (n_samples, n_times-1)

    # ==================================================================
    # Step 3: Weighted aggregation using LOCAL fit quality
    # ==================================================================
    # Use weights at midpoint times (average of adjacent time weights)
    local_weights_mid = (local_weights[:, 1:] + local_weights[:, :-1]) / 2.0

    # Weight each increment by its local fit quality
    weighted_inc = normalized_inc_sum * local_weights_mid

    # Sum over samples (weights already normalized per time)
    agg_increments = jnp.sum(weighted_inc, axis=0)  # (n_times-1,)

    # ==================================================================
    # Step 4: Accumulate across batches using EMA
    # ==================================================================
    if accumulated_increments is not None and accumulated_increments.shape == agg_increments.shape:
        # Exponential moving average: keeps memory of past detections
        agg_increments = accumulation_decay * accumulated_increments + (1.0 - accumulation_decay) * agg_increments

    # ==================================================================
    # Step 5: Temporal smoothing and peak detection
    # ==================================================================
    # Apply stronger temporal smoothing (5-point weighted average)
    kernel = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
    agg_smoothed = jnp.convolve(agg_increments, kernel, mode='same')

    # Compute weights: higher for larger increments above threshold
    threshold = jnp.quantile(agg_smoothed, detection_quantile)
    weights = jnp.maximum(agg_smoothed - threshold, 0.0)

    # Add baseline proportional to the signal (not fixed 0.01)
    # This ensures uniform fallback when no clear peaks exist
    baseline = 0.05 * jnp.mean(agg_smoothed)
    weights = weights + baseline
    weights = weights / jnp.sum(weights)

    # ==================================================================
    # Step 6: Compute new reference times as weighted quantiles
    # ==================================================================
    cum_weights = jnp.cumsum(weights)

    # Sample reference times as quantiles of the weighted distribution
    quantile_targets = jnp.linspace(0.0, 1.0, n_reference_times + 2)[1:-1]
    new_reference_times = jnp.interp(quantile_targets, cum_weights, times_mid)

    # Enforce minimum spacing to prevent collapse
    min_spacing = min_spacing_fraction * period
    new_reference_times = _enforce_min_spacing(new_reference_times, min_spacing)

    # Ensure bounded within [0, period]
    new_reference_times = jnp.clip(new_reference_times, time_sequence[0], time_sequence[-1])

    return jnp.sort(new_reference_times), agg_increments


def _enforce_min_spacing(times: jnp.ndarray, min_spacing: float) -> jnp.ndarray:
    """
    Enforce minimum spacing between consecutive times.

    Uses iterative adjustment to push times apart if too close.
    """
    times = jnp.sort(times)
    n = len(times)

    # Simple approach: if spacing is too small, interpolate to spread out
    diffs = jnp.diff(times)
    min_diff = jnp.min(diffs)

    # If minimum spacing violated, blend toward uniform
    uniform_times = jnp.linspace(times[0], times[-1], n)
    blend_factor = jnp.clip(min_spacing / (min_diff + 1e-8) - 1.0, 0.0, 1.0)

    return (1.0 - blend_factor) * times + blend_factor * uniform_times


def compute_jump_locations_from_observations(
    observations: jnp.ndarray,
    obs_times: jnp.ndarray,
    n_reference_times: int,
    alpha: float = 1.5,
    baseline_weight: float = 0.1,
) -> jnp.ndarray:
    """
    Identify times with large increments directly from observations.

    For heavy-tailed processes (stable Lévy), large increments ARE the signal.
    This function places reference times with density proportional to
    increment magnitude, so more reference times cluster near jumps.

    The concentration behavior adapts to alpha:
    - α → 1.0 (heavy tails): Few extreme jumps. We use higher power to
      concentrate reference times on these rare large jumps.
    - α → 2.0 (lighter tails): Many moderate jumps. We use lower power
      to spread reference times more evenly.

    Parameters
    ----------
    observations : array, shape (n_obs, state_dim)
        Observed data values
    obs_times : array, shape (n_obs,)
        Times at which observations were made
    n_reference_times : int
        Number of reference times to return
    alpha : float
        Stability parameter of the Lévy process (1 < alpha < 2).
        Controls concentration: smaller alpha → more concentration on large jumps.
    baseline_weight : float
        Minimum weight as fraction of mean weight. Ensures some reference times
        exist in quiet regions. Default 0.1 means quiet regions get at least
        10% of the average weight.

    Returns
    -------
    reference_times : array, shape (n_reference_times,)
        Reference time positions, clustered near large increments

    Notes
    -----
    The power transformation is: weight = increment^power where
    power = 2 - alpha. This gives:
    - α = 1.2 → power = 0.8 (strong concentration on large jumps)
    - α = 1.5 → power = 0.5 (moderate concentration)
    - α = 1.9 → power = 0.1 (nearly uniform, slight concentration)

    The baseline_weight prevents complete absence of reference times in
    quiet regions, which is important for the attention mechanism to
    have some representation everywhere.
    """
    # Compute absolute increments
    obs_increments = jnp.abs(jnp.diff(observations, axis=0))  # (n_obs-1, state_dim)
    obs_midpoints = (obs_times[1:] + obs_times[:-1]) / 2.0

    # Sum across dimensions
    total_increments = jnp.sum(obs_increments, axis=1)  # (n_obs-1,)

    # Power transformation: higher power = more concentration on large jumps
    # For α → 1: rare extreme jumps, concentrate heavily (power → 1)
    # For α → 2: many small jumps, spread out (power → 0)
    power = 2.0 - alpha
    power = jnp.clip(power, 0.1, 1.0)  # Keep in reasonable range

    weights = jnp.power(total_increments + 1e-8, power)

    # Add baseline to ensure coverage in quiet regions
    baseline = baseline_weight * jnp.mean(weights)
    weights = weights + baseline

    # Normalize
    weights = weights / jnp.sum(weights)

    # Place reference times as weighted quantiles
    cum_weights = jnp.cumsum(weights)
    quantile_targets = jnp.linspace(0.0, 1.0, n_reference_times + 2)[1:-1]
    reference_times = jnp.interp(quantile_targets, cum_weights, obs_midpoints)

    return jnp.sort(reference_times)


def compute_jump_locations_from_error(
    paths: jnp.ndarray,
    time_sequence: jnp.ndarray,
    observations: jnp.ndarray,
    obs_times: jnp.ndarray,
    n_reference_times: int,
    accumulated_signal: jnp.ndarray = None,
    accumulation_decay: float = 0.9
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find times where posterior paths have highest error relative to observations.
    Place reference_times at these high-error regions.

    This is a minimal, interpretable alternative to compute_jump_locations_from_paths.

    Logic:
    1. For each time point, compute squared error between paths and nearest observation
    2. Average over samples to get error profile over time
    3. Accumulate error signal across training steps (EMA)
    4. Place reference_times at high-error regions (weighted quantiles)

    The key insight: smooth posteriors have highest error at jump times because
    they "split the difference" between pre-jump and post-jump states.

    Parameters
    ----------
    paths : array, shape (n_samples, n_times, state_dim)
        Simulated posterior paths (reused from loss computation)
    time_sequence : array, shape (n_times,)
        Time points corresponding to path indices
    observations : array, shape (n_obs, state_dim)
        Observed data values
    obs_times : array, shape (n_obs,)
        Times at which observations were made
    n_reference_times : int
        Number of reference times to return
    accumulated_signal : array or None
        Previous accumulated error signal (for EMA across batches)
    accumulation_decay : float
        Decay factor for exponential moving average (0.9 = 90% old, 10% new)
        Higher values = longer memory, more stable but slower to adapt

    Returns
    -------
    new_reference_times : array, shape (n_reference_times,)
        Reference time positions, concentrated at high-error regions
    new_accumulated_signal : array, shape (n_times,)
        Updated accumulated error signal for next call
    """
    n_samples, n_times, state_dim = paths.shape

    # ======================================================================
    # Step 1: For each time point, find nearest observation
    # ======================================================================
    obs_indices = jnp.searchsorted(obs_times, time_sequence)
    obs_indices = jnp.clip(obs_indices, 0, len(obs_times) - 1)

    # Also check previous observation (whichever is closer)
    obs_indices_prev = jnp.clip(obs_indices - 1, 0, len(obs_times) - 1)
    dist_to_current = jnp.abs(time_sequence - obs_times[obs_indices])
    dist_to_prev = jnp.abs(time_sequence - obs_times[obs_indices_prev])
    closest_obs_idx = jnp.where(dist_to_current < dist_to_prev, obs_indices, obs_indices_prev)

    # Get observations at closest times: (n_times, state_dim)
    closest_obs = observations[closest_obs_idx]

    # ======================================================================
    # Step 2: Compute error at each time (averaged over samples)
    # ======================================================================
    # Error = squared distance between path and nearest observation
    # Shape: (n_samples, n_times)
    per_sample_errors = jnp.sum((paths - closest_obs[None, :, :]) ** 2, axis=2)

    # Average over samples to get error profile: (n_times,)
    errors = jnp.mean(per_sample_errors, axis=0)

    # ======================================================================
    # Step 3: Accumulate over training steps (EMA)
    # ======================================================================
    if accumulated_signal is not None and accumulated_signal.shape == errors.shape:
        accumulated_signal = accumulation_decay * accumulated_signal + (1.0 - accumulation_decay) * errors
    else:
        accumulated_signal = errors

    # ======================================================================
    # Step 4: Convert to weights (high error = high weight)
    # ======================================================================
    # Normalize to get probability distribution
    weights = accumulated_signal / (jnp.sum(accumulated_signal) + 1e-8)

    # ======================================================================
    # Step 5: Place reference times at weighted quantiles
    # ======================================================================
    cum_weights = jnp.cumsum(weights)

    # Generate quantile targets (exclude 0 and 1 to avoid boundary issues)
    quantile_targets = jnp.linspace(0.0, 1.0, n_reference_times + 2)[1:-1]

    # Interpolate to find times corresponding to these quantiles
    new_reference_times = jnp.interp(quantile_targets, cum_weights, time_sequence)

    return jnp.sort(new_reference_times), accumulated_signal
