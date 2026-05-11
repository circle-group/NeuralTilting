# Third-party imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
from typing import Tuple, Dict, Any

# Library-specific imports
from utils.training_utils import build_simulation_grid


# ============================================================================
# Helper functions for loss computation
# ============================================================================

def _compute_kl_term(kl_total: jnp.ndarray, quantile: float = 0.99, clip_value: float = 1e7) -> jnp.ndarray:
    """
    Compute KL divergence term with path-level Winsorization.

    This provides robust estimation for heavy-tailed KL distributions by:
    1. Winsorizing top outliers (clamping to quantile)
    2. Applying tanh clipping for numerical stability

    Parameters
    ----------
    kl_total : array, shape (n_samples,)
        Per-path KL divergence values
    quantile : float
        Upper quantile for Winsorization (default: 0.99)
    clip_value : float
        Soft clipping value for tanh (default: 1e7)

    Returns
    -------
    kl_term : float
        Robust mean KL divergence
    """
    # Winsorize: clamp extreme outliers to maintain gradient flow
    kl_upper_quantile = jnp.quantile(kl_total, q=quantile)
    kl_winsorized = jnp.clip(kl_total, a_min=None, a_max=kl_upper_quantile)

    # Tanh clipping for numerical stability (prevent inf/nan)
    kl_term_per_sample = clip_value * jnp.tanh(kl_winsorized / clip_value)

    return kl_term_per_sample.mean()


def _compute_likelihood_term(
    observations: jnp.ndarray,
    paths_at_obs: jnp.ndarray,
    obs_std: float,
    quantile: float = 0.99
) -> jnp.ndarray:
    """
    Compute likelihood term with time-marginal quantile filtering.

    This allows paths to contribute partial information by filtering outlier
    jumps at specific times while keeping good path segments.

    Parameters
    ----------
    observations : array, shape (n_obs, state_dim)
        Observed values at observation times
    paths_at_obs : array, shape (n_samples, n_obs, state_dim)
        Simulated paths evaluated at observation times
    obs_std : float
        Observation noise standard deviation
    quantile : float
        Upper quantile for time-marginal filtering (default: 0.99)

    Returns
    -------
    likelihood_term : float
        Robust mean negative log-likelihood
    """
    # Compute squared errors at each (time, state) location
    # Shape: (n_samples, n_obs, state_dim)
    squared_errors = 0.5 * jnp.power((observations - paths_at_obs) / obs_std, 2)

    # Time-marginal quantile filtering: compute quantile at each time point
    # across all sample paths. Shape: (n_obs, state_dim)
    upper_quantile = jnp.nanquantile(squared_errors, q=quantile, axis=0)

    # Winsorize: clamp extreme values to time-marginal threshold
    # This maintains gradient flow and unbiased estimation
    # Shape: (n_samples, n_obs, state_dim)
    winsorized_errors = jnp.clip(
        squared_errors,
        a_min=None,
        a_max=upper_quantile[None, :, :]
    )

    # Aggregate: sum over time, mean over state_dim and sample paths
    return jnp.sum(winsorized_errors, axis=1).mean()


def _compute_likelihood_term_no_clipping(
    observations: jnp.ndarray,
    paths_at_obs: jnp.ndarray,
    obs_std: float
) -> jnp.ndarray:
    """
    Compute likelihood term without any clipping.

    This provides unbiased gradient signals from all samples, including
    outliers. Use when variance is controlled via other mechanisms (e.g.,
    simulation-level clipping).

    Parameters
    ----------
    observations : array, shape (n_obs, state_dim)
        Observed values at observation times
    paths_at_obs : array, shape (n_samples, n_obs, state_dim)
        Simulated paths evaluated at observation times
    obs_std : float
        Observation noise standard deviation

    Returns
    -------
    likelihood_term : float
        Mean negative log-likelihood (unclipped)
    """
    # Compute squared errors at each (time, state) location
    # Shape: (n_samples, n_obs, state_dim)
    squared_errors = 0.5 * jnp.power((observations - paths_at_obs) / obs_std, 2)

    # Aggregate: sum over time, mean over state_dim and sample paths
    # No clipping - all samples contribute equally
    return jnp.sum(squared_errors, axis=1).mean()


def _compute_phi_regularization(model, regularisation_strength: float) -> jnp.ndarray:
    """
    Compute L2 regularization on phi (tilting potential) network.

    Handles both attention-based and standard architectures.

    Parameters
    ----------
    model : TiltedStableDrivenSDE
        The SDE model
    regularisation_strength : float
        L2 regularization strength

    Returns
    -------
    phi_regularization : float
        L2 penalty on phi network parameters
    """
    if hasattr(model.phi, 'temporal_attention') and model.phi.temporal_attention is not None:
        # Attention-based architecture: QuadraticNeuralPotential with TemporalAttentionEncoding
        attn = model.phi.temporal_attention

        # Attention sharpness: regularize log-space parameters
        sharpness_reg = regularisation_strength * jnp.sum(jnp.square(attn.attention_sharpness))

        # Reference embeddings: standard L2 to prevent unbounded growth
        embeddings_reg = regularisation_strength * jnp.sum(jnp.square(attn.reference_embeddings))

        attention_regularization = sharpness_reg + embeddings_reg

        # MLP parameters (A and B networks)
        mlp_A_params = eqx.filter(model.phi.mlp_A, eqx.is_inexact_array)
        mlp_B_params = eqx.filter(model.phi.mlp_B, eqx.is_inexact_array)
        mlp_l2 = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(mlp_A_params))
        mlp_l2 += sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(mlp_B_params))

        return regularisation_strength * mlp_l2 + attention_regularization
    else:
        # Fallback: regularize all phi parameters uniformly
        phi_params = eqx.filter(model.phi, eqx.is_inexact_array)
        phi_l2 = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(phi_params))
        return regularisation_strength * phi_l2


def _compute_drift_regularization(
    model,
    time_sequence: jnp.ndarray,
    drift_regularization: float
) -> jnp.ndarray:
    """
    Compute L2 regularization on drift network parameters.

    Scaled by time duration to represent integral over [0, T].

    Parameters
    ----------
    model : TiltedStableDrivenSDE
        The SDE model
    time_sequence : array
        Time grid for simulation
    drift_regularization : float
        L2 regularization strength

    Returns
    -------
    drift_regularization_term : float
        L2 penalty on drift parameters, scaled by duration
    """
    if not model.trainable_drift:
        return 0.0

    # Calculate total duration
    duration = time_sequence[-1] - time_sequence[0]

    drift = model.drift

    if hasattr(drift, 'theta') and hasattr(drift, 'mu'):
        # OU drift (OUDiagonalLinearFunction): L2 with stationarity term
        theta = drift.theta
        mu = drift.mu
        drift_l2_sq = (
            jnp.sum(jnp.square(theta))
            + jnp.sum(jnp.square(mu))
            + jnp.sum(theta * jnp.square(mu))
        )
    elif hasattr(drift, 'theta1') and hasattr(drift, 'theta2'):
        # Double well drift (DoubleWellDriftFunction): plain L2 on both parameters
        drift_l2_sq = (
            jnp.sum(jnp.square(drift.theta1))
            + jnp.sum(jnp.square(drift.theta2))
        )
    else:
        # Generic fallback: L2 on all trainable parameters
        drift_params = eqx.filter(drift, eqx.is_inexact_array)
        drift_l2_sq = sum(
            jnp.sum(jnp.square(p))
            for p in jax.tree_util.tree_leaves(drift_params)
        )

    # Scale by duration to represent integral over [0, T]
    return drift_regularization * duration * drift_l2_sq


def _compute_coefficient_regularization(
    model,
    time_sequence: jnp.ndarray,
    coeff_A_regularization: float,
    coeff_B_regularization: float
) -> jnp.ndarray:
    """
    Compute L2 regularization on A(t) and B(t) coefficient values.

    Parameters
    ----------
    model : TiltedStableDrivenSDE
        The SDE model
    time_sequence : array
        Time grid for evaluating coefficients
    coeff_A_regularization : float
        L2 regularization strength for A(t)
    coeff_B_regularization : float
        L2 regularization strength for B(t)

    Returns
    -------
    coeff_regularization : float
        L2 penalty on coefficient magnitudes
    """
    # Evaluate coefficients at all time points
    A_values, B_values = jax.vmap(model.phi.get_coefficients)(time_sequence)

    # L2 penalty on coefficient magnitudes
    A_reg = coeff_A_regularization * jnp.mean(jnp.square(A_values))
    B_reg = coeff_B_regularization * jnp.mean(jnp.square(B_values))

    return A_reg + B_reg


# ============================================================================
# Main loss function
# ============================================================================

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def loss_and_grad_for_tilted_stable_sde(
    model,
    n_loss_samples,
    state_init,
    obs_times,
    observations,
    obs_std,
    key,
    n_latent_steps,
    T=None,
    regularisation_strength=1.0,
    drift_regularization=0.01,
    coeff_A_regularization=0.0,
    coeff_B_regularization=0.0):
    """
    Loss function for the tilted stable SDE model.

    Parameters
    ----------
    model : TiltedStableDrivenSDE
        The SDE model to train
    n_loss_samples : int
        Number of posterior path samples for Monte Carlo estimation
    state_init : array
        Initial state of the SDE, shape (state_dim,)
    obs_times : array
        Times at which observations are made, shape (n_obs,)
    observations : array
        Observed values at obs_times, shape (n_obs, state_dim)
    obs_std : float or array
        Observation noise standard deviation
    key : PRNGKey
        Random key for sampling
    n_latent_steps : int, optional
        Number of time steps for latent SDE simulation grid (default: 100)
    T : float, optional
        Final time. If None, inferred from max(obs_times)
    regularisation_strength : float, optional
        L2 regularization strength for phi network (default: 1.0)
    drift_regularization : float, optional
        L2 regularization strength for drift network (default: 0.01)
    coeff_A_regularization : float, optional
        L2 regularization strength for A(t) coefficient values (default: 0.0)
    coeff_B_regularization : float, optional
        L2 regularization strength for B(t) coefficient values (default: 0.0)

    Returns
    -------
    total_loss : float
        Combined loss value (KL + likelihood + regularization)
    aux_data : dict
        Auxiliary data containing loss components for monitoring:
        - 'kl_term': KL divergence component
        - 'likelihood_term': Negative log-likelihood component
        - 'phi_regularization': L2 regularization on tilting network
        - 'drift_regularization': L2 regularization on drift
        - 'coeff_regularization': L2 regularization on A(t) and B(t) values
    """

    # Infer final time if not provided
    if T is None:
        T = jnp.max(obs_times)

    # Build unified time grid: merge latent simulation grid with observation times
    time_sequence = build_simulation_grid(
        T_start=0.0,
        T_end=T,
        n_steps=n_latent_steps,
        obs_times=obs_times
    )

    # Find indices where observation times appear in unified grid
    obs_indices = jnp.searchsorted(time_sequence, obs_times)

    # Simulate paths from variational posterior (with KL contributions)
    all_keys = random.split(key, num=n_loss_samples + 1)
    key = all_keys[0]
    keys = all_keys[1:]
    random_posterior_paths, loss_estimates, kl_total = vmap(
        model.simulate_posterior_and_loss, in_axes=[None, None, 0]
    )(state_init, time_sequence, keys)

    # Extract paths only at observation times
    # random_posterior_paths shape: (n_loss_samples, len(time_sequence), state_dim)
    # obs_indices shape: (n_obs,)
    # paths_at_obs shape: (n_loss_samples, n_obs, state_dim)
    paths_at_obs = random_posterior_paths[:, obs_indices, :]

    # Compute loss components using modular helper functions
    kl_term = _compute_kl_term(kl_total, quantile=0.99, clip_value=1e7)
    likelihood_term = _compute_likelihood_term_no_clipping(observations, paths_at_obs, obs_std)
    phi_regularization = _compute_phi_regularization(model, regularisation_strength)
    drift_regularization_term = _compute_drift_regularization(model, time_sequence, drift_regularization)
    coeff_regularization = _compute_coefficient_regularization(
        model, time_sequence, coeff_A_regularization, coeff_B_regularization
    )

    # Total regularization
    total_regularization = phi_regularization + drift_regularization_term + coeff_regularization

    # Total loss
    total_loss = kl_term + likelihood_term + total_regularization

    # Package auxiliary data for monitoring
    aux_data = {
        'kl_term': kl_term,
        'likelihood_term': likelihood_term,
        'phi_regularization': phi_regularization,
        'drift_regularization': drift_regularization_term,
        'coeff_regularization': coeff_regularization,
        # Include paths and time sequence for EM updates
        'paths': random_posterior_paths,  # (n_loss_samples, len(time_sequence), state_dim)
        'time_sequence': time_sequence,   # (len(time_sequence),)
    }

    return total_loss, aux_data


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_and_grad_for_gaussian_sde(
    model,
    n_loss_samples,
    state_init,
    obs_times,
    observations,
    obs_std,
    key,
    n_latent_steps,
    T=None,
    regularisation_strength=0.1,
    drift_regularization=0.01,
    cholesky_regularisation=0.0):
    """
    Loss function for the Gaussian SDE model with drift control.

    Parameters
    ----------
    model : GaussianDrivenSDE
        The SDE model to train
    n_loss_samples : int
        Number of posterior path samples for Monte Carlo estimation
    state_init : array
        Initial state of the SDE, shape (state_dim,)
    obs_times : array
        Times at which observations are made, shape (n_obs,)
    observations : array
        Observed values at obs_times, shape (n_obs, state_dim)
    obs_std : float or array
        Observation noise standard deviation
    key : PRNGKey
        Random key for sampling
    n_latent_steps : int
        Number of time steps for latent SDE simulation grid
    T : float, optional
        Final time. If None, inferred from max(obs_times)
    regularisation_strength : float, optional
        L2 regularization strength for control network (default: 0.1)
    drift_regularization : float, optional
        L2 regularization strength for drift network (default: 0.01)
    cholesky_regularisation : float, optional
        L2 regularization strength for CholeskyDiffusion MLP weights (default: 0.0).
        Only applied when model.diffusion is a CholeskyDiffusion (has an 'mlp' attribute).
        Keeps L(t,X) close to its identity initialisation, preventing covariance blow-up.

    Returns
    -------
    total_loss : float
        Combined loss value (KL + likelihood + regularization)
    """

    # Infer final time if not provided
    if T is None:
        T = jnp.max(obs_times)

    # Build unified time grid: merge latent simulation grid with observation times
    time_sequence = build_simulation_grid(
        T_start=0.0,
        T_end=T,
        n_steps=n_latent_steps,
        obs_times=obs_times
    )

    # Find indices where observation times appear in unified grid
    obs_indices = jnp.searchsorted(time_sequence, obs_times)

    # Simulate paths from variational posterior (with KL contributions)
    keys = random.split(key, num=n_loss_samples)
    random_posterior_paths, loss_estimates, kl_total = vmap(
        model.simulate_posterior_and_loss, in_axes=[None, None, 0]
    )(state_init, time_sequence, keys)

    # Extract paths only at observation times
    # random_posterior_paths shape: (n_loss_samples, len(time_sequence), state_dim)
    # obs_indices shape: (n_obs,)
    # paths_at_obs shape: (n_loss_samples, n_obs, state_dim)
    paths_at_obs = random_posterior_paths[:, obs_indices, :]

    # KL divergence term (mean over samples with soft clip on each path)
    clip_value_kl = 1e7
    kl_term_per_sample = clip_value_kl * jnp.tanh(kl_total / clip_value_kl) # soft clip each sample using tanh
    kl_term = kl_term_per_sample.mean()

    # Likelihood term per sample (now using only observation time points)
    # observations shape: (n_obs, state_dim)
    # paths_at_obs shape: (n_loss_samples, n_obs, state_dim)
    per_sample_likelihood = jnp.nansum(
        0.5 * jnp.power((observations - paths_at_obs)/obs_std, 2),
        axis=1
    ).mean(axis=1)

    # Soft clip per sample (tanh) to remove paths with bad jumps
    clip_value = 1e7
    per_sample_likelihood = clip_value * jnp.tanh(per_sample_likelihood / clip_value)

    # Average over samples
    likelihood_term = per_sample_likelihood.mean()

    # L2 regularization on control NN parameters
    control_params = eqx.filter(model.control, eqx.is_inexact_array)
    control_l2_reg = sum(jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(control_params))
    control_regularization = regularisation_strength * control_l2_reg

    if model.trainable_drift:
        drift_regularization_term = _compute_drift_regularization(
            model, time_sequence, drift_regularization
        )
    else:
        drift_regularization_term = 0.0

    # L2 regularization on CholeskyDiffusion MLP weights (if present)
    if cholesky_regularisation > 0.0 and hasattr(model.diffusion, 'mlp'):
        chol_params = eqx.filter(model.diffusion.mlp, eqx.is_inexact_array)
        chol_l2_reg = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(chol_params))
        cholesky_regularisation_term = cholesky_regularisation * chol_l2_reg
    else:
        cholesky_regularisation_term = 0.0

    # Total regularization
    total_regularization = control_regularization + drift_regularization_term + cholesky_regularisation_term

    # Total loss
    total_loss = kl_term + likelihood_term + total_regularization

    return total_loss