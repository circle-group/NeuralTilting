"""
Loss functions and evaluation metrics for SDE models.

This module provides modular, composable metric functions for evaluating
posterior inference quality and drift parameter learning.

All metric functions are pure, JIT-compatible, and follow a consistent signature.
"""

from typing import Dict, Union, Callable, Optional
from functools import partial
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np


# =============================================================================
# POSTERIOR PATH ERROR METRICS
# =============================================================================

def mean_squared_error(
    posterior_mean: Array,
    ground_truth: Array,
    config: Optional[Dict] = None
) -> float:
    """
    Compute mean squared error between posterior mean and ground truth.

    Parameters
    ----------
    posterior_mean : Array
        Posterior mean path, shape (n_times, state_dim)
    ground_truth : Array
        Ground truth (observations or latent path), shape (n_times, state_dim)
    config : Dict, optional
        Additional configuration (not used for MSE)

    Returns
    -------
    mse : float
        Mean squared error across all time points and dimensions
    """
    return float(jnp.mean((posterior_mean - ground_truth) ** 2))


def mean_absolute_error(
    posterior_mean: Array,
    ground_truth: Array,
    config: Optional[Dict] = None
) -> float:
    """
    Compute mean absolute error between posterior mean and ground truth.

    Parameters
    ----------
    posterior_mean : Array
        Posterior mean path, shape (n_times, state_dim)
    ground_truth : Array
        Ground truth (observations or latent path), shape (n_times, state_dim)
    config : Dict, optional
        Additional configuration (not used for MAE)

    Returns
    -------
    mae : float
        Mean absolute error across all time points and dimensions
    """
    return float(jnp.mean(jnp.abs(posterior_mean - ground_truth)))


def root_mean_squared_error(
    posterior_mean: Array,
    ground_truth: Array,
    config: Optional[Dict] = None
) -> float:
    """
    Compute root mean squared error between posterior mean and ground truth.

    Parameters
    ----------
    posterior_mean : Array
        Posterior mean path, shape (n_times, state_dim)
    ground_truth : Array
        Ground truth (observations or latent path), shape (n_times, state_dim)
    config : Dict, optional
        Additional configuration (not used for RMSE)

    Returns
    -------
    rmse : float
        Root mean squared error across all time points and dimensions
    """
    return float(jnp.sqrt(jnp.mean((posterior_mean - ground_truth) ** 2)))


def time_varying_squared_error(
    posterior_mean: Array,
    ground_truth: Array,
    config: Optional[Dict] = None
) -> Array:
    """
    Compute per-timestep squared error (useful for plotting error evolution).

    Parameters
    ----------
    posterior_mean : Array
        Posterior mean path, shape (n_times, state_dim)
    ground_truth : Array
        Ground truth (observations or latent path), shape (n_times, state_dim)
    config : Dict, optional
        Additional configuration (not used)

    Returns
    -------
    time_errors : Array
        Squared error at each time point, shape (n_times,)
        Averaged over state dimensions
    """
    # Average over state dimensions
    return jnp.mean((posterior_mean - ground_truth) ** 2, axis=1)


def time_varying_absolute_error(
    posterior_mean: Array,
    ground_truth: Array,
    config: Optional[Dict] = None
) -> Array:
    """
    Compute per-timestep absolute error (useful for plotting error evolution).

    Parameters
    ----------
    posterior_mean : Array
        Posterior mean path, shape (n_times, state_dim)
    ground_truth : Array
        Ground truth (observations or latent path), shape (n_times, state_dim)
    config : Dict, optional
        Additional configuration (not used)

    Returns
    -------
    time_errors : Array
        Absolute error at each time point, shape (n_times,)
        Averaged over state dimensions
    """
    # Average over state dimensions
    return jnp.mean(jnp.abs(posterior_mean - ground_truth), axis=1)


# =============================================================================
# PROBABILISTIC METRICS
# =============================================================================

def negative_log_likelihood(
    posterior_samples: Array,
    observations: Array,
    obs_std: float,
    config: Optional[Dict] = None
) -> float:
    """
    Compute average negative log-likelihood of observations under posterior.

    Assumes Gaussian observation model: y ~ N(x, obs_std^2)

    Parameters
    ----------
    posterior_samples : Array
        Posterior samples, shape (n_samples, n_times, state_dim)
    observations : Array
        Observed data, shape (n_times, state_dim)
    obs_std : float
        Observation noise standard deviation
    config : Dict, optional
        Additional configuration (not used)

    Returns
    -------
    nll : float
        Average negative log-likelihood per observation
    """
    # Compute posterior mean
    posterior_mean = jnp.mean(posterior_samples, axis=0)  # (n_times, state_dim)

    # Gaussian NLL: 0.5 * log(2π σ²) + 0.5 * (y - μ)² / σ²
    n_times, state_dim = observations.shape
    log_2pi = jnp.log(2 * jnp.pi)

    nll = 0.5 * log_2pi + jnp.log(obs_std)
    nll = nll + 0.5 * ((observations - posterior_mean) / obs_std) ** 2

    # Average over all observations
    return float(jnp.mean(nll))


def prediction_interval_coverage(
    posterior_samples: Array,
    ground_truth: Array,
    confidence_level: float = 0.95,
    config: Optional[Dict] = None
) -> float:
    """
    Compute prediction interval coverage rate.

    Measures the fraction of ground truth points that fall within the
    posterior credible interval at the specified confidence level.

    Parameters
    ----------
    posterior_samples : Array
        Posterior samples, shape (n_samples, n_times, state_dim)
    ground_truth : Array
        Ground truth values, shape (n_times, state_dim)
    confidence_level : float
        Confidence level for interval (default: 0.95 for 95% interval)
    config : Dict, optional
        Additional configuration

    Returns
    -------
    coverage : float
        Fraction of ground truth points within posterior interval [0, 1]
    """
    # Compute quantiles
    alpha = (1 - confidence_level) / 2
    lower_quantile = jnp.quantile(posterior_samples, alpha, axis=0)
    upper_quantile = jnp.quantile(posterior_samples, 1 - alpha, axis=0)

    # Check coverage
    within_interval = (ground_truth >= lower_quantile) & (ground_truth <= upper_quantile)

    # Average over all points
    return float(jnp.mean(within_interval))


def posterior_width(
    posterior_samples: Array,
    confidence_level: float = 0.95,
    config: Optional[Dict] = None
) -> float:
    """
    Compute average width of posterior credible intervals.

    Measures posterior uncertainty - narrower intervals indicate more certainty.

    Parameters
    ----------
    posterior_samples : Array
        Posterior samples, shape (n_samples, n_times, state_dim)
    confidence_level : float
        Confidence level for interval (default: 0.95)
    config : Dict, optional
        Additional configuration

    Returns
    -------
    avg_width : float
        Average interval width across all time points and dimensions
    """
    alpha = (1 - confidence_level) / 2
    lower_quantile = jnp.quantile(posterior_samples, alpha, axis=0)
    upper_quantile = jnp.quantile(posterior_samples, 1 - alpha, axis=0)

    widths = upper_quantile - lower_quantile
    return float(jnp.mean(widths))


# =============================================================================
# CONTINUOUS RANKED PROBABILITY SCORE
# =============================================================================

def continuous_ranked_probability_score(
    posterior_samples: Array,
    ground_truth: Array,
    config: Optional[Dict] = None
) -> float:
    """
    Compute the Continuous Ranked Probability Score (CRPS).

    CRPS is the standard proper scoring rule for probabilistic forecasts.
    It reduces to MAE when the forecast is deterministic, and rewards
    both accuracy and sharpness. Well-suited for heavy-tailed processes
    where MSE is dominated by rare large errors.

    Uses the energy form:
        CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from the forecast distribution F.

    Parameters
    ----------
    posterior_samples : Array
        Posterior samples, shape (n_samples, n_times, state_dim)
    ground_truth : Array
        Ground truth values, shape (n_times, state_dim)
    config : Dict, optional
        Additional configuration (not used)

    Returns
    -------
    crps : float
        Mean CRPS across all time points and dimensions
    """
    # E|X - y|: mean absolute error of each sample against ground truth
    # ground_truth: (n_times, state_dim) -> expand to (1, n_times, state_dim)
    term1 = jnp.mean(jnp.abs(posterior_samples - ground_truth[None, :, :]))

    # E|X - X'|: expected absolute difference between pairs of samples
    # Efficient computation: E|X - X'| = 2 * sum_i sum_{j>i} |x_i - x_j| / n^2
    # Equivalent via sorting trick: sum over all pairs = mean over broadcast
    n_samples = posterior_samples.shape[0]
    # (n_samples, 1, n_times, state_dim) - (1, n_samples, n_times, state_dim)
    pairwise = jnp.abs(
        posterior_samples[:, None, :, :] - posterior_samples[None, :, :, :]
    )  # shape: (n_samples, n_samples, n_times, state_dim)
    term2 = 0.5 * jnp.mean(pairwise)

    return float(term1 - term2)


# =============================================================================
# DRIFT LEARNING METRICS
# =============================================================================

def drift_parameter_error(
    trained_drift_params: Dict,
    true_drift_params: Dict,
    config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compare learned vs true drift parameters.

    For OU drift: compares theta (mean-reversion rate) and mu (long-term mean)
    For linear drift: compares weight and bias

    Parameters
    ----------
    trained_drift_params : Dict
        Learned drift parameters
        For OU: {'theta': Array, 'mu': Array}
        For linear: {'weight': Array, 'bias': Array}
    true_drift_params : Dict
        True drift parameters (same structure as trained_drift_params)
    config : Dict, optional
        Additional configuration

    Returns
    -------
    errors : Dict[str, float]
        Dictionary with parameter-wise errors and learned/true values
    """
    errors = {}

    # Check drift type (OU or linear)
    if 'theta' in trained_drift_params:
        # OU drift
        trained_theta = jnp.atleast_1d(trained_drift_params['theta'])
        true_theta = jnp.atleast_1d(true_drift_params['theta'])
        trained_mu = jnp.atleast_1d(trained_drift_params['mu'])
        true_mu = jnp.atleast_1d(true_drift_params['mu'])

        errors['theta_error'] = float(jnp.mean(jnp.abs(trained_theta - true_theta)))
        errors['mu_error'] = float(jnp.mean(jnp.abs(trained_mu - true_mu)))
        errors['trained_theta'] = float(jnp.mean(trained_theta))
        errors['true_theta'] = float(jnp.mean(true_theta))
        errors['trained_mu'] = float(jnp.mean(trained_mu))
        errors['true_mu'] = float(jnp.mean(true_mu))

    elif 'theta1' in trained_drift_params:
        # Double well drift: f(x) = theta1*x - theta2*x^3
        trained_theta1 = jnp.atleast_1d(trained_drift_params['theta1'])
        true_theta1 = jnp.atleast_1d(true_drift_params['theta1'])
        trained_theta2 = jnp.atleast_1d(trained_drift_params['theta2'])
        true_theta2 = jnp.atleast_1d(true_drift_params['theta2'])

        errors['theta1_error'] = float(jnp.mean(jnp.abs(trained_theta1 - true_theta1)))
        errors['theta2_error'] = float(jnp.mean(jnp.abs(trained_theta2 - true_theta2)))
        errors['trained_theta1'] = float(jnp.mean(trained_theta1))
        errors['true_theta1'] = float(jnp.mean(true_theta1))
        errors['trained_theta2'] = float(jnp.mean(trained_theta2))
        errors['true_theta2'] = float(jnp.mean(true_theta2))

    elif 'weight' in trained_drift_params:
        # Linear drift
        trained_weight = jnp.atleast_1d(trained_drift_params['weight'])
        true_weight = jnp.atleast_1d(true_drift_params['weight'])
        trained_bias = jnp.atleast_1d(trained_drift_params['bias'])
        true_bias = jnp.atleast_1d(true_drift_params['bias'])

        errors['weight_error'] = float(jnp.mean(jnp.abs(trained_weight - true_weight)))
        errors['bias_error'] = float(jnp.mean(jnp.abs(trained_bias - true_bias)))
        errors['trained_weight'] = float(jnp.mean(trained_weight))
        errors['true_weight'] = float(jnp.mean(true_weight))
        errors['trained_bias'] = float(jnp.mean(trained_bias))
        errors['true_bias'] = float(jnp.mean(true_bias))

    return errors


# =============================================================================
# JUMP-CONDITIONED METRICS HELPER
# =============================================================================

def compute_jump_mask(observations: np.ndarray, percentile: float) -> np.ndarray:
    """
    Compute a boolean mask identifying time steps where a large jump occurred.

    A "jump" at time t is defined by |y_t - y_{t-1}| exceeding the given
    percentile threshold of all absolute increments in the sequence.
    The threshold is computed from the mean absolute increment across state
    dimensions, so the mask is a single boolean per time step regardless of
    state dimensionality.

    Parameters
    ----------
    observations : np.ndarray
        Observation sequence, shape (n_times, state_dim)
    percentile : float
        Percentile threshold (e.g. 90, 95, 97.5, 99)

    Returns
    -------
    mask : np.ndarray
        Boolean array of shape (n_times,). True at time steps where the
        increment exceeds the threshold. The first time step is always False
        (no predecessor).
    """
    obs_2d = np.atleast_2d(observations).T if observations.ndim == 1 else observations
    increments = np.abs(np.diff(obs_2d, axis=0))             # (n_times-1, state_dim)
    increment_magnitudes = increments.mean(axis=1)           # (n_times-1,)
    threshold = np.percentile(increment_magnitudes, percentile)
    mask = np.concatenate([[False], increment_magnitudes > threshold])
    return mask


# =============================================================================
# METRIC REGISTRY
# =============================================================================

# Registry of all available metrics
METRIC_REGISTRY: Dict[str, Callable] = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'rmse': root_mean_squared_error,
    'crps': continuous_ranked_probability_score,
    'time_se': time_varying_squared_error,
    'time_ae': time_varying_absolute_error,
    'nll': negative_log_likelihood,
    'coverage_95': partial(prediction_interval_coverage, confidence_level=0.95),
    'coverage_90': partial(prediction_interval_coverage, confidence_level=0.90),
    'coverage_99': partial(prediction_interval_coverage, confidence_level=0.99),
    'posterior_width_95': partial(posterior_width, confidence_level=0.95),
    'drift_error': drift_parameter_error,
}


def get_metric_function(metric_name: str) -> Callable:
    """
    Get metric function by name from registry.

    Parameters
    ----------
    metric_name : str
        Name of the metric (e.g., 'mse', 'mae', 'coverage_95')

    Returns
    -------
    metric_fn : Callable
        Metric function

    Raises
    ------
    ValueError
        If metric name is not registered
    """
    if metric_name not in METRIC_REGISTRY:
        available = ', '.join(METRIC_REGISTRY.keys())
        raise ValueError(
            f"Unknown metric: '{metric_name}'. "
            f"Available metrics: {available}"
        )
    return METRIC_REGISTRY[metric_name]


def list_available_metrics() -> list:
    """
    Get list of all available metric names.

    Returns
    -------
    metric_names : list
        List of registered metric names
    """
    return list(METRIC_REGISTRY.keys())
