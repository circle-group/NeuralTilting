"""Proper scoring rules and coverage metrics for probabilistic forecasts."""

import numpy as np

# Log-spaced coverage levels: complement (1-q) log-spaced from 0.5 to 0.001
# giving dense coverage near the tails where calibration matters most.
_complements = np.logspace(np.log10(0.5), np.log10(0.001), 20)
DENSE_COVERAGE_LEVELS = sorted(1.0 - _complements)


# =============================================================================
# Path importance weights
# =============================================================================

def compute_path_weights(forecast_samples: np.ndarray, M0: float = 20.0) -> np.ndarray:
    """Soft importance weights that downweight numerically diverged paths.

    The prior simulation can occasionally produce paths with astronomically
    large values when a stable jump sample hits a near-1 uniform draw
    (floating-point underflow of (1-u) → exact 0 → infinite jump size).
    These are numerical artefacts, not genuine draws from the theoretical
    α-stable distribution.

    We target the smoothly-truncated prior π(x) ∝ p(x) · φ(x) using
    samples from p(x) as the proposal.  The IS weight is φ(x)/Z where:

        φ(xᵢ) = 1 / (1 + (Mᵢ / M0)²)
        Mᵢ    = max over all forecast steps and dimensions of |xᵢ(t, d)|

    Weights are normalised to sum to 1.

    Parameters
    ----------
    forecast_samples : np.ndarray
        Shape (K, H, D).
    M0 : float
        Soft truncation threshold in normalised units.  A path that stays
        within ±M0 gets weight ≈ 1; one at 10·M0 gets weight ≈ 0.01.
        Default 20.0 (20 standard deviations — far beyond any real return).

    Returns
    -------
    np.ndarray
        Shape (K,).  Normalised weights summing to 1.
    """
    # Replace nan/inf with a very large finite value before computing path norms.
    # np.max propagates nan, so a single nan in any path would make M=nan for
    # that path, then phi=nan, then phi.sum()=nan, poisoning all weights.
    # Replacing with 1e10 >> M0=20 gives phi≈0 for bad paths (correct behaviour).
    clean = np.nan_to_num(forecast_samples, nan=1e10, posinf=1e10, neginf=1e10)
    M = np.max(np.abs(clean), axis=(1, 2))  # (K,)
    phi = 1.0 / (1.0 + (M / M0) ** 2)
    return phi / phi.sum()


def _weighted_quantile(x: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile of 1-D array x at probability q ∈ [0, 1].

    Uses the midpoint-interpolation convention so that the result matches
    np.percentile(..., interpolation='midpoint') for uniform weights.
    """
    idx = np.argsort(x)
    x_s = x[idx]
    w_s = weights[idx]
    cum_w = np.cumsum(w_s)
    # Place each step at the midpoint of its cumulative-weight interval
    mid = cum_w - w_s / 2.0
    return float(np.interp(q, mid, x_s))


# =============================================================================
# Scoring rules
# =============================================================================

def crps(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
    weights: np.ndarray = None,
) -> np.ndarray:
    """Continuous Ranked Probability Score (univariate, per time step).

    Uses the energy form:
        CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]

    With importance weights w = (w₁, …, wK) summing to 1, this becomes:
        CRPS_w(y) = Σᵢ wᵢ|xᵢ - y| - ½ Σᵢⱼ wᵢwⱼ|xᵢ - xⱼ|

    Parameters
    ----------
    forecast_samples : np.ndarray
        Shape (K, H) or (K, H, 1).
    observations : np.ndarray
        Shape (H,) or (H, 1).
    weights : np.ndarray or None
        Shape (K,). Normalised path importance weights from
        compute_path_weights().  None → uniform weights, O(K log K) formula.

    Returns
    -------
    np.ndarray
        Shape (H,). CRPS at each forecast step.
    """
    if forecast_samples.ndim == 3 and forecast_samples.shape[2] == 1:
        forecast_samples = forecast_samples[:, :, 0]
    if observations.ndim == 2 and observations.shape[1] == 1:
        observations = observations[:, 0]

    K, H = forecast_samples.shape

    if weights is None:
        # Uniform weights: O(K log K) sorted formula
        mae_term = np.mean(np.abs(forecast_samples - observations[None, :]), axis=0)
        sorted_s = np.sort(forecast_samples, axis=0)           # (K, H)
        w = 2 * np.arange(1, K + 1) - K - 1                   # (K,)
        spread_term = (
            np.sum(w[:, None] * sorted_s, axis=0) / (K * (K - 1))
            if K > 1 else np.zeros(H)
        )
    else:
        # Weighted: O(K²) pairwise computation
        # mae:  Σᵢ wᵢ |xᵢ(h) - y(h)|
        mae_term = np.einsum(
            'k,kh->h', weights,
            np.abs(forecast_samples - observations[None, :])
        )
        # spread: ½ Σᵢⱼ wᵢwⱼ |xᵢ(h) - xⱼ(h)|
        diff = np.abs(
            forecast_samples[:, None, :] - forecast_samples[None, :, :]
        )                                                       # (K, K, H)
        ww = weights[:, None] * weights[None, :]               # (K, K)
        spread_term = 0.5 * np.einsum('ij,ijh->h', ww, diff)

    return mae_term - spread_term


def energy_score(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
    weights: np.ndarray = None,
) -> np.ndarray:
    """Energy Score — multivariate generalisation of CRPS (per time step).

        ES(F, y) = E[‖X - y‖] - 0.5 * E[‖X - X'‖]

    Parameters
    ----------
    forecast_samples : np.ndarray
        Shape (K, H, D).
    observations : np.ndarray
        Shape (H, D).
    weights : np.ndarray or None
        Shape (K,). Normalised path importance weights.
        None → uniform weights with random-pair Monte Carlo for efficiency.

    Returns
    -------
    np.ndarray
        Shape (H,). Energy score at each forecast step.
    """
    K, H, D = forecast_samples.shape

    if weights is None:
        mae_term = np.mean(
            np.linalg.norm(forecast_samples - observations[None, :, :], axis=-1),
            axis=0,
        )
        idx1 = np.random.randint(0, K, size=K)
        idx2 = np.random.randint(0, K, size=K)
        spread_term = 0.5 * np.mean(
            np.linalg.norm(
                forecast_samples[idx1] - forecast_samples[idx2], axis=-1
            ),
            axis=0,
        )
    else:
        mae_term = np.einsum(
            'k,kh->h', weights,
            np.linalg.norm(forecast_samples - observations[None, :, :], axis=-1),
        )
        diff_norm = np.linalg.norm(
            forecast_samples[:, None, :, :] - forecast_samples[None, :, :, :],
            axis=-1,
        )                                                       # (K, K, H)
        ww = weights[:, None] * weights[None, :]               # (K, K)
        spread_term = 0.5 * np.einsum('ij,ijh->h', ww, diff_norm)

    return mae_term - spread_term


def coverage(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
    quantile_levels: list,
    weights: np.ndarray = None,
) -> dict:
    """Marginal interval coverage at multiple quantile levels.

    For level q the interval is [Q_{(1-q)/2}, Q_{(1+q)/2}] per dimension.
    Coverage = fraction of forecast steps where the observation falls inside,
    averaged over dimensions.

    With importance weights the interval boundaries are weighted quantiles of
    the forecast distribution, so diverged paths do not inflate the intervals.

    Parameters
    ----------
    forecast_samples : np.ndarray
        Shape (K, H, D).
    observations : np.ndarray
        Shape (H, D).
    quantile_levels : list of float
    weights : np.ndarray or None
        Shape (K,). Normalised path importance weights.

    Returns
    -------
    dict
        Keys are quantile levels (float), values are empirical coverage (float).
    """
    K, H, D = forecast_samples.shape
    result = {}
    for q in quantile_levels:
        lo_p = (1 - q) / 2
        hi_p = (1 + q) / 2
        if weights is None:
            lo = np.percentile(forecast_samples, lo_p * 100, axis=0)  # (H, D)
            hi = np.percentile(forecast_samples, hi_p * 100, axis=0)
        else:
            lo = np.array([
                [_weighted_quantile(forecast_samples[:, h, d], weights, lo_p)
                 for d in range(D)]
                for h in range(H)
            ])                                                  # (H, D)
            hi = np.array([
                [_weighted_quantile(forecast_samples[:, h, d], weights, hi_p)
                 for d in range(D)]
                for h in range(H)
            ])
        inside = (observations >= lo) & (observations <= hi)   # (H, D)
        result[q] = float(inside.mean())
    return result


# =============================================================================
# Combined entry point
# =============================================================================

def compute_all_metrics(
    forecast_samples: np.ndarray,
    observations: np.ndarray,
    quantile_levels: list,
    path_weight_threshold: float = 20.0,
) -> dict:
    """Compute importance-weighted CRPS, Energy Score, and coverage.

    Path weights downweight numerically diverged samples (see
    compute_path_weights).  The effective sample size and number of
    near-zero-weight paths are stored for diagnostics.

    Parameters
    ----------
    forecast_samples : np.ndarray
        Shape (K, H, D).
    observations : np.ndarray
        Shape (H, D).
    quantile_levels : list of float
    path_weight_threshold : float
        M0 passed to compute_path_weights.  Default 20.0.

    Returns
    -------
    dict with keys:
        'mean_crps'            : float
        'mean_energy_score'    : float
        'crps_per_step'        : list (H,)
        'energy_score_per_step': list (H,)
        'coverage'             : dict {str(q): float}
        'n_effective_samples'  : float — 1/Σwᵢ² (≈K when no divergence)
        'n_diverged_samples'   : int  — paths with weight < 1/(10K)
        'path_weight_threshold': float
    """
    K, H, D = forecast_samples.shape

    weights = compute_path_weights(forecast_samples, M0=path_weight_threshold)

    # Sanitise forecast samples before computing scoring rules.
    # Diverged paths already have near-zero weight (from compute_path_weights),
    # so clamping their values to ±1e6 makes their CRPS contribution
    # negligible (weight ≈ 0 × finite = 0) without poisoning the arithmetic.
    # We also cast to float64 to avoid float32 overflow (e.g. 2.5e38 - (-2.5e38)
    # overflows float32 back to inf in the spread term of the CRPS formula).
    forecast_samples = np.clip(
        np.nan_to_num(forecast_samples.astype(np.float64), nan=0.0, posinf=1e6, neginf=-1e6),
        -1e6, 1e6,
    )

    # Diagnostics
    n_eff = float(1.0 / np.sum(weights ** 2))
    n_diverged = int(np.sum(weights < 1.0 / (10 * K)))

    es = energy_score(forecast_samples, observations, weights=weights)

    if D == 1:
        cs = crps(forecast_samples, observations, weights=weights)
        mean_crps = float(np.mean(cs))
        sum_crps = mean_crps  # identical for D=1
    else:
        # Per-dimension CRPS, then summed — the multivariate independence baseline.
        # Equals the Energy Score when dimensions are independent.
        per_dim_crps_per_step = np.array([
            crps(forecast_samples[:, :, d], observations[:, d], weights=weights)
            for d in range(D)
        ])                                                      # (D, H)
        cs = per_dim_crps_per_step.mean(axis=0)               # (H,) — mean over dims
        mean_crps = float(cs.mean())                           # mean per-asset per-step CRPS
        sum_crps = float(per_dim_crps_per_step.mean(axis=1).sum())  # sum of per-dim means

    cov = coverage(forecast_samples, observations, quantile_levels, weights=weights)

    return {
        'mean_crps': mean_crps,
        'sum_crps': sum_crps,
        'mean_energy_score': float(np.mean(es)),
        'crps_per_step': cs.tolist(),
        'energy_score_per_step': es.tolist(),
        'coverage': {str(q): v for q, v in cov.items()},
        'n_effective_samples': n_eff,
        'n_diverged_samples': n_diverged,
        'path_weight_threshold': path_weight_threshold,
    }
