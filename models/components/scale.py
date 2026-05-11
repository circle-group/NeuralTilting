import math
import numpy as np
import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from typing import Any
from jaxtyping import Float, Array

from models.components.utils import fourier_features

# =============================================================================
# Scale Components
# =============================================================================

class DiagonalScaleFactor(eqx.Module):
    """Diagonal scale factor σ(t, x) for stochastic differential equations.
    
    Represents the diffusion coefficient in a SDE such as:
        dx_t = f(t, x_t) dt + σ(t, x_t) dB_t
    
    This implementation uses a softplus transformation to ensure σ > 0. It is 
    constant with respect to both time and state, with independent noise scaling.
    
    Attributes:
        raw_weight: Unconstrained parameter vector of shape (state_dim,)
        state_dim: Dimension of the state space
    """
    raw_weight: jnp.ndarray
    state_dim: int = eqx.field(static=True)
    
    def __init__(self, state_dim=1, initial_weight=None, seed=0):
        """Initialize constant diagonal scale factor.
        
        Args:
            state_dim: Dimension of the state space (default: 1)
            initial_weight: Desired starting scale. If None, sampled from Gamma/10.
            seed: Random seed for initialization.
        """
        key = random.key(seed)
        key, subkey = random.split(key, num=2)

        if initial_weight is not None:
            target_weight = jnp.atleast_1d(jnp.asarray(initial_weight))
            if target_weight.shape != (state_dim,):
                raise ValueError(
                    f"initial_weight shape {target_weight.shape} doesn't match state_dim={state_dim}."
                )
        else:
            # Sample target from gamma (mean ~1.0)
            target_weight = random.gamma(subkey, a=10., shape=(state_dim,)) / 10.

        # Inverse softplus: raw = log(exp(target) - 1)
        # This ensures jax.nn.softplus(raw_weight) == target_weight at init
        self.raw_weight = jnp.log(jnp.expm1(target_weight))
        self.state_dim = state_dim
    
    @property
    def weight(self) -> jnp.ndarray:
        """The actual positive scale factor used in the SDE."""
        return jax.nn.softplus(self.raw_weight)

    def __call__(self, t: Float[Array, ""], x: Float[Array, "state_dim"], args: Any = None) -> Float[Array, "state_dim"]:
        """Evaluate the scale factor.

        Returns:
            Positive diagonal weight vector of shape (state_dim,)
        """
        return self.weight


# =============================================================================
# Cholesky Diffusion Component
# =============================================================================

class CholeskyDiffusion(eqx.Module):
    """Neural network lower-triangular Cholesky diffusion L(t, x) for multivariate SDEs.

    Returns a (D, D) lower-triangular matrix L with strictly positive diagonal,
    so the diffusion covariance Σ(t, x) = L(t, x) Lᵀ(t, x) is always SPD.

    The SDE diffusion term is `L(t, x) dW`, giving:
        posterior drift correction : −L(t,X) Lᵀ(t,X) u(t,X)
        KL per unit time           : ½ ‖Lᵀ(t,X) u(t,X)‖²

    Initialised so that L(t, x) ≈ I for all inputs:
        - diagonal entries ≈ 1 (via learnable raw_diag_bias + MLP correction)
        - off-diagonal entries ≈ 0 (MLP biases initialised near zero)

    Attributes:
        mlp: MLP mapping encoded (t, x) → D*(D+1)/2 raw lower-triangular entries.
        raw_diag_bias: Shape (D,) learnable base bias for diagonal entries,
            initialised to inv_softplus(1) ≈ 0.541 so diagonal ≈ 1 at init.
        state_dim: D — dimension of the state space (static).
        n_time_features: Number of Fourier frequency components (static).
        period: Period for Fourier encoding (static).
        _tril_rows: Static tuple of row indices for lower-triangular positions.
        _tril_cols: Static tuple of col indices for lower-triangular positions.
        _is_diag: Static bool tuple — True at diagonal positions.
    """
    mlp: eqx.nn.MLP
    raw_diag_bias: jnp.ndarray
    state_dim: int = eqx.field(static=True)
    n_time_features: int = eqx.field(static=True)
    period: float = eqx.field(static=True)
    _tril_rows: tuple = eqx.field(static=True)
    _tril_cols: tuple = eqx.field(static=True)
    _is_diag: tuple = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        width: int = 32,
        depth: int = 2,
        n_time_features: int = 0,
        period: float = 1.0,
        seed: int = 0,
    ):
        """Initialise CholeskyDiffusion.

        Args:
            state_dim: Dimension D of the state space.
            width: Hidden layer width of the MLP.
            depth: Number of hidden layers in the MLP.
            n_time_features: Fourier frequency components for time encoding.
            period: Period for Fourier encoding.
            seed: Random seed.
        """
        D = state_dim
        n_lower = D * (D + 1) // 2

        key = random.key(seed)
        in_size = 1 + 2 * n_time_features + D

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=n_lower,
            width_size=width,
            depth=depth,
            activation=nn.leaky_relu,
            key=key,
        )

        # inv_softplus(1) = log(exp(1) - 1) ≈ 0.5413
        # so softplus(raw_diag_bias + mlp_output) ≈ softplus(0.5413) ≈ 1 at init
        inv_sp1 = math.log(math.exp(1.0) - 1.0)
        self.raw_diag_bias = jnp.full((D,), inv_sp1)

        self.state_dim = D
        self.n_time_features = n_time_features
        self.period = period

        # Precompute lower-triangular index structure (static — D is fixed at init)
        rows, cols = np.tril_indices(D)
        self._tril_rows = tuple(int(r) for r in rows)
        self._tril_cols = tuple(int(c) for c in cols)
        self._is_diag = tuple(bool(r == c) for r, c in zip(rows, cols))

    def _encode(self, t: Float[Array, ""], x: Float[Array, "state_dim"]) -> jnp.ndarray:
        t_array = jnp.atleast_1d(t)
        tf = fourier_features(t, self.n_time_features, self.period)
        x = jnp.atleast_1d(x)
        return jnp.concatenate([t_array, tf, x], axis=0)

    def __call__(
        self,
        t: Float[Array, ""],
        x: Float[Array, "state_dim"],
        args: Any = None,
    ) -> Float[Array, "state_dim state_dim"]:
        """Evaluate L(t, x).

        Returns:
            Lower-triangular matrix of shape (state_dim, state_dim) with
            positive diagonal entries.
        """
        features = self._encode(t, x)
        raw = self.mlp(features)  # (n_lower,)

        rows = jnp.array(self._tril_rows)   # (n_lower,)
        cols = jnp.array(self._tril_cols)   # (n_lower,)
        is_diag = jnp.array(self._is_diag)  # (n_lower,) bool

        # At diagonal position k: softplus(raw_diag_bias[k]) + softplus(raw[k])
        #   - softplus(raw_diag_bias) is a frozen positive floor (≈1.0 at init)
        #   - softplus(raw) is the trainable positive increment (≥0 always)
        #   - diagonal is guaranteed ≥ softplus(raw_diag_bias) ≈ 1.0 for all inputs
        # At off-diagonal position k: raw[k] (unconstrained)
        diag_bias = self.raw_diag_bias[rows]  # (n_lower,) — bias at each position
        processed = jnp.where(
            is_diag,
            jax.nn.softplus(diag_bias) + jax.nn.softplus(raw),
            raw,
        )

        return jnp.zeros((self.state_dim, self.state_dim)).at[rows, cols].set(processed)