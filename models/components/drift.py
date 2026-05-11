import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from typing import Any, Optional
from jaxtyping import Float, Array, PRNGKeyArray

from models.components.utils import fourier_features

# =============================================================================
# Drift Components (Deterministic Terms)
# =============================================================================

# --- Linear Functions ---

class DiagonalLinearFunction(eqx.Module):
    """Diagonal linear function f(t, x) = w ⊙ x + b for drift term in SDEs:

        dx_t = f(t, x_t) dt + σ(t, x_t) dB_t
    
    This implementation assumes each dimension evolves independently with
    a linear relationship: f_i(x) = w_i * x_i + b_i
    
    This is simpler than full matrix multiplication and often sufficient for
    systems with decoupled dimensions (e.g., Ornstein-Uhlenbeck processes).
    
    Attributes:
        weight: Learnable diagonal weight vector w of shape (state_dim,)
        bias: Learnable bias vector b of shape (state_dim,)
        state_dim: Dimension of the state space
    """
    weight: Float[Array, "state_dim"]
    bias: Float[Array, "state_dim"]
    state_dim: int = eqx.field(static=True)
    
    def __init__(self, state_dim: int = 1, initial_weight: Optional[Array] = None, initial_bias: Optional[Array] = None, seed: int = 0):
        """Initialize diagonal linear function.
        
        Args:
            state_dim: Dimension of the state space (default: 1)
            initial_weight: Initial weight values. If None, small random initialization
            initial_bias: Initial bias values. If None, initializes to zero
            seed: Random seed for initialization
        """
        key = random.key(seed)
        key1, key2 = random.split(key)
        
        # Initialize weight
        if initial_weight is not None:
            self.weight = jnp.atleast_1d(jnp.asarray(initial_weight))
            if self.weight.shape != (state_dim,):
                raise ValueError(
                    f"initial_weight shape {self.weight.shape} doesn't match state_dim={state_dim}. "
                    f"Expected shape ({state_dim},)"
                )
        else:
            # Small random initialization for stable training
            self.weight = 5.0 * random.normal(key1, shape=(state_dim,))

        # Initialize bias
        if initial_bias is not None:
            self.bias = jnp.atleast_1d(jnp.asarray(initial_bias))
            if self.bias.shape != (state_dim,):
                raise ValueError(
                    f"initial_bias shape {self.bias.shape} doesn't match state_dim={state_dim}. "
                    f"Expected shape ({state_dim},)"
                )
        else:
            # Initialize to small random values
            self.bias = 5.0 * random.normal(key2, shape=(state_dim,))
        
        self.state_dim = state_dim
    
    def __call__(self, t: Float[Array, ""], x: Float[Array, "state_dim"], args: Any = None) -> Float[Array, "state_dim"]:
        """Evaluate the drift function f(t, x) = w ⊙ x + b.
        
        Args:
            t: Time point (scalar)
            x: State vector of shape (state_dim,)
            args: Optional arguments (ignored, kept for Diffrax compatibility)
            
        Returns:
            Drift vector of shape (state_dim,)
        """
        x = jnp.atleast_1d(x)
        return self.weight * x + self.bias
    
class OUDiagonalLinearFunction(DiagonalLinearFunction):
    """Diagonal linear function for Ornstein-Uhlenbeck mean-reverting processes.
    
    Implements the OU drift: f(t, x) = θ ⊙ (μ - x) = -θ ⊙ x + θ ⊙ μ
    where θ > 0 are mean-reversion rates and μ are mean-reversion levels.
    
    This corresponds to the SDE: dx_t = θ(μ - x_t) dt + σ dB_t
    
    The softplus transformation ensures θ remains positive during optimization.
    
    Attributes:
        weight: Raw weight values (transformed to positive via softplus)
        bias: Mean-reversion levels μ
        state_dim: Dimension of the state space
        theta: Mean-reversion rates θ > 0 (property, computed as softplus(weight))
        mu: Alias for bias (property, for clearer OU semantics)
    
    Example:
        >>> f = OUDiagonalLinearFunction(state_dim=2, initial_weight=[0.5, 1.0], initial_bias=[0.0, 1.0])
        >>> f.theta  # Access mean-reversion rates
        Array([0.5, 1.0], dtype=float32)
    """

    def __init__(self, state_dim: int = 1, initial_weight=None, initial_bias=None, seed: int = 0):
        """Initialize OU diagonal linear function with inverse-softplus mapping."""
        super().__init__(state_dim, initial_weight, initial_bias, seed)
        
        # Ensure raw weight maps to target initial_weight via softplus
        eps = 1e-7
        self.weight = jnp.log(jnp.expm1(jnp.abs(self.weight) + eps))
    
    @property
    def theta(self) -> Float[Array, "state_dim"]:
        """Mean-reversion rates θ > 0, computed as softplus(weight)."""
        return jax.nn.softplus(self.weight)
    
    @property
    def mu(self) -> Float[Array, "state_dim"]:
        """Mean-reversion levels μ (alias for bias)."""
        return self.bias
    
    def __call__(self, t: Float[Array, ""], x: Float[Array, "state_dim"], args: Any = None) -> Float[Array, "state_dim"]:
        """Evaluate the OU drift function.
        
        Args:
            t: Time point (ignored for static/time-invariant function)
            x: State vector of shape (state_dim,)
            args: Optional arguments (ignored, kept for Diffrax compatibility)
            
        Returns:
            Drift vector θ ⊙ (μ - x) of shape (state_dim,)
        """
        x = jnp.atleast_1d(x)
        return self.theta * (self.mu - x)
    
# --- Neural Network Functions ---

class NeuralNetFunction(eqx.Module):
    """Neural network-based drift function for time-dependent SDEs.
    
    Attributes:
        mlp: Equinox MLP for function approximation.
        n_time_features: Number of Fourier frequency components (static).
        state_dim: Dimension of the state space (static).
        period: Time period for Fourier feature normalization (static).
    """
    mlp: eqx.nn.MLP
    n_time_features: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    period: float = eqx.field(static=True)
    
    def __init__(self, width: int, depth: int, n_time_features: int = 0, state_dim: int = 1, period: float = 1.0, seed: int = 0):
        # We type the key internally for clarity
        key: PRNGKeyArray = random.key(seed)

        # Input dimension: 1 (raw t) + 2*N (sin/cos Fourier) + state_dim (x)
        in_size: int = 1 + 2 * n_time_features + state_dim

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=state_dim,
            width_size=width,
            depth=depth,
            activation=nn.leaky_relu,
            key=key
        )
        self.n_time_features = n_time_features
        self.state_dim = state_dim
        self.period = period
    
    def _encode(self, t: Float[Array, ""], x: Float[Array, "state_dim"]) -> Float[Array, "1+2*n_time_features+state_dim"]:
        """Encode time and state into a single feature vector.
        
        Returns:
            Features: [t, cos(ω₁t), sin(ω₁t), ..., x₁, x₂, ...]
        """
        t_array: Float[Array, "1"] = jnp.atleast_1d(t)
        
        # Fourier encoding for t
        tf: Float[Array, "2*n_time_features"] = fourier_features(t, self.n_time_features, self.period)
        
        x = jnp.atleast_1d(x)

        return jnp.concatenate([t_array, tf, x], axis=0)
    
    def __call__(self, t: Float[Array, ""], x: Float[Array, "state_dim"], args: Any = None) -> Float[Array, "state_dim"]:
        """Evaluate the neural network drift function f(t, x).

        Args:
            t: Time point
            x: State vector of shape (state_dim,)

        Returns:
            Drift vector of shape (state_dim,)
        """
        features: Float[Array, "1+2*n_time_features+state_dim"] = self._encode(t, x)
        return self.mlp(features)


# --- Non-linear Functions ---

class DoubleWellDriftFunction(eqx.Module):
    """Double well potential drift function for bistable SDEs.

    Implements the drift: f(t, x) = θ₁ ⊙ x - θ₂ ⊙ x³
    derived from the potential U(x) = -θ₁/2 · x² + θ₂/4 · x⁴ via f = -dU/dx.

    This creates a bistable system with two stable fixed points at x = ±√(θ₁/θ₂)
    (per dimension) and an unstable fixed point at x = 0.

    Both θ₁, θ₂ > 0 are enforced via softplus during optimisation.

    Attributes:
        raw_theta1: Unconstrained parameter mapped to θ₁ via softplus.
        raw_theta2: Unconstrained parameter mapped to θ₂ via softplus.
        state_dim: Dimension of the state space.
        theta1: Linear coefficient θ₁ > 0 (property).
        theta2: Cubic coefficient θ₂ > 0 (property).
    """
    raw_theta1: Float[Array, "state_dim"]
    raw_theta2: Float[Array, "state_dim"]
    state_dim: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int = 1,
        initial_theta1: Optional[Array] = None,
        initial_theta2: Optional[Array] = None,
        seed: int = 0,
    ):
        """Initialize double well drift function.

        Args:
            state_dim: Dimension of the state space.
            initial_theta1: Initial values for θ₁ > 0 (linear coefficient). If None,
                small positive random values are drawn using `seed`.
            initial_theta2: Initial values for θ₂ > 0 (cubic coefficient). If None,
                small positive random values are drawn using `seed`.
            seed: Random seed used when initial values are not provided.
        """
        key = random.key(seed)
        key1, key2 = random.split(key)
        eps = 1e-7

        def to_raw(value, rng_key, name):
            if value is not None:
                arr = jnp.atleast_1d(jnp.asarray(value, dtype=jnp.float32))
                if arr.shape != (state_dim,):
                    raise ValueError(
                        f"{name} shape {arr.shape} doesn't match state_dim={state_dim}. "
                        f"Expected shape ({state_dim},)"
                    )
                if jnp.any(arr <= 0):
                    raise ValueError(f"All values of {name} must be strictly positive.")
                # Inverse softplus so that softplus(raw) ≈ value
                return jnp.log(jnp.expm1(arr + eps))
            else:
                # Small positive random initialization: sample magnitudes, then remap
                theta_init = jnp.abs(random.normal(rng_key, shape=(state_dim,))) + eps
                return jnp.log(jnp.expm1(theta_init + eps))

        self.raw_theta1 = to_raw(initial_theta1, key1, "initial_theta1")
        self.raw_theta2 = to_raw(initial_theta2, key2, "initial_theta2")
        self.state_dim = state_dim

    @property
    def theta1(self) -> Float[Array, "state_dim"]:
        """Linear coefficient θ₁ > 0."""
        return jax.nn.softplus(self.raw_theta1)

    @property
    def theta2(self) -> Float[Array, "state_dim"]:
        """Cubic coefficient θ₂ > 0."""
        return jax.nn.softplus(self.raw_theta2)

    def __call__(self, t: Float[Array, ""], x: Float[Array, "state_dim"], args: Any = None) -> Float[Array, "state_dim"]:
        """Evaluate the double well drift f(t, x) = θ₁ ⊙ x - θ₂ ⊙ x³.

        Args:
            t: Time point (ignored; drift is time-invariant).
            x: State vector of shape (state_dim,).
            args: Optional arguments (ignored, kept for Diffrax compatibility).

        Returns:
            Drift vector of shape (state_dim,).
        """
        x = jnp.atleast_1d(x)
        return self.theta1 * x - self.theta2 * x ** 3