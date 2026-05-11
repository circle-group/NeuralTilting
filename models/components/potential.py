"""Quadratic neural potential for tilting jump measures in stable-driven SDEs.

This module implements time-dependent quadratic potentials used for exponential
tilting of Lévy jump measures in variational inference for stable processes.
"""

import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
import jax.tree_util as jtu
import equinox as eqx
from jax.lax import stop_gradient
from jaxtyping import Float, Array
from typing import Optional, Tuple

from models.components.utils import fourier_features, scale_mlp_params
from models.components.attention import TemporalAttentionEncoding


# =============================================================================
# Quadratic Neural Potential with Temporal Attention
# =============================================================================

class QuadraticNeuralPotential(eqx.Module):
    """Time-dependent quadratic potential with temporal attention for jump tilting.

    This module parameterizes a quadratic potential φ(t, x) = A(t)·x² + B(t)·x
    used to exponentially tilt the jump measure of α-stable processes. The tilting
    is crucial for variance reduction in variational inference.

    Architecture:
        - Uses temporal attention to learn important time points (e.g., jump locations)
        - Combines Fourier features and attention embeddings for time encoding
        - Two MLPs output coefficients A(t) and B(t):
            * A(t) < 0: Negative curvature ensures finite jump variance (unbounded below)
            * B(t) ∈ ℝ: Fully unbounded linear coefficient for sharp jump features

    Parameterization (simple, gradient-friendly):
        A(t) = -(a_min + adaptive_scale_A * exp(f_A(features)))
        B(t) = adaptive_scale_B * f_B(features)

    where f_A, f_B are MLPs operating on [t, fourier_features(t), attention(t)].

    Key advantages:
        - No gradient vanishing: exp has non-vanishing gradients proportional to value
        - A(t) unbounded below: can suppress jumps strongly if data requires
        - B(t) fully unbounded: can represent sharp, localized mean shifts for jumps
        - Mathematically valid: A(t) < 0 ensures integrability; B(t) only shifts mean

    Adaptive scaling (optional):
        When use_adaptive_scaling=True, scales adapt based on process parameters (α, τ, σ):
            - A network output scaled by sqrt(expected_jump_variance) (a_min stays constant)
            - B network output scaled by expected_jump_scale
        This can help with different problem scales but is optional.
        Default: False for more principled and interpretable optimization.

    Attributes:
        mlp_A: Neural network for quadratic coefficients A(t)
        mlp_B: Neural network for linear coefficients B(t)
        temporal_attention: Attention mechanism for learning temporal embeddings (optional)
        a_min: Minimum absolute curvature magnitude |A| (floor to prevent collapse) (static)
        state_dim: Dimension of the state space (static)
        n_time_features: Number of Fourier frequency components (static)
        n_attention_references: Number of learnable attention reference times (static)
        attention_embed_dim: Dimension of attention embedding output (static)
        period: Time period for Fourier encoding (static)
        use_adaptive_scaling: Whether to scale based on process parameters (static)
        alpha: Stability parameter α ∈ (1, 2] (only used if use_adaptive_scaling=True) (static)
        tau: Jump truncation parameter τ > 0 (only used if use_adaptive_scaling=True) (static)
        sigma: Scale parameter σ > 0 (only used if use_adaptive_scaling=True) (static)

    Example:
        >>> key = random.key(0)
        >>> potential = QuadraticNeuralPotential(
        ...     width=64, depth=3, state_dim=1, period=1.0,
        ...     n_time_features=4, n_attention_references=50, seed=0
        ... )
        >>> A, B = potential.get_coefficients(t=0.5)
        >>> A.shape, B.shape
        ((1,), (1,))
        >>> phi = potential(t=0.5, x=jnp.array([1.0]))
        >>> phi.shape
        (1,)
    """
    mlp_A: eqx.nn.MLP
    mlp_B: eqx.nn.MLP
    temporal_attention: Optional[TemporalAttentionEncoding]
    a_min: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    n_time_features: int = eqx.field(static=True)
    n_attention_references: int = eqx.field(static=True)
    attention_embed_dim: int = eqx.field(static=True)
    period: float = eqx.field(static=True)
    use_adaptive_scaling: bool = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    sigma: float = eqx.field(static=True)

    def __init__(
        self,
        width: int,
        depth: int,
        state_dim: int = 1,
        a_min: float = 1e-3,
        n_time_features: int = 0,
        n_attention_references: int = 50,
        attention_embed_dim: int = 32,
        attention_sharpness: float = 100.0,
        period: float = 1.0,
        use_adaptive_scaling: bool = False,
        alpha: float = 1.5,
        tau: float = 0.01,
        sigma: float = 1.0,
        seed: int = 0,
        **kwargs  # Accept extra parameters for compatibility (e.g., a_max, b_max from old configs)
    ):
        """Initialize quadratic neural potential with temporal attention.

        Args:
            width: Hidden layer width for MLPs
            depth: Number of hidden layers for MLPs
            state_dim: Dimension of the state space
            a_min: Minimum absolute curvature magnitude |A| (floor to prevent collapse)
            n_time_features: Number of Fourier frequency components for time encoding
            n_attention_references: Number of learnable attention reference times (0 = disable attention)
            attention_embed_dim: Dimension of attention embedding output
            attention_sharpness: Attention sharpness value. Controls how peaked the attention is.
                                Typical values: 50-200. Default: 100.0
            period: Time period for Fourier and attention encoding
            use_adaptive_scaling: If True, scale A and B network outputs based on process parameters.
                                  If False, use unit scaling for more principled interpretation.
                                  Default: False (recommended with attention mechanism)
            alpha: Stability parameter α ∈ (1, 2] (only used if use_adaptive_scaling=True)
            tau: Jump truncation parameter τ > 0 (only used if use_adaptive_scaling=True)
            sigma: Scale parameter σ > 0 (only used if use_adaptive_scaling=True)
            seed: Random seed for initialization
            **kwargs: Extra parameters (ignored, for backward compatibility - e.g., a_max, b_max)
        """
        # Warn about unused parameters (e.g., a_max, b_max from old configs, or DeepSets-specific params)
        if kwargs:
            import warnings
            ignored_params = list(kwargs.keys())
            warnings.warn(
                f"QuadraticNeuralPotential ignoring unused parameters: {ignored_params}.",
                UserWarning
            )

        key = random.key(seed)
        key_A, key_B, key_attention = random.split(key, 3)

        # Initialize temporal attention if requested
        if n_attention_references > 0:
            self.temporal_attention = TemporalAttentionEncoding(
                embed_dim=attention_embed_dim,
                n_reference_times=n_attention_references,
                period=period,
                initial_sharpness=attention_sharpness,
                key=key_attention
            )
            attention_feature_dim = attention_embed_dim
        else:
            self.temporal_attention = None
            attention_feature_dim = 0

        # Input dimension includes: [t, fourier_features, attention_embedding]
        input_dim = 1 + 2 * n_time_features + attention_feature_dim

        # =====================================================================
        # NETWORK INITIALIZATION FOR A COEFFICIENT
        # =====================================================================
        # Initialize with small weights so exp(f_A(·)) ≈ exp(0) = 1.0 at start
        # This gives A(t) ≈ -(a_min + 1.0) initially
        # ReLU activation: Creates piecewise linear functions ideal for sharp jump features
        mlp_A_temp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=state_dim,
            width_size=width,
            depth=depth,
            activation=nn.relu,
            key=key_A,
        )
        # Small initialization: w_scale=0.1 keeps outputs near 0, so exp(A_raw) ≈ 1.0
        # This gives A(t) ≈ -(a_min + 1.0) initially (stable starting point)
        self.mlp_A = scale_mlp_params(mlp_A_temp, w_scale=0.1, b_scale=1e-4)

        # =====================================================================
        # NETWORK INITIALIZATION FOR B COEFFICIENT
        # =====================================================================
        # ReLU activation: Creates piecewise linear functions ideal for sharp jump features
        mlp_B_temp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=state_dim,
            width_size=width,
            depth=depth,
            activation=nn.relu,
            key=key_B,
        )
        # Small initialization: B(t) ≈ 0 initially (E[B(t)] = 0 over interval)
        # w_scale=0.1 provides sufficient gradient signal while keeping initial tilting small
        # b_scale=1e-5 (epsilon) ensures biases ≈ 0 for zero mean, with numerical safety
        self.mlp_B = scale_mlp_params(mlp_B_temp, w_scale=0.1, b_scale=1e-5)

        self.a_min = a_min
        self.state_dim = state_dim
        self.n_time_features = n_time_features
        self.n_attention_references = n_attention_references
        self.attention_embed_dim = attention_embed_dim
        self.period = period
        self.use_adaptive_scaling = use_adaptive_scaling
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma

    def _encode_time(self, t: Float[Array, ""]) -> Float[Array, "input_dim"]:
        """Encode time using Fourier features and optional attention mechanism.

        Combines multiple representations of time:
            1. Raw time value t
            2. Fourier features: [cos(2πt/T), sin(2πt/T), cos(4πt/T), ...]
            3. Attention embedding (if enabled): learned representation

        Args:
            t: Time point (scalar)

        Returns:
            Feature vector of shape (1 + 2*n_time_features + attention_embed_dim,)
        """
        t_scalar = jnp.atleast_1d(jnp.asarray(t))
        fourier_feat = fourier_features(t, self.n_time_features, self.period)

        if self.temporal_attention is not None:
            attention_feat = self.temporal_attention(t)
            return jnp.concatenate([t_scalar, fourier_feat, attention_feat], axis=0)
        else:
            return jnp.concatenate([t_scalar, fourier_feat], axis=0)

    def _compute_adaptive_scales(self) -> Tuple[Float[Array, ""], Float[Array, ""]]:
        """Compute adaptive scales based on process parameters.

        The adaptive scales ensure the potential can handle different jump intensities
        by adjusting a_min and the B network output based on the expected jump
        characteristics of the truncated α-stable process.

        For a truncated α-stable process with parameters (α, τ, σ):
            - Expected jump scale: E[|Z|] ≈ τ^(1-α) / (α-1)
            - Variance scale: Var[σ·Z] ≈ σ² · (E[|Z|])²

        Uses stop_gradient to prevent these process parameters from being optimized,
        as they define the problem rather than being learnable parameters.

        Returns:
            Tuple of (adaptive_scale_A, adaptive_scale_B) where:
                - adaptive_scale_A: Multiplier for A network output based on expected variance
                - adaptive_scale_B: Multiplier for B network output based on expected jump scale
        """
        # Use stop_gradient to prevent optimization of process parameters
        alpha_sg = stop_gradient(self.alpha)
        tau_sg = stop_gradient(self.tau)
        sigma_sg = stop_gradient(self.sigma)

        # Expected jump scale for truncated α-stable: E[|Z|] ∝ τ^(1-α) / (α-1)
        expected_jump_scale = jnp.power(tau_sg, 1.0 - alpha_sg) / (alpha_sg - 1.0)

        # Variance scale: Var[σ·Z] = σ² · E[|Z|]²
        var_scale = (sigma_sg * expected_jump_scale) ** 2

        # Clip to reasonable ranges for numerical stability
        adaptive_scale_A = jnp.clip(jnp.sqrt(var_scale), min=0.1, max=100.0)
        adaptive_scale_B = jnp.clip(expected_jump_scale, min=0.1, max=20.0)

        return adaptive_scale_A, adaptive_scale_B

    def get_coefficients(
        self,
        t: Float[Array, ""]
    ) -> Tuple[Float[Array, "state_dim"], Float[Array, "state_dim"]]:
        """Compute coefficients A(t) < 0 and B(t) ∈ ℝ for jump measure tilting.

        Mathematical formulation:

        For A(t) - Negative curvature coefficient (unbounded below):
            A(t) = -(a_min + adaptive_scale_A * exp(mlp_A(features)))

            - Ensures A(t) < 0 always (required for integrability)
            - a_min is constant floor; adaptive_scale_A scales network contribution
            - Unbounded below: can become arbitrarily negative if data requires
            - No gradient vanishing: ∂exp(x)/∂x = exp(x) is always positive and proportional to value
            - At init (mlp_A ≈ 0): A(t) ≈ -(a_min + adaptive_scale_A * 1.0)

        For B(t) - Linear coefficient (fully unbounded):
            B(t) = adaptive_scale_B * mlp_B(features)

            - No constraints: B ∈ ℝ (integrability guaranteed by A < 0)
            - Can represent sharp, localized shifts for jumps
            - adaptive_scale_B sets output magnitude based on problem scale

        Tilting weight analysis:
            exp(φ(t,x+z) - φ(t,x)) = exp(A(t)·z² + [2A(t)·x + B(t)]·z)

            - A(t) < 0 ensures exponential integrability ✓
            - B(t) only affects the mean shift of jumps
            - Large |B(t)| at jump times → strong localized tilting

        Adaptive scaling (if enabled):
            - adaptive_scale_A ∝ sqrt(expected_jump_variance)
            - adaptive_scale_B ∝ expected_jump_scale
            where expected_jump_scale = τ^(1-α) / (α-1) for truncated α-stable process

        Args:
            t: Time point (scalar)

        Returns:
            Tuple (A, B) where:
                A: Array of shape (state_dim,) with A < 0 (unbounded below)
                B: Array of shape (state_dim,) fully unbounded
        """
        features = self._encode_time(t)
        A_raw = self.mlp_A(features)
        B_raw = self.mlp_B(features)

        # Optionally compute adaptive scales based on process parameters
        if self.use_adaptive_scaling:
            adaptive_scale_A, adaptive_scale_B = self._compute_adaptive_scales()
        else:
            adaptive_scale_A = 1.0
            adaptive_scale_B = 1.0

        # =====================================================================
        # A COEFFICIENT: UNBOUNDED BELOW WITH EXPONENTIAL
        # =====================================================================
        # Ensures A < 0 while maintaining gradient flow
        # a_min provides a constant floor to prevent A ≈ 0 (which would disable tilting)
        # adaptive_scale_A scales the network's contribution for different problem scales
        # Exponential ensures: (1) A unbounded below, (2) no vanishing gradients
        A = -(self.a_min + adaptive_scale_A * nn.softplus(A_raw))

        # =====================================================================
        # B COEFFICIENT: FULLY UNBOUNDED
        # =====================================================================
        # B can take any real value - integrability ensured by A < 0
        # Network initialization (w_scale=0.1, b_scale=1e-5) keeps E[B(t)] ≈ 0 initially
        B = adaptive_scale_B * B_raw

        return A, B

    def __call__(
        self,
        t: Float[Array, ""],
        x: Float[Array, "state_dim"]
    ) -> Float[Array, "state_dim"]:
        """Evaluate the quadratic potential φ(t, x) = A(t)·x² + B(t)·x.

        Computes per-dimension potentials:
            φ_i(t, x_i) = A_i(t) * x_i² + B_i(t) * x_i

        This potential is used in exponential tilting of the jump measure:
            ν_tilted(dz) ∝ exp(φ(t, x+z) - φ(t, x)) · ν_prior(dz)

        Args:
            t: Time point (scalar)
            x: State vector of shape (state_dim,)

        Returns:
            Potential values of shape (state_dim,) for each dimension
        """
        A, B = self.get_coefficients(t)
        x = jnp.atleast_1d(x)

        # Compute quadratic potential per dimension
        return A * x**2 + B * x
