"""Temporal attention mechanisms for learning important time points in SDEs.

This module implements attention-based time encoding that learns to identify
and represent important temporal locations (e.g., where jumps occur) without
being told in advance.
"""

import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray
from typing import Optional


# =============================================================================
# Temporal Attention Encoding
# =============================================================================

class TemporalAttentionEncoding(eqx.Module):
    """Attention-based time encoding for learning important temporal locations.

    This module learns to identify and represent important time points (e.g., where
    jumps occur) without being told in advance. During training, the reference times
    migrate to locations that are important for the loss, and the embeddings learn
    to represent patterns at those times.

    Architecture:
        1. Maintains learnable "reference times" (initially spread uniformly)
        2. For each query time t, computes attention weights over reference times
        3. Outputs weighted combination of learned embeddings

    The key insight: Rather than using fixed basis functions (Fourier, RBF), this
    learns BOTH where to focus (reference_times) AND what patterns to look for
    (reference_embeddings) via gradient descent on the loss.

    Mathematical formulation:
        For query time t, compute:
            attention_weights[i] = softmax(-|t - τ_i| * sharpness)
            output = Σ_i attention_weights[i] * embedding_i

    Attributes:
        reference_times: Learnable time points, shape (n_reference_times,)
        reference_embeddings: Learnable embeddings for each reference, shape (n_reference_times, embed_dim)
        attention_sharpness: Learnable scale controlling attention focus (in log-space for positivity)
        embed_dim: Dimension of output embedding vector (static)
        n_reference_times: Number of learnable reference time points (static)
        period: Time period for initialization (static)

    Example:
        >>> key = random.key(0)
        >>> # Default sharpness = 100.0
        >>> encoder = TemporalAttentionEncoding(embed_dim=32, n_reference_times=50, period=1.0, key=key)
        >>> embedding = encoder(t=0.5)
        >>> embedding.shape
        (32,)
        >>> # Custom sharpness for sharper/softer attention
        >>> encoder = TemporalAttentionEncoding(
        ...     embed_dim=32, n_reference_times=100, period=10.0,
        ...     initial_sharpness=200.0, key=key  # Sharper for discontinuous jumps
        ... )
    """
    reference_times: Float[Array, "n_refs"]
    reference_embeddings: Float[Array, "n_refs embed_dim"]
    attention_sharpness: Float[Array, "1"]
    embed_dim: int = eqx.field(static=True)
    n_reference_times: int = eqx.field(static=True)
    period: float = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int = 32,
        n_reference_times: int = 50,
        period: float = 1.0,
        initial_sharpness: float = 100.0,
        key: Optional[PRNGKeyArray] = None
    ):
        """Initialize temporal attention encoding.

        Args:
            embed_dim: Dimension of output embedding vector
            n_reference_times: Number of learnable reference time points
            period: Time period (for initialization of reference times)
            initial_sharpness: Attention sharpness value. Controls how peaked the attention is.
                              Typical values: 50-200. Higher = more peaked attention.
                              Default: 100.0
            key: Random key for initialization. If None, uses key(0)
        """
        if key is None:
            key = random.key(0)
        key_embed = random.split(key, 2)[0]

        # Initialize reference times uniformly across period
        # These will learn to migrate to important time points during training
        self.reference_times = jnp.linspace(0, period, n_reference_times)

        # Initialize embeddings with standard scaling: 1/sqrt(embed_dim)
        # This ensures output variance ≈ 1 initially via attention-weighted sum
        # These will learn to represent patterns at each reference time
        init_scale = 1.0 / jnp.sqrt(embed_dim)
        self.reference_embeddings = random.normal(
            key_embed, (n_reference_times, embed_dim)
        ) * init_scale

        # Initialize attention sharpness (stored in log-space for positivity)
        # Higher values = more focused attention (peaked around nearest reference)
        # Lower values = more diffuse attention (considers multiple references)
        self.attention_sharpness = jnp.array([jnp.log(initial_sharpness)])

        self.embed_dim = embed_dim
        self.n_reference_times = n_reference_times
        self.period = period

    def __call__(self, t: Float[Array, ""]) -> Float[Array, "embed_dim"]:
        """Compute attention-weighted embedding for time t.

        Process:
            1. Compute absolute distances between t and each reference time
            2. Apply softmax with learned sharpness to get attention weights (sum to 1)
            3. Return weighted sum of reference embeddings

        The gradient flows through:
            - reference_times: Adjusts WHERE to focus
            - reference_embeddings: Adjusts WHAT patterns to represent
            - attention_sharpness: Adjusts HOW focused the attention is

        Args:
            t: Query time point (scalar)

        Returns:
            Embedding vector of shape (embed_dim,)
        """
        t_scalar = jnp.asarray(t)

        # Compute absolute distances to all reference times
        # Shape: (n_reference_times,)
        abs_distances = jnp.abs(t_scalar - self.reference_times)

        # Get learned sharpness (positive via exp transform)
        sharpness = jnp.exp(self.attention_sharpness[0])

        # Compute attention scores (higher score = closer time)
        # Negative distance scaled by sharpness gives peaked distribution
        attention_scores = -abs_distances * sharpness

        # Softmax to get attention weights that sum to 1
        # Shape: (n_reference_times,)
        attention_weights = nn.softmax(attention_scores)

        # Weighted combination of embeddings
        # Shape: (embed_dim,)
        output = jnp.dot(attention_weights, self.reference_embeddings)

        return output
