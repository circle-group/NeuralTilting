import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

# =============================================================================
# Utility Functions
# =============================================================================

def fourier_features(t, n_freq=4, period=1.0):
    """Fourier positional encoding for scalar time t.

    Generates [cos, sin] features at multiple frequencies to capture
    high-frequency temporal dynamics in SDEs. Features are fixed (not learned)
    to provide a consistent multi-scale time representation.

    Args:
        t: Scalar time point (use vmap for batched inputs)
        n_freq: Number of frequency components (output dim = 2 * n_freq)
        period: Time period for normalization (default: 1.0)
                If your time range is [0, T], set period=T

    Returns:
        Fourier features of shape (2 * n_freq,)
        [cos(2πt/T), cos(4πt/T), ..., sin(2πt/T), sin(4πt/T), ...]

    Example:
        >>> # For time in [0, 10]
        >>> feat = fourier_features(t=5.0, n_freq=4, period=10.0)
        >>> feat.shape
        (8,)

        >>> # Batched via vmap
        >>> batch_feat = jax.vmap(lambda t: fourier_features(t, n_freq=4, period=10.0))
        >>> times = jnp.array([0.0, 2.5, 5.0, 7.5, 10.0])
        >>> batch_feat(times).shape
        (5, 8)
    """
    if n_freq == 0:
        return jnp.array([])
    frequencies = jnp.arange(1, n_freq + 1, dtype=t.dtype)
    ang = 2.0 * jnp.pi * frequencies * t / period
    return jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=0)


def scale_mlp_params(mlp: eqx.nn.MLP, w_scale: float = 1.0, b_scale: float = 0.1) -> eqx.nn.MLP:
    """Scale MLP parameters for custom initialization.

    Applies different scaling factors to weights (2D arrays) and biases (1D arrays)
    to achieve desired initialization scales. This is useful for controlling the
    initial output magnitude of neural networks.

    The function uses tree_map to traverse the MLP structure and:
        - Scales 1D arrays (biases) by b_scale
        - Scales 2D+ arrays (weights) by w_scale
        - Leaves non-arrays unchanged (e.g., activation functions)

    Args:
        mlp: Equinox MLP to scale
        w_scale: Scaling factor for weight matrices (default: 1.0)
        b_scale: Scaling factor for bias vectors (default: 0.1)

    Returns:
        New MLP with scaled parameters

    Example:
        >>> key = random.key(0)
        >>> mlp = eqx.nn.MLP(in_size=10, out_size=5, width_size=32, depth=2, key=key)
        >>> # Scale down for small initialization
        >>> small_mlp = scale_mlp_params(mlp, w_scale=0.01, b_scale=0.001)
    """
    return jtu.tree_map(
        lambda x: b_scale * x if (eqx.is_array(x) and x.ndim == 1) else (
                w_scale * x if eqx.is_array(x) else x),
        mlp
    )