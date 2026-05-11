"""N-HiTS model for financial log-price forecasting.

Implements the N-HiTS (Neural Hierarchical Interpolation for Time Series)
architecture from:
  Challu et al. "N-HiTS: Neural Hierarchical Interpolation for Time Series
  Forecasting." AAAI 2023.  https://arxiv.org/abs/2201.12886

Architecture overview:
  - S stacks, each targeting a different frequency scale.
  - Each stack has B blocks. Each block:
      1. MaxPool the residual with a stack-specific kernel (compresses input).
      2. MLP: pooled input → theta of size (lookback_len + n_knots).
      3. Split theta: backcast (full lookback_len) + forecast knots (n_knots).
      4. Interpolate n_knots → pred_len via linear interpolation.
  - Doubly residual: input to next block = residual − backcast; total forecast
    accumulates block forecasts initialised from the last observed value (level).

Channel-independent: the same weights are applied independently to each
observation dimension (vmap over D channels).

Produces a deterministic point forecast. The caller (run_window_nhits.py)
converts it to degenerate samples (K identical copies) so CRPS = MAE —
the standard convention for comparing probabilistic and deterministic models.
"""

import math

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _maxpool1d(x: jnp.ndarray, kernel_size: int) -> jnp.ndarray:
    """1-D max-pooling with ceil_mode=True (no batch dimension).

    Parameters
    ----------
    x : (L,)
    kernel_size : int

    Returns
    -------
    out : (ceil(L / kernel_size),)
    """
    L = x.shape[0]
    n_out = math.ceil(L / kernel_size)
    pad = n_out * kernel_size - L
    if pad > 0:
        x = jnp.concatenate([x, jnp.full(pad, -jnp.inf)])
    return jnp.max(x.reshape(n_out, kernel_size), axis=1)


def _linear_interp(knots: jnp.ndarray, pred_len: int) -> jnp.ndarray:
    """Linearly interpolate `knots` to `pred_len` output points.

    Parameters
    ----------
    knots : (n_knots,)
    pred_len : int

    Returns
    -------
    out : (pred_len,)
    """
    n_knots = knots.shape[0]
    if n_knots == pred_len:
        return knots
    xp = jnp.linspace(0.0, 1.0, n_knots)
    x  = jnp.linspace(0.0, 1.0, pred_len)
    return jnp.interp(x, xp, knots)


# ---------------------------------------------------------------------------
# NHiTSBlock
# ---------------------------------------------------------------------------

class NHiTSBlock(eqx.Module):
    """Single N-HiTS block.

    Parameters
    ----------
    input_size : int
        Lookback length L (before pooling).
    pred_len : int
        Forecast horizon H.
    n_pool_kernel : int
        Max-pooling kernel size for this stack.
    n_freq_down : int
        Frequency downsampling factor: forecast knots = max(H // n_freq_down, 1).
    mlp_width : int
        Hidden width of the MLP.
    mlp_depth : int
        Number of hidden layers in the MLP (equinox convention).
    key : jax.Array
    """

    mlp:        eqx.nn.MLP
    input_size: int = eqx.field(static=True)
    pred_len:   int = eqx.field(static=True)
    pool_kernel: int = eqx.field(static=True)
    n_knots:    int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        pred_len:   int,
        n_pool_kernel: int,
        n_freq_down:   int,
        mlp_width:     int,
        mlp_depth:     int,
        *,
        key: jax.Array,
    ):
        self.input_size  = input_size
        self.pred_len    = pred_len
        self.pool_kernel = n_pool_kernel
        self.n_knots     = max(pred_len // n_freq_down, 1)

        pooled_size = math.ceil(input_size / n_pool_kernel)
        n_theta     = input_size + self.n_knots   # backcast + forecast knots

        self.mlp = eqx.nn.MLP(
            in_size=pooled_size,
            out_size=n_theta,
            width_size=mlp_width,
            depth=mlp_depth,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, residual: jnp.ndarray):
        """Forward pass for a single channel.

        Parameters
        ----------
        residual : (L,)
            Current residual (in reverse-time order, as per N-HiTS convention).

        Returns
        -------
        backcast : (L,)
            Backcast in the same (reversed) domain as the input.
        forecast : (H,)
            Block forecast contribution in forward-time order.
        """
        pooled   = _maxpool1d(residual, self.pool_kernel)    # (pooled_size,)
        theta    = self.mlp(pooled)                           # (input_size + n_knots,)
        backcast = theta[: self.input_size]                   # (L,)
        knots    = theta[self.input_size :]                   # (n_knots,)
        forecast = _linear_interp(knots, self.pred_len)       # (H,)
        return backcast, forecast


# ---------------------------------------------------------------------------
# NHiTSFinancial
# ---------------------------------------------------------------------------

class NHiTSFinancial(eqx.Module):
    """N-HiTS for financial window forecasting.

    Parameters
    ----------
    lookback_len : int
        Fixed input lookback length L.
    pred_len : int
        Forecast horizon H.
    state_dim : int
        Number of observed dimensions D. Channels are processed independently
        with shared weights (channel-independent mode).
    n_stacks : int
        Number of stacks (must equal len(n_pool_kernel_size) =
        len(n_freq_downsample)).  Default 3.
    n_blocks_per_stack : int
        Number of blocks per stack.  Default 1.
    mlp_width : int
        Hidden width for all block MLPs.  Default 512.
    mlp_depth : int
        Number of hidden layers in each block MLP.  Default 2.
    n_pool_kernel_size : list[int]
        Max-pooling kernel size per stack.  Default [2, 2, 1].
    n_freq_downsample : list[int]
        Forecast knot downsampling factor per stack.  Default [4, 2, 1].
    key : jax.Array
    """

    blocks:       list
    lookback_len: int = eqx.field(static=True)
    pred_len:     int = eqx.field(static=True)
    state_dim:    int = eqx.field(static=True)

    def __init__(
        self,
        lookback_len: int,
        pred_len:     int,
        state_dim:    int,
        n_stacks:           int       = 3,
        n_blocks_per_stack: int       = 1,
        mlp_width:          int       = 512,
        mlp_depth:          int       = 2,
        n_pool_kernel_size: list      = None,
        n_freq_downsample:  list      = None,
        *,
        key: jax.Array,
    ):
        if n_pool_kernel_size is None:
            n_pool_kernel_size = [2, 2, 1]
        if n_freq_downsample is None:
            n_freq_downsample = [4, 2, 1]

        assert len(n_pool_kernel_size) == n_stacks, (
            f"len(n_pool_kernel_size)={len(n_pool_kernel_size)} "
            f"must equal n_stacks={n_stacks}"
        )
        assert len(n_freq_downsample) == n_stacks, (
            f"len(n_freq_downsample)={len(n_freq_downsample)} "
            f"must equal n_stacks={n_stacks}"
        )

        self.lookback_len = lookback_len
        self.pred_len     = pred_len
        self.state_dim    = state_dim

        all_blocks = []
        for s in range(n_stacks):
            for _ in range(n_blocks_per_stack):
                key, subkey = jax.random.split(key)
                all_blocks.append(
                    NHiTSBlock(
                        input_size=lookback_len,
                        pred_len=pred_len,
                        n_pool_kernel=n_pool_kernel_size[s],
                        n_freq_down=n_freq_downsample[s],
                        mlp_width=mlp_width,
                        mlp_depth=mlp_depth,
                        key=subkey,
                    )
                )
        self.blocks = all_blocks

    # ------------------------------------------------------------------
    def _forecast_one_channel(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run the full N-HiTS forward pass for a single channel.

        Parameters
        ----------
        x : (L,)
            Normalised lookback sequence for one channel.

        Returns
        -------
        forecast : (H,)
            Point forecast.
        """
        # Initialise residual in reverse-time order (N-HiTS convention).
        residual = x[::-1]                          # (L,)

        # Level initialisation: forecast starts at the last observed value.
        level    = x[-1]
        forecast = jnp.full((self.pred_len,), level) # (H,)

        for block in self.blocks:
            backcast, delta = block(residual)        # (L,), (H,)
            residual = residual - backcast
            forecast = forecast + delta

        return forecast  # (H,)

    # ------------------------------------------------------------------
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass (channel-independent).

        Parameters
        ----------
        x : (lookback_len, D)
            Normalised lookback sequence.

        Returns
        -------
        pred : (pred_len, D)
            Point forecast.
        """
        # vmap over channels: x.T is (D, L), output is (D, H), then .T -> (H, D)
        return jax.vmap(self._forecast_one_channel)(x.T).T  # (H, D)
