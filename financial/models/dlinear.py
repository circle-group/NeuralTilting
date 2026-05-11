"""DLinear model for financial price forecasting.

Implements the DLinear architecture from:
  "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023)

The model decomposes the input sequence into trend (moving average) and
remainder, applies one linear layer to each component, and sums the outputs.

Each window gets its own independently trained model (seq_len and pred_len are
set at construction time to match the window). This is consistent with how
TiltedStableSDEFinancial and GaussianSDEFinancial are used.

Produces a deterministic point forecast. The caller (run_window_dlinear.py)
converts this to degenerate samples (K identical copies) so that the CRPS
equals the MAE — the standard convention for comparing probabilistic and
deterministic forecasters.
"""

import jax
import jax.numpy as jnp
import equinox as eqx


class DLinearFinancial(eqx.Module):
    """DLinear for financial window forecasting.

    Parameters
    ----------
    lookback_len : int
        Fixed input lookback length. Shorter than T_train so that multiple
        training sub-windows can be extracted from the 30-day training window.
        At test time, the last lookback_len steps of training data are used.
    pred_len : int
        Forecast horizon (H).
    state_dim : int
        Number of observed dimensions (D). Channels are processed independently
        with shared linear weights (channel-independent mode).
    kernel_size : int
        Moving-average kernel size for trend extraction. Default 25.
    key : jax.random.PRNGKey
        Initialisation key.
    """

    kernel_size:  int = eqx.field(static=True)
    lookback_len: int = eqx.field(static=True)
    pred_len:     int = eqx.field(static=True)
    state_dim:    int = eqx.field(static=True)

    trend_linear:     eqx.nn.Linear
    remainder_linear: eqx.nn.Linear

    def __init__(self, lookback_len, pred_len, state_dim, kernel_size=25, *, key):
        self.kernel_size  = kernel_size
        self.lookback_len = lookback_len
        self.pred_len     = pred_len
        self.state_dim    = state_dim

        k1, k2 = jax.random.split(key)
        self.trend_linear     = eqx.nn.Linear(lookback_len, pred_len, use_bias=True, key=k1)
        self.remainder_linear = eqx.nn.Linear(lookback_len, pred_len, use_bias=True, key=k2)

    # ------------------------------------------------------------------
    def _moving_average(self, x: jnp.ndarray) -> jnp.ndarray:
        """Symmetric moving average with edge-value padding.

        Parameters
        ----------
        x : (T, D)

        Returns
        -------
        trend : (T, D)
        """
        k = self.kernel_size
        front_pad = (k - 1) // 2
        back_pad  = k - 1 - front_pad

        front = jnp.repeat(x[0:1],  front_pad, axis=0)
        back  = jnp.repeat(x[-1:],  back_pad,  axis=0)
        x_pad = jnp.concatenate([front, x, back], axis=0)  # (T + k - 1, D)

        kernel = jnp.ones(k) / k

        # Apply 1-D convolution independently to each channel.
        # jnp.convolve mode='valid' gives output length = T.
        def conv_channel(ch):   # (T + k - 1,)
            return jnp.convolve(ch, kernel, mode='valid')  # (T,)

        return jax.vmap(conv_channel, in_axes=1, out_axes=1)(x_pad)  # (T, D)

    # ------------------------------------------------------------------
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Parameters
        ----------
        x : (lookback_len, D)
            Normalised input lookback sequence.

        Returns
        -------
        pred : (pred_len, D)
            Point forecast.
        """
        trend     = self._moving_average(x)   # (lookback_len, D)
        remainder = x - trend                  # (lookback_len, D)

        # Apply shared linear independently to each channel:
        #   x.T  : (D, T)  ->  vmap linear  ->  (D, H)  ->  .T  ->  (H, D)
        trend_out = jax.vmap(self.trend_linear)(trend.T).T          # (H, D)
        rem_out   = jax.vmap(self.remainder_linear)(remainder.T).T  # (H, D)

        return trend_out + rem_out  # (H, D)
