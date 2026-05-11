"""DeepAR model for financial log-price forecasting.

Architecture:
  - Multi-layer LSTM encoder (teacher-forcing during training)
  - Linear or MLP decoder: LSTM hidden state → (loc, log_scale) of Normal distribution
  - Autoregressive inference: K independent sample trajectories from the
    final LSTM state after encoding the training window

Training loss:
  Negative log-likelihood under Normal(loc, scale) at each time step
  using teacher-forcing (input[t] = obs[t-1], obs[-1] = 0).  This is
  computed over the full training sequence; no sub-windowing is used.

Inference:
  1. Encode obs[0..T-1] with teacher-forcing → final LSTM state.
  2. Autoregressively sample H steps starting from obs[T-1]:
       y_t ~ Normal(loc_t, scale_t),  y_t used as input for step t+1.
  3. Repeat K times (jax.vmap over independent PRNGKeys).

Reference:
  Salinas et al. "DeepAR: Probabilistic forecasting with autoregressive
  recurrent networks." International Journal of Forecasting, 2020.
"""

import jax
import jax.numpy as jnp
import equinox as eqx


class DeepARFinancial(eqx.Module):
    """DeepAR for financial window forecasting.

    Parameters
    ----------
    obs_dim : int
        Observation dimension D (1 for a single ticker).
    pred_len : int
        Forecast horizon H (number of steps to predict).
    lstm_hidden_size : int
        LSTM hidden-state dimension.  Default 128.
    lstm_n_layers : int
        Number of stacked LSTM layers.  Default 2.
    decoder_hidden_layers : int
        Number of hidden layers in the distribution-parameter decoder
        (0 = single linear projection).  Default 0.
    decoder_hidden_size : int
        Hidden width of the decoder MLP (used only when
        decoder_hidden_layers > 0).  Default 64.
    key : jax.random.PRNGKey
        Initialisation key.
    """

    # ------------------------------------------------------------------
    # Static (compile-time) fields
    # ------------------------------------------------------------------
    obs_dim:          int = eqx.field(static=True)
    pred_len:         int = eqx.field(static=True)
    lstm_hidden_size: int = eqx.field(static=True)
    lstm_n_layers:    int = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Learnable fields
    # ------------------------------------------------------------------
    lstm_cells: list   # list of eqx.nn.LSTMCell (length = lstm_n_layers)
    decoder:    eqx.Module  # hidden_size → 2*obs_dim  (loc + log_scale)

    # ------------------------------------------------------------------
    def __init__(
        self,
        obs_dim:              int,
        pred_len:             int,
        lstm_hidden_size:     int = 128,
        lstm_n_layers:        int = 2,
        decoder_hidden_layers: int = 0,
        decoder_hidden_size:  int = 64,
        *,
        key: jax.Array,
    ):
        self.obs_dim          = obs_dim
        self.pred_len         = pred_len
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_n_layers    = lstm_n_layers

        # One key per LSTM layer + one for the decoder
        keys = jax.random.split(key, lstm_n_layers + 1)

        # Build LSTM cells: first layer input is obs_dim, rest are hidden_size
        cells = []
        for i in range(lstm_n_layers):
            in_size = obs_dim if i == 0 else lstm_hidden_size
            cells.append(
                eqx.nn.LSTMCell(in_size, lstm_hidden_size, key=keys[i])
            )
        self.lstm_cells = cells

        # Decoder: lstm_hidden_size → 2 * obs_dim
        out_size = 2 * obs_dim
        if decoder_hidden_layers == 0:
            self.decoder = eqx.nn.Linear(
                lstm_hidden_size, out_size, key=keys[-1]
            )
        else:
            self.decoder = eqx.nn.MLP(
                in_size=lstm_hidden_size,
                out_size=out_size,
                width_size=decoder_hidden_size,
                depth=decoder_hidden_layers,
                key=keys[-1],
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zero_state(self):
        """Return all-zero (h, c) tuple for each LSTM layer."""
        return tuple(
            (jnp.zeros(self.lstm_hidden_size), jnp.zeros(self.lstm_hidden_size))
            for _ in range(self.lstm_n_layers)
        )

    def _lstm_step(self, state, x):
        """Single forward step through all LSTM layers.

        Parameters
        ----------
        state : tuple of (h, c) pairs, one per layer
        x : (obs_dim,)

        Returns
        -------
        new_state : tuple of (h, c)
        h_last : (lstm_hidden_size,)  — hidden state of the final layer
        """
        inp = x
        new_state_list = []
        for i in range(self.lstm_n_layers):
            h, c = state[i]
            h_new, c_new = self.lstm_cells[i](inp, (h, c))
            new_state_list.append((h_new, c_new))
            inp = h_new
        return tuple(new_state_list), inp

    def _decode(self, h):
        """Map LSTM hidden state to Normal distribution parameters.

        Parameters
        ----------
        h : (lstm_hidden_size,)

        Returns
        -------
        loc   : (obs_dim,)
        scale : (obs_dim,)  — positive, via softplus + epsilon
        """
        params    = self.decoder(h)                              # (2*obs_dim,)
        loc       = params[: self.obs_dim]                       # (obs_dim,)
        scale     = jax.nn.softplus(params[self.obs_dim :]) + 1e-4  # (obs_dim,)
        return loc, scale

    def _encode(self, obs):
        """Encode an observation sequence with teacher-forcing.

        Input at step t is obs[t-1]; obs[-1] is treated as a zero vector.

        Parameters
        ----------
        obs : (T, obs_dim)

        Returns
        -------
        final_state : tuple of (h, c)  — LSTM state after the last step
        all_h       : (T, lstm_hidden_size)  — hidden states at every step
        """
        # Shifted input: [0, obs[0], obs[1], ..., obs[T-2]]
        zero    = jnp.zeros((1, self.obs_dim))
        shifted = jnp.concatenate([zero, obs[:-1]], axis=0)  # (T, obs_dim)

        def scan_fn(state, x_t):
            new_state, h = self._lstm_step(state, x_t)
            return new_state, h

        init_state = self._zero_state()
        final_state, all_h = jax.lax.scan(scan_fn, init_state, shifted)
        # all_h: (T, lstm_hidden_size)
        return final_state, all_h

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, obs):
        """Teacher-forcing forward pass (used during training).

        Parameters
        ----------
        obs : (T, obs_dim)

        Returns
        -------
        loc   : (T, obs_dim)
        scale : (T, obs_dim)
        """
        _, all_h = self._encode(obs)                    # (T, lstm_hidden_size)
        loc, scale = jax.vmap(self._decode)(all_h)      # (T, obs_dim) each
        return loc, scale

    def forecast(self, obs, n_samples, key):
        """Autoregressively sample forecast trajectories.

        Parameters
        ----------
        obs      : (T, obs_dim)  — full training-window observations
        n_samples : int          — number of independent sample paths (K)
        key      : jax.random.PRNGKey

        Returns
        -------
        samples : (n_samples, pred_len, obs_dim)
        """
        # Encode the full training sequence to get the conditioning state.
        final_state, _ = self._encode(obs)   # tuple of (h, c) per layer
        last_obs        = obs[-1]             # (obs_dim,) — seed for AR steps

        def sample_one(key_i):
            """Generate one trajectory of length pred_len."""

            def ar_step(carry, rng):
                state, prev = carry
                new_state, h = self._lstm_step(state, prev)
                loc, scale   = self._decode(h)
                y_t          = loc + scale * jax.random.normal(rng, shape=loc.shape)
                return (new_state, y_t), y_t

            step_keys = jax.random.split(key_i, self.pred_len)
            _, trajectory = jax.lax.scan(
                ar_step, (final_state, last_obs), step_keys
            )
            return trajectory   # (pred_len, obs_dim)

        sample_keys = jax.random.split(key, n_samples)
        all_samples = jax.vmap(sample_one)(sample_keys)  # (n_samples, pred_len, obs_dim)
        return all_samples
