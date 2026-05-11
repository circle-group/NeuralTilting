"""Neural Non-Stationary Merton Jump Diffusion for financial forecasting.

Reference:
    "Neural Non-Stationary Merton Jump Diffusion" (NeurIPS 2025)
    arXiv:2506.04542

Architecture:
    MLP encoder (channel-independent): past L observations -> embedding.
    Parameter head: embedding -> (H, 5) MJD parameters per dimension.
    Sampling: Euler-Maruyama with restart for K probabilistic trajectories.

The five MJD parameters per forecast step (activated in _predict_one_channel):
    col 0  mu         : drift            — identity (unbounded)
    col 1  sigma      : diffusion std    — sigmoid  -> (0, 1)
    col 2  log_lambda : log jump rate    — identity (exp applied in loss/sample)
    col 3  nu         : log-jump mean    — tanh x 0.5 -> (-0.5, 0.5)
    col 4  gamma      : log-jump std     — sigmoid  -> (0, 1)

Channel-independent: the same encoder weights are vmapped independently
over the D observation dimensions (same convention as N-HiTS / DLinear).
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx


# ---------------------------------------------------------------------------
# MJD log-likelihood (scalar, one step, one dimension)
# ---------------------------------------------------------------------------

def _mjd_step_log_prob(
    delta:      float,
    mu:         float,
    log_lambda: float,
    sigma:      float,
    nu:         float,
    gamma:      float,
    kappa:      int = 5,
) -> float:
    """Log P(delta | MJD params) as a truncated Gaussian mixture.

    Mixture: sum_{n=0}^{kappa} Poisson(n; lambda) x N(delta; a_n, b_n^2)
      a_n  = (mu - lambda*k - sigma^2/2) + n*nu
      b_n^2 = sigma^2 + n*gamma^2
      k    = exp(nu + gamma^2/2) - 1   (log-normal mean correction)
    """
    lambda_ = jnp.exp(jnp.minimum(log_lambda, 0.0))     # in (0, 1]
    k       = jnp.exp(nu + 0.5 * gamma**2) - 1.0

    ns = jnp.arange(kappa + 1, dtype=jnp.float32)       # (kappa+1,)

    log_w = (-lambda_
             + ns * jnp.log(lambda_ + 1e-8)
             - jax.scipy.special.gammaln(ns + 1.0))      # (kappa+1,)

    a_n  = (mu - lambda_ * k - 0.5 * sigma**2) + ns * nu   # (kappa+1,)
    b_sq = jnp.maximum(sigma**2 + ns * gamma**2, 1e-8)      # (kappa+1,)

    log_norm = (-0.5 * jnp.log(2.0 * jnp.pi * b_sq)
                - 0.5 * (delta - a_n)**2 / b_sq)            # (kappa+1,)

    return jax.scipy.special.logsumexp(log_w + log_norm)    # scalar


# ---------------------------------------------------------------------------
# _EncoderMLP: MLP with optional inter-layer dropout
# ---------------------------------------------------------------------------

class _EncoderMLP(eqx.Module):
    """MLP with optional dropout between hidden layers.

    Matches eqx.nn.MLP(in_size, out_size, width_size=width, depth=depth,
    activation=relu) in architecture, but supports per-layer dropout during
    training by accepting an optional PRNG key.

    Call with key=None (inference): dropout disabled.
    Call with key=<jax.Array> (training): dropout applied after each hidden
    layer's activation.  The key is split depth times internally so each
    layer gets independent noise.
    """

    layers:  list
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_size:      int,
        out_size:     int,
        width:        int,
        depth:        int,
        dropout_rate: float,
        *,
        key: jax.Array,
    ):
        ks = jrandom.split(key, depth + 1)
        self.layers = [
            eqx.nn.Linear(
                in_size if i == 0 else width,
                out_size if i == depth else width,
                key=ks[i],
            )
            for i in range(depth + 1)
        ]
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x: jnp.ndarray, key=None) -> jnp.ndarray:
        """Forward pass.

        Parameters
        ----------
        x   : (in_size,)
        key : jax.Array or None
            If None, dropout is skipped (inference mode).
            If provided, dropout is applied after each hidden activation.
        """
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
            if key is not None:
                key, subkey = jrandom.split(key)
                x = self.dropout(x, key=subkey)
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# NeuralMJDFinancial
# ---------------------------------------------------------------------------

class NeuralMJDFinancial(eqx.Module):
    """NeuralMJD for our rolling-window financial forecasting pipeline.

    The model produces K probabilistic forecast trajectories at inference
    time via Euler-Maruyama simulation with restart (Algorithm 2 from the
    paper): at each forecast step tau, simulation restarts from the predicted
    conditional mean to prevent error accumulation over the horizon.

    dropout_rate controls inter-layer dropout in the MLP encoder.
    dropout_rate=0.0 (default) reproduces the original no-dropout behaviour.
    The paper's transformer uses dropout=0.1; use the same here to prevent
    sigma/gamma from collapsing to near-zero during training.
    """

    encoder:      _EncoderMLP
    lookback_len: int = eqx.field(static=True)
    pred_len:     int = eqx.field(static=True)
    state_dim:    int = eqx.field(static=True)
    n_mjd_kappa:  int = eqx.field(static=True)

    def __init__(
        self,
        lookback_len:  int,
        pred_len:      int,
        state_dim:     int,
        mlp_width:     int   = 256,
        mlp_depth:     int   = 3,
        n_mjd_kappa:   int   = 5,
        dropout_rate:  float = 0.0,
        *,
        key: jax.Array,
    ):
        self.lookback_len = lookback_len
        self.pred_len     = pred_len
        self.state_dim    = state_dim
        self.n_mjd_kappa  = n_mjd_kappa

        self.encoder = _EncoderMLP(
            in_size=lookback_len,
            out_size=pred_len * 5,
            width=mlp_width,
            depth=mlp_depth,
            dropout_rate=dropout_rate,
            key=key,
        )

    # ------------------------------------------------------------------
    # Parameter prediction
    # ------------------------------------------------------------------

    def _predict_one_channel(self, x: jnp.ndarray, key=None) -> jnp.ndarray:
        """Predict activated MJD parameters for a single channel.

        Parameters
        ----------
        x   : (L,)
        key : jax.Array or None — passed to encoder for dropout

        Returns
        -------
        params : (H, 5)  — [mu, sigma, log_lambda, nu, gamma]
        """
        raw     = self.encoder(x, key=key).reshape(self.pred_len, 5)
        mu      = raw[:, 0]                         # drift — unbounded
        sigma   = jax.nn.sigmoid(raw[:, 1])         # diffusion std in (0, 1)
        log_lam = raw[:, 2]                         # log jump rate — raw
        nu      = jnp.tanh(raw[:, 3]) * 0.5        # log-jump mean in (-0.5, 0.5)
        gamma   = jax.nn.sigmoid(raw[:, 4])         # log-jump std in (0, 1)
        return jnp.stack([mu, sigma, log_lam, nu, gamma], axis=-1)  # (H, 5)

    def __call__(self, x: jnp.ndarray, key=None) -> jnp.ndarray:
        """Predict MJD parameters for all channels.

        Parameters
        ----------
        x   : (L, D)
        key : jax.Array or None — if provided, dropout is active (training)

        Returns
        -------
        params : (H, 5, D)
        """
        if key is not None:
            # Training mode: give each channel an independent dropout key.
            D = x.shape[1]
            channel_keys = jrandom.split(key, D)
            # vmap over channels (D,L) and per-channel keys (D,2)
            return jax.vmap(self._predict_one_channel)(x.T, channel_keys).transpose(1, 2, 0)
        else:
            # Inference mode: no dropout keys.
            return jax.vmap(self._predict_one_channel)(x.T).transpose(1, 2, 0)

    # ------------------------------------------------------------------
    # Sampling: Euler-Maruyama with restart
    # ------------------------------------------------------------------

    def sample(
        self,
        x:              jnp.ndarray,   # (L, D) lookback context
        n_samples:      int,
        steps_per_unit: int,
        key:            jax.Array,
    ) -> jnp.ndarray:
        """Generate K forecast trajectories via EM with restart.

        For each of the K trajectories and each of the H forecast steps,
        the simulation resets to the predicted conditional mean E[log S_tau]
        before applying M stochastic substeps.  Because restart bases are
        deterministic, the H forecast steps are fully independent and can be
        computed in parallel via vmap.

        Returns
        -------
        samples : (K, H, D)
        """
        # Inference mode — dropout disabled (key=None to __call__).
        params  = self(x)               # (H, 5, D)
        s0      = x[-1]                 # (D,) last training obs
        H, _, D = params.shape
        M       = steps_per_unit
        dt      = 1.0 / M

        mu      = params[:, 0, :]       # (H, D)
        sigma   = params[:, 1, :]
        log_lam = params[:, 2, :]
        nu      = params[:, 3, :]
        gamma   = params[:, 4, :]

        lambda_ = jnp.exp(jnp.minimum(log_lam, 0.0))       # (H, D)
        k       = jnp.exp(nu + 0.5 * gamma**2) - 1.0        # (H, D)
        # Deterministic drift contribution per substep
        alpha   = (mu - lambda_ * k - 0.5 * sigma**2) * dt  # (H, D)

        # Restart bases: s0 for step 0, conditional mean at step tau-1 for tau >= 1
        log_mean  = s0[None, :] + jnp.cumsum(mu, axis=0)            # (H, D)
        prev_mean = jnp.concatenate([s0[None, :], log_mean[:-1]], axis=0)  # (H, D)

        def simulate_one(key_k: jax.Array) -> jnp.ndarray:
            # (3 * H * M) keys: one triple (diff, pois, jump) per substep per step
            all_keys  = jrandom.split(key_k, 3 * H * M)
            k_diff    = all_keys[:H*M].reshape(H, M)           # (H, M)
            k_pois    = all_keys[H*M:2*H*M].reshape(H, M)      # (H, M)
            k_jmag    = all_keys[2*H*M:].reshape(H, M)         # (H, M)

            def simulate_step(args) -> jnp.ndarray:
                """Run M EM substeps for one forecast step, restarting from prev_mean."""
                tau, kd_tau, kp_tau, kj_tau = args
                # kd_tau, kp_tau, kj_tau : (M,) keys for M substeps

                def substep(cur: jnp.ndarray, keys_j) -> tuple:
                    kd, kp, kj = keys_j
                    diff = sigma[tau] * jrandom.normal(kd, (D,)) * jnp.sqrt(dt)
                    n_j  = jrandom.poisson(kp, lambda_[tau] / M).astype(float)  # (D,)
                    jump = (n_j * nu[tau]
                            + jnp.sqrt(n_j) * gamma[tau] * jrandom.normal(kj, (D,)))
                    return cur + alpha[tau] + diff + jump, None

                final, _ = jax.lax.scan(
                    substep,
                    prev_mean[tau],                        # restart base (D,)
                    (kd_tau, kp_tau, kj_tau),
                )
                return final   # (D,)

            # vmap over H forecast steps (independent due to restart)
            tau_idx = jnp.arange(H)
            path = jax.vmap(simulate_step)((tau_idx, k_diff, k_pois, k_jmag))
            return path   # (H, D)

        sample_keys = jrandom.split(key, n_samples)
        return jax.vmap(simulate_one)(sample_keys)   # (K, H, D)
