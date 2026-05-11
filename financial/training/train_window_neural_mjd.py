"""Train a NeuralMJD model on a single rolling window.

Training objective (Algorithm 1 from the paper):
  loss = NLL + w_cond_mean * MSE_of_conditional_mean

NLL uses the discretised MJD likelihood (truncated Gaussian mixture, kappa=5
terms per step).  The conditional mean is predicted analytically as
  E[log S_tau | context] = s0 + cumsum(mu)[tau]
and is used for both the MSE regulariser and as the "bootstrapped" previous
state in the NLL (replacing the true previous observation to reduce training-
inference mismatch).

Sub-windows are extracted identically to N-HiTS / DLinear.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import equinox as eqx

from financial.models.neural_mjd import NeuralMJDFinancial, _mjd_step_log_prob


def make_sub_windows(
    train_obs:    jnp.ndarray,
    lookback_len: int,
    pred_len:     int,
) -> tuple:
    """Extract overlapping (input, target) pairs from training observations.

    Parameters
    ----------
    train_obs : (T, D)
    lookback_len : int
    pred_len : int

    Returns
    -------
    inputs  : (N, lookback_len, D)
    targets : (N, pred_len, D)
    """
    T = train_obs.shape[0]
    n = T - lookback_len - pred_len + 1
    if n <= 0:
        raise ValueError(
            f"Training window too short: T={T}, need T >= "
            f"lookback_len + pred_len = {lookback_len + pred_len}."
        )
    inputs  = jnp.stack([train_obs[i : i + lookback_len]                         for i in range(n)])
    targets = jnp.stack([train_obs[i + lookback_len : i + lookback_len + pred_len] for i in range(n)])
    return inputs, targets


def train_neural_mjd(
    train_obs:      jnp.ndarray,
    forecast_obs:   jnp.ndarray,
    lookback_len:   int,
    mlp_width:      int,
    mlp_depth:      int,
    n_mjd_kappa:    int,
    dropout_rate:   float,
    w_cond_mean:    float,
    learning_rate:  float,
    training_steps: int,
    key:            jax.Array,
    verbose:        bool = False,
) -> tuple:
    """Train NeuralMJD on one rolling window.

    Parameters
    ----------
    train_obs : (T, D)
        Normalised cumulative log-prices over the training window.
    forecast_obs : (H, D)
        Normalised forecast observations — shape determines pred_len H.
    lookback_len : int
    mlp_width : int
    mlp_depth : int
    n_mjd_kappa : int
        Truncation order kappa for the Gaussian mixture likelihood.
    dropout_rate : float
        Inter-layer dropout rate for the MLP encoder.  0.0 = no dropout.
    w_cond_mean : float
        Weight of the MSE-of-conditional-mean loss term.
    learning_rate : float
    training_steps : int
    key : jax.Array
    verbose : bool

    Returns
    -------
    model : NeuralMJDFinancial
    metrics : dict
    """
    T, D = train_obs.shape
    H    = forecast_obs.shape[0]

    # Split key: model initialisation uses model_key, training loop uses key.
    key, model_key = jrandom.split(key)

    inputs, targets = make_sub_windows(train_obs, lookback_len, H)
    # inputs  : (N, lookback_len, D)
    # targets : (N, H, D)
    N = inputs.shape[0]

    model = NeuralMJDFinancial(
        lookback_len=lookback_len,
        pred_len=H,
        state_dim=D,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        n_mjd_kappa=n_mjd_kappa,
        dropout_rate=dropout_rate,
        key=model_key,
    )

    # Gradient clipping + Adam
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, key):
        def loss_fn(model):
            # Per-sample dropout keys: one key per sub-window.
            keys_per_sample = jrandom.split(key, N)

            # Predict MJD parameters for all sub-windows: (N, H, 5, D)
            all_params = jax.vmap(model)(inputs, keys_per_sample)

            mu      = all_params[:, :, 0, :]   # (N, H, D)
            sigma   = all_params[:, :, 1, :]
            log_lam = all_params[:, :, 2, :]
            nu      = all_params[:, :, 3, :]
            gamma   = all_params[:, :, 4, :]

            # Conditional mean: s0 + cumsum(mu)
            s0       = inputs[:, -1, :]                              # (N, D)
            log_mean = s0[:, None, :] + jnp.cumsum(mu, axis=1)      # (N, H, D)

            # MSE of conditional mean prediction
            mse = jnp.mean((log_mean - targets) ** 2)

            # NLL: increments from bootstrapped previous conditional mean
            prev_mean   = jnp.concatenate(
                [s0[:, None, :], log_mean[:, :-1, :]], axis=1)      # (N, H, D)
            delta_prime = targets - prev_mean                         # (N, H, D)

            kappa = model.n_mjd_kappa
            log_prob = jax.vmap(jax.vmap(jax.vmap(
                lambda d, m, ll, s, n, g: _mjd_step_log_prob(d, m, ll, s, n, g, kappa)
            )))(delta_prime, mu, log_lam, sigma, nu, gamma)          # (N, H, D)

            nll = -jnp.mean(log_prob)
            return nll + w_cond_mean * mse, (nll, mse)

        (total_loss, (nll, mse)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, total_loss, nll, mse

    initial_loss = None
    loss = nll = mse = None
    for i in range(training_steps):
        key, step_key = jrandom.split(key)
        model, opt_state, loss, nll, mse = step(model, opt_state, step_key)
        loss_val = float(loss)
        if i == 0:
            initial_loss = loss_val
        if verbose and (i % 100 == 0 or i == training_steps - 1):
            print(f"  step {i:4d}/{training_steps}  "
                  f"loss={loss_val:.4f}  nll={float(nll):.4f}  mse={float(mse):.6f}  "
                  f"(N={N} sub-windows)")

    return model, {
        'initial_loss':  initial_loss,
        'final_loss':    float(loss),
        'final_nll':     float(nll),
        'final_mse':     float(mse),
        'n_sub_windows': N,
    }
