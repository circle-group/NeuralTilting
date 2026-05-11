"""Train a DLinear model on a single rolling window.

DLinear is a deterministic linear model. Training uses MSE on multiple
overlapping sub-windows extracted from the 30-day training sequence, which
is how the original paper trains (sliding window over training data).

At test time the model sees the last `lookback_len` observations from the
training window and predicts the `pred_len`-step forecast horizon.
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np

from financial.models.dlinear import DLinearFinancial


def make_sub_windows(
    train_obs: jnp.ndarray,
    lookback_len: int,
    pred_len: int,
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
            f"Training window too short: T={T}, lookback_len={lookback_len}, "
            f"pred_len={pred_len}. Need T >= lookback_len + pred_len = "
            f"{lookback_len + pred_len}."
        )
    inputs  = jnp.stack([train_obs[i : i + lookback_len]           for i in range(n)])
    targets = jnp.stack([train_obs[i + lookback_len : i + lookback_len + pred_len] for i in range(n)])
    return inputs, targets   # (N, lookback_len, D), (N, pred_len, D)


def train_dlinear(
    train_obs: jnp.ndarray,
    forecast_obs: jnp.ndarray,
    kernel_size: int,
    lookback_len: int,
    learning_rate: float,
    training_steps: int,
    key: jax.Array,
    verbose: bool = False,
) -> tuple:
    """Train DLinear on one window.

    Parameters
    ----------
    train_obs : (T, D)
        Normalised training observations (cumulative log-prices).
    forecast_obs : (H, D)
        Normalised forecast observations (used only for shape, not training).
    kernel_size : int
        Moving-average kernel size.
    lookback_len : int
        Fixed input lookback length. Must be < T - pred_len.
    learning_rate : float
    training_steps : int
    key : jax.Array
    verbose : bool

    Returns
    -------
    model : DLinearFinancial
    metrics : dict
    """
    T, D = train_obs.shape
    H    = forecast_obs.shape[0]

    inputs, targets = make_sub_windows(train_obs, lookback_len, H)
    # inputs:  (N, lookback_len, D)
    # targets: (N, H, D)
    N = inputs.shape[0]

    model = DLinearFinancial(
        lookback_len=lookback_len,
        pred_len=H,
        state_dim=D,
        kernel_size=kernel_size,
        key=key,
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state):
        def loss_fn(model):
            # Batch MSE over all sub-windows
            preds = jax.vmap(model)(inputs)   # (N, H, D)
            return jnp.mean((preds - targets) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    initial_loss = None
    for i in range(training_steps):
        model, opt_state, loss = step(model, opt_state)
        loss_val = float(loss)
        if i == 0:
            initial_loss = loss_val
        if verbose and (i % 100 == 0 or i == training_steps - 1):
            print(f"  step {i:4d}/{training_steps}  loss={loss_val:.6f}  "
                  f"(N={N} sub-windows)")

    return model, {
        'initial_loss': initial_loss,
        'final_loss': float(loss),
        'n_sub_windows': N,
    }
