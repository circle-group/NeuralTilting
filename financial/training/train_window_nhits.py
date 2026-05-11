"""Train an N-HiTS model on a single rolling window.

N-HiTS is a deterministic model.  Training uses MSE over overlapping
sub-windows extracted from the 30-day training sequence (same approach as
DLinear).  At inference time the last `lookback_len` steps of training data
are fed through the model to produce a single point forecast.
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from financial.models.nhits import NHiTSFinancial


def make_sub_windows(
    train_obs:   jnp.ndarray,
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
            f"Training window too short: T={T}, lookback_len={lookback_len}, "
            f"pred_len={pred_len}. Need T >= lookback_len + pred_len = "
            f"{lookback_len + pred_len}."
        )
    inputs  = jnp.stack([train_obs[i : i + lookback_len]                    for i in range(n)])
    targets = jnp.stack([train_obs[i + lookback_len : i + lookback_len + pred_len] for i in range(n)])
    return inputs, targets  # (N, lookback_len, D), (N, pred_len, D)


def train_nhits(
    train_obs:          jnp.ndarray,
    forecast_obs:       jnp.ndarray,
    lookback_len:       int,
    n_stacks:           int,
    n_blocks_per_stack: int,
    mlp_width:          int,
    mlp_depth:          int,
    n_pool_kernel_size: list,
    n_freq_downsample:  list,
    learning_rate:      float,
    training_steps:     int,
    key:                jax.Array,
    verbose:            bool = False,
) -> tuple:
    """Train N-HiTS on one rolling window.

    Parameters
    ----------
    train_obs : (T, D)
        Normalised training observations (cumulative log-prices).
    forecast_obs : (H, D)
        Normalised forecast observations.  Shape used only for pred_len H.
    lookback_len : int
        Fixed input lookback length.  Must satisfy T >= lookback_len + H.
    n_stacks : int
    n_blocks_per_stack : int
    mlp_width : int
    mlp_depth : int
    n_pool_kernel_size : list[int]
    n_freq_downsample : list[int]
    learning_rate : float
    training_steps : int
    key : jax.Array
    verbose : bool

    Returns
    -------
    model : NHiTSFinancial
    metrics : dict
        keys: initial_loss, final_loss, n_sub_windows
    """
    T, D = train_obs.shape
    H    = forecast_obs.shape[0]

    inputs, targets = make_sub_windows(train_obs, lookback_len, H)
    # inputs:  (N, lookback_len, D)
    # targets: (N, H, D)
    N = inputs.shape[0]

    model = NHiTSFinancial(
        lookback_len=lookback_len,
        pred_len=H,
        state_dim=D,
        n_stacks=n_stacks,
        n_blocks_per_stack=n_blocks_per_stack,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        n_pool_kernel_size=n_pool_kernel_size,
        n_freq_downsample=n_freq_downsample,
        key=key,
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state):
        def loss_fn(model):
            preds = jax.vmap(model)(inputs)   # (N, H, D)
            return jnp.mean((preds - targets) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    initial_loss = None
    loss = None
    for i in range(training_steps):
        model, opt_state, loss = step(model, opt_state)
        loss_val = float(loss)
        if i == 0:
            initial_loss = loss_val
        if verbose and (i % 100 == 0 or i == training_steps - 1):
            print(f"  step {i:4d}/{training_steps}  loss={loss_val:.6f}  "
                  f"(N={N} sub-windows)")

    return model, {
        'initial_loss':  initial_loss,
        'final_loss':    float(loss),
        'n_sub_windows': N,
    }
