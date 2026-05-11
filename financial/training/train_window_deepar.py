"""Train a DeepAR model on a single rolling window.

DeepAR is trained with teacher-forcing: the LSTM receives the shifted
observation sequence (input[t] = obs[t-1], obs[-1] = 0) and outputs
Normal-distribution parameters (loc, scale) at every step.  The training
loss is the mean negative log-likelihood across all T time steps.

At inference time the model encodes the full training window and then
autoregressively samples K independent forecast trajectories of length H.
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from financial.models.deepar import DeepARFinancial


def train_deepar(
    train_obs:             jnp.ndarray,
    forecast_obs:          jnp.ndarray,
    lstm_hidden_size:      int,
    lstm_n_layers:         int,
    decoder_hidden_layers: int,
    decoder_hidden_size:   int,
    learning_rate:         float,
    training_steps:        int,
    key:                   jax.Array,
    verbose:               bool = False,
) -> tuple:
    """Train DeepAR on one rolling window.

    Parameters
    ----------
    train_obs : (T, D)
        Normalised training observations (cumulative log-prices).
    forecast_obs : (H, D)
        Normalised forecast observations.  Shape is used only to extract
        pred_len H; the values are not used during training.
    lstm_hidden_size : int
    lstm_n_layers : int
    decoder_hidden_layers : int
        0 → linear decoder; >0 → MLP with that many hidden layers.
    decoder_hidden_size : int
        Width of the decoder MLP (only used when decoder_hidden_layers > 0).
    learning_rate : float
    training_steps : int
    key : jax.Array
    verbose : bool

    Returns
    -------
    model : DeepARFinancial
    metrics : dict
        keys: initial_loss, final_loss, n_train_obs
    """
    T, D = train_obs.shape
    H    = forecast_obs.shape[0]

    model = DeepARFinancial(
        obs_dim=D,
        pred_len=H,
        lstm_hidden_size=lstm_hidden_size,
        lstm_n_layers=lstm_n_layers,
        decoder_hidden_layers=decoder_hidden_layers,
        decoder_hidden_size=decoder_hidden_size,
        key=key,
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state):
        def loss_fn(model):
            loc, scale = model(train_obs)   # (T, D), (T, D)
            # Mean NLL under Normal distribution across all steps and dims
            nll = -jax.scipy.stats.norm.logpdf(train_obs, loc, scale)  # (T, D)
            return jnp.mean(nll)

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
            print(f"  step {i:4d}/{training_steps}  loss={loss_val:.6f}")

    return model, {
        'initial_loss': initial_loss,
        'final_loss':   float(loss),
        'n_train_obs':  T,
    }
