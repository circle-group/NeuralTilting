"""Train a Neural Jump SDE model on a single rolling window."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import equinox as eqx

from financial.models.neural_jump_sde import NeuralJumpSDE


def train_njsde(
    train_obs: jnp.ndarray,
    dt: float,
    dim_c: int,
    dim_h: int,
    n_mixtures: int,
    hidden_width: int,
    hidden_depth: int,
    ortho: bool,
    n_ode_steps: int,
    learning_rate: float,
    training_steps: int,
    key: jax.Array,
    weight_decay: float = 0.0,
    verbose: bool = False,
) -> tuple:
    """Train a NeuralJumpSDE on one window.

    Parameters
    ----------
    train_obs : (T, obs_dim)
        Normalised log-prices.  obs_dim=1 for univariate, >1 for multivariate.
    dt : float
        Time step between observations (in normalised time).
    dim_c, dim_h : int
        Internal state and memory dimensions.
    n_mixtures : int
        Number of Gaussian mixture components.
    hidden_width, hidden_depth : int
        MLP width and depth (hidden layers) for all four networks.
    ortho : bool
        Whether to enforce the sphere constraint on c(t).
    n_ode_steps : int
        RK4 substeps per observation interval.
    learning_rate : float
    training_steps : int
    key : jax.Array
    verbose : bool

    Returns
    -------
    model : NeuralJumpSDE
    metrics : dict
    """
    obs_dim = train_obs.shape[1]
    model = NeuralJumpSDE(
        dim_c=dim_c,
        dim_h=dim_h,
        n_mixtures=n_mixtures,
        hidden_width=hidden_width,
        hidden_depth=hidden_depth,
        ortho=ortho,
        obs_dim=obs_dim,
        key=key,
    )

    if weight_decay > 0.0:
        optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    T = train_obs.shape[0]

    @eqx.filter_jit
    def step(model, opt_state):
        def loss_fn(model):
            # Negative log-likelihood normalised by sequence length
            log_lik = model.training_log_likelihood(train_obs, dt, n_ode_steps)
            return -log_lik / (T - 1)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss

    initial_loss = None
    for i in range(training_steps):
        model, opt_state, loss = step(model, opt_state)
        loss_val = float(loss)
        if i == 0:
            initial_loss = loss_val
        if verbose and (i % 100 == 0 or i == training_steps - 1):
            print(f"  step {i:4d}/{training_steps}  nll={loss_val:.6f}")

    return model, {
        'initial_loss': initial_loss,
        'final_loss': float(loss),
        'n_train_obs': T,
    }
