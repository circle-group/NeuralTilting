"""Generate predictive samples by extending the posterior beyond the training window."""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np


def generate_forecast_samples(
    model,
    window: dict,
    n_samples: int,
    key,
    n_latent_steps: int = 1000,
    use_posterior_for_forecast: bool = False,
) -> np.ndarray:
    """Generate forecast samples for the forecast horizon.

    Algorithm
    ---------
    1. Draw n_samples posterior paths on the training window.
    2. Take x_T = path[-1] from each sample (final state distribution).
    3. For each x_T, simulate the SDE forward over the forecast times using a
       fine latent grid that matches the training step size (~0.001 model
       time). The grid is built automatically so that dt ≈ T_train /
       n_latent_steps regardless of the forecast horizon length.
    4. Return the forecast paths at the observation times.

    Step 3 uses the PRIOR SDE by default (``use_posterior_for_forecast=False``).
    Setting ``use_posterior_for_forecast=True`` uses ``simulate_posterior``
    instead for both model types.  The rationale: beyond the training window,
    the posterior (with no new observations to condition on) degenerates to the
    unconditional tilted/controlled prior predictive, keeping the drift in the
    same context it was trained in.  Both models are treated symmetrically.

    Parameters
    ----------
    model : TiltedStableSDEFinancial or GaussianSDEFinancial
    window : dict
        As produced by prepare_windows.py. Must contain train_times,
        train_returns, and forecast_times.
    n_samples : int
        Number of forecast sample paths to generate.
    key : jax.random.PRNGKey
    n_latent_steps : int
        Number of latent steps used during training. Used to derive the
        target step size: dt_target = train_times[-1] / n_latent_steps.
        The forecast grid uses ceil(forecast_duration / dt_target) steps
        so the discretisation is consistent with training.
    use_posterior_for_forecast : bool
        If True, use simulate_posterior (tilted stable process) for the
        forecast extension instead of simulate_prior (untilted).  Default
        False for backward compatibility.

    Returns
    -------
    forecast_samples : np.ndarray
        Shape (n_samples, H, D) where H = len(forecast_times), D = state_dim.
    """
    train_times = jnp.array(window['train_times'])        # (T,)
    train_returns = jnp.array(window['train_returns'])    # (T, D)
    forecast_times = jnp.array(window['forecast_times'])  # (H,)

    state_dim = train_returns.shape[1]
    state_init = jnp.zeros(state_dim)

    # -------------------------------------------------------------------------
    # Step 1: posterior paths on training window → x_T samples
    # -------------------------------------------------------------------------
    key, *posterior_keys = random.split(key, n_samples + 1)
    posterior_keys = jnp.array(posterior_keys)

    posterior_paths = jax.vmap(
        lambda k: model.simulate_posterior(state_init, train_times, k)
    )(posterior_keys)
    # posterior_paths: (n_samples, T, D)

    x_T_samples = posterior_paths[:, -1, :]  # (n_samples, D)

    # -------------------------------------------------------------------------
    # Step 2: build fine forecast grid matching training step size
    #
    # Training step size: dt_target = T_train / n_latent_steps  (~0.001)
    # Forecast latent steps: ceil(forecast_duration / dt_target)
    # The fine grid is merged with the actual observation times so we can
    # extract predictions at exactly those times without interpolation.
    # -------------------------------------------------------------------------
    last_train_time = float(train_times[-1])
    T_train = last_train_time
    dt_target = T_train / n_latent_steps

    forecast_end = float(forecast_times[-1])
    forecast_duration = forecast_end - last_train_time
    n_forecast_latent = max(int(np.ceil(forecast_duration / dt_target)), 1)

    fine_grid = np.linspace(last_train_time, forecast_end, n_forecast_latent + 1)
    obs_times_np = np.array(forecast_times)
    merged = np.unique(np.concatenate([fine_grid, obs_times_np]))
    # merged[0] == last_train_time; simulate_prior scans over all elements,
    # with delta_t=0 on the first element (x_T is reproduced unchanged).

    obs_indices = np.searchsorted(merged, obs_times_np)  # indices into merged path

    merged_jax = jnp.array(merged)

    # -------------------------------------------------------------------------
    # Step 3: simulate from x_T on fine merged grid.
    # Default (use_posterior_for_forecast=False): simulate_prior — the
    # untilted/no-control process.  This is the current baseline.
    # Alternative (True): simulate_posterior — applies the learned
    # tilting/control in the forecast region.  The rationale is that, beyond
    # the training window with no new observations, the posterior degenerates
    # to the "unconditional tilted/controlled prior predictive", keeping the
    # drift operating in the same context it was trained in.  Both model types
    # are treated symmetrically.
    # -------------------------------------------------------------------------
    key, *forecast_keys = random.split(key, n_samples + 1)
    forecast_keys = jnp.array(forecast_keys)

    simulate_fn = model.simulate_posterior if use_posterior_for_forecast else model.simulate_prior

    full_paths = jax.vmap(
        lambda x_T, k: simulate_fn(x_T, merged_jax, k)
    )(x_T_samples, forecast_keys)
    # full_paths: (n_samples, len(merged), D)

    # Extract only the forecast observation times
    forecast_paths = full_paths[:, obs_indices, :]
    # forecast_paths: (n_samples, H, D)

    return np.array(forecast_paths)
