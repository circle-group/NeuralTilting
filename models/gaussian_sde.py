# Third-party imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap, grad
from jax.lax import scan, stop_gradient
import equinox as eqx
import jax.nn as nn

# Library-specific imports
from models.components import (
    DiagonalLinearFunction,
    OUDiagonalLinearFunction,
    NeuralNetFunction,
    DiagonalScaleFactor,
)

class GaussianDrivenSDE(eqx.Module):
    sigma : float = eqx.field(static=True)
    state_dim : int = eqx.field(static=True)
    trainable_drift : bool = eqx.field(static=True)
    drift : eqx.Module
    diffusion : eqx.Module
    control : eqx.Module

    def __init__(self, state_dim, sigma, drift_seed, diffusion_seed, control_seed, control_width, control_depth, n_time_features, period=1.0, trainable_drift=True, **kwargs):
        self.state_dim = state_dim
        self.sigma = sigma
        self.trainable_drift = trainable_drift

        # Extract drift-specific kwargs
        drift_kwargs = {}
        if 'initial_drift_weight' in kwargs:
            drift_kwargs['initial_weight'] = kwargs.pop('initial_drift_weight')
        if 'initial_drift_bias' in kwargs:
            drift_kwargs['initial_bias'] = kwargs.pop('initial_drift_bias')

        self.drift = OUDiagonalLinearFunction(
            state_dim=state_dim,
            seed=drift_seed,
            **drift_kwargs
        )

        # Extract diffusion-specific kwargs
        diffusion_kwargs = {}
        if 'initial_diffusion_weight' in kwargs:
            diffusion_kwargs['initial_weight'] = kwargs.pop('initial_diffusion_weight')

        # Set the noise scale function to constant
        self.diffusion = DiagonalScaleFactor(
            state_dim=state_dim,
            seed=diffusion_seed,
            **diffusion_kwargs
        )

        self.control = NeuralNetFunction(
            width=control_width,
            depth=control_depth,
            n_time_features=n_time_features,
            period=period,
            seed=control_seed,
            state_dim=state_dim
        )

    @eqx.filter_jit
    def simulate_prior(self, state_init, time_sequence, key):
        """Simulate prior SDE: dX_t = drift(X_t)dt + diffusion(X_t)dW_t"""
        def simulate_forward(state, time):
            X, past_time, key = state
            dt = time - past_time

            key, subkey = random.split(key=key, num=2)
            gaussian_noise = random.normal(subkey, shape=(self.state_dim,))
            scaled_noise = self.sigma * jnp.sqrt(dt) * gaussian_noise

            X = X + self.drift(time, X) * dt + self.diffusion(time, X) * scaled_noise

            return (X, time, key), X

        key, subkey = random.split(key=key, num=2)
        time_init = time_sequence[0]

        # Generate SDE path using the Euler-Maruyama method
        state, path = scan(simulate_forward, init=(state_init, time_init, subkey), xs=time_sequence)
        return path

    @eqx.filter_jit
    def simulate_posterior(self, state_init, time_sequence, key):
        """Simulate posterior SDE with control: dX_t = (drift(X_t) - diffusion(X_t)^2 * control(t,X_t))dt + diffusion(X_t)dW_t"""
        def simulate_forward(state, time):
            X, past_time, key = state
            dt = time - past_time

            key, subkey = random.split(key=key, num=2)
            gaussian_noise = random.normal(subkey, shape=(self.state_dim,))
            scaled_noise = self.sigma * jnp.sqrt(dt) * gaussian_noise

            # Posterior drift includes control term
            posterior_drift = self.drift(time, X) - jnp.power(self.diffusion(time, X), 2) * self.control(time, X)
            X = X + posterior_drift * dt + self.diffusion(time, X) * scaled_noise

            return (X, time, key), X

        key, subkey = random.split(key=key, num=2)
        time_init = time_sequence[0]

        # Generate SDE path using the Euler-Maruyama method
        state, path = scan(simulate_forward, init=(state_init, time_init, subkey), xs=time_sequence)
        return path

    @eqx.filter_jit
    def simulate_posterior_and_loss(self, state_init, time_sequence, key):
        """Simulate posterior SDE and compute KL divergence loss"""
        def simulate_forward(state, time):
            X, past_time, key, kl_accum = state
            dt = time - past_time

            key, subkey = random.split(key, 2)
            gaussian_noise = random.normal(subkey, shape=(self.state_dim,))
            scaled_noise = self.sigma * jnp.sqrt(dt) * gaussian_noise

            # Compute control value
            control_val = self.control(time, X)

            # Posterior drift includes control term
            posterior_drift = self.drift(time, X) - jnp.power(self.diffusion(time, X), 2) * control_val

            # Euler-Maruyama update
            X = X + posterior_drift * dt + self.diffusion(time, X) * scaled_noise

            # KL divergence contribution: ∫ (1/2) * diffusion(X_t)^2 * control(t,X_t)^2 dt
            diffusion_val = self.diffusion(time, X)
            kl_contrib = 0.5 * jnp.sum(jnp.power(diffusion_val, 2) * jnp.power(control_val, 2)) * dt
            kl_accum = kl_accum + kl_contrib

            return (X, time, key, kl_accum), (X, kl_contrib)

        key, subkey = random.split(key, 2)
        time_init = time_sequence[0]

        # Run scan
        (X_final, _, _, kl_total), (path, kl_terms) = scan(
            simulate_forward,
            init=(state_init, time_init, subkey, 0.0),
            xs=time_sequence
        )

        return path, kl_terms, kl_total