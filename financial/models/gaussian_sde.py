import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random
import jax.tree_util as jtu
from jax.lax import scan
import equinox as eqx

from models.components import (
    NeuralNetFunction,
    DiagonalScaleFactor,
    CholeskyDiffusion,
)


class GaussianSDEFinancial(eqx.Module):
    """Gaussian SDE for financial data with MLP drift.

    Identical to GaussianDrivenSDE except the drift is a NeuralNetFunction
    (MLP over time and state) rather than an OUDiagonalLinearFunction.
    """
    sigma: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    trainable_drift: bool = eqx.field(static=True)
    use_gradient_checkpointing: bool = eqx.field(static=True)

    drift: eqx.Module
    diffusion: eqx.Module
    control: eqx.Module

    def __init__(
        self,
        state_dim,
        sigma,
        drift_seed,
        diffusion_seed,
        control_seed,
        control_width,
        control_depth,
        n_time_features=0,
        period=1.0,
        trainable_drift=True,
        drift_width=64,
        drift_depth=2,
        use_gradient_checkpointing=False,
        **kwargs
    ):
        self.state_dim = state_dim
        self.sigma = sigma
        self.trainable_drift = trainable_drift
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # MLP drift: f(t, x) — no parametric assumption on drift shape
        self.drift = NeuralNetFunction(
            width=drift_width,
            depth=drift_depth,
            n_time_features=n_time_features,
            state_dim=state_dim,
            period=period,
            seed=drift_seed,
        )

        # Diffusion: constant diagonal scale
        diffusion_kwargs = {}
        if 'initial_diffusion_weight' in kwargs:
            diffusion_kwargs['initial_weight'] = kwargs['initial_diffusion_weight']

        self.diffusion = DiagonalScaleFactor(
            state_dim=state_dim,
            seed=diffusion_seed,
            **diffusion_kwargs
        )

        # Control network (unchanged from base model)
        self.control = NeuralNetFunction(
            width=control_width,
            depth=control_depth,
            n_time_features=n_time_features,
            period=period,
            seed=control_seed,
            state_dim=state_dim,
        )

    # -------------------------------------------------------------------------
    # All methods below are identical to GaussianDrivenSDE
    # -------------------------------------------------------------------------

    @eqx.filter_jit
    def simulate_prior(self, state_init, time_sequence, key):
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
        state, path = scan(simulate_forward, init=(state_init, time_init, subkey), xs=time_sequence)
        return path

    @eqx.filter_jit
    def simulate_posterior(self, state_init, time_sequence, key):
        def simulate_forward(state, time):
            X, past_time, key = state
            dt = time - past_time

            key, subkey = random.split(key=key, num=2)
            gaussian_noise = random.normal(subkey, shape=(self.state_dim,))
            scaled_noise = self.sigma * jnp.sqrt(dt) * gaussian_noise

            posterior_drift = self.drift(time, X) - jnp.power(self.diffusion(time, X), 2) * self.control(time, X)
            X = X + posterior_drift * dt + self.diffusion(time, X) * scaled_noise
            return (X, time, key), X

        key, subkey = random.split(key=key, num=2)
        time_init = time_sequence[0]
        state, path = scan(simulate_forward, init=(state_init, time_init, subkey), xs=time_sequence)
        return path

    @eqx.filter_jit
    def simulate_posterior_and_loss(self, state_init, time_sequence, key):
        def simulate_forward(state, time):
            X, past_time, key, kl_accum = state
            dt = time - past_time

            key, subkey = random.split(key, 2)
            gaussian_noise = random.normal(subkey, shape=(self.state_dim,))
            scaled_noise = self.sigma * jnp.sqrt(dt) * gaussian_noise

            control_val = self.control(time, X)
            posterior_drift = self.drift(time, X) - jnp.power(self.diffusion(time, X), 2) * control_val
            X = X + posterior_drift * dt + self.diffusion(time, X) * scaled_noise

            diffusion_val = self.diffusion(time, X)
            kl_contrib = 0.5 * jnp.sum(jnp.power(diffusion_val, 2) * jnp.power(control_val, 2)) * dt
            kl_accum = kl_accum + kl_contrib

            return (X, time, key, kl_accum), (X, kl_contrib)

        key, subkey = random.split(key, 2)
        time_init = time_sequence[0]
        scan_fn = (
            jax.checkpoint(simulate_forward)
            if self.use_gradient_checkpointing
            else simulate_forward
        )
        (X_final, _, _, kl_total), (path, kl_terms) = scan(
            scan_fn,
            init=(state_init, time_init, subkey, 0.0),
            xs=time_sequence
        )
        return path, kl_terms, kl_total


class GaussianSDEFinancialMultivariate(eqx.Module):
    """Multivariate Gaussian SDE with learnable full-covariance diffusion L(t, x).

    Extends GaussianSDEFinancial by replacing the fixed diagonal DiagonalScaleFactor
    with a CholeskyDiffusion that outputs a time- and state-dependent lower-triangular
    matrix L(t, x).  The diffusion covariance Σ(t, x) = L(t, x) Lᵀ(t, x) is always SPD.

    SDE (under posterior measure Q):
        dX = [f(t,X) − σ² L(t,X) Lᵀ(t,X) u(t,X)] dt + σ L(t,X) dW^Q

    KL divergence contribution per time step dt:
        ½ σ² ‖Lᵀ(t,X) u(t,X)‖² dt

    At initialisation L(t,X) ≈ I so behaviour matches GaussianSDEFinancial
    with frozen unit-diagonal diffusion.
    """
    sigma: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    trainable_drift: bool = eqx.field(static=True)
    use_gradient_checkpointing: bool = eqx.field(static=True)

    drift: eqx.Module
    diffusion: CholeskyDiffusion
    control: eqx.Module

    def __init__(
        self,
        state_dim,
        sigma,
        drift_seed,
        diffusion_seed,
        control_seed,
        control_width,
        control_depth,
        cholesky_width=32,
        cholesky_depth=2,
        n_time_features=0,
        period=1.0,
        trainable_drift=True,
        drift_width=64,
        drift_depth=2,
        use_gradient_checkpointing=False,
        **kwargs
    ):
        self.state_dim = state_dim
        self.sigma = sigma
        self.trainable_drift = trainable_drift
        self.use_gradient_checkpointing = use_gradient_checkpointing

        n_time_features = kwargs.get('n_time_features', n_time_features)
        period = kwargs.get('period', period)

        self.drift = NeuralNetFunction(
            width=drift_width,
            depth=drift_depth,
            n_time_features=n_time_features,
            state_dim=state_dim,
            period=period,
            seed=drift_seed,
        )

        self.diffusion = CholeskyDiffusion(
            state_dim=state_dim,
            width=cholesky_width,
            depth=cholesky_depth,
            n_time_features=n_time_features,
            period=period,
            seed=diffusion_seed,
        )

        self.control = NeuralNetFunction(
            width=control_width,
            depth=control_depth,
            n_time_features=n_time_features,
            period=period,
            seed=control_seed,
            state_dim=state_dim,
        )

    @eqx.filter_jit
    def simulate_prior(self, state_init, time_sequence, key):
        def simulate_forward(state, time):
            X, past_time, key = state
            dt = time - past_time

            key, subkey = random.split(key, 2)
            gaussian_noise = random.normal(subkey, shape=(self.state_dim,))

            L = self.diffusion(time, X)  # (D, D)
            X = X + self.drift(time, X) * dt + self.sigma * jnp.sqrt(dt) * (L @ gaussian_noise)
            return (X, time, key), X

        key, subkey = random.split(key, 2)
        time_init = time_sequence[0]
        state, path = scan(simulate_forward, init=(state_init, time_init, subkey), xs=time_sequence)
        return path

    @eqx.filter_jit
    def simulate_posterior(self, state_init, time_sequence, key):
        def simulate_forward(state, time):
            X, past_time, key = state
            dt = time - past_time

            key, subkey = random.split(key, 2)
            gaussian_noise = random.normal(subkey, shape=(self.state_dim,))

            L = self.diffusion(time, X)           # (D, D)
            control_val = self.control(time, X)   # (D,)
            Lt_u = L.T @ control_val              # (D,)
            posterior_drift = self.drift(time, X) - self.sigma ** 2 * (L @ Lt_u)
            X = X + posterior_drift * dt + self.sigma * jnp.sqrt(dt) * (L @ gaussian_noise)
            return (X, time, key), X

        key, subkey = random.split(key, 2)
        time_init = time_sequence[0]
        state, path = scan(simulate_forward, init=(state_init, time_init, subkey), xs=time_sequence)
        return path

    @eqx.filter_jit
    def simulate_posterior_and_loss(self, state_init, time_sequence, key):
        def simulate_forward(state, time):
            X, past_time, key, kl_accum = state
            dt = time - past_time

            key, subkey = random.split(key, 2)
            gaussian_noise = random.normal(subkey, shape=(self.state_dim,))

            L = self.diffusion(time, X)           # (D, D)
            control_val = self.control(time, X)   # (D,)
            Lt_u = L.T @ control_val              # (D,)
            posterior_drift = self.drift(time, X) - self.sigma ** 2 * (L @ Lt_u)
            X = X + posterior_drift * dt + self.sigma * jnp.sqrt(dt) * (L @ gaussian_noise)

            kl_contrib = 0.5 * self.sigma ** 2 * jnp.dot(Lt_u, Lt_u) * dt
            kl_accum = kl_accum + kl_contrib

            return (X, time, key, kl_accum), (X, kl_contrib)

        key, subkey = random.split(key, 2)
        time_init = time_sequence[0]
        scan_fn = (
            jax.checkpoint(simulate_forward)
            if self.use_gradient_checkpointing
            else simulate_forward
        )
        (X_final, _, _, kl_total), (path, kl_terms) = scan(
            scan_fn,
            init=(state_init, time_init, subkey, 0.0),
            xs=time_sequence
        )
        return path, kl_terms, kl_total
