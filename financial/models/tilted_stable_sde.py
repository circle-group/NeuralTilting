import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
from jax.lax import scan
import equinox as eqx

from models.components import (
    NeuralNetFunction,
    DiagonalScaleFactor,
    QuadraticNeuralPotential,
)


class TiltedStableSDEFinancial(eqx.Module):
    """Tilted Stable SDE for financial data with MLP drift.

    Identical to TiltedStableDrivenSDE except the drift is a NeuralNetFunction
    (MLP over time and state) rather than an OUDiagonalLinearFunction. This is
    appropriate for financial data where we do not assume a specific parametric
    drift form.
    """
    tau: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    sigma: float = eqx.field(static=True)
    loss_sample_size: int = eqx.field(static=True)
    max_rejection_attempts: int = eqx.field(static=True)
    max_jumps: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    trainable_drift: bool = eqx.field(static=True)
    use_gradient_checkpointing: bool = eqx.field(static=True)

    drift: eqx.Module
    diffusion: eqx.Module
    phi: eqx.Module

    def __init__(
        self,
        state_dim,
        alpha,
        tau,
        sigma,
        loss_sample_size,
        max_rejection_attempts,
        max_jumps,
        tilting_width,
        tilting_depth,
        drift_seed,
        diffusion_seed,
        phi_seed,
        drift_width=64,
        drift_depth=2,
        trainable_drift=True,
        use_gradient_checkpointing=False,
        **kwargs
    ):
        self.state_dim = state_dim
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma
        self.loss_sample_size = loss_sample_size
        self.max_rejection_attempts = max_rejection_attempts
        self.max_jumps = max_jumps
        self.trainable_drift = trainable_drift
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Extract shared time-encoding params used by both drift and phi
        n_time_features = kwargs.get('n_time_features', 0)
        period = kwargs.get('period', 1.0)

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
            diffusion_kwargs['initial_weight'] = kwargs.pop('initial_diffusion_weight')

        self.diffusion = DiagonalScaleFactor(
            state_dim=state_dim,
            seed=diffusion_seed,
            **diffusion_kwargs
        )

        # Tilting potential (unchanged from base model)
        phi_params = [
            'n_time_features',
            'period',
            'a_min',
            'n_attention_references',
            'attention_embed_dim',
            'attention_sharpness',
            'use_adaptive_scaling',
        ]
        phi_kwargs = {p: kwargs[p] for p in phi_params if p in kwargs}

        self.phi = QuadraticNeuralPotential(
            width=tilting_width,
            depth=tilting_depth,
            state_dim=state_dim,
            alpha=alpha,
            tau=tau,
            sigma=sigma,
            seed=phi_seed,
            **phi_kwargs
        )

        remaining = {k: v for k, v in kwargs.items()
                     if k not in phi_params and k != 'initial_diffusion_weight'}
        if remaining:
            warnings.warn(f"Unused kwargs: {list(remaining.keys())}")

    # -------------------------------------------------------------------------
    # All methods below are identical to TiltedStableDrivenSDE
    # -------------------------------------------------------------------------

    def generate_truncated_stable_jumps(self, key, shape=()):
        key, subkey = random.split(key, num=2)
        u = random.uniform(subkey, shape=shape)
        return self.tau * jnp.power(1 - u, -1 / self.alpha)

    def total_prior_jump_rate(self, delta_t):
        return (1 / self.alpha) * jnp.power(self.tau, -self.alpha) * delta_t

    @eqx.filter_jit
    def simulate_prior(self, state_init, time_sequence, key):
        def simulate_forward(state, time):
            X, past_time, key = state
            delta_t = time - past_time
            past_time = time

            key, subkey1, subkey2, subkey3 = random.split(key, num=4)
            total_rate = 2 * self.total_prior_jump_rate(delta_t)
            n_jumps = random.poisson(subkey1, lam=total_rate, shape=(self.state_dim,))
            stable_jumps = self.generate_truncated_stable_jumps(subkey2, shape=(self.max_jumps, self.state_dim))
            gaussian_noise = random.normal(subkey3, (self.max_jumps, self.state_dim))
            random_jumps = stable_jumps * self.sigma * gaussian_noise

            mask = jnp.arange(self.max_jumps)[:, None] < n_jumps[None, :]
            sum_of_jumps = jnp.sum(random_jumps * mask, axis=0)

            X = X + self.drift(time, X) * delta_t + self.diffusion(time, X) * sum_of_jumps
            return (X, past_time, key), X

        key, subkey = random.split(key=key, num=2)
        time_init = time_sequence[0]
        state, path = scan(simulate_forward, init=(state_init, time_init, subkey), xs=time_sequence)
        return path

    def compute_envelope_bound(self, At, Bt, state):
        K1 = 2.0 * At * state + Bt
        M = jnp.exp(-K1 ** 2 / (4.0 * At))
        return M

    def compute_tilting_factor(self, r, At, Bt, state):
        r_sq = r ** 2
        K1 = 2.0 * At * state + Bt
        denom = 1.0 - 2.0 * At * r_sq * self.sigma ** 2
        eps = 1e-6
        denom_safe = jnp.maximum(denom, eps)
        sqrt_term = 1.0 / jnp.sqrt(denom_safe)
        exp_numerator = (K1 ** 2 * r_sq * self.sigma ** 2) / (2.0 * denom_safe)
        exp_term = jnp.exp(exp_numerator)
        return sqrt_term * exp_term

    def generate_mixing_variable_rejection(self, time, state, At, Bt, n_samples, key):
        M = self.compute_envelope_bound(At, Bt, state)

        def sample_one(subkey):
            def attempt_sample(carry, _):
                r_current, accepted, subkey = carry
                subkey, prop_key, accept_key = random.split(subkey, 3)
                r_prop = self.generate_truncated_stable_jumps(prop_key, shape=(self.state_dim,))
                C = self.compute_tilting_factor(r_prop, At, Bt, state)
                accept_prob = jnp.minimum(C / M, 1.0)
                u = random.uniform(accept_key, shape=(self.state_dim,))
                newly_accepted = u < accept_prob
                should_update = newly_accepted & (~accepted)
                r_updated = jnp.where(should_update, r_prop, r_current)
                accepted_updated = accepted | newly_accepted
                return (r_updated, accepted_updated, subkey), None

            init_r = jnp.ones(self.state_dim) * self.tau
            init_accepted = jnp.zeros(self.state_dim, dtype=bool)
            (r_final, accepted_final, _), _ = scan(
                attempt_sample,
                init=(init_r, init_accepted, subkey),
                xs=None,
                length=self.max_rejection_attempts
            )
            return r_final, jnp.mean(accepted_final.astype(float))

        keys = random.split(key, n_samples)
        samples, acceptance_rates = vmap(sample_one)(keys)
        return samples, jnp.mean(acceptance_rates)

    def generate_conditional_Gaussian(self, time, state, At, Bt, mixing_batch, key):
        state = jnp.atleast_1d(state)
        mixing_batch = jnp.atleast_2d(mixing_batch)
        n_samples = mixing_batch.shape[0]

        A_expanded = At[None, :]
        B_expanded = Bt[None, :]
        state_expanded = state[None, :]

        K1 = 2 * A_expanded * state_expanded + B_expanded
        K2 = A_expanded - 1 / (2 * jnp.power(mixing_batch, 2) * jnp.power(self.sigma, 2))

        eps = 1e-6
        K2_safe = jnp.where(K2 < -eps, K2, -eps)

        mean = -K1 / (2 * K2_safe)
        var = -1 / (2 * K2_safe)

        noise = random.normal(key, shape=(n_samples, self.state_dim))
        return mean + jnp.sqrt(var) * noise

    @eqx.filter_jit
    def simulate_posterior_and_loss(self, state_init, time_sequence, key):
        def simulate_forward(state, time):
            X, past_time, key, kl_accum = state
            dt = time - past_time

            key, subkey = random.split(key=key, num=2)
            raw_stable_jumps = self.generate_truncated_stable_jumps(subkey, shape=(self.loss_sample_size, self.state_dim))
            scaled_stable_jumps = self.diffusion(time, X) * raw_stable_jumps
            normalisation = self.total_prior_jump_rate(1.)

            A_t, B_t = self.phi.get_coefficients(time)
            log_Ht = A_t * scaled_stable_jumps ** 2 + (2 * A_t * X + B_t) * scaled_stable_jumps
            negative_log_Ht = A_t * scaled_stable_jumps ** 2 - (2 * A_t * X + B_t) * scaled_stable_jumps

            max_exp_arg = 80.0
            log_Ht = max_exp_arg * jnp.tanh(log_Ht / max_exp_arg)
            negative_log_Ht = max_exp_arg * jnp.tanh(negative_log_Ht / max_exp_arg)

            f_loss = (jnp.exp(log_Ht) * (log_Ht - 1) + 1
                      + jnp.exp(negative_log_Ht) * (negative_log_Ht - 1) + 1)

            per_sample_loss = jnp.sum(f_loss, axis=1)
            loss_quantile = jnp.nanquantile(per_sample_loss, 0.95)
            clipped_loss = jnp.minimum(per_sample_loss, loss_quantile)
            loss_estimate = dt * normalisation * jnp.nanmean(clipped_loss)
            kl_accum = kl_accum + loss_estimate

            key, subkey1, subkey2, subkey3 = random.split(key, num=4)
            tilting_weights = jnp.exp(log_Ht) + jnp.exp(negative_log_Ht)
            tilt_quantile = jnp.nanquantile(tilting_weights, 0.95, axis=0)
            clipped_tilt = jnp.minimum(tilting_weights, tilt_quantile[None, :])
            tilting_factor = jnp.nanmean(clipped_tilt, axis=0)
            total_rate = tilting_factor * self.total_prior_jump_rate(dt)
            n_jumps = random.poisson(subkey1, lam=total_rate)

            mixing_variable_sample, _ = self.generate_mixing_variable_rejection(
                time, X, A_t, B_t, self.max_jumps, subkey2)
            cond_Gaussian_sample = self.generate_conditional_Gaussian(
                time, X, A_t, B_t, mixing_variable_sample, subkey3)

            mask = jnp.arange(self.max_jumps)[:, None] < n_jumps[None, :]
            sum_of_jumps = jnp.nansum(cond_Gaussian_sample * mask, axis=0)
            clip_scale = 50.0 * self.sigma
            sum_of_jumps = clip_scale * jnp.tanh(sum_of_jumps / clip_scale)
            effective_sum_of_jumps = self.diffusion(time, X) * sum_of_jumps

            X = X + self.drift(time, X) * dt + effective_sum_of_jumps
            return (X, time, key, kl_accum), (X, loss_estimate)

        key, subkey = random.split(key=key, num=2)
        time_init = time_sequence[0]
        scan_fn = (
            jax.checkpoint(simulate_forward)
            if self.use_gradient_checkpointing
            else simulate_forward
        )
        (X_final, _, _, kl_total), (sde_path, loss_estimates) = scan(
            scan_fn,
            init=(state_init, time_init, subkey, 0.0),
            xs=time_sequence
        )
        return sde_path, loss_estimates, kl_total

    @eqx.filter_jit
    def simulate_posterior(self, state_init, time_sequence, key):
        def simulate_forward(state, time):
            X, past_time, key = state
            dt = time - past_time

            key, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)

            raw_stable_jumps = self.generate_truncated_stable_jumps(subkey1, shape=(self.loss_sample_size, self.state_dim))
            scaled_stable_jumps = self.diffusion(time, X) * raw_stable_jumps

            A_t, B_t = self.phi.get_coefficients(time)
            log_Ht = A_t * scaled_stable_jumps ** 2 + (2 * A_t * X + B_t) * scaled_stable_jumps
            negative_log_Ht = A_t * scaled_stable_jumps ** 2 - (2 * A_t * X + B_t) * scaled_stable_jumps

            max_exp_arg = 80.0
            log_Ht = max_exp_arg * jnp.tanh(log_Ht / max_exp_arg)
            negative_log_Ht = max_exp_arg * jnp.tanh(negative_log_Ht / max_exp_arg)

            tilting_weights = jnp.exp(log_Ht) + jnp.exp(negative_log_Ht)
            tilt_quantile = jnp.nanquantile(tilting_weights, 0.95, axis=0)
            clipped_tilt = jnp.minimum(tilting_weights, tilt_quantile[None, :])
            tilting_factor = jnp.nanmean(clipped_tilt, axis=0)
            total_rate = tilting_factor * self.total_prior_jump_rate(dt)
            n_jumps = random.poisson(subkey2, lam=total_rate)

            mixing_variable_sample, _ = self.generate_mixing_variable_rejection(
                time, X, A_t, B_t, n_samples=self.max_jumps, key=subkey3)
            cond_Gaussian_sample = self.generate_conditional_Gaussian(
                time, X, A_t, B_t, mixing_variable_sample, subkey4)

            mask = jnp.arange(self.max_jumps)[:, None] < n_jumps[None, :]
            sum_of_jumps = jnp.nansum(cond_Gaussian_sample * mask, axis=0)
            clip_scale = 50.0 * self.sigma
            sum_of_jumps = clip_scale * jnp.tanh(sum_of_jumps / clip_scale)
            effective_sum_of_jumps = self.diffusion(time, X) * sum_of_jumps

            X = X + self.drift(time, X) * dt + effective_sum_of_jumps
            return (X, time, key), X

        key, subkey = random.split(key=key, num=2)
        time_init = time_sequence[0]
        state, sde_path = scan(simulate_forward, init=(state_init, time_init, subkey), xs=time_sequence)
        return sde_path
