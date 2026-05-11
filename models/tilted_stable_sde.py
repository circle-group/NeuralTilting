# Third-party imports
import warnings
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap, grad
from jax.lax import scan, stop_gradient
import equinox as eqx
import jax.nn as nn

# Library-specific imports
from models.components import (
    DiagonalScaleFactor,
    DiagonalLinearFunction,
    OUDiagonalLinearFunction,
    QuadraticNeuralPotential,
)

class TiltedStableDrivenSDE(eqx.Module):
    """Tilted Stable SDE with learned jump measure tilting via temporal attention.

    The tilting is controlled by a QuadraticNeuralPotential that adjusts the
    jump measure of the underlying α-stable Lévy process. The neural potential
    learns time-dependent quadratic coefficients A(t) < 0 and linear coefficients
    B(t) ∈ ℝ using temporal attention to identify important time points (e.g., jumps).

    Key components:
    - Drift: Ornstein-Uhlenbeck mean-reverting dynamics
    - Diffusion: Constant diagonal scale factor
    - Tilting: QuadraticNeuralPotential with:
        * Temporal attention mechanism with learnable reference times
        * Neural MLPs for A(t) and B(t) with Fourier time encoding
        * All parameters learned via gradients on variational bound
    """
    tau : float = eqx.field(static=True)
    alpha : float = eqx.field(static=True)
    sigma : float = eqx.field(static=True)
    loss_sample_size : int = eqx.field(static=True)
    max_rejection_attempts : int = eqx.field(static=True)
    max_jumps : int = eqx.field(static=True)
    state_dim : int = eqx.field(static=True)
    trainable_drift : bool = eqx.field(static=True)

    # Parameters (trainable or static depending on config)
    drift : eqx.Module
    diffusion : eqx.Module
    phi : eqx.Module

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
        trainable_drift=True,
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

        # Extract phi-specific kwargs for QuadraticNeuralPotential
        phi_kwargs = {}

        # Relevant parameters for QuadraticNeuralPotential
        phi_params = [
            'n_time_features',
            'period',
            'a_min',
            'n_attention_references',
            'attention_embed_dim',
            'attention_sharpness',
            'use_adaptive_scaling',
        ]

        for param in phi_params:
            if param in kwargs:
                phi_kwargs[param] = kwargs.pop(param)

        # Initialize tilting potential with temporal attention
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

        # Warn about unused kwargs
        if kwargs:
            warnings.warn(f"Unused kwargs: {list(kwargs.keys())}")

    def generate_truncated_stable_jumps(self, key, shape=()):
        """Generates jump(s) from a truncated stable distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
        shape : tuple, optional
            Shape of output array. Default () generates a scalar jump.

        Returns
        -------
        jnp.ndarray
            Jumps from truncated stable distribution with specified shape.
        """
        key, subkey = random.split(key, num=2)
        u = random.uniform(subkey, shape=shape)
        return self.tau * jnp.power(1-u, -1/self.alpha)

    def total_prior_jump_rate(self, delta_t):
        """This is the total intensity of the non-negative truncated stable density for time interval dt.
        """
        return (1/self.alpha) * jnp.power(self.tau, -self.alpha) * delta_t

    @eqx.filter_jit
    def simulate_prior(self, state_init, time_sequence, key):
        """Returns a column vector that represents the position of the SDE at the given time_sequence points.
        """
        def simulate_forward(state, time):
            X, past_time, key = state
            delta_t = time - past_time
            past_time = time

            key, subkey1, subkey2, subkey3 = random.split(key, num=4)
            total_rate = 2*self.total_prior_jump_rate(delta_t) # The factor of 2 is due to the symmetric nature of the jumps.
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

        # Generate random stable driven SDE path using the Euler method.
        state, path = scan(simulate_forward, init=(state_init, time_init, subkey), xs=time_sequence)
        return path

    def compute_envelope_bound(self, At, Bt, state):
        """Compute the envelope bound M(t, X_t) for rejection sampling.

        M(t, X_t) = exp(-(2*A_t*X_t + B_t)^2 / (4*A_t))

        This bound is derived from the maximum of C(r, t, X_t) over r >= tau.
        Since A_t <= 0, the denominator 4*A_t is negative, ensuring a finite bound.

        Parameters
        ----------
        At : array, shape (state_dim,)
            Quadratic coefficient from phi network (must be negative)
        Bt : array, shape (state_dim,)
            Linear coefficient from phi network
        state : array, shape (state_dim,)
            Current state X_t

        Returns
        -------
        M : array, shape (state_dim,)
            Envelope bound for each dimension
        """
        K1 = 2.0 * At * state + Bt
        # A_t is negative, so we divide by negative number
        M = jnp.exp(-K1**2 / (4.0 * At))
        return M

    def compute_tilting_factor(self, r, At, Bt, state):
        """Compute C(r, t, X_t) for given mixing variables.

        C(r, t, X_t) = 1/sqrt(1 - 2*A_t*r^2*sigma^2) *
                       exp((2*A_t*X_t + B_t)^2 * r^2 * sigma^2 / (2*(1 - 2*A_t*r^2*sigma^2)))

        Parameters
        ----------
        r : array, shape (..., state_dim)
            Mixing variables
        At : array, shape (state_dim,)
            Quadratic coefficient from phi network (must be negative)
        Bt : array, shape (state_dim,)
            Linear coefficient from phi network
        state : array, shape (state_dim,)
            Current state X_t

        Returns
        -------
        C : array, shape (..., state_dim)
            Tilting factor for each mixing variable
        """
        r_sq = r**2
        K1 = 2.0 * At * state + Bt

        # Denominator: 1 - 2*A_t*r^2*sigma^2
        # Since A_t <= 0, this is >= 1, ensuring numerical stability
        denom = 1.0 - 2.0 * At * r_sq * self.sigma**2
        eps = 1e-6
        denom_safe = jnp.maximum(denom, eps)

        # C(r) = 1/sqrt(denom) * exp(K1^2 * r^2 * sigma^2 / (2*denom))
        sqrt_term = 1.0 / jnp.sqrt(denom_safe)
        exp_numerator = (K1**2 * r_sq * self.sigma**2) / (2.0 * denom_safe)
        exp_term = jnp.exp(exp_numerator)

        C = sqrt_term * exp_term
        return C

    def generate_mixing_variable_rejection(self, time, state, At, Bt, n_samples, key):
        """Generate n_samples mixing variables using rejection sampling.

        Uses the truncated stable distribution as proposal q(r) and applies
        exact rejection sampling with envelope bound M(t, X_t).

        Algorithm:
        1. Propose r* ~ truncated_stable(alpha, tau)
        2. Compute acceptance probability a(r*) = C(r*, t, X_t) / M(t, X_t)
        3. Accept with probability a(r*), otherwise reject and retry

        This is exact sampling (no discretization bias) and gradient-free.

        Parameters
        ----------
        time : float
            Current time point (unused, kept for API compatibility)
        state : array, shape (state_dim,)
            Current state X_t
        At : array, shape (state_dim,)
            Quadratic coefficient from phi network (must be negative)
        Bt : array, shape (state_dim,)
            Linear coefficient from phi network
        n_samples : int
            Number of independent samples to generate
        key : PRNGKey
            Random key for sampling

        Returns
        -------
        samples : array, shape (n_samples, state_dim)
            Mixing variable samples from the tilted distribution
        acceptance_rate : float
            Mean acceptance rate across all samples (for diagnostics)
        """
        # Compute envelope bound M(t, X_t) once (same for all samples)
        M = self.compute_envelope_bound(At, Bt, state)  # shape: (state_dim,)

        def sample_one(subkey):
            """Sample one mixing variable using rejection sampling.

            Each sample goes through up to max_rejection_attempts rejection trials.
            We track which dimensions have been accepted and update accordingly.
            """

            def attempt_sample(carry, _):
                """Single rejection sampling attempt for all dimensions."""
                r_current, accepted, subkey = carry
                subkey, prop_key, accept_key = random.split(subkey, 3)

                # Propose from truncated stable distribution
                r_prop = self.generate_truncated_stable_jumps(prop_key, shape=(self.state_dim,))

                # Compute C(r_prop, t, X_t)
                C = self.compute_tilting_factor(r_prop, At, Bt, state)  # shape: (state_dim,)

                # Acceptance probability per dimension
                accept_prob = jnp.minimum(C / M, 1.0)

                # Accept/reject per dimension
                u = random.uniform(accept_key, shape=(self.state_dim,))
                newly_accepted = u < accept_prob

                # Update: use new sample for dimensions that just got accepted
                should_update = newly_accepted & (~accepted)
                r_updated = jnp.where(should_update, r_prop, r_current)
                accepted_updated = accepted | newly_accepted

                return (r_updated, accepted_updated, subkey), None

            # Initialize: no samples accepted yet
            init_r = jnp.ones(self.state_dim) * self.tau
            init_accepted = jnp.zeros(self.state_dim, dtype=bool)

            # Run up to max_rejection_attempts rejection trials
            (r_final, accepted_final, _), _ = scan(
                attempt_sample,
                init=(init_r, init_accepted, subkey),
                xs=None,
                length=self.max_rejection_attempts
            )

            # Compute acceptance rate for this sample
            sample_acceptance_rate = jnp.mean(accepted_final.astype(float))

            return r_final, sample_acceptance_rate

        # Generate n_samples in parallel using vmap
        keys = random.split(key, n_samples)
        samples, acceptance_rates = vmap(sample_one)(keys)  # shapes: (n_samples, state_dim), (n_samples,)

        # Compute mean acceptance rate for diagnostics
        mean_acceptance_rate = jnp.mean(acceptance_rates)

        return samples, mean_acceptance_rate

    def generate_conditional_Gaussian(self, time, state, At, Bt, mixing_batch, key):
        """Generate conditional Gaussian samples using efficient broadcasting."""
        # Ensure inputs are properly shaped
        state = jnp.atleast_1d(state)  # shape: (state_dim,)
        mixing_batch = jnp.atleast_2d(mixing_batch)  # shape: (n_samples, state_dim)
        n_samples = mixing_batch.shape[0]

        # Broadcasting: expand to (1, state_dim) for broadcasting with (n_samples, state_dim)
        A_expanded = At[None, :]  # shape: (1, state_dim)
        B_expanded = Bt[None, :]  # shape: (1, state_dim)
        state_expanded = state[None, :]  # shape: (1, state_dim)

        # Compute per-sample parameters using broadcasting
        K1 = 2*A_expanded*state_expanded + B_expanded  # shape: (n_samples, state_dim)
        K2 = A_expanded - 1/(2*jnp.power(mixing_batch, 2)*jnp.power(self.sigma, 2))  # shape: (n_samples, state_dim)

        eps = 1e-6
        K2_safe = jnp.where(K2 < -eps, K2, -eps)

        # Compute mean and variance for all samples
        mean = -K1/(2*K2_safe)  # shape: (n_samples, state_dim)
        var = -1/(2*K2_safe)    # shape: (n_samples, state_dim)

        # Generate noise for entire batch at once
        noise = random.normal(key, shape=(n_samples, self.state_dim))  # shape: (n_samples, state_dim)

        # Return batch of samples
        samples = mean + jnp.sqrt(var) * noise
        return samples

    @eqx.filter_jit
    def simulate_posterior_and_loss(self, state_init, time_sequence, key):
        """Simulate posterior with loss computation using current learned parameters."""
        def simulate_forward(state, time):
            X, past_time, key, kl_accum = state
            dt = time - past_time

            # Sample generation
            key, subkey = random.split(key=key, num=2)
            raw_stable_jumps = self.generate_truncated_stable_jumps(subkey, shape=(self.loss_sample_size, self.state_dim))
            scaled_stable_jumps = self.diffusion(time, X) * raw_stable_jumps  # shape: (loss_sample_size, state_dim)
            normalisation = self.total_prior_jump_rate(1.)

            # Loss estimation using phi network
            # Use deterministic reference selection (key=None) for stable gradients
            # All samples in this batch use the same top-k references, reducing gradient variance
            A_t, B_t = self.phi.get_coefficients(time)  # shapes: (state_dim,) each
            log_Ht = A_t * scaled_stable_jumps**2 + (2*A_t*X + B_t) * scaled_stable_jumps  # shape: (loss_sample_size, state_dim)
            negative_log_Ht = A_t * scaled_stable_jumps**2 - (2*A_t*X + B_t) * scaled_stable_jumps  # shape: (loss_sample_size, state_dim)

            # Soft clipping to prevent overflow
            max_exp_arg = 80.0
            log_Ht = max_exp_arg * jnp.tanh(log_Ht / max_exp_arg)
            negative_log_Ht = max_exp_arg * jnp.tanh(negative_log_Ht / max_exp_arg)

            # Compute loss for each jump (multidimensional):
            f_loss = jnp.exp(log_Ht)*(log_Ht - 1) + 1 + jnp.exp(negative_log_Ht)*(negative_log_Ht - 1) + 1  # shape: (loss_sample_size, state_dim)

            # Quantile-based clipping for heavy-tailed loss samples (97.5th percentile)
            per_sample_loss = jnp.sum(f_loss, axis=1)  # shape: (loss_sample_size,)
            loss_quantile = jnp.nanquantile(per_sample_loss, 0.95)
            clipped_loss = jnp.minimum(per_sample_loss, loss_quantile)
            loss_estimate = dt * normalisation * jnp.nanmean(clipped_loss)
            kl_accum = kl_accum + loss_estimate

            # Forward simulation using learned drift and alpha
            key, subkey1, subkey2, subkey3 = random.split(key, num=4)
            # Quantile-based clipping for heavy-tailed tilting weights (97.5th percentile per dimension)
            tilting_weights = jnp.exp(log_Ht) + jnp.exp(negative_log_Ht)  # shape: (loss_sample_size, state_dim)
            tilt_quantile = jnp.nanquantile(tilting_weights, 0.95, axis=0)  # shape: (state_dim,)
            clipped_tilt = jnp.minimum(tilting_weights, tilt_quantile[None, :])  # shape: (loss_sample_size, state_dim)
            tilting_factor = jnp.nanmean(clipped_tilt, axis=0)  # shape: (state_dim,)
            total_rate = tilting_factor * self.total_prior_jump_rate(dt)  # shape: (state_dim,)
            n_jumps = random.poisson(subkey1, lam=total_rate)  # shape: (state_dim,)

            # Generate mixing variable using rejection sampling
            mixing_variable_sample, mean_acceptance = self.generate_mixing_variable_rejection(time, X, A_t, B_t, self.max_jumps, subkey2)
            cond_Gaussian_sample = self.generate_conditional_Gaussian(time, X, A_t, B_t, mixing_variable_sample, subkey3)
            # jax.debug.print("rejection acceptance rate: {mean_acceptance:.3f}", mean_acceptance=mean_acceptance)
            # jax.debug.print("---")

            # Masking: each dimension has its own jump count
            mask = jnp.arange(self.max_jumps)[:, None] < n_jumps[None, :]  # shape: (max_jumps, state_dim)
            sum_of_jumps = jnp.nansum(cond_Gaussian_sample * mask, axis=0)  # sum over jumps, shape: (state_dim,)
            # Soft clipping to prevent extreme jumps while preserving gradients
            clip_scale = 50.0 * self.sigma
            sum_of_jumps = clip_scale * jnp.tanh(sum_of_jumps / clip_scale)
            effective_sum_of_jumps = self.diffusion(time, X) * sum_of_jumps

            # Use learned drift function
            X = X + self.drift(time, X) * dt + effective_sum_of_jumps

            return (X, time, key, kl_accum), (X, loss_estimate)

        key, subkey = random.split(key=key, num=2)
        time_init = time_sequence[0]
        init_kl_accum = 0.0

        # Generate TiltedStableDrivenSDE path using the Euler method with accumulated loss.
        (X_final, _, _, kl_total), (sde_path, loss_estimates) = scan(
            simulate_forward,
            init=(state_init, time_init, subkey, init_kl_accum),
            xs=time_sequence
        )

        return sde_path, loss_estimates, kl_total

    @eqx.filter_jit
    def simulate_posterior(self, state_init, time_sequence, key):
        """Simulate posterior paths using learned parameters."""
        def simulate_forward(state, time):
            X, past_time, key = state
            dt = time - past_time

            key, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)

            # Generate jump sizes
            raw_stable_jumps = self.generate_truncated_stable_jumps(subkey1, shape=(self.loss_sample_size, self.state_dim))
            scaled_stable_jumps = self.diffusion(time, X) * raw_stable_jumps  # shape: (loss_sample_size, state_dim)

            # Use deterministic reference selection (key=None) for stable gradients during simulation
            A_t, B_t = self.phi.get_coefficients(time)
            log_Ht = A_t * scaled_stable_jumps**2 + (2*A_t*X + B_t) * scaled_stable_jumps
            negative_log_Ht = A_t * scaled_stable_jumps**2 - (2*A_t*X + B_t) * scaled_stable_jumps

            # Soft clipping to prevent overflow
            max_exp_arg = 80.0
            log_Ht = max_exp_arg * jnp.tanh(log_Ht / max_exp_arg)
            negative_log_Ht = max_exp_arg * jnp.tanh(negative_log_Ht / max_exp_arg)

            # Quantile-based clipping for heavy-tailed tilting weights (97.5th percentile per dimension)
            tilting_weights = jnp.exp(log_Ht) + jnp.exp(negative_log_Ht)  # shape: (loss_sample_size, state_dim)
            tilt_quantile = jnp.nanquantile(tilting_weights, 0.95, axis=0)  # shape: (state_dim,)
            clipped_tilt = jnp.minimum(tilting_weights, tilt_quantile[None, :])  # shape: (loss_sample_size, state_dim)
            tilting_factor = jnp.nanmean(clipped_tilt, axis=0)  # shape: (state_dim,)
            total_rate = tilting_factor * self.total_prior_jump_rate(dt)  # shape: (state_dim,)
            n_jumps = random.poisson(subkey2, lam=total_rate)  # shape: (state_dim,)
            # jax.debug.print("total_rate {total_rate}", total_rate=total_rate)
            # jax.debug.print("n_jumps {n_jumps}", n_jumps=n_jumps)
            # jax.debug.print("---")
            mixing_variable_sample, mean_acceptance = self.generate_mixing_variable_rejection(time, X, A_t, B_t, n_samples=self.max_jumps, key=subkey3)
            cond_Gaussian_sample = self.generate_conditional_Gaussian(time, X, A_t, B_t, mixing_variable_sample, subkey4)

            # jax.debug.print("rejection acceptance rate: {mean_acceptance:.3f}", mean_acceptance=mean_acceptance)
            # jax.debug.print("---")

            # Masking: each dimension has its own jump count
            mask = jnp.arange(self.max_jumps)[:, None] < n_jumps[None, :]  # shape: (max_jumps, state_dim)
            sum_of_jumps = jnp.nansum(cond_Gaussian_sample * mask, axis=0)  # sum over jumps, shape: (state_dim,)
            # Soft clipping to prevent extreme jumps while preserving gradients
            clip_scale = 50.0 * self.sigma
            sum_of_jumps = clip_scale * jnp.tanh(sum_of_jumps / clip_scale)
            effective_sum_of_jumps = self.diffusion(time, X) * sum_of_jumps

            # Use learned drift function
            X = X + self.drift(time, X) * dt + effective_sum_of_jumps

            return (X, time, key), X

        key, subkey = random.split(key=key, num=2)
        time_init = time_sequence[0]

        # Generate TiltedStableDrivenSDE path using the Euler method.
        state, sde_path = scan(simulate_forward, init=(state_init, time_init, subkey), xs=time_sequence)
        return sde_path
