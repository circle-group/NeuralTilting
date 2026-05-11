"""Neural Jump SDE for financial time-series forecasting.

Adapted from Jia & Benson, "Neural Jump Stochastic Differential Equations",
NeurIPS 2019.  Original formulation targets temporal point processes; this
adaptation treats each regularly-sampled price observation as a discrete event
with a real-valued mark (the normalised log-price), yielding a stochastic
generative model for distributional forecasting.

Architecture
------------
Latent state  z(t) = [c(t), h(t)]  where
  c(t) ∈ R^{dim_c} : internal state, evolves via neural ODE
  h(t) ∈ R^{dim_h} : event memory, decays between observations and jumps at each one

Four learnable MLPs (CELU activation throughout, matching paper):
  F : (dim_c + dim_h) → dim_c        ODE drift for c; optionally projected orthogonal to c
  G : dim_c → dim_h                  positive decay rates for h (softplus output)
  W : (dim_c + obs_dim) → dim_h      jump update Δh from mark y ∈ R^{obs_dim}
  L : (dim_c + dim_h) → M*(1+2*D)   diagonal GMM params (shared log-weights, per-dim means/log-vars)

ODE dynamics between observations:
  dc/dt = F(z) - proj_c F(z)    [sphere constraint when ortho=True]
  dh/dt = -softplus(G(c)) ⊙ h   [exponential decay]

Jump at each observation y_t ∈ R^{obs_dim}:
  Δh = W([c, y_t])
  Δc = 0

Training loss: negative GMM log-likelihood summed over observations,
normalised by sequence length.  (Intensity integral from the original
point-process formulation is dropped — it carries no timing information
on a fixed regular grid.)

Forecasting: from the final training state, autoregressively sample y_h
from the GMM, apply the jump, and repeat for H steps.  K independent paths
are generated via vmap over different random seeds.

Multivariate (obs_dim > 1)
--------------------------
The GMM output distribution is a diagonal mixture:
  - Mixture weights log_w : (M,)        — shared across all dimensions
  - Component means        : (M, obs_dim)
  - Component log-variances: (M, obs_dim)
  L outputs M * (1 + 2 * obs_dim) parameters in that order.
  Log-likelihood = logsumexp_k( log_w_k + Σ_d log N(y_d | μ_{k,d}, σ_{k,d}) )
  Sampling: draw component k ~ Categorical(softmax(log_w)),
            then y_d ~ N(μ_{k,d}, σ_{k,d}) independently for each d.
When obs_dim=1 this reduces exactly to the original scalar GMM.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx


class NeuralJumpSDE(eqx.Module):
    """Neural Jump SDE for log-price forecasting (univariate or multivariate).

    Parameters
    ----------
    dim_c : int
        Dimension of the internal ODE state c(t).
    dim_h : int
        Dimension of the event-memory state h(t).
    n_mixtures : int
        Number of Gaussian mixture components M.
    hidden_width : int
        Width of all MLP hidden layers.
    hidden_depth : int
        Depth (number of hidden layers) of all MLPs.
    ortho : bool
        If True, project dc/dt orthogonal to c (sphere constraint).
    obs_dim : int
        Observation dimension D.  Default 1 (scalar, backward-compatible).
    key : jax.Array
        Initialisation key.
    """

    dim_c:      int  = eqx.field(static=True)
    dim_h:      int  = eqx.field(static=True)
    n_mixtures: int  = eqx.field(static=True)
    ortho:      bool = eqx.field(static=True)
    obs_dim:    int  = eqx.field(static=True)

    F: eqx.nn.MLP   # ODE drift:    (dim_c + dim_h)       → dim_c
    G: eqx.nn.MLP   # decay rates:  dim_c                  → dim_h
    W: eqx.nn.MLP   # jump fn:      (dim_c + obs_dim)      → dim_h
    L: eqx.nn.MLP   # GMM params:   (dim_c + dim_h)        → M*(1 + 2*obs_dim)
    c0: jax.Array   # learnable initial internal state (dim_c,)

    def __init__(
        self,
        dim_c:        int,
        dim_h:        int,
        n_mixtures:   int,
        hidden_width: int,
        hidden_depth: int,
        ortho:        bool,
        key:          jax.Array,
        obs_dim:      int = 1,
    ):
        keys = jrandom.split(key, 5)
        self.dim_c      = dim_c
        self.dim_h      = dim_h
        self.n_mixtures = n_mixtures
        self.ortho      = ortho
        self.obs_dim    = obs_dim

        act = jax.nn.celu

        self.F = eqx.nn.MLP(
            in_size=dim_c + dim_h,
            out_size=dim_c,
            width_size=hidden_width,
            depth=hidden_depth,
            activation=act,
            key=keys[0],
        )
        self.G = eqx.nn.MLP(
            in_size=dim_c,
            out_size=dim_h,
            width_size=hidden_width,
            depth=hidden_depth,
            activation=act,
            key=keys[1],
        )
        self.W = eqx.nn.MLP(
            in_size=dim_c + obs_dim,
            out_size=dim_h,
            width_size=hidden_width,
            depth=hidden_depth,
            activation=act,
            key=keys[2],
        )
        self.L = eqx.nn.MLP(
            in_size=dim_c + dim_h,
            out_size=n_mixtures * (1 + 2 * obs_dim),
            width_size=hidden_width,
            depth=hidden_depth,
            activation=act,
            key=keys[3],
        )
        self.c0 = jrandom.normal(keys[4], shape=(dim_c,)) * 0.01

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _ode_fn(self, z: jax.Array) -> jax.Array:
        """Continuous dynamics dz/dt evaluated at state z."""
        c = z[: self.dim_c]
        h = z[self.dim_c :]

        dcdt = self.F(z)
        if self.ortho:
            # Project dcdt orthogonal to c (sphere constraint)
            c_norm_sq = jnp.sum(c * c) + 1e-8
            dcdt = dcdt - (jnp.sum(dcdt * c) / c_norm_sq) * c

        dhdt = -jax.nn.softplus(self.G(c)) * h
        return jnp.concatenate([dcdt, dhdt])

    def _rk4_step(self, z: jax.Array, dt: float) -> jax.Array:
        """Single RK4 step of the ODE over interval dt."""
        k1 = self._ode_fn(z)
        k2 = self._ode_fn(z + 0.5 * dt * k1)
        k3 = self._ode_fn(z + 0.5 * dt * k2)
        k4 = self._ode_fn(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _integrate(self, z: jax.Array, dt: float, n_steps: int) -> jax.Array:
        """Integrate ODE for one observation interval using n_steps RK4 substeps."""
        substep = dt / n_steps

        def body(z, _):
            return self._rk4_step(z, substep), None

        z, _ = jax.lax.scan(body, z, None, length=n_steps)
        return z

    def _log_gmm(self, params: jax.Array, y: jax.Array) -> jax.Array:
        """Log-likelihood of y ∈ R^{obs_dim} under the diagonal Gaussian mixture.

        params layout: [log_w (M,) | means (M*obs_dim,) | log_vars (M*obs_dim,)]
        For obs_dim=1 this is identical to the original scalar GMM.
        """
        M = self.n_mixtures
        D = self.obs_dim
        log_w    = jax.nn.log_softmax(params[:M])                      # (M,)
        means    = params[M : M + M * D].reshape(M, D)                 # (M, D)
        log_vars = params[M + M * D :].reshape(M, D)                   # (M, D)
        # Per-component log-likelihood: sum Normal contributions over dimensions
        log_comp = jnp.sum(
            -0.5 * (jnp.log(2.0 * jnp.pi) + log_vars
                    + (y - means) ** 2 / jnp.exp(log_vars)),
            axis=-1,
        )  # (M,)
        return jax.nn.logsumexp(log_w + log_comp)

    def _sample_gmm(self, params: jax.Array, key: jax.Array) -> jax.Array:
        """Draw one sample from the diagonal Gaussian mixture.

        Returns
        -------
        y : (obs_dim,)
        """
        M = self.n_mixtures
        D = self.obs_dim
        log_w    = jax.nn.log_softmax(params[:M])
        means    = params[M : M + M * D].reshape(M, D)
        log_vars = params[M + M * D :].reshape(M, D)

        key1, key2 = jrandom.split(key)
        k    = jrandom.categorical(key1, log_w)
        stds = jnp.exp(0.5 * log_vars[k])                             # (D,)
        return means[k] + stds * jrandom.normal(key2, shape=(D,))     # (D,)

    def _apply_jump(self, z: jax.Array, y: jax.Array) -> jax.Array:
        """Apply a jump to the memory h given mark y ∈ R^{obs_dim}."""
        c  = z[: self.dim_c]
        dh = self.W(jnp.concatenate([c, y]))
        return z.at[self.dim_c :].add(dh)

    def _initial_state(self, y0: jax.Array) -> jax.Array:
        """Build z_0 = [c0, 0] then apply the first-observation jump."""
        z0 = jnp.concatenate([self.c0, jnp.zeros(self.dim_h)])
        return self._apply_jump(z0, y0)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    @eqx.filter_jit
    def training_log_likelihood(
        self,
        obs: jax.Array,
        dt: float,
        n_ode_steps: int,
    ) -> jax.Array:
        """Compute total GMM log-likelihood over the training sequence.

        Parameters
        ----------
        obs : (T, obs_dim) normalised log-prices
        dt  : time step between observations (in normalised time)
        n_ode_steps : RK4 substeps per interval

        Returns
        -------
        Scalar total log-likelihood (sum over T-1 observations; first obs
        is used only to seed the initial state).
        """
        y0 = obs[0]                          # (obs_dim,)
        z  = self._initial_state(y0)

        def scan_fn(z, y_t):
            # Integrate ODE for one interval
            z     = self._integrate(z, dt, n_ode_steps)
            # Evaluate GMM *before* the jump (left-limit, matching paper)
            log_p = self._log_gmm(self.L(z), y_t)
            # Apply jump with the observed mark
            z     = self._apply_jump(z, y_t)
            return z, log_p

        _, log_ps = jax.lax.scan(scan_fn, z, obs[1:])  # obs[1:] : (T-1, obs_dim)
        return jnp.sum(log_ps)

    # -------------------------------------------------------------------------
    # Forecasting
    # -------------------------------------------------------------------------

    @eqx.filter_jit
    def forecast(
        self,
        obs: jax.Array,
        dt: float,
        n_ode_steps: int,
        forecast_steps: int,
        n_samples: int,
        key: jax.Array,
    ) -> jax.Array:
        """Generate K stochastic forecast paths by sampling from the GMM.

        Parameters
        ----------
        obs            : (T, obs_dim) normalised log-prices (full training window)
        dt             : time step between observations
        n_ode_steps    : RK4 substeps per interval
        forecast_steps : H, number of steps to forecast
        n_samples      : K, number of independent sample paths
        key            : PRNG key

        Returns
        -------
        samples : (K, H, obs_dim) forecast log-prices
        """
        # --- Assimilate training observations to build final state ---
        y0     = obs[0]                      # (obs_dim,)
        z_init = self._initial_state(y0)

        def assimilate(z, y_t):
            z = self._integrate(z, dt, n_ode_steps)
            z = self._apply_jump(z, y_t)
            return z, None

        z_final, _ = jax.lax.scan(assimilate, z_init, obs[1:])

        # --- Sample K independent paths from z_final ---
        def forecast_one(sample_key: jax.Array) -> jax.Array:
            step_keys = jrandom.split(sample_key, forecast_steps)

            def step(z, rng):
                z     = self._integrate(z, dt, n_ode_steps)
                y_hat = self._sample_gmm(self.L(z), rng)   # (obs_dim,)
                z     = self._apply_jump(z, y_hat)
                return z, y_hat

            _, ys = jax.lax.scan(step, z_final, step_keys)
            return ys  # (H, obs_dim)

        sample_keys = jrandom.split(key, n_samples)
        samples = jax.vmap(forecast_one)(sample_keys)  # (K, H, obs_dim)
        return samples
