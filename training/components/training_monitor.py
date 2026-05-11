"""
Training monitoring and diagnostics for tilted stable SDEs.

Handles verbose output, gradient statistics, memory monitoring,
drift parameter tracking, and coefficient tracking without cluttering
the main training loop.

Supports attention-based architecture with mlp_A/mlp_B networks.
"""

from typing import Optional, Any
import jax
import jax.numpy as jnp
import equinox as eqx
import resource
import platform


def get_memory_usage_gb() -> float:
    """
    Get current memory usage in GB, handling platform differences.

    macOS: ru_maxrss is in bytes
    Linux: ru_maxrss is in kilobytes
    """
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == 'Darwin':  # macOS
        return mem_usage / 1024**3  # bytes to GB
    else:  # Linux and others
        return mem_usage / 1024**2  # KB to GB


class TrainingMonitor:
    """
    Handles all training monitoring and logging.

    Extracts verbose logging from the training loop to keep it clean.
    """

    def __init__(self, verbose: bool = True, log_interval: int = 10):
        """
        Initialize monitor.

        Args:
            verbose: Whether to print any output
            log_interval: Print detailed diagnostics every N steps
        """
        self.verbose = verbose
        self.log_interval = log_interval
        self.initial_memory: Optional[float] = None

    def log_training_start(
        self,
        n_steps: int,
        lr: float,
        n_loss_samples: int,
        n_latent_steps: int,
        gc_interval: int,
        clear_cache_interval: int,
    ):
        """Log training configuration at start."""
        if not self.verbose:
            return

        print(f"Starting training for {n_steps} steps...", flush=True)
        print(f"  Learning rate: {lr}", flush=True)
        print(f"  Loss samples per step: {n_loss_samples}", flush=True)
        print(f"  Latent steps: {n_latent_steps}", flush=True)
        if gc_interval > 0:
            print(f"  Garbage collection interval: {gc_interval} steps", flush=True)
        if clear_cache_interval > 0:
            print(f"  JAX cache clear interval: {clear_cache_interval} steps", flush=True)

        self.initial_memory = get_memory_usage_gb()
        print(f"  Initial memory: {self.initial_memory:.2f} GB", flush=True)

    def log_step(self, step: int, loss: float, aux_data: dict):
        """Log basic step info."""
        if not self.verbose:
            return

        print(f"Step {step+1}: Loss: {loss:.6f}", flush=True)

        # Print loss components breakdown periodically
        if step % self.log_interval == 0:
            print(f"          KL: {aux_data['kl_term']:.2e}, Likelihood: {aux_data['likelihood_term']:.2e}", flush=True)
            print(f"          Reg (phi: {aux_data['phi_regularization']:.2e}, drift: {aux_data['drift_regularization']:.2e}, coeff: {aux_data['coeff_regularization']:.2e})", flush=True)

    def log_compilation_memory(self, step: int):
        """Log memory after first JIT compilation."""
        if not self.verbose or step != 0:
            return

        mem_gb = get_memory_usage_gb()
        if self.initial_memory is not None:
            compile_overhead = mem_gb - self.initial_memory
            print(f"          Post-compilation memory: {mem_gb:.2f} GB (overhead: +{compile_overhead:.2f} GB)", flush=True)

    def log_nan_inf_error(self, step: int, loss_is_nan: bool, loss_is_inf: bool):
        """Log NaN/Inf error in loss."""
        if not self.verbose:
            return

        print(f"        Error at step {step+1}", flush=True)
        print("          Loss is NaN:", loss_is_nan, flush=True)
        print("          Loss is Inf:", loss_is_inf, flush=True)
        print("          Skipping this step and continuing...", flush=True)

    def log_gradient_sanitization(self, step: int):
        """Log when gradients are sanitized."""
        if not self.verbose:
            return
        print(f"        Step {step+1}: Gradients contain NaN/Inf, sanitizing gradients and continuing update", flush=True)

    def log_gradients(self, step: int, grad_value: Any, model: Any):
        """Log gradient norms and quantiles for phi network (mlp_A, mlp_B)."""
        if not self.verbose or step % self.log_interval != 0:
            return None

        # Total gradient norm
        grad_norm = jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_util.tree_map(lambda g: jnp.sum(g**2) if g is not None else 0.0, grad_value)
        )
        grad_norm = jnp.sqrt(grad_norm)
        print(f"        Gradient norm: {grad_norm:.2e}", flush=True)

        # Phi network gradient logging (mlp_A and mlp_B)
        if hasattr(grad_value.phi, 'mlp_A') and hasattr(grad_value.phi, 'mlp_B'):
            A_grads = jax.tree_util.tree_leaves(eqx.filter(grad_value.phi.mlp_A, eqx.is_inexact_array))
            B_grads = jax.tree_util.tree_leaves(eqx.filter(grad_value.phi.mlp_B, eqx.is_inexact_array))

            A_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in A_grads if g is not None))
            B_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in B_grads if g is not None))

            print(f"          Phi A grad norm: {A_grad_norm:.2e}, B grad norm: {B_grad_norm:.2e}", flush=True)

            # Quantile diagnostics
            if A_grads:
                A_flat = jnp.concatenate([jnp.abs(g).ravel() for g in A_grads if g is not None])
                A_q95 = jnp.quantile(A_flat, 0.95)
                A_q99 = jnp.quantile(A_flat, 0.99)
                print(f"          Phi A grad quantiles: 95%={A_q95:.2e}, 99%={A_q99:.2e}", flush=True)

            if B_grads:
                B_flat = jnp.concatenate([jnp.abs(g).ravel() for g in B_grads if g is not None])
                B_q95 = jnp.quantile(B_flat, 0.95)
                B_q99 = jnp.quantile(B_flat, 0.99)
                print(f"          Phi B grad quantiles: 95%={B_q95:.2e}, 99%={B_q99:.2e}", flush=True)

        return grad_norm

    def log_coefficients(self, step: int, model: Any, training_times: jnp.ndarray):
        """Log A(t), B(t) coefficient ranges."""
        if not self.verbose or step % self.log_interval != 0:
            return

        sample_times = jnp.linspace(0.0, training_times[-1], 10)
        # Use deterministic mode (no key) for consistent logging
        A_vals, B_vals = jax.vmap(model.phi.get_coefficients)(sample_times)
        print(f"          A(t) range: [{jnp.min(A_vals):.2f}, {jnp.max(A_vals):.2f}], mean: {jnp.mean(A_vals):.2f}", flush=True)
        print(f"          B(t) range: [{jnp.min(B_vals):.2f}, {jnp.max(B_vals):.2f}], mean: {jnp.mean(B_vals):.2f}", flush=True)

    def log_attention(self, step: int, model: Any, grad_value: Any):
        """Log temporal attention mechanism state."""
        if not self.verbose or step % self.log_interval != 0:
            return

        if hasattr(model.phi, 'temporal_attention') and model.phi.temporal_attention is not None:
            # Temporal attention-based architecture
            ref_times = model.phi.temporal_attention.reference_times
            sharpness = jnp.exp(model.phi.temporal_attention.attention_sharpness[0])
            ref_spacing = jnp.diff(ref_times)

            print(f"          Attention - sharpness: {sharpness:.2f}, "
                  f"ref_times: [{jnp.min(ref_times):.2f}, {jnp.max(ref_times):.2f}], "
                  f"mean_spacing: {jnp.mean(ref_spacing):.3f}", flush=True)

            # Attention gradient norms
            if hasattr(grad_value.phi, 'temporal_attention') and grad_value.phi.temporal_attention is not None:
                attn_grad = grad_value.phi.temporal_attention

                times_grad_norm = 0.0
                if attn_grad.reference_times is not None:
                    times_grad_norm = jnp.sqrt(jnp.sum(attn_grad.reference_times**2))

                sharpness_grad_norm = jnp.sqrt(jnp.sum(attn_grad.attention_sharpness**2))
                embeddings_grad_norm = jnp.sqrt(jnp.sum(attn_grad.reference_embeddings**2))

                print(f"          Attention grads - times: {times_grad_norm:.2e}, "
                      f"sharpness: {sharpness_grad_norm:.2e}, "
                      f"embeddings: {embeddings_grad_norm:.2e}", flush=True)

    def log_memory(self, step: int):
        """Log memory usage."""
        if not self.verbose or step % self.log_interval != 0:
            return

        mem_gb = get_memory_usage_gb()
        print(f"          Memory: {mem_gb:.2f} GB", flush=True)

    def log_update_norm(self, step: int, updates: Any):
        """Log optimizer update norm."""
        if not self.verbose or step % self.log_interval != 0:
            return

        update_norm = jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_util.tree_map(lambda u: jnp.sum(u**2) if u is not None else 0.0, updates)
        )
        update_norm = jnp.sqrt(update_norm)
        print(f"          Total grad update norm: {update_norm:.2e}", flush=True)

    def log_diffusion_parameters(self, step: int, model: Any) -> Optional[dict]:
        """
        Log diffusion parameters and return them for history tracking.

        This monitors the diffusion scale parameter to verify it stays fixed
        when frozen (e.g., diffusion.raw_weight in frozen_params).

        Args:
            step: Current training step
            model: The SDE model

        Returns:
            Dict with diffusion weight if present, None otherwise
        """
        # Check if diffusion component exists and has weight property
        if not hasattr(model, 'diffusion') or not hasattr(model.diffusion, 'weight'):
            return None

        weight = model.diffusion.weight

        # Log to console
        if self.verbose and step % self.log_interval == 0:
            # Format as scalar or array
            if weight.ndim == 0:
                weight_val = float(weight)
                print(f"          Diffusion σ (scale): {weight_val:.6f}", flush=True)
            else:
                weight_vals = [float(w) for w in weight]
                print(f"          Diffusion σ (scale): {weight_vals}", flush=True)

        # Return for history tracking
        if weight.ndim == 0:
            current_weight = float(weight)
        else:
            current_weight = [float(w) for w in weight]

        return {
            'weight': current_weight,
            'param_type': 'diffusion'
        }

    def log_drift_parameters(self, step: int, model: Any) -> Optional[dict]:
        """
        Log drift parameters and return them for history tracking.

        Handles both OU drift (theta, mu) and linear drift (weight, bias).

        Args:
            step: Current training step
            model: The SDE model

        Returns:
            Dict with drift parameters if trainable, None otherwise
        """
        # Check if drift is trainable
        if not hasattr(model, 'trainable_drift') or not model.trainable_drift:
            return None

        # Double well drift (theta1, theta2)
        if hasattr(model.drift, 'theta1') and hasattr(model.drift, 'theta2'):
            theta1 = model.drift.theta1
            theta2 = model.drift.theta2

            if self.verbose and step % self.log_interval == 0:
                def _fmt(x):
                    return float(x.item()) if x.ndim > 0 else float(x)
                print(f"          Drift θ₁ (linear): {_fmt(theta1):.6f}", flush=True)
                print(f"          Drift θ₂ (cubic):  {_fmt(theta2):.6f}", flush=True)

            def _s(x):
                return float(x.item()) if x.ndim > 0 else float(x)
            return {
                'theta1': _s(theta1),
                'theta2': _s(theta2),
                'param_type': 'double_well',
            }

        # Check if drift has weight/bias (standard drift structure)
        if not hasattr(model.drift, 'weight') or not hasattr(model.drift, 'bias'):
            return None

        # Check if OU drift (theta, mu) or linear drift (weight, bias)
        if hasattr(model.drift, 'theta') and hasattr(model.drift, 'mu'):
            # OU drift with mean-reversion
            theta = model.drift.theta
            mu = model.drift.mu

            # Log to console
            if self.verbose and step % self.log_interval == 0:
                # Format as scalar values
                theta_val = float(theta.item()) if theta.ndim > 0 else float(theta)
                mu_val = float(mu.item()) if mu.ndim > 0 else float(mu)
                print(f"          Drift θ (mean-reversion): {theta_val:.6f}", flush=True)
                print(f"          Drift μ (equilibrium): {mu_val:.6f}", flush=True)

            # Return for history tracking
            current_theta = float(theta.item()) if theta.ndim > 0 else float(theta)
            current_mu = float(mu.item()) if mu.ndim > 0 else float(mu)
            return {
                'theta': current_theta,
                'mu': current_mu,
                'param_type': 'ou'
            }
        else:
            # Linear drift (weight and bias only)
            weight = model.drift.weight
            bias = model.drift.bias

            # Log to console
            if self.verbose and step % self.log_interval == 0:
                # Format as scalar values
                weight_val = float(weight.item()) if weight.ndim > 0 else float(weight)
                bias_val = float(bias.item()) if bias.ndim > 0 else float(bias)
                print(f"          Drift weight: {weight_val:.6f}", flush=True)
                print(f"          Drift bias: {bias_val:.6f}", flush=True)

            # Return for history tracking
            current_weight = float(weight.item()) if weight.ndim > 0 else float(weight)
            current_bias = float(bias.item()) if bias.ndim > 0 else float(bias)
            return {
                'weight': current_weight,
                'bias': current_bias,
                'param_type': 'linear'
            }

    def log_gc(self, step: int):
        """Log garbage collection."""
        if not self.verbose:
            return
        print(f"        Ran garbage collection at step {step+1}", flush=True)

    def log_cache_clear(self, step: int):
        """Log JAX cache clearing."""
        if not self.verbose:
            return
        print(f"        Cleared JAX compilation cache at step {step+1}", flush=True)

    def log_checkpoint(self, step: int, checkpoint_path: str, metrics_path: str):
        """Log checkpoint save."""
        if not self.verbose:
            return
        print(f"  → Saved checkpoint: {checkpoint_path}", flush=True)
        print(f"  → Saved metrics: {metrics_path}", flush=True)

    def log_high_loss_skip(self, step: int, loss_value: float):
        """Log when a step is skipped due to high loss."""
        if not self.verbose:
            return
        print(f"        Step {step+1}: Loss is too high ({loss_value:.2f}), skipping update", flush=True)

    def log_training_complete(self, n_steps_completed: int):
        """Log training completion."""
        if not self.verbose:
            return
        print(f"Training completed: {n_steps_completed} steps", flush=True)
