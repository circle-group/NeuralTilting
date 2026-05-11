"""
Visualization utilities for systematic dataset generation.

This module provides plotting functions for dataset visualization.
"""

import math
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, Optional


def plot_dataset(
    data_dict: Dict[str, Any],
    save_path: Path,
    plot_dpi: int = 200,
    show_title: bool = True
):
    """
    Create a visualization plot for a generated dataset.

    This function creates a plot showing:
    - Latent path (true SDE path) as a dashed line
    - All observations as scatter points

    Parameters
    ----------
    data_dict : dict
        Dataset dictionary containing:
        - 'latent_path': (n_steps+1, state_dim) array
        - 'observations': (n_steps+1, state_dim) array
        - 'time_sequence': (n_steps+1,) array
        - 'alpha', 'tau', 'sigma', 'obs_std': float parameters
    save_path : Path
        Path where the PNG file should be saved
    plot_dpi : int, optional
        DPI for the saved figure (default: 200)
    show_title : bool, optional
        Whether to show title with parameters (default: True)

    Examples
    --------
    >>> plot_dataset(data_dict, Path("experiments/datasets/.../seed_12345678.png"))
    """
    # Extract data
    latent_path = data_dict['latent_path']
    observations = data_dict['observations']
    time_sequence = data_dict['time_sequence']

    # Get parameters for title
    alpha = data_dict['alpha']
    tau = data_dict['tau']
    obs_std = data_dict['obs_std']

    model_drift = data_dict['drift_component']

    # Build drift parameter string for title based on drift type
    drift_type = data_dict.get('drift_type', '')
    if hasattr(model_drift, 'theta') and hasattr(model_drift, 'mu'):
        drift_params_str = f"θ={model_drift.theta[0]:.2f}, μ={model_drift.mu[0]:.2f}"
    elif hasattr(model_drift, 'theta1') and hasattr(model_drift, 'theta2'):
        drift_params_str = f"θ₁={model_drift.theta1[0]:.2f}, θ₂={model_drift.theta2[0]:.2f}"
    else:
        drift_params_str = f"drift={drift_type}"

    # Determine state dimensionality
    state_dim = latent_path.shape[1] if len(latent_path.shape) > 1 else 1

    # Calculate subplot grid (max 2 columns)
    n_cols = min(state_dim, 2)
    n_rows = math.ceil(state_dim / n_cols)

    # Dynamic figsize based on dimensions
    figsize = (10 * n_cols, 6 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single subplot case
    if state_dim == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten() if state_dim > 1 else [axes]

    # Set title if requested
    if show_title:
        fig.suptitle(
            f"Dataset: α={alpha:.2f}, τ={tau:.2f}, σ_ε={obs_std:.2f}, {drift_params_str}",
            fontsize=14
        )

    # Colors
    latent_color = '#1f77b4'
    obs_color = '#ff7f0e'
    marker_size = 3

    # Plot each dimension
    for dim in range(state_dim):
        ax = axes[dim]

        # Plot latent path with a dashed line
        ax.plot(
            time_sequence,
            latent_path[:, dim] if state_dim > 1 else latent_path,
            color=latent_color,
            alpha=0.9,
            linewidth=2.5,
            ls='--',
            label="Latent path",
            zorder=3
        )

        # Plot all observations with scatter markers
        ax.scatter(
            time_sequence,
            observations[:, dim] if state_dim > 1 else observations,
            marker='o',
            color=obs_color,
            alpha=0.6,
            s=marker_size,
            label="Observations",
            zorder=5
        )

        # Add legend only for first subplot
        if dim == 0:
            ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel(f"State (dim {dim+1})" if state_dim > 1 else "State", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3, zorder=1)
        ax.set_facecolor('#edebeb')  # Light background

    # Hide unused subplots if state_dim is odd and > 1
    if state_dim % 2 == 1 and state_dim > 1 and n_cols == 2:
        axes[-1].set_visible(False)

    fig.patch.set_facecolor('white')
    plt.tight_layout()

    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close(fig)


def plot_training_results(
    metrics: Dict,
    model_config: Dict,
    training_config: Dict,
    output_paths: Dict[str, Path],
    plot_dpi: int = 200,
    true_drift_component: Optional[Any] = None
):
    """
    Generate standard training plots for both Tilted Stable and Gaussian SDE models.

    Creates:
    1. Loss history plot (with model-specific network labels)
    2. Drift parameter evolution plot (if applicable)

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from train_tilted_stable_model or train_gaussian_model
    model_config : dict
        Model configuration (automatically detects model type from keys)
    training_config : dict
        Training configuration
    output_paths : dict
        Output paths dictionary
    plot_dpi : int, optional
        DPI for saved plots (default: 200)
    true_drift_component : optional
        True drift component from dataset for comparison (default: None)

    Notes
    -----
    The function automatically detects the model type:
    - Tilted Stable SDE: Uses 'tilting_width' and 'tilting_depth' keys, labels as "φ network"
    - Gaussian SDE: Uses 'control_width' and 'control_depth' keys, labels as "U network"
    """

    # ========================================================================
    # PLOT 1: LOSS HISTORY
    # ========================================================================

    loss_history = metrics['loss_history']

    fig, ax = plt.subplots(figsize=(8, 5))

    drift_str = "known drift" if not model_config.get('trainable_drift', False) else "trainable drift"

    # Detect model type and format accordingly
    if 'tilting_width' in model_config and 'tilting_depth' in model_config:
        # Tilted Stable SDE
        n_hidden = model_config['tilting_depth'] - 1
        network_str = f"φ network: width={model_config['tilting_width']}, {n_hidden} hidden layers"
    elif 'control_width' in model_config and 'control_depth' in model_config:
        # Gaussian SDE
        n_hidden = model_config['control_depth'] - 1
        network_str = f"U network: width={model_config['control_width']}, {n_hidden} hidden layers"
    else:
        # Fallback
        network_str = "Network architecture unknown"

    # Title
    title_line1 = f"Loss History (lr={training_config['learning_rate']}, {training_config['n_loss_samples']} samples)"
    title_line2 = (
        f"{network_str} | "
        f"{drift_str} | "
        f"n_latent_steps={training_config['n_latent_steps']}"
    )

    ax.set_title(f"{title_line1}\n{title_line2}", fontsize=13)
    ax.set_xlabel("Training iteration", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.plot(loss_history)
    ax.grid(True)
    ax.set_yscale('log')

    plt.tight_layout()

    # Save
    loss_plot_path = output_paths['plots'] / "loss_history.png"
    plt.savefig(loss_plot_path, dpi=plot_dpi)
    plt.close(fig)  # Explicitly close this specific figure

    print(f"Saved loss plot: {loss_plot_path}", flush=True)

    # ========================================================================
    # PLOT 2: DRIFT PARAMETER EVOLUTION (if applicable)
    # ========================================================================

    if 'parameter_history' in metrics and metrics['parameter_history']:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        parameter_history = metrics['parameter_history']

        # Filter to only include drift parameters (exclude diffusion monitoring entries)
        drift_history = [p for p in parameter_history if p.get('param_type') in ['ou', 'linear']]

        if not drift_history:
            # No drift parameters tracked, skip this plot
            plt.close(fig)
        else:
            iterations = list(range(1, len(drift_history) + 1))

            # Check if we're tracking OU parameters (theta, mu) or generic linear (weight, bias)
            param_type = drift_history[0].get('param_type', 'linear')

            if param_type == 'ou':
                # Plot OU parameters: θ (mean-reversion rate) and μ (mean-reversion level)
                thetas = [p['theta'] for p in drift_history]
                mus = [p['mu'] for p in drift_history]

                # Extract true OU parameters if available
                true_theta = None
                true_mu = None
                if true_drift_component is not None:
                    if hasattr(true_drift_component, 'theta') and hasattr(true_drift_component, 'mu'):
                        true_theta = float(true_drift_component.theta.item()) if true_drift_component.theta.ndim > 0 else float(true_drift_component.theta)
                        true_mu = float(true_drift_component.mu.item()) if true_drift_component.mu.ndim > 0 else float(true_drift_component.mu)

                # Plot theta (mean-reversion rate) evolution
                ax1.plot(iterations, thetas, 'b-', linewidth=2, label='Trained')
                if true_theta is not None:
                    ax1.axhline(y=true_theta, color='k', linestyle='--', linewidth=2, label='True value')
                    ax1.legend(fontsize=11)
                ax1.set_xlabel("Training iteration", fontsize=12)
                ax1.set_ylabel(r"Mean-reversion rate $\theta$", fontsize=12)
                ax1.set_title(r"OU Parameter $\theta$ Evolution", fontsize=14)
                ax1.grid(True, alpha=0.3)

                # Plot mu (mean-reversion level) evolution
                ax2.plot(iterations, mus, 'r-', linewidth=2, label='Trained')
                if true_mu is not None:
                    ax2.axhline(y=true_mu, color='k', linestyle='--', linewidth=2, label='True value')
                    ax2.legend(fontsize=11)
                ax2.set_xlabel("Training iteration", fontsize=12)
                ax2.set_ylabel(r"Mean-reversion level $\mu$", fontsize=12)
                ax2.set_title(r"OU Parameter $\mu$ Evolution", fontsize=14)
                ax2.grid(True, alpha=0.3)

            else:  # param_type == 'linear'
                # Plot generic linear drift parameters: weight and bias
                weights = [p['weight'] for p in drift_history]
                biases = [p['bias'] for p in drift_history]

                # Extract true drift parameters if available
                true_weight = None
                true_bias = None
                if true_drift_component is not None:
                    if hasattr(true_drift_component, 'weight') and hasattr(true_drift_component, 'bias'):
                        true_weight = float(true_drift_component.weight.item()) if true_drift_component.weight.ndim > 0 else float(true_drift_component.weight)
                        true_bias = float(true_drift_component.bias.item()) if true_drift_component.bias.ndim > 0 else float(true_drift_component.bias)

                # Plot weight evolution
                ax1.plot(iterations, weights, 'b-', linewidth=2, label='Trained')
                if true_weight is not None:
                    ax1.axhline(y=true_weight, color='k', linestyle='--', linewidth=2, label='True value')
                    ax1.legend(fontsize=11)
                ax1.set_xlabel("Training iteration", fontsize=12)
                ax1.set_ylabel("Weight parameter", fontsize=12)
                ax1.set_title("Drift Weight Evolution", fontsize=14)
                ax1.grid(True, alpha=0.3)

                # Plot bias evolution
                ax2.plot(iterations, biases, 'r-', linewidth=2, label='Trained')
                if true_bias is not None:
                    ax2.axhline(y=true_bias, color='k', linestyle='--', linewidth=2, label='True value')
                    ax2.legend(fontsize=11)
                ax2.set_xlabel("Training iteration", fontsize=12)
                ax2.set_ylabel("Bias parameter", fontsize=12)
                ax2.set_title("Drift Bias Evolution", fontsize=14)
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save
            drift_plot_path = output_paths['plots'] / "drift_params_history.png"
            plt.savefig(drift_plot_path, dpi=plot_dpi)
            plt.close(fig)  # Explicitly close this specific figure

            print(f"Saved drift parameter plot: {drift_plot_path}", flush=True)