"""
Evaluation module for comparing trained SDE models.

This module provides tools for:
- Computing various evaluation metrics on trained models
- Comparing multiple models quantitatively
- Generating comparative visualizations

Usage:
    # Evaluate a single run
    python evaluation/evaluate_runs.py --run-id data_16241891_train_29699958 --metrics mse mae

    # Compare multiple runs
    python evaluation/compare_models.py --run-ids run1 run2 --output-dir evaluation/comparisons/my_comparison
"""

__version__ = "0.1.0"
