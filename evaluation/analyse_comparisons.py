"""
Aggregate and visualise cross-model comparison results.

Reads all metric_comparison.json files from evaluation/results/comparisons/,
aggregates by alpha value across data seeds, and produces:

  - Summary tables (console + CSV) for MSE, MAE, and drift error
  - 4-panel plot: MSE and MAE (all obs / held-out) vs alpha for both models

Usage:
    python evaluation/analyse_comparisons.py
    python evaluation/analyse_comparisons.py --output-dir evaluation/results/analysis
    python evaluation/analyse_comparisons.py \\
        --comparisons-dir evaluation/results/comparisons \\
        --output-dir my_analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_TYPES = [
    'tilted_stable_sde', 'gaussian_sde',
    'tilted_stable_sde_double_well', 'gaussian_sde_double_well',
]

MODEL_LABEL = {
    'tilted_stable_sde':              'Tilted Stable',
    'gaussian_sde':                   'Gaussian',
    'tilted_stable_sde_double_well':  'Tilted Stable (DW)',
    'gaussian_sde_double_well':       'Gaussian (DW)',
}
MODEL_COLOR = {
    'tilted_stable_sde':              '#1f77b4',   # blue
    'gaussian_sde':                   '#d62728',   # red
    'tilted_stable_sde_double_well':  '#2ca02c',   # green
    'gaussian_sde_double_well':       '#ff7f0e',   # orange
}
MODEL_MARKER = {
    'tilted_stable_sde':              'o',
    'gaussian_sde':                   's',
    'tilted_stable_sde_double_well':  '^',
    'gaussian_sde_double_well':       'D',
}
MODEL_LINESTYLE = {
    'tilted_stable_sde':              '-',
    'gaussian_sde':                   '--',
    'tilted_stable_sde_double_well':  '-.',
    'gaussian_sde_double_well':       ':',
}

PATH_METRICS = [
    'mse_training', 'mse_heldout', 'mse_all',
    'mae_training', 'mae_heldout', 'mae_all',
    'crps_training', 'crps_heldout', 'crps_all',
]

JUMP_PERCENTILES = [90, 95, 97.5, 99]
_JUMP_PCT_STRS = ['p90', 'p95', 'p97_5', 'p99']   # key suffixes matching evaluate_runs.py

JUMP_METRICS = [
    f'jump_{base}_{pct}_{subset}'
    for base in ['crps', 'mae', 'mse']
    for pct in _JUMP_PCT_STRS
    for subset in ['heldout', 'all']
]

DRIFT_SUB_KEYS = ['theta_error', 'mu_error', 'theta1_error', 'theta2_error']

# Comparison pairs: (tilted_stable_type, gaussian_type) — used in relative improvement plots
_COMPARISON_PAIRS = [
    ('tilted_stable_sde',             'gaussian_sde'),
    ('tilted_stable_sde_double_well', 'gaussian_sde_double_well'),
]


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate cross-model comparison results and produce plots/tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--comparisons-dir',
        default='evaluation/results/comparisons',
        help='Directory containing comparison subdirectories (default: evaluation/results/comparisons)',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for plots and CSV (default: evaluation/results/analysis)',
    )
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_comparisons(comparisons_dir: Path) -> List[Dict]:
    """
    Discover and load every metric_comparison.json under comparisons_dir.

    Returns a list of records, each containing:
        alpha      : float
        data_seed  : int
        metrics    : {model_type: {metric_name: float}}
    """
    dir_pattern = re.compile(
        r'alpha_(?P<alpha>[\d.]+)_obsstd_(?P<obs_std>[\d.]+)_data_(?P<data_seed>\d+)_'
    )

    records = []
    for comp_dir in sorted(comparisons_dir.iterdir()):
        if not comp_dir.is_dir():
            continue

        json_path = comp_dir / 'metric_comparison.json'
        if not json_path.exists():
            continue

        m = dir_pattern.match(comp_dir.name)
        if not m:
            print(f'  [SKIP] Cannot parse directory name: {comp_dir.name}')
            continue

        alpha     = float(m.group('alpha'))
        data_seed = int(m.group('data_seed'))

        with open(json_path) as f:
            comp = json.load(f)

        run_to_model = {e['run_id']: e['model_type'] for e in comp['models']}
        model_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

        for metric_name, values_by_run in comp['metrics'].items():
            for run_id, value in values_by_run.items():
                mt = run_to_model.get(run_id)
                if mt is None:
                    continue
                if isinstance(value, (int, float)):
                    model_metrics[mt][metric_name] = float(value)
                elif isinstance(value, dict):
                    # drift_error is a nested dict; flatten into drift_error_{key}
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            model_metrics[mt][f'drift_error_{k}'] = float(v)

        records.append({
            'alpha':     alpha,
            'data_seed': data_seed,
            'metrics':   dict(model_metrics),
        })

    return records


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_by_alpha(records: List[Dict], metric_names: List[str]):
    """
    Group records by alpha value and compute per-seed statistics.

    Returns:
        alphas : sorted list of alpha values
        agg    : alpha -> model_type -> metric -> {mean, std, n, values}
                 (or None if no data for that combination)
    """
    by_alpha: Dict[float, List[Dict]] = defaultdict(list)
    for rec in records:
        by_alpha[rec['alpha']].append(rec)

    alphas = sorted(by_alpha.keys())
    agg: Dict = {}

    for alpha in alphas:
        agg[alpha] = {}
        for mt in MODEL_TYPES:
            agg[alpha][mt] = {}
            for metric in metric_names:
                vals = [
                    rec['metrics'].get(mt, {}).get(metric)
                    for rec in by_alpha[alpha]
                    if rec['metrics'].get(mt, {}).get(metric) is not None
                ]
                if vals:
                    agg[alpha][mt][metric] = {
                        'mean':   float(np.mean(vals)),
                        'std':    float(np.std(vals)),
                        'n':      len(vals),
                        'values': vals,
                    }
                else:
                    agg[alpha][mt][metric] = None

    return alphas, agg


def compute_global_stats(records: List[Dict], metric_names: List[str]) -> Dict:
    """Pool all records across alpha values and compute overall mean/std."""
    global_stats: Dict[str, Dict] = {}
    for mt in MODEL_TYPES:
        global_stats[mt] = {}
        for metric in metric_names:
            vals = [
                rec['metrics'].get(mt, {}).get(metric)
                for rec in records
                if rec['metrics'].get(mt, {}).get(metric) is not None
            ]
            global_stats[mt][metric] = {
                'mean':   float(np.mean(vals)) if vals else None,
                'std':    float(np.std(vals))  if vals else None,
                'n':      len(vals),
                'values': vals,
            } if vals else None
    return global_stats


# =============================================================================
# TABLE OUTPUT
# =============================================================================

def _fmt(entry: Optional[Dict]) -> str:
    if entry is None or entry['mean'] is None:
        return 'N/A'
    return f"{entry['mean']:.4f} ± {entry['std']:.4f}"


def _fmt_iqr(entry: Optional[Dict]) -> str:
    if entry is None or not entry.get('values'):
        return 'N/A'
    vals = np.array(entry['values'])
    return f"{np.mean(vals):.4f} ({np.percentile(vals, 25):.4f}–{np.percentile(vals, 75):.4f})"


def print_and_save_table(
    title: str,
    metrics: List[str],
    metric_short_labels: Dict[str, str],
    alphas: List[float],
    agg: Dict,
    global_stats: Dict,
    output_path: Path,
    fmt_func=None,
):
    """Print a metric table to stdout and save it as a CSV file."""
    if fmt_func is None:
        fmt_func = _fmt

    print(f'\n{"="*80}')
    print(title)
    print(f'{"="*80}')

    # Column structure: alpha | (TS metric, GS metric) for each metric
    col_headers = ['alpha']
    for metric in metrics:
        short = metric_short_labels.get(metric, metric)
        for mt in MODEL_TYPES:
            col_headers.append(f'{MODEL_LABEL[mt]} {short}')

    rows = []

    for alpha in alphas:
        row = {'alpha': f'{alpha:.2f}'}
        for metric in metrics:
            for mt in MODEL_TYPES:
                key = f'{MODEL_LABEL[mt]} {metric_short_labels.get(metric, metric)}'
                row[key] = fmt_func(agg[alpha][mt].get(metric))
        rows.append(row)

    # Global row
    global_row = {'alpha': 'All α'}
    for metric in metrics:
        for mt in MODEL_TYPES:
            key = f'{MODEL_LABEL[mt]} {metric_short_labels.get(metric, metric)}'
            global_row[key] = fmt_func(global_stats[mt].get(metric))
    rows.append(global_row)

    # Console: compute column widths
    col_widths = {
        col: max(len(col), max(len(r.get(col, '')) for r in rows))
        for col in col_headers
    }
    header_line = '  '.join(c.ljust(col_widths[c]) for c in col_headers)
    print(header_line)
    print('-' * len(header_line))
    for i, row in enumerate(rows):
        if i == len(rows) - 1:
            print('-' * len(header_line))  # separator before global row
        print('  '.join(row.get(c, '').ljust(col_widths[c]) for c in col_headers))
    print()

    # CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=col_headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Saved: {output_path}')


# =============================================================================
# PLOTS
# =============================================================================

def plot_metrics_vs_alpha(
    alphas: List[float],
    agg: Dict,
    output_path: Path,
    log_scale: bool = False,
):
    """
    4-panel figure: MSE all, MSE heldout, MAE all, MAE heldout vs alpha.
    Each panel shows both model types with ±1 std shaded band.
    """
    panels = [
        ('mse_all',     'MSE (all obs)'),
        ('mse_heldout', 'MSE (held-out obs)'),
        ('mae_all',     'MAE (all obs)'),
        ('mae_heldout', 'MAE (held-out obs)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    alpha_arr = np.array(alphas)

    for idx, (metric, label) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]

        for mt in MODEL_TYPES:
            medians, q25s, q75s, valid = [], [], [], []
            for alpha in alphas:
                entry = agg[alpha][mt].get(metric)
                if entry is not None and entry.get('values'):
                    vals = np.array(entry['values'])
                    medians.append(np.median(vals))
                    q25s.append(np.percentile(vals, 25))
                    q75s.append(np.percentile(vals, 75))
                    valid.append(alpha)

            if not medians:
                continue

            medians = np.array(medians)
            q25s    = np.array(q25s)
            q75s    = np.array(q75s)
            valid   = np.array(valid)

            ax.plot(
                valid, medians,
                color=MODEL_COLOR[mt],
                linestyle=MODEL_LINESTYLE[mt],
                marker=MODEL_MARKER[mt],
                markersize=7,
                linewidth=2.0,
                label=MODEL_LABEL[mt],
            )
            ax.fill_between(valid, q25s, q75s, color=MODEL_COLOR[mt], alpha=0.12)

        ax.set_xlabel('α (stability index)', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} vs α', fontsize=13, fontweight='bold')
        ax.set_xticks(alphas)
        ax.set_xticklabels([f'{a:.1f}' for a in alphas])
        if log_scale:
            ax.set_yscale('log')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')

    scale_note = 'log scale' if log_scale else 'linear scale'
    plt.suptitle(
        f'Tilted Stable SDE vs Gaussian SDE — Performance vs α\n'
        f'(median with IQR across data seeds, {scale_note})',
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_drift_error_vs_alpha(
    alphas: List[float],
    agg: Dict,
    output_path: Path,
    log_scale: bool = False,
):
    """
    N-panel figure: one panel per drift error metric present in the data.

    OU drift:          theta_error, mu_error
    Double-well drift: theta1_error, theta2_error
    """
    _ALL_DRIFT_PANELS = [
        ('drift_error_theta_error',  'θ error (mean reversion)'),
        ('drift_error_mu_error',     'μ error (long-run mean)'),
        ('drift_error_theta1_error', 'θ₁ error (linear term)'),
        ('drift_error_theta2_error', 'θ₂ error (cubic term)'),
    ]

    panels = [
        (metric, label)
        for metric, label in _ALL_DRIFT_PANELS
        if any(
            agg[alpha][mt].get(metric) is not None
            for alpha in alphas for mt in MODEL_TYPES
        )
    ]

    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5))
    fig.patch.set_facecolor('white')
    if len(panels) == 1:
        axes = [axes]

    for idx, (metric, label) in enumerate(panels):
        ax = axes[idx]

        for mt in MODEL_TYPES:
            medians, q25s, q75s, valid = [], [], [], []
            for alpha in alphas:
                entry = agg[alpha][mt].get(metric)
                if entry is not None and entry.get('values'):
                    vals = np.array(entry['values'])
                    medians.append(np.median(vals))
                    q25s.append(np.percentile(vals, 25))
                    q75s.append(np.percentile(vals, 75))
                    valid.append(alpha)

            if not medians:
                continue

            medians = np.array(medians)
            q25s    = np.array(q25s)
            q75s    = np.array(q75s)
            valid   = np.array(valid)

            ax.plot(
                valid, medians,
                color=MODEL_COLOR[mt],
                linestyle=MODEL_LINESTYLE[mt],
                marker=MODEL_MARKER[mt],
                markersize=7,
                linewidth=2.0,
                label=MODEL_LABEL[mt],
            )
            ax.fill_between(valid, q25s, q75s, color=MODEL_COLOR[mt], alpha=0.12)

        ax.set_xlabel('α (stability index)', fontsize=12)
        ax.set_ylabel(f'|error|', fontsize=12)
        ax.set_title(f'{label} vs α', fontsize=13, fontweight='bold')
        ax.set_xticks(alphas)
        ax.set_xticklabels([f'{a:.1f}' for a in alphas])
        if log_scale:
            ax.set_yscale('log')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')

    scale_note = 'log scale' if log_scale else 'linear scale'
    plt.suptitle(
        f'Drift Parameter Recovery vs α\n(median with IQR across data seeds, {scale_note})',
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_crps_vs_alpha(
    alphas: List[float],
    agg: Dict,
    output_path: Path,
    log_scale: bool = False,
):
    """
    2-panel figure: CRPS (all obs) and CRPS (held-out obs) vs alpha for both models.
    """
    panels = [
        ('crps_all',     'CRPS (all obs)'),
        ('crps_heldout', 'CRPS (held-out obs)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    for idx, (metric, label) in enumerate(panels):
        ax = axes[idx]

        for mt in MODEL_TYPES:
            medians, q25s, q75s, valid = [], [], [], []
            for alpha in alphas:
                entry = agg[alpha][mt].get(metric)
                if entry is not None and entry.get('values'):
                    vals = np.array(entry['values'])
                    medians.append(np.median(vals))
                    q25s.append(np.percentile(vals, 25))
                    q75s.append(np.percentile(vals, 75))
                    valid.append(alpha)

            if not medians:
                continue

            medians = np.array(medians)
            q25s    = np.array(q25s)
            q75s    = np.array(q75s)
            valid   = np.array(valid)

            ax.plot(
                valid, medians,
                color=MODEL_COLOR[mt],
                linestyle=MODEL_LINESTYLE[mt],
                marker=MODEL_MARKER[mt],
                markersize=7,
                linewidth=2.0,
                label=MODEL_LABEL[mt],
            )
            ax.fill_between(valid, q25s, q75s, color=MODEL_COLOR[mt], alpha=0.12)

        ax.set_xlabel('α (stability index)', fontsize=12)
        ax.set_ylabel('CRPS', fontsize=12)
        ax.set_title(f'{label} vs α', fontsize=13, fontweight='bold')
        ax.set_xticks(alphas)
        ax.set_xticklabels([f'{a:.1f}' for a in alphas])
        if log_scale:
            ax.set_yscale('log')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8f8')

    scale_note = 'log scale' if log_scale else 'linear scale'
    plt.suptitle(
        f'Tilted Stable SDE vs Gaussian SDE — CRPS vs α\n'
        f'(median with IQR across data seeds, {scale_note})',
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_jump_metrics_vs_alpha(
    alphas: List[float],
    agg: Dict,
    output_path: Path,
    log_scale: bool = False,
):
    """
    3×4 grid: rows = MSE / MAE / CRPS, columns = p90 / p95 / p97.5 / p99.

    Each cell shows jump-conditioned metric (held-out) vs alpha for both models,
    making it easy to see how the advantage changes across both metric type and
    jump severity threshold.
    """
    row_bases = [
        ('jump_mse',  'Jump-MSE'),
        ('jump_mae',  'Jump-MAE'),
        ('jump_crps', 'Jump-CRPS'),
    ]
    col_pcts = [
        ('p90',   'p90'),
        ('p95',   'p95'),
        ('p97_5', 'p97.5'),
        ('p99',   'p99'),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(28, 16), sharey='row')
    fig.patch.set_facecolor('white')

    for row_idx, (base, row_label) in enumerate(row_bases):
        for col_idx, (pct_key, pct_label) in enumerate(col_pcts):
            ax = axes[row_idx][col_idx]
            metric = f'{base}_{pct_key}_heldout'

            for mt in MODEL_TYPES:
                medians, q25s, q75s, valid = [], [], [], []
                for alpha in alphas:
                    entry = agg[alpha][mt].get(metric)
                    if entry is not None and entry.get('values'):
                        vals = np.array(entry['values'])
                        medians.append(np.median(vals))
                        q25s.append(np.percentile(vals, 25))
                        q75s.append(np.percentile(vals, 75))
                        valid.append(alpha)

                if not medians:
                    continue

                medians = np.array(medians)
                q25s    = np.array(q25s)
                q75s    = np.array(q75s)
                valid   = np.array(valid)

                ax.plot(
                    valid, medians,
                    color=MODEL_COLOR[mt],
                    linestyle=MODEL_LINESTYLE[mt],
                    marker=MODEL_MARKER[mt],
                    markersize=6,
                    linewidth=1.8,
                    label=MODEL_LABEL[mt],
                )
                ax.fill_between(valid, q25s, q75s, color=MODEL_COLOR[mt], alpha=0.12)

            ax.set_xlabel('α', fontsize=10)
            ax.set_ylabel(row_label, fontsize=10)
            ax.set_title(f'{row_label} — {pct_label} (held-out)', fontsize=10, fontweight='bold')
            ax.set_xticks(alphas)
            ax.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=8)
            if log_scale:
                ax.set_yscale('log')
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f8f8')

    scale_note = 'log scale' if log_scale else 'linear scale'
    plt.suptitle(
        f'Tilted Stable SDE vs Gaussian SDE — Jump-conditioned Metrics vs α\n'
        f'(median with IQR across data seeds, {scale_note})',
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_relative_improvement_vs_alpha(
    alphas: List[float],
    agg: Dict,
    output_path: Path,
):
    """
    3-panel figure showing relative improvement (%) of tilted stable over Gaussian
    for MSE, MAE, and CRPS on held-out observations. Positive = tilted stable is better.

    Automatically detects the comparison pair (original or double well) from the data.
    """
    panels = [
        ('mse_heldout',  'MSE (held-out)'),
        ('mae_heldout',  'MAE (held-out)'),
        ('crps_heldout', 'CRPS (held-out)'),
    ]

    # Detect which comparison pair is present in the aggregated data
    ts_type, gs_type = _COMPARISON_PAIRS[0]  # default: original
    for ts_candidate, gs_candidate in _COMPARISON_PAIRS:
        if alphas and agg[alphas[0]].get(ts_candidate) and any(
            agg[alpha][ts_candidate].get('mse_heldout') is not None for alpha in alphas
        ):
            ts_type, gs_type = ts_candidate, gs_candidate
            break

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.patch.set_facecolor('white')

    for idx, (metric, label) in enumerate(panels):
        ax = axes[idx]

        rel_improv, valid_alphas = [], []
        for alpha in alphas:
            ts = agg[alpha][ts_type].get(metric)
            gs = agg[alpha][gs_type].get(metric)
            if ts is None or gs is None:
                continue
            if gs['mean'] == 0:
                continue
            # positive = tilted stable is better (lower error)
            pct = (gs['mean'] - ts['mean']) / gs['mean'] * 100
            rel_improv.append(pct)
            valid_alphas.append(alpha)

        if not rel_improv:
            continue

        rel_improv    = np.array(rel_improv)
        valid_alphas  = np.array(valid_alphas)

        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in rel_improv]
        ax.bar(valid_alphas, rel_improv, color=colors, alpha=0.75, width=0.07)
        ax.axhline(0, color='black', linewidth=1.0, linestyle='-')

        ax.set_xlabel('α (stability index)', fontsize=12)
        ax.set_ylabel('Relative improvement (%)', fontsize=12)
        ax.set_title(
            f'{MODEL_LABEL[ts_type]} vs {MODEL_LABEL[gs_type]} — {label}\n(positive = tilted stable wins)',
            fontsize=12, fontweight='bold',
        )
        ax.set_xticks(valid_alphas)
        ax.set_xticklabels([f'{a:.1f}' for a in valid_alphas])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#f8f8f8')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_arguments()

    comparisons_dir = Path(args.comparisons_dir)
    if not comparisons_dir.exists():
        print(f'Error: comparisons directory not found: {comparisons_dir}')
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path('evaluation/results/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f'\nLoading comparisons from: {comparisons_dir}')
    records = load_all_comparisons(comparisons_dir)
    print(f'  Found {len(records)} comparison records')

    if not records:
        print('No records found — nothing to analyse.')
        sys.exit(1)

    all_metrics = PATH_METRICS + JUMP_METRICS + [f'drift_error_{k}' for k in DRIFT_SUB_KEYS]

    # ── Aggregate ─────────────────────────────────────────────────────────────
    alphas, agg    = aggregate_by_alpha(records, all_metrics)
    global_stats   = compute_global_stats(records, all_metrics)

    print(f'\n  Alpha values: {[f"{a:.1f}" for a in alphas]}')
    first_mt = next((mt for mt in MODEL_TYPES if agg[alphas[0]].get(mt, {}).get(PATH_METRICS[0])), None)
    n_seeds = agg[alphas[0]][first_mt][PATH_METRICS[0]]['n'] if first_mt else '?'
    print(f'  Seeds per alpha: {n_seeds}')

    # ── Tables ────────────────────────────────────────────────────────────────
    _mse_labels = {
        'mse_training': 'mse_train',
        'mse_heldout':  'mse_heldout',
        'mse_all':      'mse_all',
    }
    print_and_save_table(
        title='MSE by alpha',
        metrics=['mse_training', 'mse_heldout', 'mse_all'],
        metric_short_labels=_mse_labels,
        alphas=alphas, agg=agg, global_stats=global_stats,
        output_path=output_dir / 'table_mse.csv',
    )
    print_and_save_table(
        title='MSE by alpha (mean with IQR)',
        metrics=['mse_training', 'mse_heldout', 'mse_all'],
        metric_short_labels=_mse_labels,
        alphas=alphas, agg=agg, global_stats=global_stats,
        output_path=output_dir / 'table_mse_iqr.csv',
        fmt_func=_fmt_iqr,
    )

    _mae_labels = {
        'mae_training': 'mae_train',
        'mae_heldout':  'mae_heldout',
        'mae_all':      'mae_all',
    }
    print_and_save_table(
        title='MAE by alpha',
        metrics=['mae_training', 'mae_heldout', 'mae_all'],
        metric_short_labels=_mae_labels,
        alphas=alphas, agg=agg, global_stats=global_stats,
        output_path=output_dir / 'table_mae.csv',
    )
    print_and_save_table(
        title='MAE by alpha (mean with IQR)',
        metrics=['mae_training', 'mae_heldout', 'mae_all'],
        metric_short_labels=_mae_labels,
        alphas=alphas, agg=agg, global_stats=global_stats,
        output_path=output_dir / 'table_mae_iqr.csv',
        fmt_func=_fmt_iqr,
    )

    drift_metrics_present = [
        f'drift_error_{k}' for k in DRIFT_SUB_KEYS
        if any(
            agg[a][mt].get(f'drift_error_{k}') is not None
            for a in alphas for mt in MODEL_TYPES
        )
    ]
    if drift_metrics_present:
        _drift_labels = {f'drift_error_{k}': k for k in DRIFT_SUB_KEYS}
        print_and_save_table(
            title='Drift parameter recovery by alpha',
            metrics=drift_metrics_present,
            metric_short_labels=_drift_labels,
            alphas=alphas, agg=agg, global_stats=global_stats,
            output_path=output_dir / 'table_drift_error.csv',
        )
        print_and_save_table(
            title='Drift parameter recovery by alpha (mean with IQR)',
            metrics=drift_metrics_present,
            metric_short_labels=_drift_labels,
            alphas=alphas, agg=agg, global_stats=global_stats,
            output_path=output_dir / 'table_drift_error_iqr.csv',
            fmt_func=_fmt_iqr,
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating plots...')

    plot_metrics_vs_alpha(alphas, agg, output_dir / 'metrics_vs_alpha.png')
    plot_metrics_vs_alpha(alphas, agg, output_dir / 'metrics_vs_alpha_log.png', log_scale=True)

    if drift_metrics_present:
        plot_drift_error_vs_alpha(alphas, agg, output_dir / 'drift_error_vs_alpha.png')
        plot_drift_error_vs_alpha(alphas, agg, output_dir / 'drift_error_vs_alpha_log.png', log_scale=True)

    crps_metrics_present = [
        m for m in ['crps_training', 'crps_heldout', 'crps_all']
        if any(
            agg[a][mt].get(m) is not None
            for a in alphas for mt in MODEL_TYPES
        )
    ]
    if crps_metrics_present:
        _crps_labels = {
            'crps_training': 'crps_train',
            'crps_heldout':  'crps_heldout',
            'crps_all':      'crps_all',
        }
        print_and_save_table(
            title='CRPS by alpha',
            metrics=crps_metrics_present,
            metric_short_labels=_crps_labels,
            alphas=alphas, agg=agg, global_stats=global_stats,
            output_path=output_dir / 'table_crps.csv',
        )
        print_and_save_table(
            title='CRPS by alpha (mean with IQR)',
            metrics=crps_metrics_present,
            metric_short_labels=_crps_labels,
            alphas=alphas, agg=agg, global_stats=global_stats,
            output_path=output_dir / 'table_crps_iqr.csv',
            fmt_func=_fmt_iqr,
        )
        plot_crps_vs_alpha(alphas, agg, output_dir / 'crps_vs_alpha.png')
        plot_crps_vs_alpha(alphas, agg, output_dir / 'crps_vs_alpha_log.png', log_scale=True)

    jump_crps_metrics_present = [
        m for m in JUMP_METRICS if m.startswith('jump_crps_') and m.endswith('_heldout')
        and any(
            agg[a][mt].get(m) is not None
            for a in alphas for mt in MODEL_TYPES
        )
    ]
    if jump_crps_metrics_present:
        _jump_labels = {m: m.replace('jump_crps_', '').replace('_heldout', '') for m in jump_crps_metrics_present}
        print_and_save_table(
            title='Jump-conditioned CRPS (held-out) by alpha',
            metrics=jump_crps_metrics_present,
            metric_short_labels=_jump_labels,
            alphas=alphas, agg=agg, global_stats=global_stats,
            output_path=output_dir / 'table_jump_crps.csv',
        )
        print_and_save_table(
            title='Jump-conditioned CRPS (held-out) by alpha (mean with IQR)',
            metrics=jump_crps_metrics_present,
            metric_short_labels=_jump_labels,
            alphas=alphas, agg=agg, global_stats=global_stats,
            output_path=output_dir / 'table_jump_crps_iqr.csv',
            fmt_func=_fmt_iqr,
        )
        plot_jump_metrics_vs_alpha(alphas, agg, output_dir / 'jump_metrics_vs_alpha.png')
        plot_jump_metrics_vs_alpha(alphas, agg, output_dir / 'jump_metrics_vs_alpha_log.png', log_scale=True)

    plot_relative_improvement_vs_alpha(
        alphas, agg, output_dir / 'relative_improvement_vs_alpha.png'
    )

    print(f'\nAll outputs written to: {output_dir}')


if __name__ == '__main__':
    main()
