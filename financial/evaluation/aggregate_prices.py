"""Aggregate per-window metrics for prices experiments and produce comparison plots.

Supports an arbitrary number of models.  The legacy --ts-name / --gauss-name
flags are still fully supported for backward compatibility and automatically
map to display labels "Tilted Stable" / "Gaussian".

Usage (legacy — identical behaviour to before):
    python financial/evaluation/aggregate_prices.py \\
        --ts-name   nvda_ts_a1.90_sig1.0_obsstd0.10_train30d_fc2d_d2w32_prices \\
        --gauss-name nvda_gaussian_sig1.0_obsstd0.10_train30d_fc2d_prices

Usage (multi-model):
    python financial/evaluation/aggregate_prices.py \\
        --model nvda_ts_a1.90_..._prices:"Tilted Stable" \\
        --model nvda_gaussian_..._prices:Gaussian \\
        --model nvda_deepar_..._prices:DeepAR \\
        --model nvda_dlinear_..._prices:DLinear
"""

import sys
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from financial.evaluation.metrics import DENSE_COVERAGE_LEVELS

RESULTS_DIR     = Path(__file__).parent.parent / 'results'
QUANTILE_LEVELS = DENSE_COVERAGE_LEVELS

PALETTE    = ['steelblue', 'tomato', 'forestgreen', 'gold', 'mediumpurple', 'saddlebrown', 'deeppink']


def _analysis_dir(names: list, max_len: int = 200) -> Path:
    """Return analysis output directory, hashing the name if it would be too long."""
    joined = '_vs_'.join(names)
    if len(joined) <= max_len:
        return RESULTS_DIR / 'analysis' / joined
    h = hashlib.sha1(joined.encode()).hexdigest()[:8]
    short = f"{names[0][:60]}_vs_{len(names)-1}_others_{h}"
    return RESULTS_DIR / 'analysis' / short
MARKERS    = ['o', 's', '^', 'D', 'v', 'P', 'X']
BOX_COLORS = ['lightblue', 'lightsalmon', 'lightgreen', 'moccasin', 'thistle', 'burlywood']
COL_W = 22   # characters per model column in the comparison table


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(config_name: str, metrics_suffix: str = '') -> list:
    run_dir  = RESULTS_DIR / config_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Results not found: {run_dir}")
    filename = f'metrics{metrics_suffix}.json' if metrics_suffix else 'metrics.json'
    records  = []
    for window_dir in sorted(run_dir.glob('window_*')):
        p = window_dir / filename
        if p.exists():
            with open(p) as f:
                records.append(json.load(f))
    print(f"  {config_name}: {len(records)} windows loaded")
    return records


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def summarise(records: list) -> dict:
    crps_vals     = [r['mean_crps'] for r in records if not np.isnan(r['mean_crps'])]
    sum_crps_vals = [r['sum_crps'] for r in records if 'sum_crps' in r and not np.isnan(r['sum_crps'])]
    es_vals       = [r['mean_energy_score'] for r in records]
    total_div     = sum(r.get('n_diverged_samples', 0) for r in records)
    n_eff_vals    = [r['n_effective_samples'] for r in records if 'n_effective_samples' in r]

    summary = {
        'n_windows':             len(records),
        'crps_mean':             float(np.mean(crps_vals))       if crps_vals     else float('nan'),
        'crps_std':              float(np.std(crps_vals))        if crps_vals     else float('nan'),
        'crps_median':           float(np.median(crps_vals))     if crps_vals     else float('nan'),
        'sum_crps_mean':         float(np.mean(sum_crps_vals))   if sum_crps_vals else float('nan'),
        'sum_crps_std':          float(np.std(sum_crps_vals))    if sum_crps_vals else float('nan'),
        'sum_crps_median':       float(np.median(sum_crps_vals)) if sum_crps_vals else float('nan'),
        'energy_score_mean':     float(np.mean(es_vals)),
        'energy_score_std':      float(np.std(es_vals)),
        'total_diverged_samples': int(total_div),
        'mean_n_effective':      float(np.mean(n_eff_vals)) if n_eff_vals else float('nan'),
        'coverage': {},
    }
    for q in QUANTILE_LEVELS:
        q_key    = str(float(q))
        cov_vals = [r['coverage'][q_key] for r in records if q_key in r['coverage']]
        if cov_vals:
            summary['coverage'][q] = {
                'mean': float(np.mean(cov_vals)),
                'std':  float(np.std(cov_vals)),
            }
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_ms(mean, std):
    return f"{mean:.4f} \u00b1 {std:.4f}"   # "mean ± std"


def print_comparison_table(models: list, summaries: list):
    """Print a multi-model comparison table.

    Parameters
    ----------
    models    : list of (result_name, display_label)
    summaries : list of summary dicts (same order)
    """
    labels  = [lbl for _, lbl in models]
    row_w   = 30
    total_w = row_w + COL_W * len(models)

    print(f"\n{'='*total_w}")
    print(f"{'Metric':<{row_w}}" + ''.join(f"{lbl:>{COL_W}}" for lbl in labels))
    print(f"{'='*total_w}")

    def row(name, vals):
        return f"{name:<{row_w}}" + ''.join(f"{v:>{COL_W}}" for v in vals)

    print(row('Windows evaluated',
              [str(s['n_windows']) for s in summaries]))
    print(row('CRPS (mean \u00b1 std)',
              [_fmt_ms(s['crps_mean'], s['crps_std']) for s in summaries]))
    print(row('CRPS (median)',
              [f"{s['crps_median']:.4f}" for s in summaries]))
    print(row('Sum-CRPS (mean \u00b1 std)',
              [_fmt_ms(s['sum_crps_mean'], s['sum_crps_std']) for s in summaries]))
    print(row('Sum-CRPS (median)',
              [f"{s['sum_crps_median']:.4f}" for s in summaries]))
    print(row('Energy Score (mean \u00b1 std)',
              [_fmt_ms(s['energy_score_mean'], s['energy_score_std']) for s in summaries]))
    print(row('Total diverged samples',
              [str(s['total_diverged_samples']) for s in summaries]))
    print(row('Mean eff. sample size',
              [f"{s['mean_n_effective']:.1f}" for s in summaries]))

    print(f"\n{'Coverage (nominal \u2192 actual)':}")
    for q in QUANTILE_LEVELS:
        covs     = [s['coverage'].get(q, {}).get('mean', float('nan')) for s in summaries]
        cov_strs = ''.join(f"{c:>{COL_W}.3f}" for c in covs)
        print(f"  {q:.0%} interval: {cov_strs}  (nominal={q:.3f})")
    print(f"{'='*total_w}\n")
    print("Note: CRPS computed in normalised cumulative log-price space.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability(models: list, summaries: list, output_path: Path):
    nominal = QUANTILE_LEVELS
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    for i, ((_, label), summary) in enumerate(zip(models, summaries)):
        actual = [summary['coverage'].get(q, {}).get('mean', np.nan) for q in nominal]
        ax.plot(nominal, actual,
                marker=MARKERS[i % len(MARKERS)],
                color=PALETTE[i % len(PALETTE)],
                linewidth=2, markersize=7, label=label,
                zorder=3 if i == 0 else 2)
    ax.set_xlabel('Nominal coverage level', fontsize=14)
    ax.set_ylabel('Empirical coverage', fontsize=14)
    ax.set_title('Reliability diagram (prices — normalised log-price space)')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_crps_comparison(models: list, all_records: list, output_path: Path,
                         log_scale: bool = False, clip_percentile: float = None):
    labels   = [lbl for _, lbl in models]
    all_crps = [
        [r['mean_crps'] for r in records if not np.isnan(r['mean_crps'])]
        for records in all_records
    ]
    fig_w = max(5, 2.5 * len(models))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    bp = ax.boxplot(all_crps, tick_labels=labels,
                    patch_artist=True,
                    medianprops=dict(color='navy', linewidth=2))
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(BOX_COLORS[i % len(BOX_COLORS)])
    if log_scale:
        ax.set_yscale('log')
    elif clip_percentile is not None:
        all_vals = [v for crps in all_crps for v in crps]
        if all_vals:
            ax.set_ylim(bottom=0, top=float(np.percentile(all_vals, clip_percentile)))
    ax.set_ylabel('CRPS (normalised log-price space)')
    title = 'Per-window CRPS distribution (prices)'
    if log_scale:
        title += ' — log scale'
    elif clip_percentile is not None:
        title += f' — y clipped at p{clip_percentile:.0f}'
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_models(args) -> list:
    """Build ordered list of (result_name, display_label) from parsed args."""
    models = []
    if args.ts_name:
        models.append((args.ts_name, 'Tilted Stable'))
    if args.gauss_name:
        models.append((args.gauss_name, 'Gaussian'))
    for spec in (args.model or []):
        if ':' in spec:
            name, label = spec.split(':', 1)
        else:
            name, label = spec, spec
        models.append((name, label))
    return models


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate per-window metrics for financial prices experiments.')
    # Legacy backward-compat flags
    parser.add_argument('--ts-name',    default='',
                        help='(legacy) result name for Tilted Stable; '
                             'equivalent to --model NAME:"Tilted Stable"')
    parser.add_argument('--gauss-name', default='',
                        help='(legacy) result name for Gaussian; '
                             'equivalent to --model NAME:Gaussian')
    # New multi-model flag
    parser.add_argument('--model', action='append', default=[],
                        metavar='NAME[:LABEL]',
                        help='Add a model by result name with an optional display '
                             'label after the first colon.  May be repeated.')
    parser.add_argument('--metrics-suffix', default='',
                        help='Load metrics{suffix}.json instead of metrics.json.')
    args = parser.parse_args()

    models = _parse_models(args)
    if len(models) < 2:
        parser.error('At least two models are required.  Use --ts-name/--gauss-name '
                     'or --model NAME[:LABEL] (repeated).')

    print("Loading results...")
    all_records = [load_metrics(name, metrics_suffix=args.metrics_suffix)
                   for name, _ in models]
    summaries   = [summarise(records) for records in all_records]

    print_comparison_table(models, summaries)

    out_dir = _analysis_dir([name for name, _ in models])
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.metrics_suffix or ''
    for (name, _), summary in zip(models, summaries):
        with open(out_dir / f'summary_{name}{suffix}.json', 'w') as f:
            json.dump(summary, f, indent=2)

    plot_reliability(models, summaries,       out_dir / f'reliability_diagram{suffix}.png')
    plot_crps_comparison(models, all_records, out_dir / f'crps_comparison{suffix}.png')
    plot_crps_comparison(models, all_records, out_dir / f'crps_comparison_log{suffix}.png',
                         log_scale=True)
    plot_crps_comparison(models, all_records, out_dir / f'crps_comparison_p95{suffix}.png',
                         clip_percentile=95)
    print(f"Analysis outputs in {out_dir}")


if __name__ == '__main__':
    main()
