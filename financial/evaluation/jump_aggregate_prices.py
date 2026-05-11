"""Jump-conditioned evaluation for prices experiments.

Because forecast_obs are cumulative log-prices (not per-step returns), jump
magnitudes are computed from INCREMENTS: |forecast_obs[h] - forecast_obs[h-1]|.
This recovers the individual price move at each step, consistent with the
jump-conditioned metrics in the returns experiment.

Step h=0 is excluded (would need the last training observation to compute
its increment).  Typical forecast horizon is 13-14 steps so this is minor.

Supports an arbitrary number of models.  The legacy --ts-name / --gauss-name
flags are still fully supported.

Usage (legacy):
    python financial/evaluation/jump_aggregate_prices.py \\
        --ts-name   nvda_ts_a1.90_sig4.0_obsstd0.10_train30d_fc2d_d2w32_prices \\
        --gauss-name nvda_gaussian_sig4.0_obsstd0.10_train30d_fc2d_prices

Usage (multi-model):
    python financial/evaluation/jump_aggregate_prices.py \\
        --model nvda_ts_...:"Tilted Stable" \\
        --model nvda_gaussian_...:Gaussian \\
        --model nvda_deepar_...:DeepAR
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

RESULTS_DIR    = Path(__file__).parent.parent / 'results'
PERCENTILES    = [90, 95, 97.5, 99]
PERCENTILE_KEYS = ['p90', 'p95', 'p97_5', 'p99']

PALETTE    = ['steelblue', 'tomato', 'forestgreen', 'darkorange', 'mediumpurple', 'saddlebrown']


def _analysis_dir(names: list, max_len: int = 200) -> Path:
    """Return analysis output directory, hashing the name if it would be too long."""
    joined = '_vs_'.join(names)
    if len(joined) <= max_len:
        return RESULTS_DIR / 'analysis' / joined
    h = hashlib.sha1(joined.encode()).hexdigest()[:8]
    short = f"{names[0][:60]}_vs_{len(names)-1}_others_{h}"
    return RESULTS_DIR / 'analysis' / short


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(config_name: str, metrics_suffix: str = '') -> list:
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
# Jump-conditioned scoring
# ---------------------------------------------------------------------------

def compute_thresholds(records: list) -> dict:
    """Compute global jump thresholds from per-step increments |obs[h] - obs[h-1]|.

    Step h=0 is excluded (no previous step available without last train obs).
    Thresholds are computed from the first model's records and then applied
    to all models identically (model-agnostic, observation-derived).
    """
    all_magnitudes = []
    for r in records:
        obs = np.array(r['forecast_obs'])
        if obs.ndim == 1:
            obs = obs[:, None]
        if obs.shape[0] < 2:
            continue
        increments = np.diff(obs, axis=0)                       # (H-1, D)
        magnitudes = np.max(np.abs(increments), axis=1)         # (H-1,)
        all_magnitudes.append(magnitudes)
    magnitudes_all = np.concatenate(all_magnitudes)
    return {
        p_key: float(np.percentile(magnitudes_all, p))
        for p_key, p in zip(PERCENTILE_KEYS, PERCENTILES)
    }


def jump_conditioned_scores(records: list, thresholds: dict) -> dict:
    """Mean CRPS over jump-flagged steps (h >= 1) across all windows."""
    results = {p_key: [] for p_key in PERCENTILE_KEYS}
    for r in records:
        obs = np.array(r['forecast_obs'])
        if obs.ndim == 1:
            obs = obs[:, None]
        if obs.shape[0] < 2:
            continue
        increments = np.diff(obs, axis=0)
        magnitudes = np.max(np.abs(increments), axis=1)
        scores     = np.array(r['crps_per_step'])[1:]   # skip step 0
        for p_key, threshold in thresholds.items():
            mask = magnitudes > threshold
            if mask.any():
                results[p_key].extend(scores[mask].tolist())
    summary = {}
    for p_key in PERCENTILE_KEYS:
        vals = results[p_key]
        summary[p_key] = {
            'mean':    float(np.mean(vals)) if vals else float('nan'),
            'std':     float(np.std(vals))  if vals else float('nan'),
            'n_steps': len(vals),
        }
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(models: list, jc_list: list):
    """Print jump-conditioned CRPS table for N models.

    Parameters
    ----------
    models  : list of (result_name, display_label)
    jc_list : list of jump-conditioned score dicts (same order)
    """
    labels = [lbl for _, lbl in models]
    col_w  = 14
    row_w  = 12
    total_w = row_w + col_w * len(models) + 12

    print(f"\n{'='*total_w}")
    print(f"Jump-conditioned CRPS — prices (normalised log-price space, h\u22651)")
    print(f"{'='*total_w}")
    header = f"{'Threshold':<{row_w}}" + ''.join(f"{lbl:>{col_w}}" for lbl in labels)
    print(header)
    print(f"{'-'*total_w}")

    for p_key in PERCENTILE_KEYS:
        vals = [jc[p_key] for jc in jc_list]
        row  = f"{p_key:<{row_w}}" + ''.join(f"{v['mean']:>{col_w}.4f}" for v in vals)
        # Append absolute gaps vs first model for all subsequent models
        gaps = ''.join(
            f"  \u0394{v['mean'] - vals[0]['mean']:+.4f}"
            for v in vals[1:]
        )
        print(row + gaps)

    print(f"{'='*total_w}")
    n_steps_row = (f"{'n_steps':<{row_w}}" +
                   ''.join(f"{jc[PERCENTILE_KEYS[0]]['n_steps']:>{col_w}d}" for jc in jc_list))
    print(n_steps_row)
    print(f"{'='*total_w}\n")
    print("Thresholds: |obs[h] - obs[h-1]| (per-step price increments, model-agnostic).")
    print("\u0394 = difference relative to first model (positive = first model is better).")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_jump_crps(models: list, jc_list: list, output_path: Path):
    n      = len(models)
    labels = [lbl for _, lbl in models]
    x      = np.arange(len(PERCENTILE_KEYS))
    width  = min(0.7 / n, 0.3)

    fig, ax = plt.subplots(figsize=(max(7, 2 * n + 3), 4))
    for i, ((_, label), jc) in enumerate(zip(models, jc_list)):
        means  = [jc[k]['mean'] for k in PERCENTILE_KEYS]
        stds   = [jc[k]['std']  for k in PERCENTILE_KEYS]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               color=PALETTE[i % len(PALETTE)], alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(['p90', 'p95', 'p97.5', 'p99'])
    ax.set_xlabel('Jump threshold (percentile of |price increment|)')
    ax.set_ylabel('CRPS (normalised log-price space)')
    ax.set_title('Jump-conditioned CRPS — prices')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_models(args) -> list:
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
        description='Jump-conditioned CRPS for financial prices experiments.')
    parser.add_argument('--ts-name',    default='',
                        help='(legacy) result name for Tilted Stable')
    parser.add_argument('--gauss-name', default='',
                        help='(legacy) result name for Gaussian')
    parser.add_argument('--model', action='append', default=[],
                        metavar='NAME[:LABEL]',
                        help='Add a model.  May be repeated.')
    parser.add_argument('--metrics-suffix', default='')
    args = parser.parse_args()

    models = _parse_models(args)
    if len(models) < 2:
        parser.error('At least two models required.')

    print("Loading records...")
    all_records = [load_records(name, metrics_suffix=args.metrics_suffix)
                   for name, _ in models]

    # Thresholds are always derived from the first model's observations
    # (model-agnostic: observations are the same across models on the same windows)
    print("Computing jump thresholds from forecast price increments...")
    thresholds = compute_thresholds(all_records[0])
    for p_key, thresh in thresholds.items():
        print(f"  {p_key}: |\u0394price| > {thresh:.4f} (normalised)")

    print("Computing jump-conditioned scores...")
    jc_list = [jump_conditioned_scores(records, thresholds) for records in all_records]

    print_table(models, jc_list)

    out_dir = _analysis_dir([name for name, _ in models])
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.metrics_suffix or ''
    result = {
        'thresholds':       thresholds,
        'threshold_type':   'price_increments',
        'models': {
            name: jc
            for (name, _), jc in zip(models, jc_list)
        },
    }
    out_json = out_dir / f'jump_crps{suffix}.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_json}")

    plot_jump_crps(models, jc_list, out_dir / f'jump_crps_comparison{suffix}.png')


if __name__ == '__main__':
    main()
