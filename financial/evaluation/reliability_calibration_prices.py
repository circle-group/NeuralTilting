"""Reliability diagrams for prices experiments.

This script builds a single wide figure with three side-by-side reliability
panels for:
1. NVDA univariate experiments
2. GOOGL univariate experiments
3. Multivariate experiment

It uses the per-window coverage already stored in each run's ``metrics.json``
files, so plotting is lightweight and fully reproducible.

Usage:
    python financial/evaluation/reliability_calibration_prices.py

Optional:
    python financial/evaluation/reliability_calibration_prices.py \
        --metrics-suffix _dense \
        --basename reliability_calibration_prices_dense
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from financial.evaluation.metrics import DENSE_COVERAGE_LEVELS

RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "analysis"
QUANTILE_LEVELS = DENSE_COVERAGE_LEVELS

PANEL_SPECS = [
    (
        "NVDA",
        [
            ("nvda_njsde_obsstd0.10_train30d_fc2d_prices", "NJ-SDE"),
            ("nvda_dlinear_obsstd0.10_train30d_fc2d_prices", r"DLinear$^\dagger$"),
            ("nvda_nhits_obsstd0.10_train30d_fc2d_prices", r"N-HiTS$^\dagger$"),
            (
                "nvda_gaussian_sig1.0_obsstd0.20_train30d_fc2d_d2w32_creg2_prices",
                "Gaussian SDE",
            ),
            ("nvda_deepar_h256l2_obsstd0.10_train30d_fc2d_prices", "DeepAR"),
            ("nvda_neural_mjd_dropout0.1_obsstd0.10_train30d_fc2d_prices", "Neural MJD"),
            ("nvda_ts_a1.90_sig1.0_obsstd0.10_train30d_fc2d_d2w32_prices", "TS (ours)"),
        ],
    ),
    (
        "GOOGL",
        [
            ("googl_njsde_reg_obsstd0.10_train30d_fc2d_prices", "NJ-SDE"),
            ("googl_dlinear_obsstd0.10_train30d_fc2d_prices", r"DLinear$^\dagger$"),
            (
                "googl_gaussian_sig1.0_obsstd0.20_train30d_fc2d_d2w32_creg2_prices",
                "Gaussian SDE",
            ),
            ("googl_deepar_h256l2_obsstd0.10_train30d_fc2d_prices", "DeepAR"),
            ("googl_neural_mjd_dropout0.1_obsstd0.10_train30d_fc2d_prices", "Neural MJD"),
            ("googl_ts_a1.90_sig1.0_obsstd0.10_train30d_fc2d_d2w32_prices", "TS (ours)"),
        ],
    ),
    (
        "Multivariate",
        [
            (
                "aapl_amd_amzn_googl_intc_meta_msft_nflx_nvda_tsla_dlinear_train30d_fc2d_prices",
                r"DLinear$^\dagger$",
            ),
            (
                "aapl_amd_amzn_googl_intc_meta_msft_nflx_nvda_tsla_nhits_train30d_fc2d_prices",
                r"N-HiTS$^\dagger$",
            ),
            (
                "aapl_amd_amzn_googl_intc_meta_msft_nflx_nvda_tsla_gaussian_sig1.0_obsstd0.20_train30d_fc2d_d2w32_creg2_prices",
                "Gaussian SDE",
            ),
            (
                "aapl_amd_amzn_googl_intc_meta_msft_nflx_nvda_tsla_deepar_h256l2_train30d_fc2d_prices",
                "DeepAR",
            ),
            (
                "aapl_amd_amzn_googl_intc_meta_msft_nflx_nvda_tsla_neural_mjd_dropout0.1_train30d_fc2d_prices",
                "Neural MJD",
            ),
            (
                "aapl_amd_amzn_googl_intc_meta_msft_nflx_nvda_tsla_ts_a1.90_sig1.0_obsstd0.10_train30d_fc2d_d2w32_lr5_prices",
                "TS (ours)",
            ),
        ],
    ),
]

MODEL_STYLES = {
    "NJ-SDE": {"color": "#8c564b", "marker": "o", "linewidth": 2.6, "markersize": 8.2},
    r"DLinear$^\dagger$": {
        "color": "#d62728",
        "marker": "s",
        "linewidth": 2.6,
        "markersize": 8.2,
    },
    r"N-HiTS$^\dagger$": {
        "color": "#e377c2",
        "marker": "^",
        "linewidth": 2.6,
        "markersize": 8.5,
    },
    "Gaussian SDE": {"color": "#ff7f0e", "marker": "D", "linewidth": 2.6, "markersize": 8.0},
    "DeepAR": {"color": "#2ca02c", "marker": "v", "linewidth": 2.6, "markersize": 8.3},
    "Neural MJD": {"color": "#9467bd", "marker": "P", "linewidth": 2.6, "markersize": 8.3},
    "TS (ours)": {"color": "#1f77b4", "marker": "X", "linewidth": 3.1, "markersize": 9.4},
}


def load_coverage_summary(run_name: str, metrics_suffix: str = "") -> dict:
    run_dir = RESULTS_DIR / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Results not found: {run_dir}")

    metrics_name = f"metrics{metrics_suffix}.json" if metrics_suffix else "metrics.json"
    coverage_by_level = {q: [] for q in QUANTILE_LEVELS}
    n_windows = 0

    for window_dir in sorted(run_dir.glob("window_*")):
        metrics_path = window_dir / metrics_name
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            record = json.load(f)
        n_windows += 1
        coverage = record.get("coverage", {})
        for q in QUANTILE_LEVELS:
            value = coverage.get(str(float(q)))
            if value is not None and not math.isnan(value):
                coverage_by_level[q].append(float(value))

    if n_windows == 0:
        raise RuntimeError(f"No {metrics_name} files found under {run_dir}")

    mean_coverage = {}
    for q, values in coverage_by_level.items():
        mean_coverage[q] = float(np.mean(values)) if values else float("nan")

    return {
        "run_name": run_name,
        "n_windows": n_windows,
        "coverage": mean_coverage,
    }


def build_panel_summaries(metrics_suffix: str = "") -> list[tuple[str, list[tuple[str, dict]]]]:
    panels = []
    for panel_title, model_specs in PANEL_SPECS:
        model_summaries = []
        for run_name, label in model_specs:
            summary = load_coverage_summary(run_name, metrics_suffix=metrics_suffix)
            model_summaries.append((label, summary))
            print(f"{panel_title:<10} {label:<20} windows={summary['n_windows']}")
        panels.append((panel_title, model_summaries))
    return panels


def build_legend_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Perfect calibration",
        )
    ]
    for label in [
        "NJ-SDE",
        r"DLinear$^\dagger$",
        r"N-HiTS$^\dagger$",
        "Gaussian SDE",
        "DeepAR",
        "Neural MJD",
        "TS (ours)",
    ]:
        style = MODEL_STYLES[label]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                label=label,
            )
        )
    return handles


def plot_submission_figure(
    panels: list[tuple[str, list[tuple[str, dict]]]],
    output_path: Path,
    width: float,
    height: float,
    show_legend: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.labelsize": 21,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 19,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(width, height), sharex=True, sharey=True)
    tick_values = np.linspace(0.0, 1.0, 6)

    for idx, (ax, (title, model_summaries)) in enumerate(zip(axes, panels)):
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, zorder=1)
        for label, summary in model_summaries:
            style = MODEL_STYLES[label]
            y_values = [summary["coverage"][q] for q in QUANTILE_LEVELS]
            ax.plot(
                QUANTILE_LEVELS,
                y_values,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                zorder=4 if label == "TS (ours)" else 3,
            )

        ax.set_title(title, pad=8, fontweight="semibold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="x", pad=8)
        ax.tick_params(axis="y", labelleft=True, pad=8)
        ax.grid(alpha=0.40, linewidth=1.0)

    fig.supxlabel("Nominal coverage level", y=0.15, fontsize=21)
    fig.supylabel("Empirical coverage", x=0.025, fontsize=21)

    if show_legend:
        legend_handles = build_legend_handles()
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            columnspacing=1.3,
            handletextpad=0.6,
        )
        fig.tight_layout(rect=(0.016, 0.08, 1.0, 0.93), w_pad=1.0)
        fig.subplots_adjust(wspace=0.30)
    else:
        fig.tight_layout(rect=(0.016, 0.07, 1.0, 1.0), w_pad=1.0)
        fig.subplots_adjust(wspace=0.30)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def serialise_panels(panels: list[tuple[str, list[tuple[str, dict]]]]) -> dict:
    payload = {"levels": [float(q) for q in QUANTILE_LEVELS], "panels": {}}
    for panel_title, model_summaries in panels:
        payload["panels"][panel_title] = {}
        for label, summary in model_summaries:
            payload["panels"][panel_title][label] = {
                "run_name": summary["run_name"],
                "n_windows": summary["n_windows"],
                "coverage": {str(float(q)): summary["coverage"][q] for q in QUANTILE_LEVELS},
            }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 1x3 submission-ready reliability figure for prices experiments."
    )
    parser.add_argument(
        "--metrics-suffix",
        default="",
        help="Load metrics{suffix}.json instead of metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the figure and summary JSON will be written.",
    )
    parser.add_argument(
        "--basename",
        default="reliability_calibration_prices",
        help="Basename for output files, without extension.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=24.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=5.8,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Skip the shared figure legend.",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Save only a PNG file and skip the PDF copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels = build_panel_summaries(metrics_suffix=args.metrics_suffix)

    png_path = args.output_dir / f"{args.basename}.png"
    plot_submission_figure(
        panels=panels,
        output_path=png_path,
        width=args.width,
        height=args.height,
        show_legend=not args.no_legend,
    )

    if not args.png_only:
        pdf_path = args.output_dir / f"{args.basename}.pdf"
        plot_submission_figure(
            panels=panels,
            output_path=pdf_path,
            width=args.width,
            height=args.height,
            show_legend=not args.no_legend,
        )

    json_path = args.output_dir / f"{args.basename}.json"
    with open(json_path, "w") as f:
        json.dump(serialise_panels(panels), f, indent=2)
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
