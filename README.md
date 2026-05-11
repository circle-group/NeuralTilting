# NeuralTilting
Official Implementation of Variational Inference for Lévy Process-Driven SDEs via Neural Tilting

## Overview

This repository implements:

- **Tilted-Stable SDE** — the proposed variational family that learns an exponential tilt of the prior Lévy measure via a quadratic neural parametrisation.
- **Gaussian SDE** — the Gaussian-diffusion variational baseline for direct comparison.
- Both models in two drift settings: **Ornstein-Uhlenbeck (OU)** and **double-well potential (DW)**.
- Financial forecasting variants applied to rolling-window equity price prediction, with five baseline models (DeepAR, DLinear, N-HiTS, Neural Jump SDE, Neural MJD).

---

## Repository structure

```
submission/
├── models/                          # Model implementations
│   ├── tilted_stable_sde.py         #   Tilted-Stable SDE (OU drift)
│   ├── gaussian_sde.py              #   Gaussian SDE (OU drift)
│   ├── tilted_stable_sde_double_well.py
│   ├── gaussian_sde_double_well.py
│   └── components/                  #   Shared building blocks
│       ├── attention.py             #     Temporal attention / adaptive encoding
│       ├── drift.py                 #     Drift network
│       ├── potential.py             #     Tilting potential (quadratic parametrisation)
│       ├── scale.py                 #     Diffusion scale network
│       └── utils.py
│
├── training/                        # Training scripts and infrastructure
│   ├── train_tilted_stable_sde_gpu.py
│   ├── train_gaussian_sde_gpu.py
│   ├── train_tilted_stable_sde_double_well_gpu.py
│   ├── train_gaussian_sde_double_well_gpu.py
│   ├── loss.py                      #   ELBO and regularisation losses
│   ├── train_model.py               #   Generic training entry point
│   └── components/
│       ├── optimiser_config.py
│       ├── training_loop.py
│       └── training_monitor.py
│
├── evaluation/                      # Evaluation and analysis
│   ├── evaluate_runs.py             #   Per-run metrics (MSE, MAE, CRPS, jump metrics)
│   ├── compare_models.py            #   Pairwise TS vs Gaussian comparison
│   ├── analyse_comparisons.py       #   Aggregate analysis, tables, and plots
│   ├── loss_functions.py
│   └── run_utils.py
│
├── generation/                      # Dataset and posterior generation
│   ├── generate_batch_prior_tilted_stable_sde.py      # OU datasets
│   ├── generate_batch_prior_tilted_stable_sde_double_well.py  # DW datasets
│   └── generate_posteriors.py       #   Posterior sample visualisation
│
├── utils/
│   ├── dataset_utils.py
│   ├── training_utils.py
│   └── visualization_utils.py
│
├── configs/                         # Example configurations
│   ├── ou/
│   │   ├── tilted_stable.yaml       #   TS-SDE on OU dataset
│   │   └── gaussian.yaml            #   Gaussian SDE on OU dataset
│   └── double_well/
│       ├── tilted_stable.yaml       #   TS-SDE on double-well dataset
│       └── gaussian.yaml            #   Gaussian SDE on double-well dataset
│
├── generate_ou_configs.py           # Batch config generator for OU experiments
├── generate_double_well_configs.py  # Batch config generator for DW experiments
├── requirements.txt
│
└── financial/                       # Financial forecasting application
    ├── prepare_windows_prices.py    #   Build rolling-window datasets from Yahoo Finance
    ├── run_window.py                #   Train + evaluate one window (TS or Gaussian)
    ├── run_window_deepar.py         #   DeepAR baseline
    ├── run_window_dlinear.py        #   DLinear baseline
    ├── run_window_nhits.py          #   N-HiTS baseline
    ├── run_window_njsde.py          #   Neural Jump SDE baseline
    ├── run_window_neural_mjd.py     #   Neural MJD baseline
    ├── data/                        #   Download and windowing utilities
    ├── models/                      #   Financial model implementations
    ├── training/
    ├── forecasting/
    ├── evaluation/                  #   Aggregation and analysis scripts
    └── configs/                     #   Experiment configs (one per setting)
```

---

## Installation

**Python 3.10+ is required.**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### GPU acceleration (optional)

The training scripts auto-detect the available JAX backend and fall back to CPU if no GPU is found. To enable GPU support, install the appropriate JAX build after the step above:

```bash
# NVIDIA CUDA 12 (Linux):
pip install -U "jax[cuda12]"

# NVIDIA CUDA 11 (Linux):
pip install -U "jax[cuda11_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Apple Silicon / AMD GPU (macOS):
pip install jax-metal
```

---

## Synthetic experiments (OU and double-well)

The synthetic workflow has five stages: generate datasets → generate configs → train → evaluate → compare and analyse.

### 1. Generate datasets

OU datasets (used by both TS-SDE and Gaussian SDE):

```bash
python generation/generate_batch_prior_tilted_stable_sde.py
```

Double-well datasets:

```bash
python generation/generate_batch_prior_tilted_stable_sde_double_well.py
```

Both scripts write to `datasets/` and contain a parameter grid at the top of the file (alpha values, observation noise, number of realisations). Edit the grid there before running. Datasets are shared between the two model types — each dataset is generated once from the true tilted-stable process.

### 2. Generate training configs

After datasets are in place, generate per-seed, per-alpha YAML configs automatically:

```bash
# OU experiment (scans datasets/tilted_stable_sde/)
python generate_ou_configs.py --obs-std 0.10 --n-seeds 50

# Double-well experiment (scans datasets/tilted_stable_sde_double_well/)
python generate_double_well_configs.py --obs-std 0.10 --n-seeds 10
```

This writes configs to `training_configs/tilted_stable/`, `training_configs/gaussian/`, etc. To see what would be written without writing: add `--dry-run`.

For quick experiments, use the provided example configs in `configs/` directly (see step 3).

### 3. Train models

```bash
# Tilted-Stable SDE — OU
python training/train_tilted_stable_sde_gpu.py --config configs/ou/tilted_stable.yaml

# Gaussian SDE — OU
python training/train_gaussian_sde_gpu.py --config configs/ou/gaussian.yaml

# Tilted-Stable SDE — double-well
python training/train_tilted_stable_sde_double_well_gpu.py --config configs/double_well/tilted_stable.yaml

# Gaussian SDE — double-well
python training/train_gaussian_sde_double_well_gpu.py --config configs/double_well/gaussian.yaml
```

Each run saves its model, config snapshot, training metrics, and diagnostic plots to:

```
training_runs/{model_type}/{alpha}_{obs_std}/data_{data_seed}_train_{train_seed}/
```

To continue training from a checkpoint:

```bash
python training/train_tilted_stable_sde_gpu.py \
    --config configs/ou/tilted_stable.yaml \
    --parent-run "alpha_1.50_obsstd_0.10/data_42_train_713"
```

### 4. Generate posterior samples

```bash
python generation/generate_posteriors.py \
    training_runs/tilted_stable_sde/alpha_1.50_obsstd_0.10/data_42_train_713 \
    --n-posterior-samples 50
```

Pass two run directories to overlay TS and Gaussian posteriors in the same plot:

```bash
python generation/generate_posteriors.py \
    training_runs/tilted_stable_sde/alpha_1.50_obsstd_0.10/data_42_train_713 \
    training_runs/gaussian_sde/alpha_1.50_obsstd_0.10/data_42_train_321 \
    --n-posterior-samples 50
```

Posterior samples are saved as `.pkl` files alongside the model and are reused by `evaluate_runs.py`.

### 5. Evaluate a single run

```bash
python evaluation/evaluate_runs.py \
    --run-path training_runs/tilted_stable_sde/alpha_1.50_obsstd_0.10/data_42_train_713 \
    --n-posterior-samples 100
```

Metrics (MSE, MAE, CRPS, jump-conditioned variants, drift parameter error) are written to `evaluation/results/{model_type}/{param_dir}/{run_id}/evaluation_summary.json`.

### 6. Compare TS-SDE against Gaussian SDE

Once both models have been evaluated on the same dataset:

```bash
python evaluation/compare_models.py \
    --eval-dirs \
        evaluation/results/tilted_stable_sde/alpha_1.50_obsstd_0.10/<ts_run_id> \
        evaluation/results/gaussian_sde/alpha_1.50_obsstd_0.10/<gauss_run_id>
```

### 7. Aggregate across all comparisons

After running comparisons for all seeds and alpha values:

```bash
python evaluation/analyse_comparisons.py
```

This produces CSV tables and plots in `evaluation/results/analysis/`, including per-alpha CRPS, MAE, MSE, and relative improvement figures.

---

## Financial forecasting experiments

The financial experiment runs a rolling-window forecast over equity log-price data downloaded automatically from Yahoo Finance.

### 1. Prepare rolling windows

```bash
python financial/prepare_windows_prices.py \
    --config financial/configs/nvda_tilted_stable_alpha1.9_sig1.0_obsstd0.10_d2w32.yaml
```

This downloads the specified tickers, builds the rolling-window dataset, and writes it to `financial/data/prepared/<config_name>/`. The number of windows is printed on completion.

### 2. Train and evaluate per window

```bash
# Tilted-Stable SDE
python financial/run_window.py \
    --config-name nvda_tilted_stable_alpha1.9_sig1.0_obsstd0.10_d2w32 \
    --window-idx 0

# Gaussian SDE (use the corresponding Gaussian config name)
python financial/run_window.py \
    --config-name nvda_gaussian_sig1.0_obsstd0.20_d2w32_creg2 \
    --window-idx 0
```

Repeat for all window indices (0 to N−1). Results are written to `financial/results/<config_name>/window_<idx>/`.

### 3. Run baseline models

Each baseline has its own entry point with the same `--config-name` / `--window-idx` interface. The config names for the baselines are in `financial/configs/`.

```bash
python financial/run_window_deepar.py    --config-name 10ticker_deepar_h256l2_obsstd0.10_train30d_fc2d_prices --window-idx 0
python financial/run_window_dlinear.py   --config-name 10ticker_dlinear_obsstd0.10_train30d_fc2d_prices      --window-idx 0
python financial/run_window_nhits.py     --config-name 10ticker_nhits_obsstd0.10_train30d_fc2d_prices         --window-idx 0
python financial/run_window_njsde.py     --config-name 10ticker_njsde_reg_obsstd0.10_train30d_fc2d_prices     --window-idx 0
python financial/run_window_neural_mjd.py --config-name 10ticker_neural_mjd_dropout0.1_obsstd0.10_train30d_fc2d_prices --window-idx 0
```

### 4. Aggregate results

After all windows and models are complete:

```bash
python financial/evaluation/aggregate_prices.py \
    --model nvda_tilted_stable_alpha1.9_sig1.0_obsstd0.10_d2w32:"TS-SDE" \
    --model nvda_gaussian_sig1.0_obsstd0.20_d2w32_creg2:"Gaussian-SDE" \
    --model <deepar_config_name>:"DeepAR"
```

This writes per-model and comparative metric summaries (CRPS, MAE, energy score, jump CRPS) to `financial/results/analysis/`.

For jump-conditioned metrics across models:

```bash
python financial/evaluation/jump_aggregate_prices.py \
    --model nvda_tilted_stable_alpha1.9_sig1.0_obsstd0.10_d2w32:"TS-SDE" \
    --model nvda_gaussian_sig1.0_obsstd0.20_d2w32_creg2:"Gaussian-SDE"
```

---

## Configuration reference

All training configs are YAML files. Key fields shared across model types:

| Field | Description |
|---|---|
| `model.alpha` | Stability index of the driving Lévy process (1 < α < 2) |
| `data.data_seed` | Seed used to select the dataset realisation |
| `data.obs_std` | Observation noise standard deviation |
| `training.training_steps` | Number of gradient steps |
| `training.n_loss_samples` | Monte Carlo samples per ELBO gradient estimate |
| `training.n_latent_steps` | Latent time discretisation steps |

Tilted-Stable-specific fields:

| Field | Description |
|---|---|
| `model.tilting_width` / `tilting_depth` | MLP width/depth for the tilting potential |
| `model.n_attention_references` | Number of learnable temporal reference points |
| `model.a_min` | Minimum absolute value of the A coefficient (strict negativity) |
| `training.tilting_regularisation` | L2 penalty on tilting coefficients |

Gaussian-SDE-specific fields:

| Field | Description |
|---|---|
| `model.control_width` / `control_depth` | MLP width/depth for the control network |
| `training.control_regularisation` | L2 penalty on control network weights |
