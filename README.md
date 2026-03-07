# Dehazing Ultrasound using Diffusion Models

PyTorch reproduction of [Dehazing Ultrasound using Diffusion Models](https://arxiv.org/abs/2307.11204) (Stevens et al., 2023).

Uses joint posterior sampling with score-based diffusion models to remove structured haze noise from ultrasound RF data. The core implementation lives in `joint_diffusion/`.

## Datasets

| Dataset | Elements | RF Shape | Description |
|---------|----------|----------|-------------|
| ZEA | 64 | `(N, 3, 1024, 64)` | Simulated (Field II) |
| PICMUS | 128 | `(N, 3, 1024, 128)` | Experimental (L11-4v probe) |

## Project Structure

```
dehazing-diffusion/
├── joint_diffusion/       # Main codebase (PyTorch)
│   ├── train.py           # Training script
│   ├── inference.py       # Inference script
│   ├── batch_dehaze_picmus.py  # Batch dehazing all PICMUS val frames
│   ├── datasets.py        # ZEA + PICMUS dataset loader
│   ├── gcnr_eval.py       # gCNR quantitative evaluation
│   ├── sweep_picmus.py    # Hyperparameter sweep
│   ├── score_quality_check.py  # Score model quality benchmark
│   ├── configs/
│   │   ├── training/      # Training configs (ZEA + PICMUS)
│   │   └── inference/     # Inference configs
│   ├── generators/        # Score models (NCSNv2), SDE, sampling, guidance
│   └── utils/             # Corruptors, metrics, B-mode conversion, plotting
├── data_conversion/       # PICMUS data conversion & haze generation
├── reproduce_helpers/     # Data synthesis & visualization scripts
└── paper/                 # LaTeX source & plan docs
```

## Quick Start

### 1. Generate Data

```bash
cd reproduce_helpers
python synthesize_dataset.py --out-dir /path/to/data/zea_synth --n-train 1000 --n-val 100
```

### 2. Train Models

Two score models (tissue + haze) are trained independently:

| Dataset | Config | Epochs | Notes |
|---------|--------|--------|-------|
| ZEA tissue | `score_zea_tissue.yaml` | 350 | `reduce_mean: false` |
| ZEA haze | `score_zea_haze.yaml` | 350 | `reduce_mean: false` |
| PICMUS tissue | `score_picmus_tissue_v4.yaml` | 1000 | `reduce_mean: true`, ch=64, cosine LR |
| PICMUS haze | `score_picmus_haze.yaml` | 500 | `reduce_mean: true` |

```bash
cd joint_diffusion
bash train_run.sh tissue          # ZEA
bash train_run.sh picmus_tissue   # PICMUS
```

Checkpoints are saved every 10 epochs to `wandb/run-<id>/files/training_checkpoints/`.

> **Note**: PICMUS (128-wide images) requires `reduce_mean: true` to normalize gradients. Without it, gradient clipping effectively halves the learning rate per pixel.

### 3. Run Inference

Update checkpoint paths in the inference config, then:

```bash
# ZEA (single frames)
python inference.py -e paper/zea_dehaze_pigdm -t denoise

# PICMUS (single frames)
python inference.py -e paper/picmus_dehaze_v4 -t denoise

# PICMUS (all 43 val frames)
python batch_dehaze_picmus.py --inf_config configs/inference/paper/picmus_dehaze_v4.yaml
```

Output figures are saved to `figures/` (single) or `results/` (batch).

### 4. Evaluate

```bash
# gCNR on contrast_speckle val frames
python gcnr_eval.py --inf_config configs/inference/paper/picmus_dehaze_v4.yaml

# Score model quality benchmark (val loss at t=0.01)
python score_quality_check.py --config configs/training/score_picmus_tissue_v4.yaml
```

### Config Options

| Key | Description |
|-----|-------------|
| `noise_stddev` | Haze strength gamma (default: 0.05 for PICMUS) |
| `num_scales` | Number of SDE time steps (default: 1000) |
| `guidance` | Guidance method: `pigdm`, `projection`, `companded_projection` |
| `lambda_coeff` | Guidance strength (0.5 for PICMUS v4) |
| `kappa_coeff` | Haze model weight (0.5 for PICMUS v4) |
| `ccdf` | Fraction of diffusion steps to run (0.7 for PICMUS v4) |
| `use_ema` | Use EMA weights — set `false` (EMA is worse for both datasets) |
| `keep_track` | Save intermediate steps — set `false` for PICMUS (OOM otherwise) |
| `display_bmode` | Convert RF output to B-mode for plotting |
| `bmode_module` | `zea` or `picmus` (probe-specific B-mode conversion) |

## Architecture

- **Score model**: NCSNv2 backbone (32 channels ZEA, 64 channels PICMUS)
- **SDE**: Simple VE-SDE with sigma=25
- **Sampling**: Predictor-Corrector (Euler-Maruyama predictor, no corrector)
- **Guidance**: Companded projection (PICMUS) / PiGDM (ZEA)
- **Data**: RF frames with mu-law companding, normalized to [0, 1]

## Results

| Dataset | gCNR (hazy → dehazed) | PSNR | SSIM |
|---------|----------------------|------|------|
| PICMUS | 0.224 → 0.314 | 30.20 dB | 0.814 |

