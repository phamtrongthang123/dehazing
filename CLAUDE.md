# Dehazing Project — Joint Diffusion for Ultrasound RF Data

Score-based diffusion models for removing haze artifacts from ultrasound RF (radio-frequency) channel data. Two datasets: ZEA (simulated, 64-element) and PICMUS (experimental, 128-element). The `joint_diffusion/` subdirectory is a git submodule from `tristan-deep/joint-diffusion`.

## Environment

- **Conda env**: `dehazing` (`/home/tp030/.conda/envs/dehazing`)
- Python 3.10, PyTorch 2.5.1+cu121
- Activate: `conda activate dehazing` or `conda run -n dehazing python ...`
- SLURM: partition `agpu`, logs in `/scrfs/storage/tp030/home/dehazing/slurm_logs/`
- Suppress wandb: `export WANDB_MODE=disabled`

## Data

All data under `--data_root /scrfs/storage/tp030/home/f2f_ldm/data`.

| Dataset | Path (relative to data_root) | Shape (train) | Shape (val) |
|---------|------------------------------|---------------|-------------|
| ZEA tissue | `zea_synth/tissue/` | `(N, 3, 1024, 64)` | same |
| ZEA haze | `zea_synth/haze/` | `(N, 3, 1024, 64)` | same |
| PICMUS tissue | `picmus/tissue/` | `(395, 3, 1024, 128)` | `(43, 3, 1024, 128)` |
| PICMUS haze | `picmus/haze/` | `(400, 3, 1024, 128)` | `(50, 3, 1024, 128)` |

Format: `.npz` files with key `rf`. PICMUS tissue from 6 datasets with stride-1 sliding window over 75 angles. PICMUS haze is synthetic bandpass-filtered noise.

## Training

### Commands
```bash
cd joint_diffusion
bash train_run.sh tissue    # or: haze, picmus_tissue, picmus_haze
```

### Configs
- ZEA: `configs/training/score_zea_{tissue,haze}.yaml`
- PICMUS: `configs/training/score_picmus_{tissue,haze}.yaml`

### Checkpoints (current best)
| Model | Run dir (under `joint_diffusion/wandb/`) | Best ckpt | Val loss (t=0.01) |
|-------|------------------------------------------|-----------|-------------------|
| ZEA tissue | `run--jm8ljbms/files/` | `ckpt-349.pt` | 8.2 |
| ZEA haze | `run--1bj42atz/files/` | `ckpt-349.pt` | — |
| PICMUS tissue | `run--xnhklvch/files/` | `ckpt-479.pt` | 28.6 |
| PICMUS haze | `run--w1lwubg5/files/` | last | — |

### Critical training lesson: `reduce_mean`
PICMUS images are 2x wider than ZEA (128 vs 64). With `reduce_mean: false`, loss is summed over pixels, so gradients are ~2x larger but clipped at `grad_clip=1.0`, making effective learning rate ~2x lower per pixel. **Always use `reduce_mean: true` for PICMUS.** ZEA uses `reduce_mean: false` (works fine at 64px width).

## Inference

### Commands
```bash
cd joint_diffusion
# ZEA
python inference.py -e paper/zea_dehaze_pigdm -t denoise
# PICMUS
python inference.py -e paper/picmus_dehaze_pigdm -t denoise
```

### Configs
- `configs/inference/paper/zea_dehaze_pigdm.yaml`
- `configs/inference/paper/picmus_dehaze_pigdm.yaml`

### Optimal parameters (PICMUS)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `lambda` | 0.01 | Weak guidance works best; 0.1 pulls toward noisy input |
| `kappa` | 0.0 | Single-model guidance (no haze model) works better |
| `noise_stddev` | 0.1 | |
| `ccdf` | 0.8 | |
| `use_ema` | false | **EMA weights are worse than raw weights for both ZEA and PICMUS** |

### Output figures
- ZEA: `figures/2026_03_01_sgm_zea_tissue_dehazing_3.png`
- PICMUS: `figures/2026_03_01_sgm_picmus_tissue_dehazing_final.png`

## Key Debugging Insights

1. **EMA is worse than raw weights** — Discovered for both ZEA and PICMUS. Set `use_ema: false` in inference configs. The `inverse.py` EMA loading was made configurable via the `use_ema` config flag.

2. **Gradient scaling with `reduce_mean: false`** — When image dimensions change, sum-based loss changes gradient magnitude. Combined with gradient clipping, this effectively lowers learning rate for larger images. PICMUS v1 models (350 epochs, `reduce_mean: false`) never converged (flat gray output). Fixed by switching to `reduce_mean: true`.

3. **Guidance strength** — Strong guidance (`lambda=0.1`) pulls the reconstruction toward the noisy input. Weak guidance (`lambda=0.01`) lets the score model denoise properly while still using measurement information.

4. **Joint vs single-model guidance** — For PICMUS, single-model guidance (`kappa=0.0`, tissue model only) outperforms joint guidance with the haze model. This may be because the haze model quality is lower.

5. **Score quality benchmarks** (validation loss at t=0.01):
   - ~8: Good quality (ZEA tissue) — clean unconditional samples
   - ~28: Acceptable (PICMUS v2) — tissue structure visible
   - ~38: Insufficient (PICMUS v1) — flat output
   - Target for high quality: ~15-20

## Code Modifications

- **`datasets.py`**: Refactored with `_get_rf_dataset(config, subdir, kind)` to parameterize data subdirectory. Dataset names `picmus_tissue`, `picmus_haze` added to `_DATASETS`.
- **`inverse.py`**: EMA loading made configurable via `use_ema` config flag. Added `_load_ema_weights()` method and corruptor EMA loading.
- **`corruptors.py`**: Haze path configurable via `haze_data_subdir` config (default `"zea_synth"`).
- **`inverse.py`**: Conditional B-mode import via `bmode_module` config (`"zea"` default, `"picmus"` for PICMUS).
- **`bmode_picmus.py`**: L11-4v probe parameters (128 el, 0.3mm pitch, 5.208 MHz, fs=20.832 MHz).
- **`processing.py`**: mu-law companding, gCNR metric.

## Project Status

| Phase | Status |
|-------|--------|
| Data preparation (ZEA + PICMUS) | DONE |
| Training (ZEA + PICMUS) | DONE |
| Inference / dehazing (ZEA + PICMUS) | DONE |
| Evaluation / metrics (gCNR, etc.) | TODO |

## Key Files

| File | Purpose |
|------|---------|
| `paper/plan.md` | Reproduction plan with phase status |
| `joint_diffusion/train_run.sh` | Training launch script |
| `joint_diffusion/inference.py` | Inference entry point |
| `joint_diffusion/sweep_picmus.py` | PICMUS hyperparameter sweep (4-phase) |
| `joint_diffusion/diagnostic_picmus.py` | Diagnostic tests for PICMUS model quality |
| `data_conversion/convert_picmus.py` | PICMUS data conversion |
| `data_conversion/generate_haze_128el.py` | PICMUS synthetic haze generation |
