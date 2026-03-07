# Dehazing Reproduction — Spec

## Goal
Reproduce ultrasound dehazing via score-based diffusion models. Remove haze artifacts from ultrasound RF data while preserving tissue structure.

## Status: COMPLETE (2026-03-07)

All acceptance criteria met:

| Criterion | Result |
|-----------|--------|
| Dehazed images show tissue structure | v4 model: clear tissue layers, bmode mean=77.9 (target=73.8) |
| Visually distinct from hazy input | hazy=115.3 → dehazed=77.9 |
| gCNR dehazed > gCNR hazy | 0.314 > 0.224 (+0.089) |
| Works for ZEA and PICMUS | Both datasets working |

## Best Configuration

**PICMUS** (v4 model, run--cs2128ri):
- Score quality: 0.1886 (val loss at t=0.01)
- Sweep-optimized params: gamma=0.05, lambda=0.5, kappa=0.5, ccdf=0.7
- PSNR=30.20 dB, SSIM=0.814
- Inference: `cd joint_diffusion && python inference.py -e paper/picmus_dehaze_v4 -t denoise`

**ZEA** (run--jm8ljbms):
- Score quality: 0.0655
- Inference: `cd joint_diffusion && python inference.py -e paper/zea_dehaze_pigdm -t denoise`

## Key Technical Decisions

1. **Score model quality was the bottleneck** — all 3 guidance methods (pigdm, projection, companded_projection) produced identical washed-out output when score loss > 0.20. Improving the model (ch=64, cosine LR, 1000 epochs) was the fix.
2. **reduce_mean=true** required for PICMUS (128-wide images vs ZEA's 64).
3. **EMA worse than raw weights** for both datasets.
4. **Higher guidance strength** (lambda=0.5) works better than weak (0.01) once the score model is good.
5. **Lower ccdf** (0.7) better than 0.8 — start diffusion earlier.

## Files

| File | Purpose |
|------|---------|
| `SCORES.md` | Full results log (append-only) |
| `implementation_plan.md` | Phase status and lessons |
| `joint_diffusion/inference.py` | Inference entry point |
| `joint_diffusion/sweep_picmus.py` | Hyperparameter sweep |
| `joint_diffusion/gcnr_eval.py` | gCNR quantitative evaluation |
| `joint_diffusion/configs/inference/paper/picmus_dehaze_v4.yaml` | Best PICMUS inference config |
| `joint_diffusion/configs/inference/paper/picmus_dehaze_v5.yaml` | v5 inference config (untested sweep) |

## Constraints
- 1 GPU (SLURM partition `agpu`), conda env `dehazing`
- Suppress wandb: `export WANDB_MODE=disabled`
- Working dir: `/scrfs/storage/tp030/home/dehazing/`
- Data root: `/scrfs/storage/tp030/home/f2f_ldm/data`
