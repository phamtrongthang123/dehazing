# Dehazing Reproduction — Spec for Autonomous Agent

## Goal
Reproduce ultrasound dehazing via score-based diffusion models. The method should remove haze artifacts from ultrasound RF data while preserving tissue structure. Currently, **dehazed output looks washed out / flat gray** — the method is not working correctly for PICMUS data.

## What "done" looks like
- Dehazed B-mode images clearly show tissue structure (layers, boundaries, cysts) matching ground truth
- Dehazed output is visually distinct from the hazy input (not just a slightly smoothed version)
- gCNR of dehazed images is higher than gCNR of hazy input (quantitative improvement)
- Works for both ZEA (simulated, 64-element) and PICMUS (experimental, 128-element) datasets

## Current state
- ZEA dehazing: **partially working** — tissue structure visible but quality could improve
- PICMUS dehazing: **broken** — output is washed out flat gray (see `figures/2026_03_01_sgm_picmus_tissue_dehazing_final.png`)
- All training and inference code runs end-to-end without errors
- Score model quality (val loss at t=0.01): ZEA tissue=8.2 (decent), PICMUS tissue=28.6 (too high, target <20)

## Constraints
- **1 GPU** for dev (no multi-GPU). SLURM partitions: only `vgpu` and `agpu`. See `example_slurm.sh`
- Conda env: `dehazing` — activate with `conda activate dehazing` or `conda run -n dehazing python ...`
- Suppress wandb: `export WANDB_MODE=disabled`
- Working directory: `/scrfs/storage/tp030/home/dehazing/`
- Inference code: `cd joint_diffusion && python inference.py -e paper/picmus_dehaze_pigdm -t denoise`
- Training code: `cd joint_diffusion && bash train_run.sh picmus_tissue`
- Data root: `/scrfs/storage/tp030/home/f2f_ldm/data`

## Diagnosis of the washed-out problem

The dehazing pipeline has two forces during sampling:
1. **Score model** (unconditional prior) — pushes samples toward realistic tissue
2. **Guidance** (data consistency) — pushes samples toward being consistent with the noisy measurement y

When the score model is weak (loss=28 instead of <15), the prior is too weak to counter the guidance, and the output collapses toward a bland average. Three attack vectors:

### A. Improve score model quality (most important)
- PICMUS tissue model trained 500 epochs on only 395 samples — likely undertrained
- Try: more epochs (1000+), data augmentation (random flips, crops), learning rate schedule
- Try: fine-tune from ZEA tissue checkpoint (transfer learning) — was tried once but needs more tuning
- Benchmark: unconditional samples from the score model should look like realistic tissue
- Target validation loss at t=0.01: < 20 (currently 28.6)
- Generate unconditional samples: `python diagnostic_picmus.py` (already exists)

### B. Verify guidance correctness
- The `companded_projection` guidance in `generators/SGM/guidance.py` was custom-written
- Compare its math against the PIGDM paper (Algorithm 1) and the original TF implementation
- The `denoise_update` method computes `loss = ||y_hat - x_pred||^2` — verify the gradient direction is correct (should push x toward data consistency, not away)
- Try simpler guidance methods first: `pigdm` or `projection` (already implemented) as a sanity check
- If `pigdm` guidance with the same score model produces better results, the `companded_projection` has a bug

### C. Tune inference hyperparameters
- Only tune AFTER score model quality is decent (loss < 20)
- Key params in `configs/inference/paper/picmus_dehaze_pigdm.yaml`:
  - `lambda_coeff`: guidance strength (current: 0.01, try 0.001-0.1)
  - `ccdf`: start diffusion fraction (current: 0.8, try 0.5-0.9)
  - `noise_stddev`: haze strength gamma (current: 0.1)
  - `guidance`: try `pigdm`, `projection`, `dps` as alternatives to `companded_projection`
- Use `sweep_picmus.py` for systematic sweeps (already exists)

## Priority order
1. First, verify guidance correctness by running inference with `pigdm` guidance (change 1 line in config)
2. If pigdm also washed out → score model is the bottleneck → focus on training
3. If pigdm works better → debug companded_projection math
4. Once dehazing visually works, compute gCNR metrics

## Key files
| File | Purpose |
|------|---------|
| `joint_diffusion/generators/SGM/guidance.py` | Guidance implementations (pigdm, companded_projection, dps, projection) |
| `joint_diffusion/generators/SGM/sampling.py` | Score sampler (predictor-corrector) |
| `joint_diffusion/utils/inverse.py` | Denoiser wrapper, plotting |
| `joint_diffusion/inference.py` | Inference entry point |
| `joint_diffusion/train.py` | Training script |
| `joint_diffusion/diagnostic_picmus.py` | Diagnostic: unconditional samples, score quality |
| `joint_diffusion/sweep_picmus.py` | Hyperparameter sweep |
| `joint_diffusion/configs/inference/paper/picmus_dehaze_pigdm.yaml` | PICMUS inference config |
| `joint_diffusion/configs/training/score_picmus_tissue.yaml` | PICMUS tissue training config |

## What NOT to do
- Do not refactor working code — focus on making dehazing work
- Do not add new datasets or models — fix the existing pipeline first
- Do not change ZEA configs/code unless it breaks — ZEA already works
- Do not spend time on visualization improvements until dehazing actually works
- Do not train for more than 4 hours per run on a single GPU — if no convergence, investigate why
