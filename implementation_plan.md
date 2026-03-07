# Implementation Plan

## Status Summary

| Phase | Status | Result |
|-------|--------|--------|
| 1: Guidance sanity check | DONE 2026-03-06 | All 3 guidance methods produce same washed-out output — score model is bottleneck |
| 2: Fix guidance | SKIPPED | Irrelevant until score model quality improves |
| 3a: v3 model (ep1000, ch=32) | DONE 2026-03-06 | Score loss 0.2695; model capacity was bottleneck |
| 3b: v4 model (ep1000, ch=64) | DONE 2026-03-06 | run--cs2128ri; ep1000 score=**0.1886** (best PICMUS, below 0.20 target) |
| 3c: v5 model (GroupNorm, batch=8) | DONE ep1000 | run--n1kp0qeb; score=**0.1603** (BEST PICMUS); infer fig: 2026_03_06_sgm_picmus_tissue_dehazing_4.png; bmode mean=92.2 |
| 3d: v6 model (t-importance, scratch) | FAILED→RESUBMITTED | OOM on c2008 (zombie PID 2962654); new jobs: train=180045→score=180046→infer=180047; c2008 excluded |
| 3e: v7 model (fine-tune v5+t-importance) | FAILED→RESUBMITTED | OOM on c2008 (same); new jobs: train=180048→score=180049→infer=180050; c2008 excluded |
| intermediate inference | DONE | 180008: v4@ep730 + v5@ep535 visual quality check completed |
| 4: Hyperparameter tuning | RUNNING | Phase 4 sweep job 180038 (v4 model); also need to run with v5 model |
| 5: Quantitative evaluation | SUBMITTED | gcnr_eval.py created; job 180052 (v4 model, 13 contrast_speckle frames) |

## Next Steps (when in-flight jobs complete)

**Current in-flight jobs** (as of 2026-03-06, session 4):
- 180038: v4 Phase 4 sweep (c2008) — Phase 1 DONE: gamma=0.05 best (PSNR=28.50); Phases 2-3 pending
  - v4 Phase 1 results: gamma 0.05→28.50, 0.1→28.27, 0.15→24.80, 0.2→23.11, 0.3→TBD
- 180039: v5 inference — PENDING (Resources)
- 180045→180046→180047: v6 train→score→infer (c2008 excluded) — PENDING Priority
- 180048→180049→180050: v7 train→score→infer (c2008 excluded) — PENDING Priority
- 180052: gCNR eval (v4 model, 13 contrast_speckle frames) — PENDING Priority

**Already completed (session 2-4)**:
- v4 inference: bmode mean=77.9 → **Row 1: CLEAR tissue layers!** (best result so far)
- v5 score quality: 0.1603 (BEST score quality; better than v4's 0.1886)
- gcnr_eval.py + slurm_gcnr_eval.sh created
- All in-flight scripts verified (no bugs found)
- v4 vs v5 puzzle: v4 better dehazing (bmode=77.9 vs 92.2) despite worse score quality — likely params not tuned for v5

1. **When Phase 4 sweep (180038) completes**: update SCORES.md with best params; submit v5 sweep:
   `cd joint_diffusion && sbatch slurm_sweep_phase4.sh configs/inference/paper/picmus_dehaze_v5.yaml`
2. **When v5 inference (180039) completes**: submit v5 gCNR eval:
   `cd joint_diffusion && sbatch slurm_gcnr_eval.sh configs/inference/paper/picmus_dehaze_v5.yaml`
3. **When gCNR eval (180052) completes**: update SCORES.md with gCNR tissue/hazy/dehazed values
4. **When v6/v7 complete**: update SCORES.md with score quality and inference bmode means
5. **After all sweeps done**: run gCNR eval on best model; compare v4 vs v5 gCNR

```bash
# Check job status
squeue -u tp030 --format="%.8i %.20j %.2t %.15R" | grep -v gazelens

# Check inference outputs (after jobs complete)
ls joint_diffusion/figures/ | grep -E "v4|v5|v6|v7"

# Check Phase 4 sweep results
cat joint_diffusion/sweep_results/picmus/results.csv | tail -20
```

## Model Configs Quick Reference

| Model | Config file | Run dir | Key change vs prev |
|-------|------------|---------|-------------------|
| v3 | score_picmus_tissue_v3.yaml | run--ez1mqfie | Extended from v2 + horiz flip; ch=32 |
| v4 | score_picmus_tissue_v4.yaml | run--cs2128ri | ch=64, cosine LR 1e-4→1e-6 |
| v5 | score_picmus_tissue_v5.yaml | run--n1kp0qeb | GroupNorm(8), ELU, batch=8 |
| v6 | score_picmus_tissue_v6.yaml | unknown (runtime) | Same as v5 + t_importance_alpha=0.3 |
| v7 | score_picmus_tissue_v7.yaml | unknown (runtime) | Fine-tunes v5@ep1000; epochs=1300 (300 new); t_importance_alpha=0.3 |

Inference configs: `configs/inference/paper/picmus_dehaze_v{4,5}.yaml` (static); v6/v7 written at runtime to `/scrfs/.../picmus_dehaze_v{6,7}_runtime.yaml`.

## Phase 4: Hyperparameter Tuning [TODO]

Only after inference shows tissue structure (bmode mean < 120, std > 2).

- Sweep `lambda_coeff`: [0.001, 0.005, 0.01, 0.05, 0.1]
- Sweep `ccdf`: [0.5, 0.6, 0.7, 0.8, 0.9]
- Record all results in SCORES.md with params and visual quality notes
- Use `sweep_picmus.py` with `--inf_config` argument

## Phase 5: Quantitative Evaluation [READY TO RUN]

- gCNR infrastructure: `gcnr_eval.py` + `slurm_gcnr_eval.sh` (created 2026-03-06)
- Evaluates 13 contrast_speckle val frames (indices [6,14,17,19,20,21,22,26,29,32,35,36,38])
- Known cyst: center (cx=134, cz=67) px, radius 15px in 271×237 B-mode image
- Signal ROI: r=13px circle; Background ROI: annulus r_inner=17, r_outer=30px
- Run after best inference model identified: `sbatch slurm_gcnr_eval.sh configs/inference/paper/picmus_dehaze_v4.yaml`
- Also computes PSNR and SSIM (already in sweep results)
- Record gCNR results in SCORES.md

## Key Lessons

- **Score model is gating**: At loss > 0.20 (new scale), guidance method doesn't matter. Run `score_quality_check.py` early.
- **Capacity bottleneck**: ch=32@ep1000 only 7% better than ch=32@ep500. ch=64 (v4) matches v3's best at ep280.
- **t-importance sampling**: `t = eps + (T-eps)*u^(1/alpha)`, u~U[0,1]; exponent is `1/alpha` NOT `alpha`. alpha=0.3 → 25% samples at t<0.01 (vs 1% uniform). Backward-compat: alpha=1.0 = uniform.
- **Fine-tune epochs config**: `train.py --resume` runs `range(start_epoch, config.epochs)`. v7 uses `epochs: 1300` (not 300!) to run 300 new epochs from ep1000.
- **Optimizer LR on resume**: Restored optimizer has saved LR (~1e-6 at ep1000), not config's `lr`. Fine for fine-tuning.
- **num_img=2 for PICMUS inference**: `ch=64` U-Net with `torch.enable_grad()` predictor uses ~8GB per 2 images. Set `num_img: 2` (not 5) in PICMUS inference configs to avoid OOM on A100-40GB. The v4 inference job 179128 failed with 30.88GB used for 5 images + gradient tracking.
- **keep_track=false required for PICMUS inference**: `keep_track: true` stores all 1000 timestep images → 30+ GB OOM on A100-40GB. Always `keep_track: false`.
- **Static inference configs on /scrfs**: Never use `/tmp` for runtime inference configs — compute node /tmp can be full and NFS absolute paths cause doubled-path bugs in `init_config`. Write runtime configs to `/scrfs/.../picmus_dehaze_vX_runtime.yaml`.
- **v6/v7 run dir discovery**: Script snapshots wandb dirs before/after training to find new run dir; saves path to `$SCRIPT_DIR/v{6,7}_run_dir.txt` (shared NFS, not /tmp).
- **config.yaml manual copy**: `WANDB_MODE=disabled` means wandb doesn't copy config. Training scripts must `cp config.yaml wandb/run--.../files/`. Inference will fail without it.
- **GPU reliability**: Exclude broken node: `#SBATCH --exclude=c2110 --constraint=1a100`.
- **EMA worse than raw**: `use_ema: false` in all inference configs.
- **reduce_mean=true required for PICMUS**: 2x wider images + grad_clip=1.0 with reduce_mean=false halves effective LR per pixel.
- **Scale mismatch**: Training eval loss (~0.03) != t=0.01 benchmark (~0.19 new scale). Always use `score_quality_check.py` for cross-model comparisons.
- **SCORES.md**: Append-only scoreboard. Add a row whenever a score quality or inference result is obtained. Never delete rows.
- **gCNR val frames**: Contrast_speckle val indices (seed=42, 10% split) = [6,14,17,19,20,21,22,26,29,32,35,36,38] (13 frames: 6 expe + 7 simu). Cyst 1 at pixel (134,67), r=15px in 271×237 B-mode (xlims=[-20,20]mm, zlims=[5,40]mm). Cyst 2 at Z=42.8mm is outside zlims — not visible.
- **gcnr_eval.py**: Standalone evaluation script. Creates SGMDenoiser once, runs in batches of n_batch=4. Pass target_samples directly (bypasses val loader). Must set cfg.image_shape before creating denoiser.
- **Login node /tmp can fill up**: Claude agent bash tool writes output to `/tmp/claude-6294/...` on login node. If /tmp is full (ENOSPC), ALL bash commands fail. In this case: (1) Read/Edit/Write tools still work; (2) SLURM jobs on compute nodes are unaffected; (3) Fix requires clearing /tmp manually or starting fresh session.
- **v4 better dehazing than v5 despite worse score quality**: At fixed hyperparams (lambda=0.01, kappa=0.0), v4 bmode mean=77.9 (close to GT=73.8) while v5 bmode mean=92.2. Hypothesis: optimal guidance params differ per model. Need v5 sweep to find optimal params for v5.
