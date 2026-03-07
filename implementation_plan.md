# Implementation Plan

## Status Summary

| Phase | Status | Result |
|-------|--------|--------|
| 1: Guidance sanity check | DONE | All 3 guidance methods produce same washed-out output — score model was bottleneck |
| 2: Fix guidance | SKIPPED | Not needed — guidance was correct |
| 3a: v3 model (ep1000, ch=32) | DONE | Score 0.2695; model capacity was bottleneck |
| 3b: v4 model (ep1000, ch=64) | DONE | Score **0.1886**; clear tissue layers in inference (bmode=77.9, target=73.8) |
| 3c: v5 model (GroupNorm, batch=8) | DONE | Score **0.1603** (best); inference bmode=92.2 (needs param tuning) |
| 3d: v6 model (t-importance, scratch) | FAILED | OOM on c2008 (zombie process); resubmitted jobs never ran |
| 3e: v7 model (fine-tune v5+t-importance) | FAILED | Same OOM; resubmitted jobs never ran |
| 4: Hyperparameter tuning (v4) | DONE | Best: gamma=0.05, lambda=0.5, kappa=0.5, ccdf=0.7 → PSNR=30.20, SSIM=0.814 |
| 5: gCNR evaluation (v4) | DONE | **PASS**: dehazed gCNR=0.314 > hazy=0.224 (+0.089) |

## Project Completion

All spec criteria are met as of 2026-03-07:
1. Dehazed B-mode images show clear tissue structure (v4 model)
2. Dehazed output visually distinct from hazy input (bmode 77.9 vs 115.3)
3. gCNR dehazed (0.314) > gCNR hazy (0.224) — quantitative improvement confirmed
4. Works for both ZEA and PICMUS datasets

## Best Models & Configs

| Dataset | Run dir | Score quality | Inference config |
|---------|---------|---------------|-----------------|
| ZEA tissue | run--jm8ljbms | 0.0655 | paper/zea_dehaze_pigdm.yaml |
| ZEA haze | run--1bj42atz | — | paper/zea_dehaze_pigdm.yaml |
| PICMUS tissue (v4) | run--cs2128ri | 0.1886 | paper/picmus_dehaze_v4.yaml |
| PICMUS tissue (v5) | run--n1kp0qeb | 0.1603 | paper/picmus_dehaze_v5.yaml |

Best PICMUS inference params (v4 sweep): gamma=0.05, lambda=0.5, kappa=0.5, ccdf=0.7

## Optional Future Work (not required for spec)

- Run v5 hyperparameter sweep (v5 has better score quality, may beat v4 with tuned params)
- Re-run v6/v7 training on a working GPU node
- Run gCNR eval with best sweep params (current gCNR used default params, not sweep-optimized)
- Run gCNR eval on v5 model

## Key Lessons

- **Score model is gating**: At loss > 0.20, guidance method doesn't matter
- **Capacity matters**: ch=64 (v4) >> ch=32 (v3); GroupNorm+ELU (v5) >> BatchNorm+SiLU (v4) for score quality
- **Hyperparams differ per model**: v4 best at lambda=0.5, kappa=0.5 (not the original 0.01/0.0)
- **reduce_mean=true required for PICMUS**: 2x wider images + grad_clip halves effective LR
- **EMA worse than raw weights**: use_ema=false in all inference configs
- **keep_track=false for PICMUS**: storing 1000 timesteps causes OOM
- **GPU reliability**: c2008 had zombie process occupying 30.88 GB; always exclude known-bad nodes
