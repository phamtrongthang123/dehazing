# Implementation Plan

## Phase 1: Guidance sanity check [TODO]
**Goal**: Determine if the problem is the guidance method or the score model.

- [ ] Run inference with `guidance: pigdm` instead of `companded_projection` (edit picmus config, 1-line change)
- [ ] Run inference with `guidance: projection` as another baseline
- [ ] Compare all three outputs visually — save figures with descriptive names
- [ ] Record findings in SCORES.md

**Decision point**: If pigdm/projection produce better tissue structure → companded_projection has a bug (go to Phase 2). If all three are washed out → score model is the bottleneck (skip to Phase 3).

## Phase 2: Fix guidance [TODO, conditional]
Only if Phase 1 shows companded_projection is buggy.

- [ ] Compare `companded_projection.denoise_update()` math against PIGDM paper Algorithm 1
- [ ] Check gradient sign: `x = x - lambda * r_t^2 * grad_x` — verify this pushes x toward data consistency
- [ ] Write a unit test: given a known x_true and y = corrupt(x_true), does guidance push x toward x_true?
- [ ] Fix any bugs found, re-run inference, compare

## Phase 3: Improve score model [TODO, conditional]
Only if Phase 1 shows all guidance methods produce washed output.

- [ ] Run `diagnostic_picmus.py` to generate unconditional samples from current model — assess quality
- [ ] Train PICMUS tissue model for 1000 epochs (current: 500) with `reduce_mean: true`
- [ ] Try data augmentation: random horizontal flips in `datasets.py` for PICMUS
- [ ] Try learning rate warmup or cosine schedule
- [ ] After training, check val loss at t=0.01 (target: < 20, current: 28.6)
- [ ] Generate unconditional samples from new model — should show clear tissue structure
- [ ] Re-run inference with best checkpoint

## Phase 4: Hyperparameter tuning [TODO]
Only after dehazing shows tissue structure (not flat gray).

- [ ] Sweep lambda_coeff: [0.001, 0.005, 0.01, 0.05, 0.1]
- [ ] Sweep ccdf: [0.5, 0.6, 0.7, 0.8, 0.9]
- [ ] Record all results in SCORES.md with params and visual quality notes

## Phase 5: Quantitative evaluation [TODO]
- [ ] Compute gCNR on dehazed vs hazy input (use `processing.py:gcnr`)
- [ ] Compute PSNR and SSIM vs ground truth
- [ ] Record in SCORES.md
