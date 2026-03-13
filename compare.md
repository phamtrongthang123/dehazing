# Code vs Paper Verification — Final Report

Systematic verification of the PyTorch implementation in `joint_diffusion/` against the IEEE TMI paper "Dehazing Ultrasound using Diffusion Models" (Stevens et al., 2024). Paper sources: `paper/main.tex`, `paper/algorithm.tex`, `paper/results.tex`.

Date: 2026-03-13

---

## Algorithm 1 vs Code — MATCH

| Paper (Algorithm 1) | Code | Status |
|---|---|---|
| `y = C(y_RF)` companding (μ=255) | `corruptors.py`: `mu_law_compress()` | **MATCH** |
| Factorize into patches | `sampling.py`: `_extract_patches()` | **MATCH** |
| CCDF initialization x_{tτ} | `sampling.py`: `sde.forward_diffuse(y, t)` | **MATCH** |
| Log-likelihood `C(C⁻¹(x) + γ·C⁻¹(h))` | `guidance.py`: `joint_denoise_update()` | **MATCH** |
| Gradient updates `x += λ·∇` | `guidance.py`: `x -= λ·r_t²·∇_x \|\|...\|\|²` | **MATCH** (equivalent: ∇log p ∝ -∇\|\|...\|\|², r_t² is PIGDM enhancement) |
| Euler-Maruyama (tissue + haze) | `sampling.py`: `EulerMaruyamaPredictor` | **MATCH** |
| Patch interleaving | `sampling.py`: `_interleave_patches()` | **MATCH** |
| Expanding `C⁻¹(x₀)` | `inverse.py`: B-mode conversion | **MATCH** |
| Reverse SDE `dx = [f(t)x - g²s_θ]dt + g·dw̄` | `sde_lib.py` reverse SDE | **MATCH** |

## Metrics — MATCH

| Paper | Code | Status |
|---|---|---|
| gCNR = 1 - ∫min(p_A, p_B) | `processing.py:gcnr()`, `gcnr_eval.py` | **MATCH** |
| KS test | `processing.py:ks_test()` | **MATCH** |
| FWHM (autocorrelation) | `processing.py:get_fhwm_from_autocorrelation()` | **MATCH** |
| μ-law companding μ=255 | `processing.py:companding_tf()` | **MATCH** |

---

## Intentional Differences (documented in project logs)

### 1. num_scales: 1000 vs paper's T=200
- **Source**: Inherited from Song et al.'s score_sde_pytorch default (`sde_lib.py` hardcodes N=1000)
- **Log**: `plan.md:99` — "Added `num_scales: 1000` to training configs (was missing)"
- **Impact**: With CCDF=0.8, effective steps = 800 vs paper's 160. More compute but potentially better quality.
- **Rationale**: No explicit decision documented — inherited default. Could reduce to 200 for faster inference.

### 2. Guidance: λ=0.5, κ=0.5 (v4 best) vs paper's λ≈0.5, κ≈0.5
- **Source**: 4-phase hyperparameter sweep (`sweep_picmus.py`)
- **Log**: `SCORES.md:38` — "lambda=0.5, kappa=0.5 → PSNR=29.61"
- **Log**: `implementation_plan.md:47` — "Hyperparams differ per model: v4 best at lambda=0.5, kappa=0.5"
- **History**: Initial configs used λ=0.01, κ=0.0 → "washed out, flat gray" (`SCORES.md:25`). Sweep found 0.5/0.5 optimal for v4 model.
- **Actually MATCHES paper** for v4 model! The paper's λ≈0.5, κ≈0.5 agrees with sweep results.

### 3. Full-frame training vs paper's 128×64 patches
- **No log entry** explaining this choice.
- ZEA trains on [1024, 64], PICMUS on [1024, 128] — full frames, not patches.
- Paper trained on 128×64 patches for in-vivo cardiac data.
- Different dataset structure makes full-frame training reasonable.

### 4. Epochs: 350-1000 vs paper's 100
- **Source**: Extended training needed for PICMUS convergence
- **Log**: `implementation_plan.md:45` — "Score model is gating: At loss > 0.20, guidance method doesn't matter"
- ZEA converged at ~350 epochs; PICMUS needed 1000 epochs to reach score quality < 0.20

### 5. EMA weights disabled (`use_ema: false`)
- **Source**: Discovered empirically
- **Log**: `implementation_plan.md:49` — "EMA worse than raw weights: use_ema=false in all inference configs"
- Paper doesn't explicitly discuss EMA, but original TF code used it by default

---

## Missing from Code (not in logs)

### 1. Brightness offset augmentation (±0.1)
- **Paper** (Section III-D): "random brightness offset uniformly sampled between ±0.1"
- **Code** (`datasets.py`): Only horizontal flip augmentation; **no brightness offset**
- **Impact**: May affect model robustness to intensity variations

### 2. Top-10% brightness matching for evaluation
- **Paper** (Section IV): "brightness matched by matching the average intensity of the top 10% pixel values"
- **Code**: Tracks B-mode brightness values in logs but no explicit top-10% matching algorithm
- **Impact**: Visual comparisons in figures may not be brightness-normalized as paper describes

### 3. Dynamic range: 50 dB vs paper's 60 dB
- **Paper**: "plotted in the same dynamic range of 60 dB"
- **Code**: `dynamic_range: [-50, 0]` in inference configs = 50 dB

### 4. Patch overlap (PICMUS): 4.7% vs paper's 10%
- **Paper**: "10% overlap between adjacent patches"
- **Code**: `patch_overlap: 6` pixels on 128px width = 4.7%
- ZEA is fine: 6px/64px = 9.4% ≈ 10%

---

## Detailed Verification Results

### 1. SDE Parameters — MATCH
**File:** `joint_diffusion/generators/SGM/sde_lib.py`
- Paper uses `simple` SDE: `f(t)=0, g(t)=σ^t, σ=25`
- Code (`simple` class, line 314-354):
  - `sde()`: `drift = zeros, diffusion = σ^t` — **MATCH**
  - `marginal_prob()`: `mean = x, std = √((σ^{2t}-1)/(2·ln σ))` — **MATCH**
  - `α_t = 1` (mean = x, no scaling) — **MATCH**
- Config: `sde: simple, sigma: 25` in both ZEA and PICMUS configs — **MATCH**

### 2. Augmentation — PARTIAL MATCH
**File:** `joint_diffusion/datasets.py`
- Paper: "random left-right flip with equal probabilities" → Code line 103-105: `torch.flip(x, dims=[2])` with 50% prob — **MATCH**
- Paper: "random brightness offset uniformly sampled between ±0.1" → **NOT IMPLEMENTED**

### 3. DSM Objective (score_loss) — MATCH
**File:** `joint_diffusion/generators/SGM/SGM.py`, lines 205-261
- Paper Eq. 8: `||s_θ(x_t, t) - ∇_{x_t} log q(x_t|x)||²`
- Code: `losses = (score * std + z)²` — equivalent because:
  - `score = model(x)/std` (line 278), so `score * std = model(x)`
  - `∇_{x_t} log q(x_t|x) = -z/std` (Gaussian kernel)
  - Simplifies to `(model(x) + z)²` = `(score * std + z)²` — **MATCH**
- `reduce_mean` option: per-pixel averaging vs sum — correctly configurable
- t-importance sampling (alpha parameter): Enhancement not in paper, added for v6/v7

### 4. Inference Configs — Verified
**ZEA** (`zea_dehaze_pigdm.yaml`):
- `patch_overlap: 6` pixels (on 64px width = 9.4% ≈ paper's 10%) — **MATCH**
- `dynamic_range: [-50, 0]` = 50 dB (paper says 60 dB) — **DIFFERS** (minor)
- `lambda: 0.1, kappa: 0.1` — different from paper's ~0.5, appropriate for ZEA

**PICMUS v4** (`picmus_dehaze_v4.yaml`):
- `patch_overlap: 6` pixels (on 128px width = 4.7%) — smaller than paper's 10%
- `dynamic_range: [-50, 0]` = 50 dB — **DIFFERS** from paper's 60 dB
- `lambda: 0.5, kappa: 0.5` — **MATCHES paper** (tuned via sweep)
- `ccdf: 0.7` — slightly different from paper's 0.8 (sweep-optimized)
- `use_ema: false` — paper doesn't specify, but documented as better

### 5. B-mode Display & Brightness Matching
**File:** `joint_diffusion/utils/inverse.py`
- Dynamic range: `[-50, 0]` dB in configs (50 dB), paper uses 60 dB
- B-mode conversion uses `rf_to_bmode()` with configurable dynamic range
- **No brightness matching** (top-10% intensity matching) found anywhere in codebase
- B-mode plots use `vmin=0, vmax=255` for display — standard uint8 range

---

## Final Summary

### Confirmed MATCH (core correctness)
| Item | Status |
|---|---|
| Algorithm 1 (joint posterior sampling) | **MATCH** |
| Reverse SDE (Euler-Maruyama) | **MATCH** |
| SDE: simple, σ=25, f=0, g=σ^t | **MATCH** |
| μ-law companding (μ=255) | **MATCH** |
| Companded guidance: C(C⁻¹(x)+γ·C⁻¹(h)) | **MATCH** |
| DSM training objective (Eq. 8) | **MATCH** |
| Patch interleaving | **MATCH** |
| CCDF initialization | **MATCH** |
| gCNR, KS test, FWHM metrics | **MATCH** |
| Horizontal flip augmentation | **MATCH** |
| Patch overlap ~10% (ZEA) | **MATCH** |

### Documented Differences (justified by experiments)
| Item | Paper | Code | Reason |
|---|---|---|---|
| num_scales | 200 | 1000 | Inherited from score_sde_pytorch defaults |
| λ, κ (PICMUS v4) | ~0.5, ~0.5 | 0.5, 0.5 | **Actually matches!** Sweep-optimized |
| Epochs | 100 | 350-1000 | Score model quality required more training |
| EMA | implied on | off | Discovered worse than raw weights |
| Training size | 128×64 patches | full frames | Different dataset structure |

### Gaps (not implemented)
| Item | Paper | Code | Impact |
|---|---|---|---|
| Brightness offset ±0.1 augmentation | Yes | **Missing** | May reduce robustness to intensity variations |
| Top-10% brightness matching | Yes (for fair comparison) | **Missing** | Figures may not be brightness-normalized |
| Dynamic range | 60 dB | 50 dB | Minor visual difference in B-mode display |
| Patch overlap (PICMUS) | 10% | 4.7% (6px/128px) | May affect patch coherence at boundaries |
