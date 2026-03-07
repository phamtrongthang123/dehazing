# Scores & Results Log

Append-only. Never delete rows.

## Score Model Quality (val loss at t=0.01, lower = better)

| Date | Model | Epochs | reduce_mean | Loss | Notes |
|------|-------|--------|-------------|------|-------|
| 2026-03-01 | ZEA tissue | 350 | false | 8.2 | Good unconditional samples |
| 2026-03-01 | PICMUS tissue v1 | 350 | false | 38.0 | Flat output, undertrained |
| 2026-03-01 | PICMUS tissue v2 | 500 | true | 28.6 | Tissue visible but weak |
| 2026-03-06 | PICMUS tissue v3 | 1000 (ep1000) | true | 0.2695 (new scale) | Extended from v2 + horizontal flip; run--ez1mqfie |
| 2026-03-06 | PICMUS tissue v4 | ep320 | true | 0.2486 | channels=64, BatchNorm, cosine LR; run--cs2128ri |
| 2026-03-06 | PICMUS tissue v4 | ep920 (intermediate) | true | **0.1972** | BELOW 0.20 TARGET! Phase 4 may proceed |
| 2026-03-06 | PICMUS tissue v4 | ep1000 (FINAL) | true | **0.1886** | Best PICMUS model yet! Continued improvement from ep920 |
| 2026-03-06 | PICMUS tissue v5 | ep1000 | true | **0.1603** | GroupNorm(8), ELU, batch=8; run--n1kp0qeb; BEST PICMUS model! Beats v4 |
| 2026-03-06 | PICMUS tissue v6 | FAILED (OOM) | true | — | c2008 zombie proc (PID 2962654, 30.88GB); resubmitted as job 180045 excluding c2008 |
| 2026-03-06 | PICMUS tissue v7 | FAILED (OOM) | true | — | Same c2008 zombie issue; resubmitted as job 180048 excluding c2008 |

## Inference Results

| Date | Dataset | Guidance | lambda | kappa | ccdf | Score model | Visual quality | Figure |
|------|---------|----------|--------|-------|------|-------------|---------------|--------|
| 2026-03-01 | ZEA | companded_projection | 0.1 | 0.1 | 0.8 | ep350/8.2 | Good tissue structure | zea_tissue_dehazing_3.png |
| 2026-03-01 | PICMUS | companded_projection | 0.01 | 0.0 | 0.8 | ep479/28.6 | Washed out, flat gray | picmus_tissue_dehazing_final.png |
| 2026-03-06 | PICMUS | pigdm | 0.01 | 0.0 | 0.8 | ep499/28.6 | Washed out — same as companded_proj | 2026_03_06_sgm_picmus_tissue_dehazing.png |
| 2026-03-06 | PICMUS | projection | 0.01 | 0.0 | 0.8 | ep499/28.6 | Washed out — same as companded_proj | 2026_03_06_sgm_picmus_tissue_dehazing_0.png |
| 2026-03-06 | PICMUS | companded_projection | 0.01 | 0.0 | 0.8 | ep499/28.6 | Washed out — same as all methods | 2026_03_06_sgm_picmus_tissue_dehazing_1.png |
| 2026-03-06 | PICMUS | companded_projection | 0.01 | 0.0 | 0.8 | v3/ep1000 | Still washed out; bmode mean=181.6 (bright), std=0.056 (uniform) | — |
| 2026-03-06 | PICMUS | companded_projection | 0.01 | 0.0 | 0.8 | v5/ep1000 (score=0.1603) | Some structure visible; bmode mean=92.2 (target=73.8, hazy=115.3) | 2026_03_06_sgm_picmus_tissue_dehazing_4.png |
| 2026-03-06 | PICMUS | companded_projection | 0.01 | 0.0 | 0.8 | v4/ep1000 (score=0.1886) | **Row 1: CLEAR tissue layers!** bmode mean=77.9 (target=73.8, hazy=115.3) | 2026_03_06_sgm_picmus_tissue_dehazing_5.png |
