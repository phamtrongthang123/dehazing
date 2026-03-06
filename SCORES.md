# Scores & Results Log

Append-only. Never delete rows.

## Score Model Quality (val loss at t=0.01, lower = better)

| Date | Model | Epochs | reduce_mean | Loss | Notes |
|------|-------|--------|-------------|------|-------|
| 2026-03-01 | ZEA tissue | 350 | false | 8.2 | Good unconditional samples |
| 2026-03-01 | PICMUS tissue v1 | 350 | false | 38.0 | Flat output, undertrained |
| 2026-03-01 | PICMUS tissue v2 | 500 | true | 28.6 | Tissue visible but weak |

## Inference Results

| Date | Dataset | Guidance | lambda | kappa | ccdf | Score model | Visual quality | Figure |
|------|---------|----------|--------|-------|------|-------------|---------------|--------|
| 2026-03-01 | ZEA | companded_projection | 0.1 | 0.1 | 0.8 | ep350/8.2 | Good tissue structure | zea_tissue_dehazing_3.png |
| 2026-03-01 | PICMUS | companded_projection | 0.01 | 0.0 | 0.8 | ep479/28.6 | Washed out, flat gray | picmus_tissue_dehazing_final.png |
