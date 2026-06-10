# Phase 5 -- Paired-statistics summary

Generated from `analysis/runs/phase2_collapse.csv` + `analysis/runs/phase4_recovery.csv`.
All CIs are 95% percentile bootstrap (10,000 resamples). Significance: McNemar exact (< 25 discordant pairs) or chi2 continuity-corrected.

## Cell-level success rates (A / B / C)

| family | level | A | B | C |
|---|---|---|---|---|
| clean | 0 |  64.0% [ 50.0,  76.0] (n=50) |  68.0% [ 54.0,  80.0] (n=50) |  58.0% [ 44.0,  72.0] (n=50) |
| viewpoint | 2 |   0.0% [  0.0,   0.0] (n=60) |   0.0% [  0.0,   0.0] (n=60) |   0.0% [  0.0,   0.0] (n=60) |
| viewpoint | 4 |   0.0% [  0.0,   0.0] (n=60) |   0.0% [  0.0,   0.0] (n=60) |   0.0% [  0.0,   0.0] (n=60) |
| lighting | 2 |  41.7% [ 30.0,  55.0] (n=60) |  48.3% [ 35.0,  61.7] (n=60) |  38.3% [ 26.7,  50.0] (n=60) |
| lighting | 4 |  16.7% [  8.3,  26.7] (n=60) |  48.3% [ 36.7,  61.7] (n=60) |  36.7% [ 25.0,  48.3] (n=60) |
| texture | 2 |  11.7% [  5.0,  20.0] (n=60) |  31.7% [ 20.0,  43.3] (n=60) |  28.3% [ 16.7,  40.0] (n=60) |
| texture | 4 |  53.3% [ 40.0,  66.7] (n=60) |  46.7% [ 35.0,  60.0] (n=60) |  56.7% [ 43.3,  68.3] (n=60) |
| noise | 2 |  24.4% [ 13.3,  37.8] (n=45) |   3.3% [  0.0,  10.0] (n=30) |  10.0% [  0.0,  23.3] (n=30) |
| noise | 4 |   0.0% [  0.0,   0.0] (n=35) |   0.0% [  0.0,   0.0] (n=30) |   0.0% [  0.0,   0.0] (n=30) |
| layout | 2 |  16.7% [  8.3,  26.7] (n=60) |  40.0% [ 28.3,  51.7] (n=60) |  43.3% [ 31.7,  55.0] (n=60) |
| layout | 4 |  20.0% [ 10.0,  30.0] (n=60) |  26.7% [ 16.7,  38.3] (n=60) |  31.7% [ 20.0,  43.3] (n=60) |

## H1 -- LoRA + standard augmentation lifts robustness (A vs B, pooled)

- Pooled over 10 perturbed cells, n=540 matched episodes.
- SR_A = 19.6%, SR_B = 27.0%
- delta = +7.4pp, 95% CI +2.8 to +11.9
- McNemar: chi2 (continuity-corrected), p = 0.001793 (B-only wins=58, A-only wins=98)

## H2 -- Targeted augmentation per in-dist family (B vs C, Holm corrected)

| family | n | SR_B | SR_C | delta (pp) [95% CI] | p_raw | p_Holm | reject @ 0.05 |
|---|---|---|---|---|---|---|---|
| lighting | 120 | 48.3% | 37.5% | -10.8 [-22.5, +0.0] | 0.0929 | 0.279 | no |
| noise | 60 | 1.7% | 5.0% | +3.3 [-3.3, +10.0] | 0.625 | 1 | no |
| texture | 120 | 39.2% | 42.5% | +3.3 [-7.5, +14.2] | 0.658 | 1 | no |

## H3 -- Held-out generalization on layout (A vs B)

- n=120 matched episodes (layout L2+L4 pooled)
- SR_A = 18.3%, SR_B = 33.3%
- delta = +15.0pp, 95% CI +5.0 to +25.0
- McNemar: chi2 (continuity-corrected), p = 0.00719

## Headline numbers (point estimates for README/PROBES tables)

- SR_A clean = 64.0% (n=50)
- Delta_robust pooled = +7.4pp on A vs B (proxy for LoRA's net robustness gain)
- Held-out (layout) gain = +15.0pp on A vs B (LoRA's task-rep transfer)
