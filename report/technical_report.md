# VLA-Collapse-Recover — Technical Report

> **Status: skeleton.** Every result is `TBD` until filled from a real run. No fabricated numbers.
> A reproducible *study*, not a paper; no novelty claimed.

## 1. Headline — intervention comparison (lead result)

The contribution is the **A/B/C(/D) comparison under matched perturbations**, with paired
statistics — **not** the recovery magnitude (same-family recovery is expected). Perturbed SR is
averaged over the in-distribution augmented families; paired at fixed init states; 3 seeds on B/C(/D).

| Condition | Aug | Clean SR | Perturbed SR (in-dist) | Recovery | Δ_method vs B | 95% CI | Holm *p* |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| A base | — | `TBD` | `TBD` | — | — | — | — |
| B LoRA+std aug | generic | `TBD` | `TBD` | `TBD` | ref | — | — |
| C LoRA+targeted aug | aligned | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| D feature-mod (STRETCH) | FTM/FLA | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

*Narrative of the paired B-vs-C result: `TBD`.*

## 2. Setup & the reproduction-gap disclaimer

- Model: SmolVLA (~450M), base `HuggingFaceVLA/smolvla_libero`. Benchmark: LIBERO subset.
- Perturbations: LIBERO-Plus (graded L1–L5), families: viewpoint / lighting / texture / noise.
- **The base absolute clean SR here may not match the published number** (LIBERO eval setup is
  sensitive). This is *why only deltas are claimed.* Exact eval config: `TBD`.
- Env: CUDA 12.8 (cu128), PyTorch ≥ 2.7, single RTX 5090 (sm_120), serial rollouts.

## 3. Collapse (Phase 2, condition A, 1 seed) — `TBD`

Base-model SR vs perturbation level per family; Δ_robust @ L4. Plot: `TBD`.

## 4. Recovery (Phase 4) — `TBD`

Per-family SR for A/B/C with **explicit in-dist vs held-out tags**. In-dist recovery is **not**
generalization. Held-out (cross-family, STRETCH) results, if run, are labeled as such.

## 5. Statistics (Phase 5)

- Bootstrap 95% CIs (≥ 10,000 resamples) on success rates.
- Paired comparison at fixed init states (McNemar / paired bootstrap) — detects the ~5–10pp B-vs-C gap.
- Holm–Bonferroni correction across families.

## 6. Failure cases — `TBD`

Where C still fails; qualitative rollout notes; any family with no recovery.

## 7. Prior art & positioning

Cites arXiv 2510.00037 (RobustVLA *method*, closest prior work — distinct from this *study*),
2510.13626 (LIBERO-Plus), 2510.03827 (LIBERO-PRO), 2512.02902 (FTM/FLA), 2510.17640 (RESample).

## 8. Limitations & honesty notes

Single card / < 5-day budget; small task subset; in-dist headline recovery by design; absolute
numbers not comparable to papers (deltas only).
