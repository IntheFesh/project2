# VLA-Collapse-Recover — One-Pager

> Skeleton. All numbers `TBD` from real runs. Reproducible study; no novelty claimed.

**Question.** Open VLAs ace clean LIBERO but collapse under visual/viewpoint perturbations. Which
*training intervention* best recovers robustness — and is the difference statistically real?

**Approach.** SmolVLA (~450M) on a LIBERO subset; LIBERO-Plus graded perturbations (L1–L5).
Compare **A** base · **B** LoRA + standard aug · **C** LoRA + perturbation-targeted aug
(**D** feature-mod adapter, STRETCH). Single RTX 5090, < 5-day GPU budget.

**Headline (the point).** The **A/B/C comparison with paired statistics**, not the recovery size:

| | Perturbed SR (in-dist) | Recovery | Δ_method (C−B) | 95% CI | Holm *p* |
|---|:--:|:--:|:--:|:--:|:--:|
| A / B / C | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

**Rigor.** Bootstrap 95% CIs (≥10k), paired test at fixed init states, Holm–Bonferroni across families.
In-dist vs held-out always labeled. Only delta claims — base absolute SR may not match papers.

**Deliverables.** Collapse curve · recovery curve + lead comparison table · 2–3 before/after demo clips.
