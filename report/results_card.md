# VLA-Collapse-Recover — Results Card

> One-page shareable results card. **All cells `TBD`** until filled from real runs (no fabrication).
> Deltas only; in-dist vs held-out always labeled.

**Question.** Which training intervention most *provably* recovers VLA visual-perturbation
robustness — measured with paired statistics, not point success rates?

**Setup.** SmolVLA (~450M) · LIBERO + LIBERO-Plus (pre-built tasks, 1 trial each) · single RTX 5090,
< 5 days · conditions A (base) / B (LoRA+std aug) / C (LoRA+targeted aug). Base absolute SR may not
match published numbers (deltas only).

## Headline — paired intervention comparison `TBD`

| Metric (in-dist avg) | Value | 95% CI | Holm *p* |
|---|:--:|:--:|:--:|
| **Δ_method = SR_C − SR_B** (paired, fixed task IDs) | `TBD` | `TBD` | `TBD` |
| Recovery_C | `TBD` | `TBD` | — |
| Recovery_B | `TBD` | `TBD` | — |

## Collapse (base, condition A) `TBD`

| | Clean SR | Perturbed SR (avg) | Δ_robust |
|---|:--:|:--:|:--:|
| A (base) | `TBD` | `TBD` | `TBD` |

## Generalization (the overfit check) `TBD`

| | Recovery_C in-dist | Recovery_C held-out | **Generalization gap** |
|---|:--:|:--:|:--:|
| C | `TBD` | `TBD` | `TBD` |

## Mechanism — language-conditioning probe `TBD`

| correct → ablated | paired ΔSR | 95% CI | McNemar *p* |
|---|:--:|:--:|:--:|
| correct → blank / shuffled / mismatched | `TBD` | `TBD` | `TBD` |

*Generated reproducibly by `scripts/analyze_results.py` from the per-trial rollout CSV.*
