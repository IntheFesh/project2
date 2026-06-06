# VLA-Collapse-Recover — One-Pager

> Skeleton. All numbers `TBD` from real runs. Reproducible study; no novelty claimed.

**Question.** Do open VLAs that ace clean LIBERO actually perceive the scene, or exploit shortcuts
that break under a moved camera / changed lighting — and which training intervention most *provably*
repairs that brittleness?

**Two contributions.** (1) An honest measurement of visual-perturbation **collapse → recovery** from
perturbation-targeted LoRA, with paired statistics + a first-class **held-out generalization** test
+ a **language-conditioning probe**. (2) A **reusable paired-statistics evaluation harness/protocol**
for VLA robustness (`eval/stats/`, `docs/EVALUATION.md`) — the VLA analogue of rliable (arXiv:2108.13264).

**Approach.** SmolVLA (~450M, arXiv:2506.01844) on a LIBERO subset; LIBERO-Plus **pre-built**
perturbation tasks selected by `(family, level)`, run once each. Compare **A** base · **B** LoRA +
standard aug · **C** LoRA + perturbation-targeted aug (**D** feature-mod, STRETCH). Single RTX 5090,
< 5-day GPU budget (a deliberate scoping-discipline feature).

**Headline (the point).** The **A/B/C comparison with paired statistics**, not the recovery size:

| | Perturbed SR (in-dist) | Recovery | Δ_method (C−B) | 95% CI | Holm *p* | Gen. gap (in−held) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| A / B / C | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

**Rigor.** Bootstrap 95% CIs (≥10k); **paired** test at **fixed task IDs** (matched by
`(task_id, level, seed)`); Holm–Bonferroni across families; rliable-style IQM aggregate. In-dist vs
held-out always labeled. Only delta claims — base absolute SR may not match papers.

**Deliverables.** Collapse curve · recovery + lead comparison table + held-out generalization gap ·
language-probe ΔSR · 2–3 before/after demo clips · the reusable eval harness.
