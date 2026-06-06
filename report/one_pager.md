# VLA-Collapse-Recover — One-Pager

> Skeleton. All numbers `TBD` from real runs. Reproducible study; no novelty claimed.

**Question.** Do open VLAs that ace clean LIBERO actually perceive the scene, or exploit shortcuts
that break under a moved camera / changed lighting — and which training intervention most *provably*
repairs that brittleness?

**Two contributions.** (1) A reproducible measurement of visual-perturbation **collapse → recovery**
from perturbation-targeted LoRA, single GPU under 5 days. (2) **A diagnostic-probe battery** that
distinguishes **representation-level fixing from symptom patching** (`docs/PROBES.md`). The
paired-statistics machinery is **supporting infrastructure** (shared with the author's prior
**PolicyArena** project), not a headline claim.

**Probes (the contribution).**
- **Held-out cross-family generalization** — `generalization_gap` (in-dist − held-out): small ⇒ a
  real fix, large ⇒ family-specific patch.
- **Language-conditioning sensitivity** — `SR_correct − SR_ablated` (paired): ≈ 0 ⇒ ignores language;
  should rise after a real fix.
- **Visual-feature-shift** *(optional / scaffolded)* — vision-encoder cosine distance, clean vs perturbed.

**Approach.** SmolVLA (~450M, arXiv:2506.01844) on a LIBERO subset; LIBERO-Plus **pre-built**
perturbation tasks selected by `(family, level)`, run once each. Compare **A** base · **B** LoRA +
standard aug · **C** LoRA + perturbation-targeted aug (**D** feature-mod, STRETCH). Single RTX 5090,
< 5-day GPU budget (a deliberate scoping-discipline feature).

**Headline (the point).** Read the probes **via** the A/B/C comparison + paired statistics (not the recovery size):

| | Perturbed SR (in-dist) | Recovery | Δ_method (C−B) | 95% CI | Holm *p* | Gen. gap (in−held) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| A / B / C | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

**Rigor.** Bootstrap 95% CIs (≥10k); **paired** test at **fixed task IDs** (matched by
`(task_id, level, seed)`); Holm–Bonferroni across families; rliable-style IQM aggregate. In-dist vs
held-out always labeled. Only delta claims — base absolute SR may not match papers.

**Deliverables.** Collapse curve · recovery + intervention table + held-out **generalization gap** ·
**language-sensitivity** ΔSR · 2–3 before/after demo clips · diagnostic-probe report
(`scripts/analyze_results.py`).

**Caveat.** Condition-C `viewpoint` augmentation is a weak 2-D proxy for true 3-D camera-viewpoint
perturbation (see report limitations).
