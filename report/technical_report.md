# VLA-Collapse-Recover — Technical Report

> **Status: skeleton.** Every result is `TBD` until filled from a real run. No fabricated numbers.
> A reproducible *study*, not a paper; no novelty claimed.

## 0. Question & contributions

**Question.** Do open VLAs that score high on clean LIBERO actually perceive the scene, or exploit
shortcuts that break under a moved camera / changed lighting — and which training intervention most
*provably* repairs that brittleness?

Two contributions:

1. **A reproducible measurement** of visual-perturbation collapse and recovery from
   perturbation-targeted LoRA — on a single GPU, under 5 days.
2. **A diagnostic-probe battery** — held-out cross-family generalization + language-conditioning
   sensitivity + (optional) visual-feature-shift — that distinguishes **representation-level fixing
   from symptom patching**. Canonical description: `docs/PROBES.md`.

The A/B/C intervention comparison and paired statistics are the *instrument* the probes are read
with — **supporting infrastructure** (shared with the author's prior **PolicyArena** project), not
this repo's contribution.

**World-model framing.** Perturbation robustness, **language conditioning**, and **cross-family
generalization** are *joint* proxies for whether the policy's implicit representation is *causal*
(models the scene) or *shortcut-based* (latches onto spurious cues a perturbation destroys). A
representation-level fix improves all three *together*; a symptom patch raises only in-dist success.
We *test* these observable consequences rather than *claim* causality.

## 1. How the probes are read — the intervention comparison

**Question (restated).** Does perturbation-targeted LoRA fix the representation, or patch symptoms?
The **A/B/C(/D) comparison under matched perturbations**, with paired statistics, is **how we read
the probe results** — *not* the contribution itself, and **not** the recovery magnitude (same-family
recovery is expected). Perturbed SR is averaged over the in-distribution augmented families; **paired
at fixed task IDs** (matched by `(task_id, level, seed)`); 3 LoRA-training seeds on B/C(/D).

| Condition | Aug | Clean SR | Perturbed SR (in-dist) | Recovery | Δ_method vs B | 95% CI | Holm *p* |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| A base | — | `TBD` | `TBD` | — | — | — | — |
| B LoRA+std aug | generic | `TBD` | `TBD` | `TBD` | ref | — | — |
| C LoRA+targeted aug | aligned | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| D feature-mod (STRETCH) | FTM/FLA | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

*Narrative of the paired B-vs-C result: `TBD`.*

## 2. Setup & the reproduction-gap disclaimer

- Model: SmolVLA (~450M, arXiv:2506.01844), base `HuggingFaceVLA/smolvla_libero`. Benchmark: LIBERO subset.
- Perturbations: LIBERO-Plus (arXiv:2510.13626) **pre-built task instances** selected by
  `(family, level)`, run once each (`num_trials_per_task = 1`); families: viewpoint / lighting /
  texture / noise (+ held-out). See `docs/LIBERO_PLUS_NOTES.md`.
- **The base absolute clean SR here may not match the published number** (LIBERO eval setup is
  sensitive). This is *why only deltas are claimed.* Exact eval config: `TBD`.
- Env: CUDA 12.8 (cu128), PyTorch ≥ 2.7, single RTX 5090 (sm_120), serial rollouts.

## 3. Collapse (Phase 2, condition A, 1 seed) — `TBD`

Base-model SR vs difficulty level per family; Δ_robust. Plot: `TBD`.

## 4. Recovery & cross-family generalization (Phase 4) — `TBD`

Per-family SR for A/B/C with **explicit in-dist vs held-out tags** (via `classify_distribution`).
In-dist recovery is **not** generalization.

**First-class held-out test.** Condition C is evaluated on at least one family **never seen during
augmentation** (`held_out_families`, e.g. `layout`), and we report the **generalization gap** =
Recovery_C(in-dist) − Recovery_C(held-out) (`generalization_gap` in `eval/metrics.py`). A large
positive gap means C overfits its augmentation family; ``~0`` means it genuinely generalizes. This
directly addresses the "did you just train on the test perturbation?" red flag. Held-out rows are
labeled as such in every table.

## 5. Statistics (Phase 5) — the harness

- Bootstrap 95% CIs (≥ 10,000 resamples) on success rates.
- **Paired** comparison at **fixed task IDs** (McNemar / paired bootstrap, matched by
  `(task_id, level, seed)`) — detects the ~5–10pp B-vs-C gap.
- Holm–Bonferroni correction across families.
- rliable-style aggregate (pooled SR + CI + IQM of per-cell SRs).

Packaged as `eval/stats/` and driven by `scripts/analyze_results.py`; protocol in `docs/EVALUATION.md`.

This statistical infrastructure is **supporting, not a contribution** — the project makes **no claim
of a methodological contribution in evaluation statistics**. *Statistical infrastructure (paired
bootstrap, McNemar, Holm–Bonferroni) is shared with the author's prior project, **PolicyArena** — a
statistically-validated tool-calling + RAG agent for a Chinese enterprise service desk. This
repository's distinct contribution is the **visual representation quality diagnostic** (held-out
cross-family generalization + language-conditioning probe), not the statistical methodology.*

## 6. Failure cases — `TBD`

Where C still fails; qualitative rollout notes; any family with no recovery.

## 6.1 Mechanistic probe — language conditioning (Phase 6) — `TBD`

Evidence about *why* collapse happens. The same task IDs are run under correct / blank / shuffled /
mismatched instructions; we report **paired** ΔSR (`language_sensitivity`, matched per task ID;
stats via `eval/paired.py`). ΔSR ≈ 0 indicates the policy ignores language (effectively a
vision-action model), consistent with the LIBERO-Plus / LIBERO-PRO observation. An optional
vision-feature-shift probe (cosine distance of encoder features, clean vs perturbed) is scaffolded
but low-priority. Budget impact is counted as `extra_units` in the estimator.

## 7. Prior art & positioning

Method / benchmark prior art: arXiv 2510.00037 (RobustVLA *method*, closest prior work — distinct
from this *study*), 2510.13626 (LIBERO-Plus), 2510.03827 (LIBERO-PRO), 2512.02902 (FTM/FLA),
2510.17640 (RESample), 2506.01844 (SmolVLA base).

**Positioning lineage (the contribution): representation quality** — shortcut learning
(arXiv:2004.07780, Geirhos et al.), causal representation learning (arXiv:2102.11107, Schölkopf et
al.), world models (arXiv:1803.10122, Ha & Schmidhuber). The probes test for causal vs. shortcut
representations.

Methods / evaluation references (supporting infrastructure, not the contribution): rliable
(arXiv:2108.13264), SimplerEnv (arXiv:2405.05941), AutoEval (arXiv:2503.24278).

## 8. Limitations & honesty notes

- Single card / < 5-day budget; a **subset** of LIBERO-Plus's ~10k pre-built tasks; in-dist headline
  recovery is expected by design (the held-out gap is the generalization test).
- Absolute numbers are not comparable to papers (deltas only).
- **Condition-C design fork (open, documented).** torchvision augmentation — especially for
  `viewpoint` — is a **weak 2-D proxy** for LIBERO-Plus's true **3-D camera-viewpoint** perturbation:
  a single 2-D frame cannot reproduce a moved camera. The honest alternative is to fine-tune
  Condition C on LIBERO-Plus's **released perturbation training data** (RLDS / LeRobot). We keep
  torchvision augmentation as the **default** and provide an optional, clearly-marked config stub
  (`configs/lora/condition_c_realdata.yaml`) for the alternative; it is **not implemented** here.
- LIBERO-Plus integration details still marked `TODO(verify)` in `docs/LIBERO_PLUS_NOTES.md` must be
  confirmed against the installed package on the GPU box before trusting absolute task counts.
