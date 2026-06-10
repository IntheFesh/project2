# VLA-Collapse-Recover — Case Study

A single-GPU diagnostic of vision-language-action models under perturbation,
with paired statistics and honest negative findings.

**Repo:** [github.com/IntheFesh/project2](https://github.com/IntheFesh/project2) ·
**Budget:** one RTX 5090, < 1 GPU-day · **Sample:** 1,790 per-episode trials, single seed.

---

## 1. The question (and why it's not a "method paper")

Open vision-language-action models such as SmolVLA score well on clean LIBERO
benchmarks but **collapse** under modest visual perturbations — a moved camera,
changed lighting, added sensor noise. Perturbation-targeted LoRA fine-tuning is
known to lift those scores. The interesting question is **how**:

> Does perturbation-targeted LoRA fix the underlying *representation* — so the
> recovery transfers to held-out perturbation families — or does it merely
> *patch symptoms* on whichever families it was trained on?

This project is a **diagnostic, not a method**. Its contribution is the
measurement framework and the resulting finding, not a new fine-tuning algorithm.

## 2. Experimental design (one paragraph)

Three conditions on SmolVLA-LIBERO, identical in everything except the LoRA
augmentation: **A** = base policy (no fine-tuning); **B** = LoRA + the generic
photometric augmentation that ships with LeRobot; **C** = LoRA + augmentation
*aligned* to four evaluation perturbation families (viewpoint / lighting /
texture / noise). All three are evaluated on the LIBERO-spatial suite at two
LIBERO-Plus difficulty levels (L2, L4), on the four trained families *plus* a
fifth held-out family (`layout`) that no condition ever sees during
augmentation. LoRA rank 16, α 32, 20k steps; evaluation pairs episodes at fixed
`(task_id, episode_index)` so McNemar and paired bootstrap apply.

## 3. Results

![Recovery heatmap](../analysis/runs/recovery_heatmap.png)

Three pre-registered hypotheses, paired statistics, Holm-Bonferroni where multiple
comparisons are tested:

| Test | Quantity | Estimate (95% CI) | Significance |
|---|---|---|---|
| **H1** — LoRA + standard aug lifts robustness (A vs B, pooled over all perturbed cells) | ΔSR | **+7.4 pp** [+2.8, +11.9] | McNemar p ≈ **0.0018** |
| **H2** — Targeted aug beats standard aug (B vs C, per in-dist family, Holm-corrected) | ΔSR per family | lighting **−10.8**, noise +3.3, texture +3.3 | all Holm-p > 0.05 |
| **H3** — LoRA generalizes to held-out family (A vs B on `layout`) | ΔSR | **+15.0 pp** [+5.0, +25.0] | McNemar p ≈ **0.0072** |

### Cell-level success rates

| family | level | A: base | B: LoRA + standard | C: LoRA + targeted |
|---|---|---|---|---|
| clean | 0 | 64.0% | 68.0% | 58.0% |
| viewpoint | L2 / L4 | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| lighting | L2 / L4 | 41.7% / 16.7% | 48.3% / 48.3% | 38.3% / 36.7% |
| texture | L2 / L4 | 11.7% / 53.3% | 31.7% / 46.7% | 28.3% / 56.7% |
| noise | L2 / L4 | 24.4% / 0.0% | 3.3% / 0.0% | 10.0% / 0.0% |
| **layout** *(held-out)* | L2 / L4 | 16.7% / 20.0% | 40.0% / 26.7% | 43.3% / 31.7% |

## 4. The finding (read this paragraph if you read nothing else)

**LoRA fine-tuning improves task representation in a way that transfers across
perturbation families, *independently* of which family was augmented.** Layout
is never augmented, yet B recovers +15 pp on it — the same order of magnitude
as in-distribution gains. Meanwhile, perturbation-family-matched augmentation
provides *no systematic advantage* over generic photometric augmentation (all
three Holm-adjusted family-level p-values exceed 0.05; the matched lighting
augmentation actually underperforms generic at our magnitude). The recovery is
real and broad, but the credit goes to LoRA's effect on the policy's task
representation — not to family-specific augmentation matching.

This is exactly the conclusion the diagnostic battery was designed to discriminate.

## 5. Honest negative findings (these are real conclusions, not bugs)

Treating these as failures of the experiment would be the wrong reading; the
diagnostic is doing its job.

- **viewpoint stays at 0% under all three conditions.** Our targeted
  augmentation for viewpoint is a 2-D image-space proxy (`RandomPerspective` +
  `RandomAffine`), not a true 3-D camera shift. The null result confirms the
  proxy hypothesis stated up-front in `data/augment/visual_aug.py`: image-space
  warps cannot confer 3-D viewpoint invariance. A genuine fix would require
  simulator re-rendering with shifted camera extrinsics.

- **noise gets *worse* under LoRA**, dropping from A: 24.4% to B: 3.3%
  (C: 10.0%) at L2. We conjecture that photometric augmentation shifts the
  visual prior toward "clean-looking" features, increasing fragility under
  additive sensor noise. C's targeted GaussianNoise(σ=0.08) recovers part of
  the gap but does not reach A. We report this as an open question, not a
  result we claim — but it is a falsifiable hypothesis a follow-up can test.

- **C's lighting underperforms B** (38.3% vs 48.3% at L2). Our targeted lighting
  augmentation uses ±0.4 brightness/contrast jitter at full magnitude, double
  the ±0.2 used by LeRobot's default. The heavier jitter appears to hurt rather
  than help — a natural augmentation-magnitude ablation is the obvious follow-up.

## 6. Limitations

Stated explicitly so readers can calibrate.

- **Single seed.** B/C are seed-0 only. LIBERO-Plus's deterministic init states
  give within-seed paired statistics but not seed-level variance. A 3-seed
  extension is the natural next step (~10 additional GPU-h, within the original
  ≤5 GPU-day budget envelope).
- **Single suite, single backbone.** LIBERO-spatial, SmolVLA. Cross-suite and
  cross-backbone replication is future work.
- **viewpoint augmentation is a 2-D proxy by design.** Documented up-front;
  itself part of the diagnostic read-out (a proxy that *should* fail does fail).
- **noise cell has reduced sample.** Per-cell evaluation for the noise family
  uses 6 tasks × 5 episodes (vs 12 × 5 elsewhere) and `episode_length=180`
  (vs default 280), driven by wall-clock budget — sensor-noise rollouts run to
  env-max without success. B and C use the *same* reduced configuration so the
  paired comparison remains valid; statistical power on noise is reduced.
- **No language-conditioning probe yet.** Probe 2 of the diagnostic battery
  (instruction sensitivity) is scaffolded but deferred; would be the highest-ROI
  next experiment.

## 7. What this work contributes

| Contribution | Where in the repo |
|---|---|
| End-to-end pipeline: data → LoRA → eval → paired statistics → figure | `train/`, `eval/runners/`, `eval/stats/`, `analysis/` |
| Pre-registered three-hypothesis design with Holm-Bonferroni correction | `eval/runners/phase5_stats.py` |
| Diagnostic finding: LoRA's task-rep transfer ≫ augmentation-family matching | `docs/PROBES.md`, `README.md` |
| Crash-safe resume-able evaluation harness across 5 families × 2 levels | `eval/runners/phase4_recovery.py` |
| Honest negative findings preserved as conclusions, not hidden | this section, `docs/PROBES.md`, `README.md` |

Full reproduction in [`docs/REPRODUCING.md`](../docs/REPRODUCING.md); raw
per-episode CSVs in `analysis/runs/` (1,790 trials); statistics regenerable
end-to-end via `python -m eval.runners.phase5_stats`.

## 8. About me

Statistics M.A. (US), undergraduate background in Information and Computational
Science. Interested in embodied AI, RL, multi-agent control, and the
intersection of statistical rigor with empirical ML. This is the second of two
portfolio projects; the first ([PolicyArena](https://github.com/IntheFesh/project1))
applies the same paired-statistics philosophy to LLM-agent policy compliance.

Contact: `<TODO: email>` · ORCID: `<TODO>`
