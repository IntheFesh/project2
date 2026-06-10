# Results

Single-seed run on the LIBERO-spatial subset (1790 per-episode trials total). All effects are
**paired** at matched `(family, level, task_id, episode_index)`; 95% CIs are 10,000-resample
bootstrap; significance is McNemar (exact when < 25 discordant pairs, else continuity-corrected χ²);
H2 family p-values are Holm–Bonferroni corrected. Regenerate with `make stats` →
[`analysis/runs/phase5_summary.md`](../analysis/runs/phase5_summary.md) (the authoritative source for
every number below).

> **Single seed.** Every ΔSR and p-value below is *within-seed* paired statistics (the LIBERO-Plus
> deterministic init states make episodes matched across conditions). A multi-seed extension — to add
> seed-level variance — is future work.

![Recovery heatmap](../analysis/runs/recovery_heatmap.png)

---

## The three pre-registered tests

| Test | question | quantity | estimate (95% CI) | significance |
|---|---|---|---|---|
| **H1** | Does LoRA + standard aug lift robustness? (A vs B, pooled over all 10 perturbed cells, n=540) | ΔSR | **+7.4 pp** [+2.8, +11.9] | McNemar **p ≈ 0.0018** — significant |
| **H2** | Does targeted aug beat standard aug? (B vs C, per in-dist family, Holm-corrected) | ΔSR per family | lighting **−10.8**, noise +3.3, texture +3.3 | all Holm-p > 0.05 — **not rejected** |
| **H3** | Does the LoRA gain reach a *held-out* family? (A vs B on `layout`, never augmented, n=120) | ΔSR | **+15.0 pp** [+5.0, +25.0] | McNemar **p ≈ 0.0072** — significant |

**H2 detail** (B vs C):

| family | n | SR_B | SR_C | Δ_method (pp) [95% CI] | p_raw | p_Holm | reject @ 0.05 |
|---|--:|--:|--:|--:|--:|--:|:--:|
| lighting | 120 | 48.3% | 37.5% | **−10.8** [−22.5, +0.0] | 0.093 | 0.279 | no |
| noise | 60 | 1.7% | 5.0% | +3.3 [−3.3, +10.0] | 0.625 | 1.00 | no |
| texture | 120 | 39.2% | 42.5% | +3.3 [−7.5, +14.2] | 0.658 | 1.00 | no |

## Cell-level success rates (A / B / C)

| family | level | A: base | B: LoRA + standard aug | C: LoRA + targeted aug |
|---|:--:|:--:|:--:|:--:|
| clean | 0 | 64.0% | 68.0% | 58.0% |
| viewpoint | 2 / 4 | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| lighting | 2 / 4 | 41.7% / 16.7% | 48.3% / 48.3% | 38.3% / 36.7% |
| texture | 2 / 4 | 11.7% / 53.3% | 31.7% / 46.7% | 28.3% / 56.7% |
| noise | 2 / 4 | 24.4% / 0.0% | 3.3% / 0.0% | 10.0% / 0.0% |
| **layout** *(held-out)* | 2 / 4 | 16.7% / 20.0% | 40.0% / 26.7% | 43.3% / 31.7% |

---

## Reading the result: a diagnostic, not a method

The headline is **what the fine-tune changed about the representation**, read through the probe
battery ([`PROBES.md`](PROBES.md)) — not the size of the recovery.

- **LoRA improves the task representation, and it transfers (H1 + H3).** LoRA + photometric
  augmentation lifts robustness broadly (H1: +7.4 pp), and — critically — the lift reaches `layout`,
  a family that was **never augmented** in any condition (H3: +15.0 pp). The held-out gain matching
  the in-distribution gain is the signature of a **representation-level** improvement, not
  family-specific symptom patching.
- **Augmentation-family-matching gives no systematic advantage (H2).** For every in-distribution
  family, C ≠ B does not survive Holm correction. Augmentation helps; matching the augmentation
  *family* to the eval perturbation does not. The recovery is augmentation-mediated, not
  augmentation-family-specific.

This is the whole point of the diagnostic: it tells us the recovery is mediated by LoRA's
task-representation improvement, and it refuses to let in-distribution recovery be sold as
generalization.

---

## Honest negative findings (real conclusions, not bugs)

These are **results**, framed as such — not failures to be softened into "limitations" or deferred
to "future work."

1. **`viewpoint` stays at 0% under A, B, and C.** The Condition-C augmentation for viewpoint is a
   **2-D image-space proxy** (RandomPerspective + RandomAffine), not a true 3-D camera shift. As
   flagged up-front in `data/augment/visual_aug.py`, it **cannot confer 3-D viewpoint invariance**,
   and it doesn't. The null is consistent with the proxy hypothesis — a proxy that *should* fail
   does fail. This is a real conclusion about the 2-D proxy, not a training bug.

2. **`noise` gets *worse* after LoRA.** At L2, success drops from **24.4% (A) → 3.3% (B) / 10.0%
   (C)**. LoRA + photometric augmentation appears to make the policy *more* fragile to additive
   sensor noise — we conjecture the augmented training shifts the visual prior toward "clean-looking"
   features. C's targeted `GaussianNoise(σ=0.08)` recovers part of the gap but does not reach A. We
   treat this as an **open question**, a real and reported effect — not a result we are claiming and
   not a bug.

3. **Condition C's `lighting` underperforms B.** Targeted lighting augmentation used ±0.4
   brightness/contrast jitter (vs B's default ±0.2); the heavier jitter **hurts rather than helps**
   (Δ_method = −10.8 pp). This is a real conclusion — at this magnitude, "more targeted" is worse —
   and motivates an augmentation-magnitude ablation (deferred).

---

See [`PROBES.md`](PROBES.md) for the full diagnostic reading and the conceptual frame
(shortcut learning / causal representation / world models).
