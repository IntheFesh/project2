# VLA-Collapse-Recover · One-Pager

**Diagnostic eval framework for vision-language-action models, single GPU, < 1 GPU-day.**
[github.com/IntheFesh/project2](https://github.com/IntheFesh/project2)

---

**Question.** Open VLAs collapse under visual perturbation. Does LoRA fine-tuning
fix the *representation* (transfers to held-out families) or just *patch symptoms*
on whichever families it was trained on?

**Design.** Three conditions on SmolVLA-LIBERO — A: base, B: LoRA + standard
augmentation, C: LoRA + perturbation-targeted augmentation — evaluated on the same
deterministic init states across viewpoint / lighting / texture / noise at L2 / L4,
plus a held-out `layout` family no condition ever sees during augmentation. 1,790
per-episode trials, paired at `(task_id, episode_index)` for McNemar and paired
bootstrap; Holm-Bonferroni where appropriate.

**Headline finding.** LoRA improves task representation in a way that **transfers
across perturbation families, independently of which family was augmented** — yet
perturbation-family-matched augmentation provides no systematic advantage over
generic augmentation.

| Hypothesis | Result | Significance |
|---|---|---|
| H1: LoRA + std aug lifts robustness (A vs B, pooled) | ΔSR = **+7.4 pp** [+2.8, +11.9] | McNemar p ≈ **0.0018** |
| H2: Targeted aug beats standard (B vs C, per family, Holm) | lighting **−10.8**, noise +3.3, texture +3.3 pp | all Holm-p > 0.05 |
| H3: Generalization to held-out `layout` (A vs B) | ΔSR = **+15.0 pp** [+5.0, +25.0] | McNemar p ≈ **0.0072** |

**Honest negative findings** *(treated as conclusions, not hidden)*: viewpoint
stays at 0% under all three conditions (2-D augmentation cannot confer 3-D
viewpoint invariance, as flagged up-front); noise actually *worsens* under LoRA
(24.4% → 3.3% / 10.0%), suggesting photometric augmentation shifts the visual
prior toward clean-looking features; C's lighting underperforms B's at our
augmentation magnitude.

**Stack.** SmolVLA · LeRobot 0.5.2 · PEFT-LoRA · paired bootstrap + McNemar +
Holm-Bonferroni · crash-safe resume-able eval harness. Single RTX 5090.

**Reproducibility.** `make stats` regenerates the full statistical summary
byte-identically from raw per-episode CSVs (`analysis/runs/*.csv`).
[REPRODUCING.md](https://github.com/IntheFesh/project2/blob/main/docs/REPRODUCING.md)
walks the pipeline end-to-end.

**Position.** Diagnostic, not a method paper. No novelty claimed. Single-seed.
Companion to [PolicyArena](https://github.com/IntheFesh/project1) — same
paired-statistics philosophy on LLM-agent policy compliance.

*`<TODO: name>` · M.A. Statistics · `<TODO: email>`*
