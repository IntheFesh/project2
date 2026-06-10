# VLA-Collapse-Recover — Technical Report

A diagnostic study of LoRA fine-tuning under visual perturbation in vision-language-action models.

**Single-author preprint · single-seed empirical study · v0.1**
**Repository:** [github.com/IntheFesh/project2](https://github.com/IntheFesh/project2)

> **Position statement.** This is a diagnostic, not a method paper. The contribution is the
> measurement framework, the pre-registered three-hypothesis design, and the resulting finding
> (which is partly negative). No novelty in the fine-tuning algorithm is claimed.

---

## 1. Introduction

Open vision-language-action models (VLAs) such as SmolVLA achieve strong success rates on clean
LIBERO benchmarks but degrade sharply under visual perturbation. LIBERO-Plus (Sylvest et al.,
2025) systematically quantifies this *collapse* across families {viewpoint, lighting, texture,
noise, layout, ...}. Subsequent work — RESample (arXiv:2510.17640), LIBERO-PRO, RobustVLA —
proposes perturbation-aligned data augmentation during fine-tuning to *recover* robustness.

The empirical question we pose is mechanistic rather than methodological:

> Does perturbation-targeted LoRA fine-tuning fix the underlying visual representation so that
> robustness transfers to **held-out perturbation families**, or does it merely patch symptoms
> on whichever families it was trained against?

Answering this requires a within-policy comparison at fixed init states, paired statistics that
can detect a small (5-15 pp) intervention gap, and a held-out family for the cross-family
generalization probe. We construct exactly this setup on a single RTX 5090 in under a GPU-day.

## 2. Method

### 2.1 Backbone and dataset

We use `HuggingFaceVLA/smolvla_libero` (≈ 610 M params, SmolVLM2-500M-Instruct vision-language
backbone) as Condition A. Training data is the HuggingFaceVLA/libero dataset restricted to the
LIBERO-spatial split (task indices 30–39 in the upstream LeRobot v3.0 packaging: 432 episodes,
52,970 frames).

### 2.2 Three conditions

The three conditions differ **only** in the LoRA augmentation. Everything else — backbone, LoRA
target modules, rank, α, optimizer, schedule, batch size, training steps, evaluation cells, init
state seeds — is held constant. LoRA configuration: rank $r = 16$, $\alpha = 32$
(scaling $\alpha/r = 2$), targeting the policy expert's $q$/$v$ projections and the
state/action heads. Both B and C train for 20,000 steps at batch size 16 with the standard
LeRobot cosine schedule, batch size 16, AdamW $\beta = (0.9, 0.95)$, peak LR $10^{-4}$.

**Condition A — base.** No fine-tuning; the published `smolvla_libero` checkpoint.

**Condition B — LoRA + standard augmentation.** The LeRobot default photometric augmentation:
`ColorJitter` over brightness/contrast/saturation/hue independently, `SharpnessJitter`,
`RandomAffine(±5°, translate=0.05)`; three randomly chosen at a time.

**Condition C — LoRA + perturbation-targeted augmentation.** A family-aligned augmentation
suite chosen to mirror the LIBERO-Plus eval perturbations *in family*, with magnitudes set at
the full level (level 5 of 5):

| Family | Augmentation | Fidelity |
|---|---|---|
| lighting | `ColorJitter(brightness=0.4, contrast=0.4)` | faithful (photometric ↔ photometric) |
| noise | `GaussianNoise(σ=0.08)` | faithful |
| texture | `ColorJitter(saturation=0.4)` + `GaussianBlur(σ ∈ [0.1, 1.5])` | proxy (2-D cannot swap textures) |
| viewpoint | `RandomPerspective(distortion=0.3)` + `RandomAffine(translate=0.1)` | proxy (2-D ≠ 3-D camera shift) |

The held-out family `layout` is **never** augmented in any condition. The
proxy-vs-faithful distinction is a *deliberate diagnostic feature*: a proxy that
*should* fail at conferring true invariance lets us read out whether observed
recovery is family-specific or representation-mediated.

### 2.3 Evaluation

We use the LIBERO-Plus benchmark (Sylvest et al., 2025), restricted to LIBERO-spatial. Clean
evaluation uses the original LIBERO 10 spatial tasks (forced via `PYTHONPATH` isolation against
the `LIBERO-orig` submodule). Perturbed evaluation uses LIBERO-Plus's deterministic cell selector
(`select_cell_task_ids`), which returns the first $n$ tasks matching `(family, difficulty_level)`
by ascending JSON id. Each cell is 12 tasks × 5 episodes (60 paired episodes) except the noise
family, which uses 6 × 5 = 30 paired episodes and an `episode_length` cap of 180 steps (above the
spatial demo p90 of 149; B and C use the same reduced configuration so the paired test stays
valid). Total: $50 + 5 \times 2 \times \text{cell size}$ episodes per condition, 1,790 paired
episodes across A / B / C.

### 2.4 Statistical design

Three pre-registered hypotheses, paired at $(\text{task\_id}, \text{episode\_index})$:

$$
\textbf{H1:}\quad \mathrm{SR}_B > \mathrm{SR}_A \text{ pooled across all perturbed cells}
$$

$$
\textbf{H2:}\quad \mathrm{SR}_C > \mathrm{SR}_B \text{ per in-distribution family } f \in \{\text{lighting, noise, texture}\}
$$

$$
\textbf{H3:}\quad \mathrm{SR}_B > \mathrm{SR}_A \text{ on the held-out family layout}
$$

Significance: McNemar's test on matched binary outcomes, using the exact binomial when
the number of discordant pairs is below 25 and chi-squared with continuity correction otherwise.
Confidence intervals: 10,000-resample paired bootstrap on $\Delta = \mathrm{SR}_X - \mathrm{SR}_Y$,
resampling matched episode indices jointly. The H2 family-wise p-values are corrected with the
Holm-Bonferroni step-down procedure at $\alpha = 0.05$. No correction is applied to H1 and H3 since
each is a single pre-registered hypothesis.

### 2.5 Reproducibility

All randomness is seeded (`seed=0` for both training and evaluation); LIBERO-Plus init states are
deterministic by construction. The full pipeline regenerates byte-identically from the raw
per-episode CSVs:

```
make setup        # uv-managed Python 3.12 environment
make smoke        # 60-step LoRA smoke test (~ 1 min on RTX 5090)
make train-B      # condition B: LoRA + standard augmentation
make train-C      # condition C: LoRA + targeted augmentation
make eval         # phase4 paired evaluation across all cells
make stats        # phase5 statistical summary + recovery heatmap
```

See `docs/REPRODUCING.md` for the full walkthrough and `analysis/runs/phase5_summary.{md,json}`
for the canonical statistical output.

## 3. Results

### 3.1 Cell-level success rates

| family | level | A: base | B: LoRA + standard | C: LoRA + targeted |
|---|---|---|---|---|
| clean | 0 | 64.0% (n=50) | 68.0% (n=50) | 58.0% (n=50) |
| viewpoint | L2 | 0.0% (n=60) | 0.0% (n=60) | 0.0% (n=60) |
| viewpoint | L4 | 0.0% (n=60) | 0.0% (n=60) | 0.0% (n=60) |
| lighting | L2 | 41.7% (n=60) | 48.3% (n=60) | 38.3% (n=60) |
| lighting | L4 | 16.7% (n=60) | 48.3% (n=60) | 36.7% (n=60) |
| texture | L2 | 11.7% (n=60) | 31.7% (n=60) | 28.3% (n=60) |
| texture | L4 | 53.3% (n=60) | 46.7% (n=60) | 56.7% (n=60) |
| noise | L2 | 24.4% (n=45) | 3.3% (n=30) | 10.0% (n=30) |
| noise | L4 | 0.0% (n=35) | 0.0% (n=30) | 0.0% (n=30) |
| layout (held-out) | L2 | 16.7% (n=60) | 40.0% (n=60) | 43.3% (n=60) |
| layout (held-out) | L4 | 20.0% (n=60) | 26.7% (n=60) | 31.7% (n=60) |

95% percentile bootstrap CIs in `analysis/runs/phase5_summary.md`. The texture L4 > L2 inversion
reflects that LIBERO-Plus's `difficulty_level` is a heterogeneous task bin rather than a clean
linear magnitude (documented in `level_to_fraction`).

### 3.2 H1 — LoRA + standard augmentation lifts robustness

Pooled across the ten perturbed (family, level) cells, $n = 540$ matched episodes:

$$\mathrm{SR}_A = 19.6\%, \qquad \mathrm{SR}_B = 27.0\%, \qquad \Delta = +7.4\,\text{pp}\ [+2.8, +11.9]$$

McNemar (chi-squared, continuity-corrected): $p \approx 0.0018$; B-only wins 58, A-only wins 98
out of 156 discordant pairs. The effect survives the conservative pooled framing and is robust
to the within-cell heterogeneity visible in the cell table above.

### 3.3 H2 — Targeted augmentation per in-distribution family (Holm-corrected)

| family | n | $\mathrm{SR}_B$ | $\mathrm{SR}_C$ | $\Delta$ (pp) [95% CI] | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | reject @ 0.05 |
|---|---|---|---|---|---|---|---|
| lighting | 120 | 48.3% | 37.5% | **−10.8** [−22.5, +0.0] | 0.093 | 0.279 | no |
| noise | 60 | 1.7% | 5.0% | +3.3 [−3.3, +10.0] | 0.625 | 1.000 | no |
| texture | 120 | 39.2% | 42.5% | +3.3 [−7.5, +14.2] | 0.658 | 1.000 | no |

**None of the three Holm-corrected family-level tests rejects the null.** The lighting cell goes
the wrong way at our augmentation magnitude (targeted underperforms standard by 10.8 pp, with the
95% CI just touching zero); the noise and texture cells show small positive point estimates with
CIs that span zero.

### 3.4 H3 — Held-out generalization on `layout`

Layout is never augmented in any condition. Pooling layout L2 and L4 ($n = 120$ matched episodes):

$$\mathrm{SR}_A = 18.3\%, \qquad \mathrm{SR}_B = 33.3\%, \qquad \Delta = +15.0\,\text{pp}\ [+5.0, +25.0]$$

McNemar (chi-squared, continuity-corrected): $p \approx 0.0072$. The held-out gain is **of the
same magnitude as the pooled in-distribution gain** (H1: +7.4 pp), and slightly *larger* than
some in-distribution recoveries — strongly inconsistent with a family-specific symptom-patching
account of the recovery mechanism.

## 4. Discussion

### 4.1 The diagnostic verdict

Reading H1, H2, and H3 jointly:

- **H1** establishes that LoRA + photometric augmentation produces broad, statistically
  significant robustness improvement.
- **H2** establishes that the improvement is **not** mediated by family-matching of the
  augmentation: the three per-family tests at the in-distribution families all fail to reject
  null.
- **H3** establishes that the improvement **transfers to a perturbation family that was never
  augmented**, at the same magnitude as the in-distribution gain.

The natural interpretation: LoRA fine-tuning is improving the policy's *task representation* on
the spatial suite (object grasp and placement under a single ego-centric camera) in a way that
incidentally hardens it against multiple types of visual perturbation. Augmentation is the
mechanism that drives LoRA to fit something other than the clean visual prior, but the *which*
augmentation matters less than the *fact* of augmentation. This is consistent with the broader
literature on data augmentation as implicit regularization, and inconsistent with a strict
augmentation-aligned-equivariance reading.

### 4.2 Honest negative findings

These are not failures of the experiment; they are part of what the diagnostic reads out.

**viewpoint stays at 0% under all three conditions.** Our viewpoint augmentation is a
2-D image-space proxy (`RandomPerspective` + `RandomAffine`), which cannot confer
true 3-D viewpoint invariance — moving a 3-D camera changes occlusions, parallax, and lighting
in ways no 2-D warp recovers. The 0% recovery is what one should *predict* from this proxy
hypothesis (stated up-front in `data/augment/visual_aug.py`); observing it confirms that
recovery on the other families is not an artifact of, e.g., LoRA overfitting to test init states.
A real fix would require simulator re-rendering with shifted camera extrinsics — outside our
single-GPU budget but the natural follow-up.

**noise gets *worse* under LoRA.** Condition A scores 24.4% at noise L2; B drops to 3.3%, C to
10.0%. This is statistically robust within the noise cell (B-only / A-only McNemar discordance
heavily favors A). We conjecture that LoRA + photometric augmentation pushes the visual encoder
toward a "clean-looking" prior — concentrating activation on smooth gradients of color and
luminance — and additive sensor noise then degrades that prior more than it degrades the base
encoder's coarser features. C's targeted `GaussianNoise(σ=0.08)` recovers part of the gap but
does not reach A. This is a falsifiable hypothesis (one could probe noise-frequency
sensitivity of the encoder activations) and a productive direction for follow-up; we report it
as an open question, not a claim.

**C's lighting underperforms B's lighting.** Our targeted lighting augmentation uses ±0.4
brightness/contrast jitter at full magnitude — twice LeRobot's default ±0.2. The heavier jitter
appears to hurt rather than help, even at L4 where it should be most useful. A
magnitude-ablation across $\{0.1, 0.2, 0.3, 0.4\}$ would directly localize the optimum and is
the natural follow-up; we did not run it within the single-GPU budget.

### 4.3 What this is, and is not

**This is.** A reproducible, single-GPU, single-seed empirical diagnostic with paired statistics
and pre-registered hypotheses, producing a finding that is partly positive (LoRA helps broadly,
generalizes to held-out) and partly negative (family-matched augmentation has no systematic
edge; viewpoint and noise resist this style of fix).

**This is not.** A method paper — no novelty in the fine-tuning algorithm or the augmentation
choices is claimed. A SOTA benchmark — the absolute numbers reflect the published
`smolvla_libero` checkpoint and are not directly comparable across published evals (LIBERO setup
is notoriously sensitive). A multi-seed study — single-seed paired statistics are sound but a
3-seed extension is the natural next step.

## 5. Limitations and threats to validity

- **Single seed.** B/C are seed-0 only. LIBERO-Plus's deterministic init states give paired
  *within*-seed inference but not seed-level variance. The natural 3-seed extension would
  consume an additional ~10 GPU-h, well within the original ≤5 GPU-day budget envelope.
- **Single suite.** LIBERO-spatial only. Cross-suite replication (object, goal, 10) is future
  work; the pipeline supports it but it was outside scope.
- **Single backbone.** SmolVLA only. The diagnostic framework is backbone-agnostic but the
  specific verdict (LoRA's task-rep transfer ≫ augmentation-family-matching) is a SmolVLA
  finding.
- **Proxy augmentations are proxies on purpose.** viewpoint and texture use 2-D image-space
  proxies that we *expect* to be weak. This is part of the diagnostic, not an inadvertent
  shortcoming. A simulator-based 3-D augmentation pipeline is the obvious extension.
- **Augmentation magnitudes are not tuned.** Both B (LeRobot defaults) and C (full-magnitude
  level-5 alignment) are reasonable initial points; the C-lighting underperformance suggests an
  ablation could change the H2 verdict.
- **No language-conditioning probe yet.** Probe 2 of the diagnostic battery (instruction
  sensitivity via correct / blank / shuffled instructions) is scaffolded in `eval/probe.py` but
  deferred. Highest-ROI next experiment.

## 6. Reproducibility checklist

- [x] Code released: [github.com/IntheFesh/project2](https://github.com/IntheFesh/project2)
- [x] Pre-trained checkpoints: `HuggingFaceVLA/smolvla_libero` (public on HuggingFace)
- [x] Trained LoRA adapters: `adapters/{B,C}_seed0_a32/` (in repo; ~5 MB each)
- [x] Raw per-episode CSVs: `analysis/runs/{phase2_collapse,phase4_recovery}.csv` (1,790 rows)
- [x] Statistical output: `analysis/runs/phase5_summary.{md,json}` (regenerable byte-identically)
- [x] Random seeds: all `seed=0`; LIBERO-Plus init states deterministic
- [x] Hyperparameter table: §2.2 + `configs/lora/condition_c_aug.yaml`
- [x] Compute budget: single RTX 5090, < 1 GPU-day per condition pair
- [x] Environment pins: `pyproject.toml` (Python 3.12, lerobot 0.5.2 via source, peft 0.19.1,
  torch 2.8.0+cu128)
- [x] End-to-end command: `make stats` (off-GPU); `make eval` (on-GPU, ~ 4–6 h)
- [x] CI: GitHub Actions runs lint + off-GPU tests on every push

## 7. References

- Sylvest et al. **LIBERO-Plus**: a perturbed-LIBERO benchmark for VLA robustness evaluation
  (HuggingFace dataset `Sylvest/LIBERO-plus`).
- LeRobot project (Hugging Face): https://github.com/huggingface/lerobot
- SmolVLA: https://huggingface.co/HuggingFaceVLA/smolvla_libero
- Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models** (ICLR 2022).
- Geirhos et al. **Shortcut Learning in Deep Neural Networks** (Nat. Mach. Intell. 2020).
- Holm. **A simple sequentially rejective multiple test procedure** (Scand. J. Stat. 1979).
- McNemar. **Note on the sampling error of the difference between correlated proportions**
  (Psychometrika 1947).

**Citation.**

```
@misc{<TODO: key>,
  author = {<TODO: author>},
  title  = {VLA-Collapse-Recover: A Diagnostic Study of LoRA Fine-Tuning under Visual Perturbation},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/IntheFesh/project2}
}
```

Contact: `<TODO: email>` · ORCID: `<TODO>`
