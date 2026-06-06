# Diagnostic-probe battery — the headline contribution

This is the canonical description of what VLA-Collapse-Recover *contributes*: a small battery of
**mechanistic diagnostics** of VLA visual-representation quality and language conditioning. It is a
**diagnostic, not a method** — it measures *what* a fine-tune changed about the representation, not a
new way to fine-tune. All results are **delta-only**, in-domain (a LIBERO subset), and every cell is
`TBD` until a real run fills it.

## 1. The scientific question

Open VLAs score high on clean LIBERO but **collapse** under visual perturbation. Two competing
explanations:

- **(a) Shortcut features.** The vision encoder relies on spurious, non-causal features that a moved
  camera / changed lighting / new texture destroys (shortcut learning).
- **(b) Language-blind lookup.** The policy largely ignores the instruction and behaves as a
  vision→action lookup that degrades whenever the visual input shifts.

Perturbation-targeted LoRA fine-tuning (Condition C) clearly raises success on the trained families.
**But which thing did it actually fix?** This project asks:

> **Does perturbation-targeted LoRA fix the underlying representation — so that robustness
> *generalizes to held-out perturbation families* and *language sensitivity increases* — or does it
> only paper over symptoms on the families it was trained on?**

The A/B/C intervention comparison and the paired statistics (`eval/stats/`, `docs/EVALUATION.md`) are
the *instrument* we read the probes with — not the contribution itself.

## 2. The diagnostic battery

### Probe 1 — Held-out cross-family generalization  *(first-class; implemented)*
Train Condition C with augmentation on families `F_train` (`trained_families`); evaluate on a
**disjoint** held-out family `F_held` (`held_out_families`, e.g. `layout`). Report the
**`generalization_gap` = Recovery_C(in-dist) − Recovery_C(held-out)** (`eval/metrics.py`). A *small*
gap is evidence of representation-level fixing; a *large* gap is evidence of family-specific symptom
patching. Always labelled in-dist vs held-out (`classify_distribution`).

### Probe 2 — Language-conditioning sensitivity  *(first-class; implemented)*
Run the **same task-ID set** under `correct`, `blank` (empty), and `shuffled`/`mismatched`
instructions; report **`language_sensitivity = SR_correct − SR_ablated`**, paired at task IDs
(`eval/probe.py`, stats via `eval/paired.py`). Near-zero language sensitivity ⇒ the policy ignores
language. A representation-level fix should *increase* language sensitivity; a symptom patch should
leave it near zero.

### Probe 3 — Visual-feature-shift  *(optional; scaffolded seam only)*
Cosine distance of the **vision encoder's** features, clean vs perturbed. A *decrease* after LoRA ⇒
the encoder has been pushed toward perturbation-invariant features. This needs model-internal hooks,
is fragile, and is intentionally only a clearly-marked optional seam
(`eval/probe.py::feature_shift_probe`, raises `NotImplementedError`). Do not over-invest.

## 3. What each probe answers

| Probe | Question it answers | Decision rule (`TBD` until real runs) |
|---|---|---|
| 1 — held-out generalization | Did the fix transfer beyond the trained families? | small `generalization_gap` ⇒ representation-level; large ⇒ symptom patch |
| 2 — language sensitivity | Does the policy use the instruction at all (and more after LoRA)? | `language_sensitivity ≈ 0` ⇒ language-blind; ↑ after LoRA ⇒ better conditioning |
| 3 — visual-feature shift *(optional)* | Did the encoder become perturbation-invariant? | clean-vs-perturbed cosine distance ↓ after LoRA ⇒ invariance |

Read together: robustness recovery **with** a small generalization gap **and** rising language
sensitivity is (weak) evidence of a more causal, world-model-like representation; recovery **without**
those is evidence of shortcut/symptom patching.

## 4. Limitations & honest non-claims

- **Diagnostic, not a method.** No new fine-tuning algorithm is claimed; Condition C borrows
  published augmentation ideas. The contribution is the *measurement*.
- **Cannot adjudicate every hypothesis.** The probes give converging evidence, not proof; they cannot
  rule out every shortcut/confound. Probe 3 in particular is encoder-architecture dependent.
- **Delta-only, in-domain.** A LIBERO subset on a single RTX 5090 in < 5 days; absolute numbers are
  not comparable to papers. In-dist recovery is expected by design — Probe 1's held-out gap is the
  generalization test.
- **Condition-C augmentation caveat.** For `viewpoint`, 2-D torchvision warps are a *weak proxy* for
  true 3-D camera-viewpoint perturbation (see the report's limitations + the optional
  `configs/lora/condition_c_realdata.yaml` stub).

### Conceptual frame (citations)

- **Shortcut learning** — Geirhos et al., *Shortcut Learning in Deep Neural Networks*, Nature Machine
  Intelligence 2020 ([arXiv:2004.07780](https://arxiv.org/abs/2004.07780)).
- **Causal representation learning** — Schölkopf et al., *Towards Causal Representation Learning*,
  Proc. IEEE 2021 ([arXiv:2102.11107](https://arxiv.org/abs/2102.11107)).
- **World models** — Ha & Schmidhuber, *World Models*, 2018
  ([arXiv:1803.10122](https://arxiv.org/abs/1803.10122)).

(VLA-specific perturbation/robustness prior art — LIBERO-Plus, LIBERO-PRO, RobustVLA, etc. — is in
the README "Prior art" section.)
