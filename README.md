# VLA-Collapse-Recover

**A reproducible *mechanistic probe* of VLA visual-representation quality under perturbation.** Open
VLAs ace clean LIBERO but collapse under a moved camera or changed lighting. **Does
perturbation-targeted LoRA fix the underlying representation (cross-family generalization, language
conditioning), or only paper over symptoms on the trained families?** Single RTX 5090, < 5 days,
delta-only.

**Two contributions:**

1. **A reproducible measurement** of VLA visual-perturbation *collapse* and the *recovery* from
   perturbation-targeted LoRA fine-tuning — on a single GPU, under 5 days.
2. **A diagnostic-probe battery** — held-out cross-family generalization + language-conditioning
   sensitivity + (optional) visual-feature-shift — that distinguishes **representation-level fixing
   from symptom patching**. Canonical description: [`docs/PROBES.md`](docs/PROBES.md).

The paired-statistics machinery (bootstrap CIs, McNemar, Holm–Bonferroni) is **supporting
infrastructure** — shared with the author's prior project, PolicyArena (see
[Honesty guards](#-honesty-guards-read-first)) — **not** a headline claim of this repo.

> **Headline = the probe battery**, read *via* the A/B/C intervention comparison + paired statistics
> (not the recovery magnitude — same-family recovery is expected by design). A research-engineering
> portfolio project, **not a paper**; no novelty claimed — see [prior art](#prior-art-cited-no-novelty-claimed).

```bash
git clone <this-repo> && cd vla-collapse-recover
./run.sh            # one-click: set up the env, run tests + lint, check GPU readiness
```

---

## Contents

- [Honesty guards](#-honesty-guards-read-first)
- [Results tables (all `TBD`)](#headline-result--intervention-comparison-phase-45----tbd)
- [📚 Tutorial](#-tutorial) — the detailed, step-by-step guide
  - [1. Prerequisites](#1-prerequisites)
  - [2. The two-stage environment](#2-the-two-stage-environment-why--how)
  - [3. Repository map](#3-repository-map-what-each-piece-does)
  - [4. Experimental design (concepts)](#4-experimental-design-the-concepts-you-need)
  - [5. What you can run TODAY (off-GPU)](#5-what-you-can-run-today-off-gpu)
  - [6. Renting the GPU & deploying](#6-renting-the-gpu--deploying)
  - [7. Phase-by-phase execution guide](#7-phase-by-phase-execution-guide)
  - [8. Statistics deep-dive](#8-statistics-deep-dive)
- [Metrics](#metrics) · [Environment pins](#environment-pin-these) · [Budget](#budget-hard-limits)
- [Build status](#build-status) · [Layout](#repository-layout) · [Stack](#stack) · [Prior art](#prior-art-cited-no-novelty-claimed)

---

## ⚠️ Honesty guards (read first)

1. **No fabricated results.** Every number in the tables below is `TBD` until a real run fills it.
2. **Only relative / delta claims in headlines** (Δ_robust, Recovery, Δ_method). We do **not** claim
   to reproduce any paper's absolute SOTA. The base SmolVLA-LIBERO **absolute** success rate here
   **may not match the published number** (LIBERO eval setup is sensitive) — which is *exactly why
   only deltas are reported*.
3. **In-distribution vs held-out is always labeled.** A perturbation family seen during augmentation
   is *in-dist*; one never seen is *held-out generalization*. In-dist recovery is **never** presented
   as generalization. (This is enforced in code by `classify_distribution`.)
4. **Headline = the diagnostic-probe battery** (representation quality — see
   [`docs/PROBES.md`](docs/PROBES.md)), read *via* the A/B/C comparison + paired statistics — not the
   recovery magnitude, and not the statistics harness itself.

> **Methodology & honesty (PolicyArena cross-reference).** Statistical infrastructure (paired
> bootstrap, McNemar, Holm–Bonferroni) is shared with the author's prior project, **PolicyArena** — a
> statistically-validated tool-calling + RAG agent for a Chinese enterprise service desk. This
> repository's distinct contribution is the **visual representation quality diagnostic** (held-out
> cross-family generalization + language-conditioning probe), not the statistical methodology.

---

## Headline result — Intervention comparison *(Phase 4–5)*  ·  `TBD`

Perturbed success rate is averaged over the **in-distribution** augmented families
(viewpoint / lighting / texture / noise). Paired at fixed **task IDs** (matched `(task_id, level, seed)`);
3 LoRA-training seeds on the B/C(/D) comparison.

| Condition | Training augmentation | Clean SR | Perturbed SR (in-dist avg) | Recovery | Δ_method vs **B** | 95% CI (Δ_method) | Holm *p* |
|-----------|-----------------------|:--------:|:--------------------------:|:--------:|:-----------------:|:-----------------:|:--------:|
| **A** — base (no FT) | — (pretrained `smolvla_libero`) | `TBD` | `TBD` | — | — | — | — |
| **B** — LoRA + standard aug | generic torchvision aug | `TBD` | `TBD` | `TBD` | *(reference)* | — | — |
| **C** — LoRA + perturbation-targeted aug | aug aligned to eval families | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **D** — feature-modulation adapter *(STRETCH)* | FTM/FLA-style | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

*Δ_method (C − B), paired at fixed task IDs, is the lead statistic — the 5–10pp gap a paired test makes detectable.*

### Headline diagnostic probes  ·  `TBD`  — see [`docs/PROBES.md`](docs/PROBES.md)

These two rows — **not** the recovery magnitude — are the contribution: they say *whether* LoRA fixed
the representation or only patched symptoms.

| Probe | Quantity | Value |
|-------|----------|:-----:|
| **Held-out cross-family generalization** | `generalization_gap` = Recovery_C(in-dist) − Recovery_C(held-out) | `TBD` |
| **Language-conditioning sensitivity** | `SR_correct − SR_ablated` (paired at task IDs) | `TBD` |

*Small gen-gap **and** rising language sensitivity ⇒ representation-level fix; otherwise symptom patching.*

## Collapse curve — base model under perturbation *(Phase 2, condition A, 1 seed)*  ·  `TBD`

| Family (LIBERO-Plus) | Clean | L2 | L4 | Δ_robust @ L4 (collapse) |
|----------------------|:-----:|:--:|:--:|:------------------------:|
| viewpoint  | `TBD` | `TBD` | `TBD` | `TBD` |
| lighting   | `TBD` | `TBD` | `TBD` | `TBD` |
| texture    | `TBD` | `TBD` | `TBD` | `TBD` |
| noise      | `TBD` | `TBD` | `TBD` | `TBD` |

## Recovery by family *(Phase 4)*  ·  `TBD`

| Family | in-dist / held-out | A (base) SR | B SR | C SR | Recovery (C) |
|--------|:------------------:|:-----------:|:----:|:----:|:------------:|
| viewpoint | in-dist  | `TBD` | `TBD` | `TBD` | `TBD` |
| lighting  | in-dist  | `TBD` | `TBD` | `TBD` | `TBD` |
| texture   | in-dist  | `TBD` | `TBD` | `TBD` | `TBD` |
| noise     | in-dist  | `TBD` | `TBD` | `TBD` | `TBD` |
| **layout (held-out)** | **held-out** | `TBD` | `TBD` | `TBD` | `TBD` |

**Generalization gap** (Recovery_C on in-dist − Recovery_C on held-out): `TBD`. A **first-class
result**: it tests whether Condition C genuinely generalizes or merely overfits its augmentation
family (the "did you just train on the test perturbation?" red flag). Families are tagged in-dist
vs held-out by `classify_distribution`; see `generalization_gap` in [`eval/metrics.py`](eval/metrics.py).

## Language-conditioning probe — *why* collapse happens *(mechanism, Phase 6)*  ·  `TBD`

A cheap, high-signal probe of the collapse *mechanism*: run the **same task IDs** under
correct / blank / shuffled / mismatched instructions and measure **paired** ΔSR
(`language_sensitivity` in [`eval/probe.py`](eval/probe.py), matched per task ID). ΔSR ≈ 0 ⇒ the
policy effectively ignores language (a vision-action model), echoing the LIBERO-Plus / LIBERO-PRO
finding that VLAs largely ignore instructions.

| Instruction | SR | paired ΔSR vs correct | 95% CI | McNemar *p* |
|---|:--:|:--:|:--:|:--:|
| correct | `TBD` | — | — | — |
| blank | `TBD` | `TBD` | `TBD` | `TBD` |
| shuffled | `TBD` | `TBD` | `TBD` | `TBD` |
| mismatched | `TBD` | `TBD` | `TBD` | `TBD` |

---

# 📚 Tutorial

A guided walk-through, from a clean clone to the rented-GPU pipeline. It is honest about what
runs **today** (the off-GPU core: statistics, budget gating, tests) versus what is **scaffolded
with clearly-marked seams** that you implement on the rental (the rollouts, training, demos).

> **Legend used throughout:** ✅ implemented & tested · 🟡 scaffolded (CLI + a `NotImplementedError`
> seam tagged with its phase) · ⬜ pending data.

## 1. Prerequisites

- **[uv](https://docs.astral.sh/uv/)** (Python package/env manager). Install once:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Python 3.10–3.12** (uv will fetch one if needed). Linux only (the cu128 stack is Linux-only).
- **No GPU needed** for the off-GPU core. The full pipeline needs a **single RTX 5090** (Blackwell,
  `sm_120`) on a rental — see [§6](#6-renting-the-gpu--deploying).

## 2. The two-stage environment (why & how)

Rollouts are the time bottleneck and the GPU rental is capped at **< 5 days**, so we split the
environment so that *most engineering happens off-GPU for free*:

| Stage | Command | Contents | Where |
|------|---------|----------|-------|
| **Core** | `uv sync` | numpy / scipy / pandas / matplotlib / pyyaml / tqdm + pytest / ruff | anywhere (off-GPU) |
| **GPU** | `uv pip install -r requirements-gpu.txt` then `uv sync --extra gpu` | cu128 `torch≥2.7` + torchvision, transformers, peft, … | rented 5090 only |
| **From source** | `lerobot[smolvla]`, `LIBERO-Plus` | SmolVLA/π0.5 stack + drop-in perturbations | rented 5090 only |

The one-click script wraps both stages:

```bash
./run.sh            # Core only: uv sync + tests + lint + GPU-readiness check (no big downloads)
./run.sh deploy     # On the rental: Core + GPU stack + LeRobot + LIBERO-Plus + verify_env
./run.sh help       # all commands
```

> The default `./run.sh` **never** triggers a multi-GB download — the GPU stack is installed only by
> the explicit `gpu` / `deploy` commands.

## 3. Repository map (what each piece does)

```
vla-collapse-recover/
├── run.sh                      ✅ one-click setup / test / verify / deploy (uv, general)
├── deploy.py                   ✅ integrated one-click deploy (AutoDL-aware, single file)
├── pyproject.toml  uv.lock     ✅ two-stage deps (core + `gpu` extra); reproducible lock
├── requirements-gpu.txt        ✅ pinned cu128 torch/torchvision (rental)
├── configs/                    ✅ config-driven (YAML): model/ lora/ perturb/ eval/
├── perturb/
│   ├── libero_plus_constants.py ✅ single source of truth: family↔category, levels, JSON path
│   └── libero_plus_wrapper.py  ✅ task-selection (parse/group/select) + in-dist/held-out tagger
│                               🟡 make_perturbed_env(task_id) (Phase 2 env seam)
├── data/
│   ├── prepare_libero_subset.py ✅ parse task_classification.json + select  🟡 locate-JSON seam
│   └── augment/visual_aug.py    ✅ aligned_magnitude  🟡 torchvision transforms (Phase 3)
├── train/
│   ├── train_lora.py           🟡 LoRA fine-tune (Phase 3 seam)
│   └── feature_mod.py          🟡 feature-mod adapter (Phase 8 / STRETCH seam)
├── eval/
│   ├── metrics.py              ✅ SR, Δ_robust, Recovery, Δ_method, generalization_gap
│   ├── bootstrap.py paired.py holm.py  ✅ stats primitives
│   ├── stats/                  ✅ public paired-stats facade + report builder (the harness)
│   ├── budget.py               ✅ GPU-day budget estimator (task-instance units)
│   ├── probe.py                ✅ language-conditioning probe (rollout = seam)
│   ├── run_rollout.py          ✅ plan/row/CSV pure · 🟡 policy+sim seam (1 trial/task)
│   └── record_demo.py          🟡 demo recorder (Phase 6 seam)
├── scripts/
│   ├── verify_env.py           ✅ asserts sm_120 / cu128 / torch≥2.7 / LeRobot
│   ├── estimate_budget.py · smoke_timing.py  ✅ budget gate + smoke timing
│   └── analyze_results.py      ✅ rollout CSV → full paired-stats report (text + JSON)
├── docs/                       LIBERO_PLUS_NOTES.md · EVALUATION.md
├── analysis/                   eval summaries (CSV/JSON) + demos/ (a few clips)
├── report/                     technical_report.md · one_pager.md · results_card.md
└── tests/                      ✅ off-GPU tests + fixtures/ (run with `./run.sh test`)
```

## 4. Experimental design (the concepts you need)

**The four conditions** (what differs is *only the training intervention*):

| | Condition | What it is |
|---|-----------|-----------|
| **A** | base | the pretrained `smolvla_libero`, no fine-tuning — used to reproduce the *collapse* |
| **B** | LoRA + standard aug | LoRA fine-tune with *generic* torchvision augmentation |
| **C** | LoRA + targeted aug | LoRA fine-tune with augmentation whose **family & magnitude mirror the eval perturbations** |
| **D** | feature-mod (STRETCH) | a small FTM/FLA-style feature-modulation adapter instead of LoRA |

> **Caveat (Condition C).** torchvision augmentation is a *weak 2-D proxy* for LIBERO-Plus's true
> **3-D `viewpoint`** perturbation — a single frame cannot reproduce a moved camera. It remains the
> default; the honest alternative (fine-tuning C on LIBERO-Plus's released perturbation training
> data) is documented in the report's limitations and stubbed in `configs/lora/`.

**In-distribution vs held-out.** Training augmentation and evaluation perturbation share a *family*
(e.g. both use "lighting") but live on *separate data splits*. A family used in augmentation is
**in-dist**; a family the model never trained on is **held-out** (true generalization). The repo tags
every result via `classify_distribution(spec, trained_families) → {"clean","in_dist","held_out"}`, so
the README/report can never accidentally sell in-dist recovery as generalization.

**Why paired statistics.** All conditions are rolled out from **identical initial states** (`seed`),
so each episode is *matched* across conditions. The B-vs-C gap is expected to be small (~5–10pp); a
**paired** test (McNemar / paired bootstrap) detects it where an unpaired test would not.

See [Metrics](#metrics) for the exact formulas.

## 5. What you can run TODAY (off-GPU)

Everything here needs only `uv sync` (no GPU, no big downloads).

**(a) Tests + lint** — confirms the scaffold is healthy:
```bash
./run.sh test          # ruff check . && pytest -q   (currently: all green)
```

**(b) The statistics API** — fully implemented; this is the project's differentiator. Example:
```python
from eval.metrics import success_rate, delta_method
from eval.paired import mcnemar, paired_bootstrap_delta
from eval.holm  import holm_bonferroni
from perturb.libero_plus_wrapper import PerturbSpec, classify_distribution

# per-episode 0/1 outcomes at FIXED init states -> episodes are matched across conditions
base = [0, 0, 1, 0, 0, 0, 1, 0]   # condition A under a perturbation
C    = [1, 1, 1, 0, 1, 0, 1, 1]   # condition C, SAME init states / order

print(delta_method(success_rate(C), success_rate(base)))  # point gap  C - A
print(paired_bootstrap_delta(C, base))                    # (delta, lo, hi)  paired 95% CI
print(mcnemar(C, base)["pvalue"])                         # paired significance

# correct family-level p-values for multiplicity:
for name, p_raw, p_adj, reject in holm_bonferroni(
        {"viewpoint": 0.001, "lighting": 0.03, "texture": 0.2, "noise": 0.04}):
    print(name, round(p_adj, 4), "REJECT" if reject else "keep")

# honesty tag (was this eval family trained on?):
print(classify_distribution(PerturbSpec("viewpoint", 4), ["viewpoint", "lighting"]))  # -> in_dist
```

**(c) Budget gating** — confirm the planned matrix fits the **< 5-day** cap *before* renting:
```bash
./run.sh budget                          # uses SEC_PER_EP=30, TRAIN_H=4 by default
SEC_PER_EP=45 TRAIN_H=6 ./run.sh budget  # plug in your own (smoke-tested) timings
# what-if directly:
uv run python -m scripts.estimate_budget --sec-per-episode 30 --train-hours-per-run 4
```
If the matrix exceeds the cap, the tool **exits non-zero** and tells you the largest
`episodes/task` that would fit (or to cut tasks/levels/seeds).

## 6. Renting the GPU & deploying

On the rented **RTX 5090** (or any `sm_120` box):

```bash
./run.sh deploy        # Core + cu128 torch + GPU stack + LeRobot + LIBERO-Plus + verify_env
# (equivalently, step by step:)
uv pip install -r requirements-gpu.txt
uv sync --extra gpu
uv pip install "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot"
git clone <LIBERO-Plus> third_party/LIBERO-Plus && uv pip install -e third_party/LIBERO-Plus
python scripts/verify_env.py
```

`verify_env.py` must report `device capability == (12, 0) sm_120`, `torch >= 2.7`, CUDA build
`12.8`, and a successful `lerobot import`. It **fails by design off-GPU** (no torch/CUDA) — that
is expected; it passes only on the card.

> **Why these pins:** the 5090 is Blackwell `sm_120`; CUDA 12.8 is the minimum and PyTorch 2.7.0 is
> the first stable with native `sm_120`/cu128 wheels. Older CUDA-12.x (≤12.6) fails with
> *"no kernel image is available for execution on the device."* **Do NOT use CUDA 13.x.** Prefer the
> `pytorch/pytorch:2.x-cuda12.8-cudnn9` image and avoid runtime JIT of custom CUDA kernels.

### 6.1 AutoDL one-click — `deploy.py`

On an [AutoDL](https://www.autodl.com/) RTX 5090 instance, use the integrated, AutoDL-aware
deployer. It **reuses the image's pre-installed cu128 PyTorch** (no multi-GB re-download),
routes all model/dataset caches to the big data disk `/root/autodl-tmp`, turns on AutoDL
**academic network acceleration** + the `hf-mirror.com` endpoint, then installs LeRobot +
LIBERO-Plus, verifies `sm_120`, runs the tests, and gates the budget.

The LIBERO-Plus step does the full install its README requires (not just `pip install -e`):
apt system libs (`libexpat1`, `libfontconfig1-dev`, `libpython3-stdlib`, `libmagickwand-dev` —
ImageMagick/Wand backs the texture/background perturbations; installed only if root/sudo, else the
exact command is printed), `pip install -r extra_requirements.txt`, and the **`assets.zip`**
download from the HF dataset `Sylvest/LIBERO-plus` unzipped to `<checkout>/libero/libero/assets/`
(idempotent, mirror-aware). **Run deploy + the asset download in 无卡模式 (no-GPU mode)** to keep
these multi-GB downloads off the GPU clock, then switch to GPU mode for rollouts.

```bash
# clone onto the big data disk; use tmux so an SSH drop won't kill a long run
cd /root/autodl-tmp && git clone -b claude/confident-lovelace-4ddWN <your-repo-url> vcr && cd vcr
tmux new -s vcr

python3 deploy.py --check     # DRY RUN: detect GPU/torch/AutoDL + print the plan (installs nothing)
python3 deploy.py             # full deploy with AutoDL-friendly defaults
```

Tips: choose an AutoDL image with **CUDA 12.8 / PyTorch ≥ 2.7** (otherwise `deploy.py` warns and you
can add `--install-torch`); do downloads/setup in **无卡模式 (no-GPU mode)** to save GPU-hours, then
switch to GPU mode for rollouts. Useful flags: `--stage core` (off-GPU only), `--budget-only`,
`--no-accel`, `--no-hf-mirror`, `--skip-lerobot`, `--skip-libero-plus`, `--skip-apt`,
`--skip-assets`, `--data-dir PATH`.

## 7. Phase-by-phase execution guide

Always **smoke-test per-episode wall-clock first** (`./run.sh smoke …`) and re-check the budget
before scaling. The rollout/training entry points are scaffolded; each raises a
`NotImplementedError` naming the phase to implement.

| Phase | Goal | Command | Status |
|------:|------|---------|--------|
| **1** | base clean SR on a task subset | `uv run python -m eval.run_rollout --tasks … --n-episodes 20 --seed 0` | 🟡 CLI ready; rollout loop = seam |
| **2** | collapse curve (base × families×levels) | same, with `--perturb viewpoint:4` | 🟡 `PerturbSpec` ready; env = seam |
| **3** | LoRA train B & C (adapter only) | `uv run python -m train.train_lora --condition C --aug-families viewpoint lighting` | 🟡 aug magnitudes ready; train = seam |
| **4** | recovery + the headline A/B/C table | re-run eval per condition; fill tables | ⬜ needs Phase 1–3 data |
| **5** | bootstrap CI · paired · Holm | `eval.bootstrap` / `eval.paired` / `eval.holm` | ✅ implemented & tested; ⬜ awaiting data |
| **6** | before/after demo clips | `uv run python -m eval.record_demo --robust-adapter … --perturb viewpoint:4` | 🟡 seam |
| **7** | report + README polish | edit `report/*.md`; this README | 🟡 in progress |
| **8** | STRETCH: cross-family gen · π0.5 · cond. D · OpenVLA-OFT | — | ⬜ |

**Worked example — the smoke-then-budget loop (Phase 1 → gate):**
```bash
# Off-GPU, plan with assumed/known per-episode seconds:
uv run python -m scripts.smoke_timing --durations 28 31 30 29 33 --train-hours-per-run 4
# On the rental (once the Phase-1 rollout lands), measure live then gate automatically:
uv run python -m scripts.smoke_timing --smoke-tasks 2 --smoke-episodes 5 --train-hours-per-run 4
```

## 8. Statistics deep-dive

The statistics module is the part a domain reviewer scrutinises, so it is implemented and unit-tested
*before* any GPU run. The pipeline for one perturbation family:

1. **Point estimates** — `eval.metrics`: `success_rate`, then `Δ_robust`, `Recovery`, `Δ_method`.
2. **Uncertainty** — `eval.bootstrap.bootstrap_ci(outcomes, n_resamples=10_000)` → 95% percentile CI.
3. **Significance (paired)** — `eval.paired`:
   - `mcnemar(C, B)` on matched episodes (exact binomial when discordant pairs are few, else
     continuity-corrected χ²);
   - `paired_bootstrap_delta(C, B)` for a CI on `SR_C − SR_B`.
4. **Multiplicity** — `eval.holm.holm_bonferroni({family: p})` corrects the per-family p-values; a
   family is significant iff its adjusted p ≤ α.

Reading the lead table: report **Δ_method (C − B)** with its 95% CI and **Holm-corrected** p-value,
per family **and** aggregated, each tagged in-dist or held-out. Lead with this — *not* the recovery size.

---

## Metrics

Let `SR` = success rate = (successful episodes / total episodes). For a model *M* and base model *A*:

- **Δ_robust** (collapse magnitude) = `SR_clean − SR_perturbed` for a fixed model. Larger ⇒ bigger collapse.
- **Recovery** of intervention *M* on a perturbed family =
  `(SR_M^pert − SR_A^pert) / (SR_A^clean − SR_A^pert)` — fraction of the base model's lost performance
  restored. `1.0` ⇒ back to base's clean level; can exceed 1.0. (Zero-collapse denominator returns NaN.)
- **Δ_method** = `SR_C^pert − SR_B^pert`, computed **paired at fixed task IDs** (matched `(task_id, level, seed)`).

Implemented in [`eval/metrics.py`](eval/metrics.py); CIs/tests in [`eval/bootstrap.py`](eval/bootstrap.py),
[`eval/paired.py`](eval/paired.py), [`eval/holm.py`](eval/holm.py).

## Environment (pin these)

This project targets the **RTX 5090 (Blackwell, `sm_120`)**: **CUDA 12.8 (cu128)** + **PyTorch ≥ 2.7.0**
(first stable with native `sm_120`/cu128 wheels). **Do NOT use CUDA 13.x.** Prefer the official
**`pytorch/pytorch:2.x-cuda12.8-cudnn9`** image; avoid runtime JIT of custom CUDA kernels (use AOT wheels).
Verify with [`scripts/verify_env.py`](scripts/verify_env.py). Full install: [§6](#6-renting-the-gpu--deploying).

## Budget (hard limits)

| Resource | Limit |
|---|---|
| GPU | **single** RTX 5090 (32 GB), **serial** rollouts |
| GPU rental | **< 5 days** total — gate the matrix with `./run.sh budget` first |
| Disk | **50 GB** — eval CSV/JSON only; a *handful* of demo clips; subset LIBERO + subset LIBERO-Plus (full ≈ 6.4 GB) |
| Calendar | ~10 weeks |

Off-GPU (no rental): scaffold, stats, budget gating, full smoke tests. On-GPU (< 5 days, serial): all
LoRA training + all rollout evals + the 3-seed comparison.

## Build status

✅ implemented & tested · 🟡 scaffolded (CLI + phase-tagged `NotImplementedError` seam) · ⬜ pending

| Phase | Description | Status |
|------:|-------------|--------|
| **0** | Scaffold + environment | ✅ (GPU/sm_120 verify deferred to the rental) |
| 1 | Base rollout eval (clean SR) | 🟡 CLI + seam |
| 2 | Perturbation suite + collapse curve (output #1) | 🟡 selector done; env seam |
| 3 | Augmentation + LoRA training | 🟡 aug magnitudes done; train seam |
| 4 | Recovery curve + intervention comparison (**headline**, output #2) | ⬜ needs 1–3 |
| **5** | Statistics: bootstrap / paired / Holm (**differentiator**) | ✅ implemented & tested; ⬜ awaiting data |
| 6 | Demo videos (output #3) | 🟡 seam |
| 7 | Report + repo polish | 🟡 README + report skeletons |
| 8 | STRETCH: cross-family · π0.5 · feature-mod D · OpenVLA-OFT | ⬜ |
| — | **Tooling** | ✅ `run.sh` + `deploy.py` (AutoDL) one-click · budget gate · smoke harness |

## Repository layout

See the annotated [Repository map](#3-repository-map-what-each-piece-does) above for what each file does.

## Stack

| Role | Choice |
|---|---|
| Primary model (CORE) | **SmolVLA** (~450M) via LeRobot; base = `HuggingFaceVLA/smolvla_libero` |
| Modern comparator (STRETCH) | **π0.5** via LeRobot-native `pi05_libero` (flow-matching, ~3B, LoRA) |
| Benchmark | **LIBERO** (Spatial / Object / Goal / Long subset) |
| Perturbations (PRIMARY) | **LIBERO-Plus** (arXiv 2510.13626) — drop-in for `libero`, 7 dims, graded L1–L5 |
| Perturbations (stress/citation) | **LIBERO-PRO** (arXiv 2510.03827) — ~0% floor by design; citation/stress only |
| Fine-tuning | LoRA/PEFT (SmolVLA r=16/32; π0.5 LoRA; 7B QLoRA only) |
| Augmentation | torchvision + viewpoint/lighting generators aligned to LIBERO-Plus families |
| Statistics | numpy/scipy + bootstrap CI / paired (fixed init) / Holm–Bonferroni |

## Prior art (cited; no novelty claimed)

- **arXiv 2510.00037 — "RobustVLA"** *(closest prior work; a robustness **method**)*. This repo is a
  reproducible **study**, not that method; the project was renamed `VLA-Collapse-Recover` to avoid the clash.
- **arXiv 2510.13626 — LIBERO-Plus**: graded perturbation benchmark used here as the primary suite.
- **arXiv 2510.03827 — LIBERO-PRO**: memorization-exposing benchmark (≈0% floor); stress/citation only.
- **arXiv 2512.02902 — FTM/FLA-style feature modulation**: basis for STRETCH condition D.
- **arXiv 2510.17640 — RESample-style augmentation**: basis for perturbation-aligned augmentation.
- **SmolVLA — arXiv 2506.01844**: the ~450M base model (LeRobot); chosen so the whole study fits a
  single RTX 5090 in < 5 days — the tight scope is a deliberate *scoping-discipline* feature.

**Positioning lineage (the contribution): representation quality.** The diagnostic battery sits in
the lineage of work on *what a network actually represents*:

- **Shortcut learning — arXiv 2004.07780** (Geirhos et al., Nat. Mach. Intell. 2020): networks
  exploit non-causal shortcuts that break under distribution shift — exactly the collapse we probe.
- **Causal representation learning — arXiv 2102.11107** (Schölkopf et al., Proc. IEEE 2021): the
  frame for "causal vs. shortcut" features and why causal representations transfer.
- **World models — arXiv 1803.10122** (Ha & Schmidhuber, 2018): the conceptual target — a policy
  whose internal representation *models the scene* rather than memorising pixels.

**Why this lens (world models).** Perturbation robustness, **language conditioning**, and
**cross-family generalization** are *joint* proxies for whether the policy's implicit representation
is **causal** (models the scene) or **shortcut-based** (latches onto spurious cues a moved camera /
new texture / shuffled instruction destroys). A representation-level fix should improve all three
*together*; a symptom patch raises in-dist success while held-out generalization and language
sensitivity stay flat. The probe battery measures these observable consequences directly — we do not
*claim* causality, we *test* for it.

**Methods & evaluation references (supporting infrastructure, not the contribution).** The statistics
follow standard reproducible-evaluation practice — **rliable / "Statistical Precipice"
(arXiv:2108.13264)**, **SimplerEnv (arXiv:2405.05941)**, **AutoEval (arXiv:2503.24278)** — and the
paired-stats code is shared with the author's prior **PolicyArena** project (see Honesty guards).

Train-time augmentation and eval-time perturbation share a *family* but stay on *separate splits*.

## License

MIT — see [`LICENSE`](LICENSE).
