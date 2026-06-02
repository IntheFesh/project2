# VLA-Collapse-Recover

**A reproducible empirical study of visual-perturbation robustness in open Vision-Language-Action (VLA) models.**

Open VLAs score high on *clean* LIBERO but **collapse** under visual / viewpoint perturbations
(documented across multiple independent 2025–2026 papers). This repo (A) reproduces that collapse,
(B) **recovers** it via perturbation-augmented LoRA fine-tuning, (C) compares training interventions
with **rigorous paired statistics**, and (D) produces before/after rollout demo videos.

> **The headline is the *intervention comparison* (A/B/C), not the recovery magnitude.** Same-family
> recovery is expected; the contribution is a clean, reproducible study with honest statistics.

This is a **research-engineering portfolio project**, not a paper. No novelty is claimed.

---

## ⚠️ Honesty guards (read first)

1. **No fabricated results.** Every number below is `TBD` until a real run fills it.
2. **Only relative / delta claims in headlines** (Δ_robust, Recovery, Δ_method). We do **not** claim
   to reproduce any paper's absolute SOTA number. The base SmolVLA-LIBERO **absolute** success rate
   here **may not match the published number** — LIBERO eval setup is sensitive — which is *exactly why
   only deltas are reported*.
3. **In-distribution vs held-out is always labeled.** A perturbation family seen during augmentation
   is *in-dist*; one never seen is *held-out generalization*. In-dist recovery is **never** presented
   as generalization.
4. **Headline = intervention comparison**, led by paired statistics, not by recovery size.

---

## Headline result — Intervention comparison *(Phase 4–5)*  ·  `TBD`

Perturbed success rate is averaged over the **in-distribution** augmented families
(viewpoint / lighting / texture / noise). Paired at fixed init states; 3 seeds on the B/C(/D) comparison.

| Condition | Training augmentation | Clean SR | Perturbed SR (in-dist avg) | Recovery | Δ_method vs **B** | 95% CI (Δ_method) | Holm *p* |
|-----------|-----------------------|:--------:|:--------------------------:|:--------:|:-----------------:|:-----------------:|:--------:|
| **A** — base (no FT) | — (pretrained `smolvla_libero`) | `TBD` | `TBD` | — | — | — | — |
| **B** — LoRA + standard aug | generic torchvision aug | `TBD` | `TBD` | `TBD` | *(reference)* | — | — |
| **C** — LoRA + perturbation-targeted aug | aug aligned to eval families | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **D** — feature-modulation adapter *(STRETCH)* | FTM/FLA-style | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

*Δ_method (C − B) at fixed init states is the lead statistic — the 5–10pp gap a paired test makes detectable.*

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
| *(held-out family, STRETCH cross-family test)* | held-out | `TBD` | `TBD` | `TBD` | `TBD` |

---

## Metrics

Let `SR` = success rate = (successful episodes / total episodes). For a model *M* and base model *A*:

- **Δ_robust** (collapse magnitude) = `SR_clean − SR_perturbed` for a fixed model. Larger ⇒ bigger collapse.
- **Recovery** of intervention *M* on a perturbed family =
  `(SR_M^pert − SR_A^pert) / (SR_A^clean − SR_A^pert)` — fraction of the base model's lost performance
  restored. `1.0` ⇒ back to base's clean level; can exceed 1.0. (Zero-collapse denominator is guarded.)
- **Δ_method** (headline) = `SR_C^pert − SR_B^pert`, computed **paired at fixed init states**.

See [`eval/metrics.py`](eval/metrics.py).

---

## Environment (pin these)

This project targets the **RTX 5090 (Blackwell, `sm_120`)**, which requires:

- **CUDA 12.8 (cu128)** — the minimum for `sm_120`. **Do NOT use CUDA 13.x.**
- **PyTorch ≥ 2.7.0** — first stable release with native `sm_120` / cu128 wheels.
- Older CUDA-12.x builds (≤ 12.6, up to `sm_90`) fail with
  `"no kernel image is available for execution on the device."`
- Prefer the official **`pytorch/pytorch:2.x-cuda12.8-cudnn9`** Docker image.
- **Avoid runtime JIT of custom CUDA kernels** (Blackwell PTX JIT has a known missing-lib issue) — use
  pre-built / AOT wheels.

### Install (two-stage, budget-aware)

```bash
# 1) OFF-GPU (no rental): lightweight core + dev tools. Enough for scaffolding,
#    the statistics modules, analysis, and all off-GPU smoke tests.
uv sync

# 2) ON-GPU (rented RTX 5090 only):
#    2a) cu128 PyTorch (pinned; needs the pytorch cu128 index, not plain PyPI):
uv pip install -r requirements-gpu.txt
#    2b) the rest of the GPU-adjacent stack (PyPI-resolvable):
uv sync --extra gpu
#    2c) LeRobot (SmolVLA / pi0.5) + LIBERO-Plus from source (fast-moving; not pinned):
uv pip install "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot"
uv pip install -e third_party/LIBERO-Plus    # drop-in replacement for `libero`

# 3) Verify the card actually exposes sm_120 (must run on the rental):
python scripts/verify_env.py
```

`scripts/verify_env.py` checks `torch.cuda.get_device_capability() == (12, 0)`, `torch >= 2.7`, the CUDA
runtime version, and that LeRobot imports. **It cannot pass in a CPU/no-GPU container** — that is expected.

---

## Budget (hard limits)

| Resource | Limit |
|---|---|
| GPU | **single** RTX 5090 (32 GB), **serial** rollouts |
| GPU rental | **< 5 days** total |
| Disk | **50 GB** — eval CSV/JSON only; a *handful* of demo clips; subset LIBERO + subset LIBERO-Plus (full ≈ 6.4 GB); clean intermediates |
| Calendar | ~10 weeks |

Off-GPU (no rental): Phases 0–2 setup, LIBERO-Plus drop-in + thin wrapper, augmentation pipeline,
stats modules, full smoke tests. On-GPU (< 5 days, serial): all LoRA training + all rollout evals +
3-seed comparison. **Always smoke-test per-episode wall-clock before scaling.**

---

## Build status

| Phase | Description | Status |
|------:|-------------|--------|
| **0** | Scaffold + environment | ✅ scaffold done; GPU/sm_120 verify deferred to rental |
| 1 | Base rollout eval (clean SR) | ⬜ |
| 2 | Perturbation suite + collapse curve (guaranteed output #1) | ⬜ |
| 3 | Augmentation + LoRA training | ⬜ |
| 4 | Recovery curve + intervention comparison (headline, output #2) | ⬜ |
| 5 | Statistics: bootstrap CI / paired / Holm–Bonferroni (differentiator) | ⬜ |
| 6 | Demo videos (output #3) | ⬜ |
| 7 | Report + repo polish | ⬜ |
| 8 | STRETCH: cross-family generalization · π0.5 comparator · feature-mod D · OpenVLA-OFT | ⬜ |

---

## Repository layout

```
vla-collapse-recover/
├── README.md  pyproject.toml  LICENSE  .gitignore  CLAUDE.md
├── configs/   model/ lora/ perturb/ eval/      # config-driven (YAML)
├── data/      prepare_libero_subset.py  augment/
├── perturb/   libero_plus_wrapper.py           # thin wrapper over LIBERO-Plus (NOT reimplemented)
├── train/     train_lora.py  feature_mod.py
├── eval/      run_rollout.py metrics.py bootstrap.py paired.py holm.py record_demo.py
├── analysis/                                    # CSVs + plots (small)
├── report/    technical_report.md  one_pager.md
├── scripts/   verify_env.py
└── tests/                                       # off-GPU smoke tests
```

---

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

---

## Prior art (cited; no novelty claimed)

- **arXiv 2510.00037 — "RobustVLA"** *(closest prior work; a robustness **method**)*. This repo is a
  reproducible **study**, not that method, and the project was renamed `VLA-Collapse-Recover` to avoid
  the name clash.
- **arXiv 2510.13626 — LIBERO-Plus**: graded perturbation benchmark used here as the primary suite.
- **arXiv 2510.03827 — LIBERO-PRO**: memorization-exposing benchmark (≈0% floor); stress/citation only.
- **arXiv 2512.02902 — FTM/FLA-style feature modulation**: basis for STRETCH condition D.
- **arXiv 2510.17640 — RESample-style augmentation**: basis for perturbation-aligned augmentation.

Train-time augmentation and eval-time perturbation share a *family* but stay on *separate splits*.

## License

MIT — see [`LICENSE`](LICENSE).
