# Reproducing VLA-Collapse-Recover

This is the end-to-end guide to go from a clean clone to
[`analysis/runs/phase5_summary.md`](../analysis/runs/phase5_summary.md) — the authoritative
paired-statistics output. It is written for *another researcher who wants to reproduce the run*.

**Scope and budget.** The full pipeline is a single-seed run on a **LIBERO-spatial subset**, one
**RTX 5090** (Blackwell, `sm_120`), **serial** rollouts, that fits in roughly a GPU-day of compute
(hard cap **< 5 days**, 50 GB disk). All headline claims are **delta-only**; absolute success rates
are not directly comparable to published numbers and are not claimed to be (see
[`PROBES.md`](PROBES.md)). Single seed: every ΔSR / p-value is paired statistics within one seed;
a multi-seed extension is future work.

---

## 0. Two stages: off-GPU vs on-GPU

| Stage | Where | What | How |
|---|---|---|---|
| **Off-GPU core** | any Linux box, no GPU | stats, analysis/heatmap, budget gating, full test suite | `make setup` |
| **On-GPU** | rented RTX 5090 (`sm_120`) | LoRA training (B/C) + all rollouts (collapse + recovery) | `uv sync --extra gpu` + cu128 torch + LeRobot + LIBERO-Plus |

You can reproduce the **statistics** (Phase 5) from the committed per-episode CSVs **off-GPU today**.
Reproducing the **rollouts and training** (Phases 2–4) that *produced* those CSVs needs the GPU.

---

## 1. Off-GPU: regenerate the statistics from the committed CSVs

```bash
git clone https://github.com/IntheFesh/project2 vla-collapse-recover && cd vla-collapse-recover
make setup          # uv sync: numpy/scipy/pandas/matplotlib + pytest/ruff (no GPU, no big downloads)
make test           # 115 pure-logic tests, all green (no torch/lerobot needed)
make stats          # regenerate phase5_summary.{md,json} + the heatmap from the CSVs
```

`make stats` runs `python -m eval.runners.phase5_stats` over
[`analysis/runs/phase2_collapse.csv`](../analysis/runs/phase2_collapse.csv) +
[`phase4_recovery.csv`](../analysis/runs/phase4_recovery.csv) and **rewrites
`phase5_summary.md` byte-for-byte identically** — this is your reproducibility check on the
analysis. (The PNG heatmap is byte-deterministic; the SVG carries matplotlib's embedded render
date, so it shows a timestamp-only diff.)

Everything below reproduces the CSVs themselves on a GPU.

---

## 2. On-GPU: rent and deploy

The 5090 is Blackwell `sm_120`: you need **CUDA 12.8 (cu128)** + **PyTorch 2.8.0** (first stable with
native `sm_120` wheels). **Do not use CUDA 13.x.** Prefer the `pytorch/pytorch:2.x-cuda12.8-cudnn9`
image and AOT wheels (avoid JIT of custom CUDA kernels).

### One-click (AutoDL-aware)

```bash
# Clone onto the big data disk; tmux so an SSH drop won't kill a long run.
cd /root/autodl-tmp && git clone https://github.com/IntheFesh/project2 vcr && cd vcr
tmux new -s vcr

python3 deploy.py --check     # DRY RUN: detect GPU/torch/AutoDL + print the plan (installs nothing)
python3 deploy.py             # full deploy
```

`deploy.py` reuses the image's pre-installed cu128 torch, routes caches to `/root/autodl-tmp`, turns
on AutoDL academic acceleration + the `hf-mirror.com` endpoint, then installs LeRobot + the full
LIBERO-Plus (apt `libmagickwand-dev`, `extra_requirements.txt`, and the **`assets.zip`** from the HF
dataset `Sylvest/LIBERO-plus` unzipped to `libero/libero/assets/`). **Run downloads in 无卡模式
(no-GPU mode)** so the multi-GB pulls don't burn GPU-hours, then switch to GPU mode.

### Manual equivalent

```bash
uv pip install -r requirements-gpu.txt                         # torch==2.8.0+cu128 + torchvision
uv sync --extra gpu                                            # peft==0.19.1, transformers, ...
uv pip install "lerobot[dataset,peft] @ git+https://github.com/huggingface/lerobot@v0.5.2"
git clone <LIBERO-Plus> third_party/LIBERO-plus && uv pip install -e third_party/LIBERO-plus
git clone <LIBERO-orig> third_party/LIBERO-orig                # clean LIBERO suites (for the clean baseline)
```

### Environment variables

| Variable | Value | Why |
|---|---|---|
| `HF_ENDPOINT` | `https://hf-mirror.com` | faster/robust HF downloads (esp. in CN) |
| `HF_HOME` | `/root/autodl-tmp/hf` | keep model/dataset caches on the big disk |
| `MUJOCO_GL` | `egl` | headless offscreen rendering for rollouts |
| `PYTHONPATH` | `third_party/LIBERO-orig` (clean only) | the installed `libero` IS LIBERO-Plus; the **clean** suites live only in the separate LIBERO-orig clone |

### Verify the card

```bash
python scripts/verify_env.py    # asserts device capability (12, 0) sm_120, torch >= 2.7, CUDA 12.8, LeRobot import
```

It **fails by design off-GPU** (no torch/CUDA) — that is expected; it passes only on the card.

---

## 3. Gate the budget, then smoke-test timing

Always project the matrix against the **< 5-day** cap *before* a long run, and **smoke-test
per-episode wall-clock** first (rollouts are the bottleneck):

```bash
make smoke                                          # 60-step LoRA smoke: validates the train/aug/CLI wiring
uv run python -m scripts.smoke_timing --smoke-tasks 2 --smoke-episodes 5 --train-hours-per-run 4
./run.sh budget                                     # SEC_PER_EP=30, TRAIN_H=4 defaults; exits non-zero if it won't fit
```

---

## 4. Phase 2 — collapse curve (condition A, base model)

```bash
uv run python -m eval.runners.phase2_collapse --instances-per-cell 12 --n-episodes 5 --seed 0
# -> analysis/runs/phase2_collapse.csv  (condition A: clean + {viewpoint,lighting,texture,noise} x {L2,L4})
```

The clean baseline runs the original LIBERO 10-task suite (via the LIBERO-orig clone); the perturbed
cells select pre-built LIBERO-Plus task IDs by `(family, level)` from `task_classification.json` and
run each once (`num_trials_per_task = 1`). Crash-safe: the CSV is flushed after every cell.

## 5. Phase 3 — train the LoRA adapters (B and C)

```bash
make train-B        # uv run python -m train.train_lora --condition B   (LoRA + LeRobot default aug)
make train-C        # uv run python -m train.train_lora --condition C   (LoRA + targeted aug, 4 CORE families)
# adapters land in adapters/{B,C}_seed0_a32/checkpoints/last/pretrained_model  (gitignored)
```

Each LoRA run is a thin wrapper over the verified `lerobot-train` CLI (no hand-written loop). Budget
the default at ~4 GPU-hours per run; smoke first. Condition C's targeted augmentation magnitudes are
derived from `data/augment/visual_aug.py::aligned_magnitude`.

## 6. Phase 4 — recovery rollouts (A held-out layout + B/C all cells)

```bash
make eval           # uv run python -m eval.runners.phase4_recovery
# -> analysis/runs/phase4_recovery.csv
```

This evaluates B and C on the **same** deterministic `(family, level)` cells as Phase 2 (so episodes
pair with A by `task_id`), and adds the **held-out `layout`** family (never augmented → the
generalization test). **Resume is automatic:** the runner records each `(condition, family, level)`
cell and **skips cells already present** in the output CSV, so an interrupted run continues where it
stopped. The `noise` family uses a reduced configuration (6 tasks/cell, `episode_length=180`) for
both B and C symmetrically — see [`PROBES.md`](PROBES.md) §4.

## 7. Phase 5 — paired statistics + heatmap

```bash
make stats          # python -m eval.runners.phase5_stats  +  analysis/make_heatmap.py
# -> analysis/runs/phase5_summary.{md,json}  +  recovery_heatmap.{png,svg}
```

This computes the three pre-registered tests (H1/H2/H3) over matched `(family, level, task_id,
episode_index)` episodes — bootstrap 95% CIs (10,000 resamples), McNemar (exact < 25 discordant
pairs, else continuity-corrected χ²), Holm–Bonferroni across the H2 families. The output should
match the committed [`phase5_summary.md`](../analysis/runs/phase5_summary.md); the interpretation is
in [`RESULTS.md`](RESULTS.md).

---

## Troubleshooting

- **`no kernel image is available for execution on the device`** — your CUDA is ≤ 12.6. The 5090
  needs cu128 + torch ≥ 2.7. Do not use CUDA 13.x.
- **`torch.load` weights-only error on LIBERO-Plus init states** — handled in
  `perturb/libero_plus_wrapper.py` (trusted pickle, `weights_only=False`); init-state files ship
  with LIBERO-Plus.
- **texture/background perturbations fail** — missing ImageMagick: install `libmagickwand-dev`
  (`deploy.py` does this when root/sudo, else prints the exact command).
- **noise-family rollouts hang / run to env-max** — expected; they are ~5× slower. The Phase 4
  runner caps `episode_length` and halves instances for `noise` only (kept symmetric across B/C).
- **clean baseline gives 0%** — you ran clean against LIBERO-Plus. The clean suites need the
  LIBERO-orig clone on `PYTHONPATH`; the official `lerobot-eval` wiring is in
  `eval/runners/lerobot_runner.py`.
