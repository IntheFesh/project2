# CLAUDE.md — VLA-Collapse-Recover

Project memory for Claude Code. Read this every session.

## What this is
A **reproducible empirical study** of visual-perturbation robustness in open VLA models:
reproduce the **collapse**, **recover** via perturbation-augmented LoRA, **compare** training
interventions with rigorous paired statistics, and ship **demo videos**. Portfolio /
research-engineering project (PhD-application signal), **not a paper**.

## Hard budget (non-negotiable)
- **single** RTX 5090 (32 GB, Blackwell `sm_120`), **serial** rollouts.
- **< 5 days** total GPU rental. 50 GB disk. ~10 weeks calendar.
- Rollouts are the time bottleneck → always smoke-test per-episode wall-clock before scaling.

## Non-negotiable rules
1. **Never fabricate results.** Every metric is `TBD` until a real run fills it.
2. **Only delta/relative claims in headlines** (Δ_robust, Recovery, Δ_method). Never claim to
   reproduce a paper's absolute SOTA; README states the base absolute SR may not match.
3. **Always label in-dist vs held-out.** Never present in-dist recovery as generalization.
4. **Headline = intervention comparison (A/B/C), not recovery size.** Lead with paired stats.
5. **No novelty claims.** Cite prior art, incl. arXiv 2510.00037 (the "RobustVLA" *method*).
6. **Env:** CUDA 12.8 (cu128) + PyTorch ≥ 2.7 (first stable with `sm_120`). No CUDA 13.x.
   Prefer `pytorch/pytorch:2.x-cuda12.8-cudnn9`; avoid JIT of custom CUDA kernels (use AOT wheels).
7. **Disk discipline:** store only eval summaries (CSV/JSON); no rollout videos except a few demo
   clips; subset LIBERO + subset LIBERO-Plus; clean intermediates.
8. **Single serial card:** smoke-test timing first; size task subset & episode count to fit < 5 days.
   3 seeds on the key B/C(/D) comparison; collapse curve (A) needs only 1 seed.
9. **Do NOT reimplement perturbations.** Use LIBERO-Plus drop-in (`pip install -e .`); thin wrapper only.
10. Type hints + docstrings + smoke tests; config-driven (YAML in `configs/`); small reviewable commits.
11. **Ask before** large downloads, long (>2h) runs, or opening the GPU rental.

## Stack (pinned)
- Model CORE: **SmolVLA** (~450M) via LeRobot; base `HuggingFaceVLA/smolvla_libero`.
- Comparator STRETCH: **π0.5** via LeRobot-native `pi05_libero`. (OpenVLA-OFT 7B → skip by default.)
- Benchmark: **LIBERO**. Perturbations PRIMARY: **LIBERO-Plus** (2510.13626); STRESS: LIBERO-PRO (2510.03827).
- FT: LoRA/PEFT. Aug: torchvision + perturbation-aligned generators. Stats: numpy/scipy + bootstrap/paired/Holm.

## Conditions
A = base (no FT) · B = LoRA + standard aug · C = LoRA + perturbation-targeted aug · D = feature-mod adapter (STRETCH).

## Two-stage env
- `uv sync` → lightweight core + dev (off-GPU work: stats, analysis, smoke tests).
- `uv sync --extra gpu` → cu128 torch stack (rental only). Then `pip install` LeRobot + LIBERO-Plus from source.
- `python scripts/verify_env.py` → asserts `sm_120 (12,0)`, torch ≥ 2.7, LeRobot import. Cannot pass off-GPU.

## Per-turn output contract
(1) plan · (2) change · (3) how to run/verify · (4) verification result · (5) CORE/STRETCH + GPU-day budget note.
Always flag in-dist vs held-out. Keep the user in the loop before heavy/irreversible steps.

## Git
Develop on branch `claude/confident-lovelace-4ddWN`. Small commits. Push with `-u origin <branch>`.
Do NOT open a PR unless explicitly asked.
