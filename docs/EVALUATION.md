# Evaluation protocol

This project's headline contribution is not a model — it is a **reusable, honest evaluation
protocol** for VLA visual-robustness, packaged in [`eval/stats/`](../eval/stats/) and driven by
[`scripts/analyze_results.py`](../scripts/analyze_results.py). Its north star is **rliable**
(Agarwal et al., *"Deep Reinforcement Learning at the Edge of the Statistical Precipice"*, NeurIPS
2021, [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)): with few seeds and noisy success
indicators, report **effect sizes with uncertainty**, not bare point success rates. This repo is
the VLA-manipulation analogue of that argument.

## The unit of evaluation

A pre-built LIBERO-Plus **task instance**, run **once** (`num_trials_per_task = 1`; see
[`LIBERO_PLUS_NOTES.md`](LIBERO_PLUS_NOTES.md)). Per `(family, level)` cell we evaluate
`instances_per_cell` task IDs. Every trial is one CSV row:

```
condition, task_id, family, level, seed, success
```

## The protocol (what `build_report` computes)

1. **Per-family success rate + bootstrap 95% CI** — `bootstrap_ci` (≥ 10,000 resamples).
2. **Headline paired effect `Δ_method = SR_C − SR_B`** — conditions A/B/C are evaluated on the
   **same task IDs**, so trials are *matched*. We pair on `(task_id, level, seed)` and report a
   **paired-bootstrap CI** plus **McNemar's test** (`eval/paired.py`). Pairing is what makes the
   small (~5–10pp) B-vs-C gap detectable.
3. **Multiplicity control** — **Holm–Bonferroni** across perturbation families (`eval/holm.py`); a
   family is significant iff its adjusted *p* ≤ α.
4. **Recovery** — fraction of the base model's collapse an intervention restores (`eval/metrics.py`).
5. **Generalization gap** — `Recovery_C(in-dist) − Recovery_C(held-out)`; a first-class check that
   Condition C is not merely overfitting its augmentation family.
6. **rliable-style aggregate** — pooled SR with a bootstrap CI **and** the **interquartile mean
   (IQM)** of per-cell success rates (robust to outlier cells).

## Honesty guards baked into the protocol

- **Paired by task, not by resampled init states.** Seeds for B/C are *independent LoRA training
  runs* evaluated on the same task-ID set (training-init variance), each row carrying its `seed`.
- **In-dist vs held-out is always labelled** (`classify_distribution`); in-dist recovery is never
  reported as generalization.
- **Deltas only.** The base absolute SR may not match published numbers; we report relative effects
  with uncertainty, never an absolute-SOTA claim.

## Reproduce

```bash
# 1) produce per-trial CSVs (on the GPU box; Phase 1+):
uv run python -m eval.run_rollout --condition A --seed 0 --out analysis/runs/A.csv
# ... B and C ...  then concatenate to analysis/runs/all.csv
# 2) analyze (off-GPU):
uv run python -m scripts.analyze_results --csv analysis/runs/all.csv --out analysis/report
```

Produces `analysis/report.txt` (human-readable) and `analysis/report.json` (machine-readable).
