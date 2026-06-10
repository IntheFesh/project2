"""Phase 2 — Collapse curve: base SmolVLA under LIBERO-Plus perturbation.

For each CORE family x level cell, select N task instances from task_classification.json,
run them via the official lerobot-eval CLI (eval/lerobot_runner.py), and emit per-trial
CSV rows matching docs/EVALUATION.md:  condition, task_id, family, level, seed, success.

README collapse matrix: families {viewpoint, lighting, texture, noise} x levels {Clean, L2, L4}.
  * Clean  -> original LIBERO 10-task suite (PYTHONPATH=LIBERO-orig isolation), level=0.
  * L2/L4  -> LIBERO-Plus perturbed variants for that family+level.

IMPORTANT index convention (verified 2026-06-08):
  task_classification.json `id` is 1-indexed; lerobot `--env.task_ids` is 0-indexed.
  So lerobot_task_id = json_id - 1.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from eval.lerobot_runner import run_eval
from perturb.libero_plus_constants import FAMILY_TO_CATEGORY

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_CLASS_JSON = REPO_ROOT / "third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"

CORE_FAMILIES = ("viewpoint", "lighting", "texture", "noise")
COLLAPSE_LEVELS = (2, 4)  # README collapse columns L2, L4
CLEAN_LEVEL = 0
POLICY_PATH = "HuggingFaceVLA/smolvla_libero"
SUITE = "libero_spatial"


def select_cell_task_ids(suite: str, family: str, level: int, n: int) -> list[int]:
    """Select N lerobot 0-indexed task_ids for a (family, level) cell, deterministically.

    Picks the first N matching instances by ascending json id. Returns lerobot task_ids
    (json_id - 1).
    """
    category = FAMILY_TO_CATEGORY[family]
    tc = json.loads(TASK_CLASS_JSON.read_text())
    matches = sorted(
        (e for e in tc[suite] if e["category"] == category and e["difficulty_level"] == level),
        key=lambda e: e["id"],
    )
    return [e["id"] - 1 for e in matches[:n]]


def run_phase2(
    *,
    instances_per_cell: int,
    n_episodes: int,
    seed: int,
    out_csv: str | Path,
    clean_task_ids: tuple[int, ...] = tuple(range(10)),
) -> Path:
    """Run the full Phase 2 collapse grid and write per-trial CSV rows."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # --- Clean baseline (original LIBERO 10 tasks, once, shared across families) ---
    print(f"[Phase2] CLEAN baseline: {SUITE} tasks {list(clean_task_ids)}, {n_episodes} ep each")
    clean_results = run_eval(
        policy_path=POLICY_PATH, suite=SUITE, task_ids=list(clean_task_ids),
        n_episodes=n_episodes, seed=seed, is_libero_plus=False,
    )
    for r in clean_results:
        rows.append({"condition": "A", "task_id": r["task_id"], "family": "clean",
                     "level": CLEAN_LEVEL, "seed": seed, "success": int(r["success"])})
    _flush_csv(rows, out_csv)
    print(f"[Phase2] clean done: {sum(r['success'] for r in clean_results)}/{len(clean_results)}")

    # --- Perturbed cells: 4 families x {L2, L4} ---
    for family in CORE_FAMILIES:
        for level in COLLAPSE_LEVELS:
            task_ids = select_cell_task_ids(SUITE, family, level, instances_per_cell)
            if not task_ids:
                print(f"[Phase2] WARN: no tasks for {family} L{level}, skipping")
                continue
            print(f"[Phase2] {family} L{level}: {len(task_ids)} tasks x {n_episodes} ep")
            results = run_eval(
                policy_path=POLICY_PATH, suite=SUITE, task_ids=task_ids,
                n_episodes=n_episodes, seed=seed, is_libero_plus=True,
            )
            for r in results:
                rows.append({"condition": "A", "task_id": r["task_id"], "family": family,
                             "level": level, "seed": seed, "success": int(r["success"])})
            _flush_csv(rows, out_csv)
            n_succ = sum(r["success"] for r in results)
            print(f"[Phase2] {family} L{level} done: {n_succ}/{len(results)} = {100*n_succ/len(results):.1f}%")

    print(f"[Phase2] COMPLETE: {len(rows)} rows -> {out_csv}")
    return out_csv


def _flush_csv(rows: list[dict], path: Path) -> None:
    """Write all rows to CSV (overwrite; called after each cell for crash-safety)."""
    fields = ["condition", "task_id", "family", "level", "seed", "success"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances-per-cell", type=int, default=12)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/runs/phase2_collapse.csv")
    args = ap.parse_args()
    run_phase2(instances_per_cell=args.instances_per_cell, n_episodes=args.n_episodes,
               seed=args.seed, out_csv=args.out)


if __name__ == "__main__":
    main()
