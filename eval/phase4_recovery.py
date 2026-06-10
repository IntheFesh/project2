"""Phase 4 -- Recovery curve & intervention comparison (B/C vs A).

Evaluates the LoRA adapters (condition B = standard aug, C = targeted aug) on the SAME
deterministic (family, level) cells as the Phase 2 collapse grid, so results pair with
condition A by ``task_id`` for McNemar / paired bootstrap (Phase 5).

Reuses ``select_cell_task_ids`` (deterministic) and ``run_eval`` from the Phase 2 modules.
Adds the held-out ``layout`` family (never augmented -> generalization test). Crash-safe:
flushes after every cell and resumes by skipping (condition, family, level) cells already
present in the output CSV.

Plan (single unattended run):
  * A : layout x {L2, L4}              (A's clean+core already in analysis/runs/phase2_collapse.csv)
  * B : clean + {core4 + layout} x {L2, L4}
  * C : clean + {core4 + layout} x {L2, L4}
Output rows match Phase 2 schema: condition, task_id, family, level, seed, success.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from eval.lerobot_runner import run_eval
from eval.phase2_collapse import (
    CLEAN_LEVEL,
    COLLAPSE_LEVELS,
    POLICY_PATH,
    SUITE,
    select_cell_task_ids,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_FIELDS = ["condition", "task_id", "family", "level", "seed", "success"]

# held-out generalization family (never augmented in condition C)
HELDOUT_FAMILIES = ("layout",)
EVAL_FAMILIES = ("viewpoint", "lighting", "texture", "noise") + HELDOUT_FAMILIES

BASE_POLICY = POLICY_PATH  # condition A
ADAPTERS = {
    "B": "adapters/B_seed0_a32/checkpoints/last/pretrained_model",
    "C": "adapters/C_seed0_a32/checkpoints/last/pretrained_model",
}
CLEAN_TASK_IDS = tuple(range(10))


def _load_done_cells(out_csv: Path) -> set[tuple[str, str, int]]:
    """Return {(condition, family, level)} already in the CSV (for resume)."""
    if not out_csv.exists():
        return set()
    done = set()
    with out_csv.open() as f:
        for r in csv.DictReader(f):
            done.add((r["condition"], r["family"], int(r["level"])))
    return done


def _append_rows(rows: list[dict], out_csv: Path) -> None:
    """Append rows to CSV (header written once); flush per cell for crash-safety."""
    new = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


# Per-episode wall-clock ceiling (s). noise/layout renders are the slowest (~12-60s/ep
# observed); 60s/ep gives generous headroom so a whole cell is never killed mid-run.
SEC_PER_EPISODE_CAP = 60
TIMEOUT_FLOOR_S = 900

# Noise-family special handling (Phase 4, post-mortem of first run):
# - On non-recovered policies a noise episode runs to env-max (~280 steps) without success,
#   making one cell ~8h. We cap episode_length to a value safely above the spatial
#   demo p90 (demo median 123 / p90 149 / max 193 steps) so success episodes are never
#   truncated while failures terminate ~36% earlier.
# - We also halve instances_per_cell for noise only, keeping B/C symmetric (paired stats
#   remain valid on the same task_ids). Documented in docs/PROBES.md.
NOISE_EPISODE_LENGTH = 180
NOISE_INSTANCES_PER_CELL = 6


def _cell_timeout_s(n_tasks: int, n_episodes: int) -> int:
    """Dynamic per-cell timeout: scales with cell size (a cell = one lerobot-eval call)."""
    return max(TIMEOUT_FLOOR_S, n_tasks * n_episodes * SEC_PER_EPISODE_CAP)


def _eval_cell(condition, policy_path, family, level, task_ids, n_episodes, seed, is_plus):
    ep_len = NOISE_EPISODE_LENGTH if family == "noise" else None
    results = run_eval(
        policy_path=policy_path, suite=SUITE, task_ids=task_ids,
        n_episodes=n_episodes, seed=seed, is_libero_plus=is_plus,
        timeout_s=_cell_timeout_s(len(task_ids), n_episodes),
        episode_length=ep_len,
    )
    rows = [{"condition": condition, "task_id": r["task_id"], "family": family,
             "level": level, "seed": seed, "success": int(r["success"])} for r in results]
    n_succ = sum(r["success"] for r in results)
    print(f"[Phase4] {condition} {family} L{level}: {n_succ}/{len(results)} = "
          f"{100 * n_succ / max(len(results), 1):.1f}%")
    return rows


def _condition_plan(condition: str) -> tuple[str, bool, tuple[str, ...]]:
    """(policy_path, eval_clean, families) for a condition."""
    if condition == "A":
        # A clean+core already measured in Phase 2; here we only fill held-out layout.
        return BASE_POLICY, False, HELDOUT_FAMILIES
    return ADAPTERS[condition], True, EVAL_FAMILIES


def run_phase4(
    *, conditions: list[str], instances_per_cell: int, n_episodes: int, seed: int,
    out_csv: str | Path,
) -> Path:
    out_csv = Path(out_csv)
    done = _load_done_cells(out_csv)
    for condition in conditions:
        policy_path, eval_clean, families = _condition_plan(condition)
        if eval_clean and (condition, "clean", CLEAN_LEVEL) not in done:
            rows = _eval_cell(condition, policy_path, "clean", CLEAN_LEVEL,
                              list(CLEAN_TASK_IDS), n_episodes, seed, is_plus=False)
            _append_rows(rows, out_csv)
        for family in families:
            for level in COLLAPSE_LEVELS:
                if (condition, family, level) in done:
                    print(f"[Phase4] skip {condition} {family} L{level} (already done)")
                    continue
                n_inst = NOISE_INSTANCES_PER_CELL if family == "noise" else instances_per_cell
                task_ids = select_cell_task_ids(SUITE, family, level, n_inst)
                if not task_ids:
                    print(f"[Phase4] WARN: no tasks for {family} L{level}, skipping")
                    continue
                rows = _eval_cell(condition, policy_path, family, level, task_ids,
                                  n_episodes, seed, is_plus=True)
                _append_rows(rows, out_csv)
    print(f"[Phase4] COMPLETE -> {out_csv}")
    return out_csv


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase 4 recovery eval for conditions A(layout)/B/C.")
    ap.add_argument("--conditions", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--instances-per-cell", type=int, default=12)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/runs/phase4_recovery.csv")
    args = ap.parse_args(argv)
    run_phase4(conditions=args.conditions, instances_per_cell=args.instances_per_cell,
               n_episodes=args.n_episodes, seed=args.seed, out_csv=args.out)


if __name__ == "__main__":
    main()
