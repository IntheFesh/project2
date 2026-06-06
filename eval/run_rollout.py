"""Rollout evaluation: run each selected LIBERO-Plus task ONCE and record success (Phase 1+).

Task-SELECTION semantics: the eval set is a list of pre-built task IDs (clean + perturbed cells);
each is run a single trial (``num_trials_per_task = 1``). Output is a per-trial CSV with the exact
schema consumed by ``scripts/analyze_results.py``::

    condition, task_id, family, level, seed, success

Pure logic (eval-plan construction, row/aggregate building, CSV/JSON writing) is implemented and
unit-tested here. Loading the policy and stepping the simulator are the GPU seams
(``_load_policy`` / ``_run_one_trial``), which mirror LIBERO-Plus ``benchmark_scripts/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from eval.metrics import success_rate
from perturb.libero_plus_wrapper import LiberoPlusTask, select_perturbed_tasks

REPO = Path(__file__).resolve().parent.parent

#: Per-trial CSV columns (exact schema consumed by scripts/analyze_results.py).
ROLLOUT_FIELDS: tuple[str, ...] = ("condition", "task_id", "family", "level", "seed", "success")

CLEAN_FAMILY = "clean"
CLEAN_LEVEL = 0


@dataclass(frozen=True)
class EvalItem:
    """One unit of evaluation: a task UID labelled with its family and difficulty level."""

    task_uid: str
    family: str
    level: int


def build_eval_plan(
    tasks: Sequence[LiberoPlusTask],
    families: Sequence[str],
    levels: Sequence[int],
    instances_per_cell: int,
    clean_uids: Sequence[str] = (),
    *,
    seed: int = 0,
) -> list[EvalItem]:
    """Build the (clean + perturbed) list of :class:`EvalItem`s to roll out (pure, deterministic).

    Clean UIDs (original LIBERO tasks) are labelled ``family="clean", level=0``; perturbed task
    UIDs are selected per ``(family, level)`` cell via :func:`select_perturbed_tasks`.
    """
    plan: list[EvalItem] = [EvalItem(uid, CLEAN_FAMILY, CLEAN_LEVEL) for uid in clean_uids]
    for family in families:
        for level in levels:
            for uid in select_perturbed_tasks(tasks, family, level, instances_per_cell, seed=seed):
                plan.append(EvalItem(uid, family, level))
    return plan


def make_row(
    condition: str, task_id: str, family: str, level: int, seed: int, success: bool | int
) -> dict:
    """Construct one per-trial CSV row dict (validated)."""
    if success not in (0, 1, True, False):
        raise ValueError(f"success must be 0/1/bool, got {success!r}")
    return {
        "condition": str(condition),
        "task_id": str(task_id),
        "family": str(family),
        "level": int(level),
        "seed": int(seed),
        "success": int(bool(success)),
    }


def aggregate_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Aggregate per-trial rows to success rate per ``(condition, family, level)`` (pure)."""
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for r in rows:
        groups[(str(r["condition"]), str(r["family"]), int(r["level"]))].append(int(r["success"]))
    out: list[dict] = []
    for (condition, family, level), successes in groups.items():
        out.append({
            "condition": condition, "family": family, "level": level,
            "n": len(successes), "n_success": int(sum(successes)),
            "success_rate": success_rate(successes),
        })
    return out


def write_rows_csv(rows: Sequence[Mapping], path: str | Path) -> Path:
    """Write per-trial rows to a CSV with the canonical :data:`ROLLOUT_FIELDS` header."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROLLOUT_FIELDS))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in ROLLOUT_FIELDS})
    return out


def write_summary(rows: Sequence[Mapping], out_path: str | Path) -> tuple[Path, Path]:
    """Write the per-trial CSV (``out_path``) + an aggregate JSON sidecar. Returns both paths."""
    csv_path = write_rows_csv(rows, out_path)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps({"n_rows": len(rows), "aggregate": aggregate_rows(rows)}, indent=2))
    return csv_path, json_path


# ----------------------------------------------------------------- GPU seams --
def _load_policy(model_config: str, adapter: str | None):
    """Load the policy (default SmolVLA base; optional LoRA adapter for B/C/D). GPU seam."""
    raise NotImplementedError(
        "Phase 1: load the policy via LeRobot (base SmolVLA; optional LoRA adapter)."
    )


def _run_one_trial(policy, task_uid: str, seed: int) -> int:
    """Run one trial for ``task_uid`` and return success (0/1). GPU/simulator seam.

    Mirrors LIBERO-Plus benchmark_scripts: build the env via
    :func:`perturb.libero_plus_wrapper.make_perturbed_env`, roll out the policy, return success.
    """
    raise NotImplementedError(
        f"Phase 1: roll out {task_uid!r} once (make_perturbed_env -> step policy -> success)."
    )


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    """Compose plan -> per-task trial -> CSV/JSON summary (only the trial itself is a GPU seam)."""
    if args.tasks:
        plan = [EvalItem(uid, "unknown", 0) for uid in args.tasks]
    else:
        from data.prepare_libero_subset import (  # local import: avoids loading at module scope
            find_task_classification,
            load_task_classification,
        )
        from perturb.libero_plus_wrapper import parse_task_classification

        src = Path(args.classification) if args.classification else find_task_classification()
        tasks = parse_task_classification(load_task_classification(src))
        plan = build_eval_plan(
            tasks, args.families, args.levels, args.instances_per_cell, args.clean_tasks or [],
            seed=args.seed,
        )

    out = Path(args.out) if args.out else REPO / "analysis" / "runs" / f"{args.condition}_seed{args.seed}.csv"
    print(f"eval plan: {len(plan)} task trials (condition {args.condition}, seed {args.seed})")

    policy = _load_policy(args.model_config, args.adapter)  # GPU seam (raises until Phase 1)
    rows = [
        make_row(args.condition, item.task_uid, item.family, item.level, args.seed,
                 _run_one_trial(policy, item.task_uid, args.seed))
        for item in plan
    ]
    return write_summary(rows, out)


def build_parser() -> argparse.ArgumentParser:
    """Construct the rollout CLI parser."""
    p = argparse.ArgumentParser(description="VLA-Collapse-Recover rollout evaluation (1 trial/task).")
    p.add_argument("--model-config", default="configs/model/smolvla.yaml")
    p.add_argument("--eval-config", default="configs/eval/default.yaml")
    p.add_argument("--condition", default="A", help="Condition label for the CSV (A/B/C/D).")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (B/C/D); None = base (A).")
    p.add_argument("--tasks", nargs="*", default=None, help="Explicit task UIDs (ad-hoc run).")
    p.add_argument("--classification", default=None, help="Path to task_classification.json.")
    p.add_argument("--families", nargs="*", default=["viewpoint", "lighting", "texture", "noise"])
    p.add_argument("--levels", nargs="*", type=int, default=[2, 4])
    p.add_argument("--instances-per-cell", type=int, default=20)
    p.add_argument("--clean-tasks", nargs="*", default=None, help="Clean/original task UIDs.")
    p.add_argument("--seed", type=int, default=0, help="Seed (a LoRA-training seed for B/C/D).")
    p.add_argument("--out", default=None, help="Output CSV path (default: analysis/runs/<auto>.csv).")
    return p


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and dispatch to :func:`run`."""
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
