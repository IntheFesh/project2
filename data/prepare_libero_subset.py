"""Select LIBERO / LIBERO-Plus tasks for evaluation (task-SELECTION semantics).

Two kinds of selection:

* **Clean / original LIBERO** suites -- :func:`select_subset` picks a small deterministic subset
  of base tasks (the clean baseline; see docs/LIBERO_PLUS_NOTES.md for why clean == original).
* **Perturbed LIBERO-Plus** tasks -- parse ``task_classification.json`` and select pre-built task
  IDs per ``(family, level)`` cell (each run once; ``num_trials_per_task = 1``).

Parsing + grouping + selection are pure and unit-tested (against ``tests/fixtures/``). Only
*locating* the installed ``task_classification.json`` is a thin seam (:func:`find_task_classification`).
Disk discipline (50 GB): select a subset rather than evaluating all ~10k pre-built tasks.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from perturb.libero_plus_constants import TASK_CLASSIFICATION_RELPATH
from perturb.libero_plus_wrapper import (
    CORE_FAMILIES,
    LiberoPlusTask,
    group_by_category_level,
    parse_task_classification,
    select_perturbed_tasks,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = "configs/eval/task_subset.yaml"  # sidecar; avoids clobbering commented configs


# ----------------------------------------------------- clean / original LIBERO subset --
def select_subset(tasks_by_suite: Mapping[str, Sequence[str]], n_per_suite: int) -> list[str]:
    """Deterministically pick the first ``n_per_suite`` tasks from each (clean/original) suite.

    Order follows the suite mapping then each suite's task order, so the subset is stable across
    runs (important for reproducibility and the paired comparison). Used for the clean baseline.

    Raises:
        ValueError: if ``n_per_suite`` is negative.
    """
    if n_per_suite < 0:
        raise ValueError("n_per_suite must be >= 0")
    subset: list[str] = []
    for tasks in tasks_by_suite.values():
        subset.extend(list(tasks)[:n_per_suite])
    return subset


def write_task_subset(path: str | Path, subset: Sequence[str]) -> Path:
    """Write ``{"task_subset": [...]}`` YAML to ``path`` (creating parents). Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"task_subset": list(subset)}, sort_keys=False))
    return out


# ------------------------------------------------ LIBERO-Plus task_classification.json --
def load_task_classification(path: str | Path) -> dict:
    """Load (parse) a ``task_classification.json`` file into a dict (pure)."""
    return json.loads(Path(path).read_text())


def find_task_classification() -> Path:
    """Locate the installed ``task_classification.json`` (thin seam).

    Searches ``third_party/LIBERO-plus`` and the importable ``libero`` package. Raises
    ``FileNotFoundError`` with guidance if LIBERO-Plus is not installed (expected off-GPU).
    """
    candidates = [REPO / "third_party" / "LIBERO-plus" / TASK_CLASSIFICATION_RELPATH]
    try:  # lazy import so this module stays importable off-GPU
        import libero  # noqa: F401

        pkg_dir = Path(libero.__file__).resolve().parent  # .../libero/libero
        candidates.append(pkg_dir / "benchmark" / "task_classification.json")
    except Exception:  # noqa: BLE001 -- libero not installed; fall through to the error below
        pass
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "task_classification.json not found. Install LIBERO-Plus (see deploy.py / README 6.1) or "
        "pass an explicit --classification path. Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def enumerate_libero_tasks(
    path: str | Path | None = None,
) -> dict[tuple[str, int], list[LiberoPlusTask]]:
    """Parse ``task_classification.json`` and group tasks by ``(category, difficulty_level)``.

    ``path=None`` locates the installed JSON (seam); pass a path for off-GPU/testing use.
    """
    src = Path(path) if path is not None else find_task_classification()
    return group_by_category_level(parse_task_classification(load_task_classification(src)))


# --------------------------------------------------------------------------------- CLI --
def build_parser() -> argparse.ArgumentParser:
    """Construct the perturbed-task-selection CLI parser."""
    p = argparse.ArgumentParser(
        description="Select LIBERO-Plus perturbed task IDs per (family, level) cell."
    )
    p.add_argument("--families", nargs="*", default=list(CORE_FAMILIES),
                   help="Perturbation families to select (default: CORE).")
    p.add_argument("--levels", nargs="*", type=int, default=[2, 4],
                   help="Difficulty levels (1-5) to select per family.")
    p.add_argument("--instances-per-cell", type=int, default=20,
                   help="Task IDs to sample per (family, level) cell.")
    p.add_argument("--classification", default=None,
                   help="Path to task_classification.json (default: locate the installed copy).")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed (deterministic selection).")
    p.add_argument("--out", default=DEFAULT_OUT, help="Sidecar YAML to write the task UIDs to.")
    return p


def run(args: argparse.Namespace) -> None:
    """Select perturbed task UIDs for the requested cells and write them to the sidecar YAML."""
    src = Path(args.classification) if args.classification else find_task_classification()
    tasks = parse_task_classification(load_task_classification(src))
    uids: list[str] = []
    for family in args.families:
        for level in args.levels:
            uids.extend(
                select_perturbed_tasks(tasks, family, level, args.instances_per_cell, seed=args.seed)
            )
    out = write_task_subset(REPO / args.out, uids)
    print(
        f"selected {len(uids)} perturbed task ids "
        f"({len(args.families)} families x {len(args.levels)} levels x "
        f"<= {args.instances_per_cell}/cell) -> {out}"
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and dispatch to :func:`run`."""
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
