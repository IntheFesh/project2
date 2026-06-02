"""Prepare a small LIBERO task subset for evaluation and training (Phase 1-2).

Disk discipline (50 GB): select an ~8-10 task subset across LIBERO Spatial/Object/Goal/Long
and a LIBERO-Plus subset (full HF assets ~6.4 GB) rather than downloading everything.

The selection + YAML writing are pure and tested now. Enumerating the available LIBERO tasks
needs LIBERO installed, so that is the Phase 1-2 seam; ``run()`` composes
enumerate -> select -> write.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = "configs/eval/task_subset.yaml"  # sidecar; avoids clobbering commented configs


def select_subset(tasks_by_suite: Mapping[str, Sequence[str]], n_per_suite: int) -> list[str]:
    """Deterministically pick the first ``n_per_suite`` tasks from each suite.

    Order follows the suite mapping then each suite's task order, so the subset is stable
    across runs (important for reproducibility and paired comparisons).

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


def enumerate_libero_tasks(suites: Sequence[str]) -> dict[str, list[str]]:
    """Return ``{suite: [task_id, ...]}`` from the installed LIBERO (Phase 1-2 seam)."""
    raise NotImplementedError(
        "Phase 1-2: enumerate tasks per suite from the installed LIBERO benchmark."
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the subset-preparation CLI parser."""
    p = argparse.ArgumentParser(description="Select an ~8-10 task LIBERO subset.")
    p.add_argument("--suites", nargs="*",
                   default=["libero_spatial", "libero_object", "libero_goal", "libero_long"])
    p.add_argument("--n-per-suite", type=int, default=2, help="Tasks per suite (~8-10 total).")
    p.add_argument("--out", default=DEFAULT_OUT, help="Sidecar YAML to write task_subset to.")
    p.add_argument("--download", action="store_true",
                   help="Download the (subset of) benchmark assets. Prompts before big pulls.")
    return p


def run(args: argparse.Namespace) -> None:
    """Enumerate tasks, select the subset, and write it out."""
    tasks_by_suite = enumerate_libero_tasks(args.suites)
    subset = select_subset(tasks_by_suite, args.n_per_suite)
    out = write_task_subset(REPO / args.out, subset)
    print(f"wrote {len(subset)} tasks to {out}")


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and dispatch to :func:`run`."""
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
