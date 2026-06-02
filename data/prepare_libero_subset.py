"""Prepare a small LIBERO task subset for evaluation and training (Phase 1-2).

Disk discipline (50 GB): select an ~8-10 task subset across LIBERO Spatial/Object/Goal/Long
and a LIBERO-Plus subset (full HF assets ~6.4 GB) rather than downloading everything.
Writes the chosen task ids into the eval config's ``task_subset`` (or a sidecar file).

GPU not required; benchmark assets are large, so this prompts before big downloads.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Construct the subset-preparation CLI parser."""
    p = argparse.ArgumentParser(description="Select an ~8-10 task LIBERO subset.")
    p.add_argument("--suites", nargs="*",
                   default=["libero_spatial", "libero_object", "libero_goal", "libero_long"])
    p.add_argument("--n-per-suite", type=int, default=2, help="Tasks per suite (~8-10 total).")
    p.add_argument("--out", default="configs/eval/default.yaml",
                   help="Eval config to update with the selected task_subset.")
    p.add_argument("--download", action="store_true",
                   help="Download the (subset of) benchmark assets. Prompts before big pulls.")
    return p


def run(args: argparse.Namespace) -> None:
    """Select tasks and (optionally) fetch the subset of benchmark assets."""
    raise NotImplementedError(
        "Phase 1-2: enumerate LIBERO tasks, pick the subset, optionally fetch LIBERO/-Plus assets."
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and dispatch to :func:`run`."""
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
