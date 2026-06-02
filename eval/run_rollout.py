"""Rollout evaluation entry point (Phase 1+).

Loads a policy (default: pretrained SmolVLA-LIBERO), runs rollouts on a LIBERO task
subset under an optional LIBERO-Plus perturbation, and writes a small success-rate
summary (CSV/JSON) to ``analysis/``.

GPU-only at runtime (LeRobot + simulator). The CLI surface is defined here so configs and
downstream scripts can be wired off-GPU; the rollout loop itself lands in Phase 1.

Example (on the rented RTX 5090)::

    python eval/run_rollout.py \\
        --model-config configs/model/smolvla.yaml \\
        --eval-config configs/eval/default.yaml \\
        --tasks libero_spatial_0 libero_spatial_1 \\
        --n-episodes 20 --seed 0 --perturb viewpoint:4 --workers 1
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Construct the rollout CLI parser."""
    p = argparse.ArgumentParser(description="VLA-Collapse-Recover rollout evaluation.")
    p.add_argument("--model-config", default="configs/model/smolvla.yaml",
                   help="Model YAML (default: SmolVLA base).")
    p.add_argument("--eval-config", default="configs/eval/default.yaml",
                   help="Evaluation YAML (task subset, episodes, seeds).")
    p.add_argument("--adapter", default=None,
                   help="Optional LoRA adapter dir (conditions B/C/D). None = base (A).")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Explicit task ids; overrides the eval config's task_subset.")
    p.add_argument("--n-episodes", type=int, default=None,
                   help="Episodes per task (overrides eval config).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (fixed init states).")
    p.add_argument("--perturb", default=None,
                   help="LIBERO-Plus perturbation as 'family:level' (e.g. 'viewpoint:4'); "
                        "omit for clean.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel rollout workers (single GPU now; reserved for a 2nd card).")
    p.add_argument("--out", default=None,
                   help="Output summary path (CSV/JSON). Default: analysis/runs/<auto>.")
    return p


def run(args: argparse.Namespace) -> None:
    """Execute the rollout evaluation and write the success-rate summary."""
    raise NotImplementedError(
        "Phase 1: load policy via LeRobot, run rollouts on the LIBERO subset, "
        "compute success rate, and write the CSV/JSON summary."
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and dispatch to :func:`run`."""
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
