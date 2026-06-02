"""Side-by-side before/after demo recorder (Phase 6).

Records a few short rollout clips: the base policy failing under a moved camera / changed
lighting next to the robust (condition C) policy succeeding. Disk discipline: only a
*handful* of curated clips are saved (to ``analysis/demos/``), never bulk rollout video.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Construct the demo-recording CLI parser."""
    p = argparse.ArgumentParser(description="Record side-by-side base-vs-robust demo clips.")
    p.add_argument("--model-config", default="configs/model/smolvla.yaml")
    p.add_argument("--robust-adapter", required=False,
                   help="LoRA adapter for the robust (condition C) policy.")
    p.add_argument("--task", required=False, help="LIBERO task id to demo.")
    p.add_argument("--perturb", default="viewpoint:4", help="'family:level' for the demo.")
    p.add_argument("--n-clips", type=int, default=2, help="How many clips to save (keep small).")
    p.add_argument("--out-dir", default="analysis/demos")
    return p


def run(args: argparse.Namespace) -> None:
    """Record and save the side-by-side demo clips."""
    raise NotImplementedError(
        "Phase 6: roll out base vs robust policy under the perturbation and save a few clips."
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and dispatch to :func:`run`."""
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
