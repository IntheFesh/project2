"""Measure per-episode wall-clock on a tiny rollout, then project the full GPU-day budget.

Rollouts are the time bottleneck on a single serial card, so we ALWAYS smoke-test
per-episode timing before scaling (and before opening the full rental). The live rollout is
Phase 1 (GPU); this harness wraps the timing + budget projection so that, on the rented
card, a 1-2 task / few-episode run immediately answers "does the matrix fit < 5 days?".

Off-GPU you can still do what-if planning by supplying measured/assumed durations directly::

    uv run python -m scripts.smoke_timing --durations 28 31 30 29 33 \\
        --train-hours-per-run 4

On the rented card (once Phase 1 lands), omit ``--durations`` to time a real smoke rollout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.budget import (
    DEFAULT_CAP_DAYS,
    format_report,
    project_budget,
    summarize_episode_times,
)
from scripts.estimate_budget import add_matrix_args, build_matrix

REPO = Path(__file__).resolve().parent.parent


def _time_smoke_rollout(n_tasks: int, n_episodes: int, perturb: str | None) -> list[float]:
    """Run a tiny rollout and return per-episode wall-clock seconds (Phase 1 seam, GPU)."""
    raise NotImplementedError(
        "Phase 1: run a tiny rollout under timing and collect per-episode durations. "
        "Off-GPU, pass --durations to do what-if budget planning instead."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smoke-time rollouts and project the GPU budget.")
    p.add_argument("--durations", type=float, nargs="*", default=None,
                   help="Per-episode seconds (what-if/off-GPU). Omit to time a live rollout.")
    p.add_argument("--smoke-tasks", type=int, default=2)
    p.add_argument("--smoke-episodes", type=int, default=5)
    p.add_argument("--perturb", default=None, help="'family:level' for the smoke rollout.")
    p.add_argument("--n-train-runs", type=int, default=None)
    p.add_argument("--train-hours-per-run", type=float, default=0.0)
    p.add_argument("--cap-days", type=float, default=DEFAULT_CAP_DAYS)
    p.add_argument("--out", default="analysis/runs/smoke_timing.json")
    add_matrix_args(p)  # --eval-config, --instances-per-cell, --clean-instances, families/levels, -d
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.durations:
        durations = list(args.durations)
        source = "supplied (what-if)"
    else:
        durations = _time_smoke_rollout(args.smoke_tasks, args.smoke_episodes, args.perturb)
        source = "measured (live smoke rollout)"

    times = summarize_episode_times(durations)
    # p95 is a safe per-episode estimate for a serial-card budget.
    sec_per_episode = times["p95"]

    matrix, default_train_runs = build_matrix(args)
    n_train_runs = args.n_train_runs if args.n_train_runs is not None else default_train_runs

    proj = project_budget(
        matrix, sec_per_episode=sec_per_episode,
        n_train_runs=n_train_runs, train_hours_per_run=args.train_hours_per_run,
        cap_days=args.cap_days,
    )

    print(f"per-episode timing [{source}]: {times}")
    print(f"using sec_per_episode = p95 = {sec_per_episode:.2f}s")
    print(format_report(proj))

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"timing": times, "projection": proj}, indent=2))
    print(f"wrote {out_path}")
    return 0 if proj["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())
