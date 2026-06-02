"""Project the rollout/eval GPU-day budget and gate the < 5-day rental cap.

Reads the eval + perturbation configs for the matrix dimensions; you supply the measured
per-episode wall-clock (from ``scripts/smoke_timing.py``) and per-train-run hours. Exits 0
if the matrix fits the cap, 1 otherwise -- so it can gate opening the rental in a script.

    uv run python -m scripts.estimate_budget --sec-per-episode 30 \\
        --n-train-runs 6 --train-hours-per-run 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from eval.budget import (
    DEFAULT_CAP_DAYS,
    RolloutMatrix,
    format_report,
    max_episodes_per_task,
    project_budget,
)

REPO = Path(__file__).resolve().parent.parent
FALLBACK_TASKS = 8  # lean single-card starting point when task_subset is unset


def _load_yaml(rel: str) -> dict:
    return yaml.safe_load((REPO / rel).read_text())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project GPU-day budget for the eval matrix.")
    p.add_argument("--eval-config", default="configs/eval/default.yaml")
    p.add_argument("--perturb-config", default="configs/perturb/libero_plus.yaml")
    p.add_argument("--sec-per-episode", type=float, required=True,
                   help="Measured serial per-episode wall-clock (s); use smoke_timing.py.")
    p.add_argument("--n-train-runs", type=int, default=None,
                   help="Default: (B,C[,D]) x comparison seeds.")
    p.add_argument("--train-hours-per-run", type=float, default=0.0)
    p.add_argument("--cap-days", type=float, default=DEFAULT_CAP_DAYS)
    p.add_argument("--include-d", action="store_true", help="Include STRETCH condition D.")
    # Optional overrides of the config-derived matrix dims.
    p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--n-families", type=int, default=None)
    p.add_argument("--n-levels", type=int, default=None)
    return p


def build_matrix(args: argparse.Namespace) -> tuple[RolloutMatrix, int]:
    """Build the matrix from configs (+ overrides). Returns ``(matrix, default_n_train_runs)``."""
    eval_cfg = _load_yaml(args.eval_config)
    perturb_cfg = _load_yaml(args.perturb_config)

    families = args.n_families if args.n_families is not None else len(
        perturb_cfg.get("families_core", []))
    levels = args.n_levels if args.n_levels is not None else len(
        perturb_cfg.get("start_levels", []))
    subset = eval_cfg.get("task_subset") or []
    n_tasks = args.n_tasks if args.n_tasks is not None else (len(subset) or FALLBACK_TASKS)
    n_eps = args.n_episodes if args.n_episodes is not None else eval_cfg.get("n_episodes", 20)
    seeds_cmp = len(eval_cfg.get("seeds_comparison", [0, 1, 2]))
    seeds_col = len(eval_cfg.get("seeds_collapse", [0]))

    seeds = {"A": seeds_col, "B": seeds_cmp, "C": seeds_cmp}
    if args.include_d:
        seeds["D"] = seeds_cmp
    default_train_runs = (3 if args.include_d else 2) * seeds_cmp  # B,C[,D] x seeds

    matrix = RolloutMatrix(
        n_tasks=n_tasks, n_episodes_per_task=n_eps,
        n_families=families, n_levels=levels, seeds_per_condition=seeds,
    )
    return matrix, default_train_runs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix, default_train_runs = build_matrix(args)
    n_train_runs = args.n_train_runs if args.n_train_runs is not None else default_train_runs

    proj = project_budget(
        matrix, sec_per_episode=args.sec_per_episode,
        n_train_runs=n_train_runs, train_hours_per_run=args.train_hours_per_run,
        cap_days=args.cap_days,
    )
    print(f"matrix: tasks={matrix.n_tasks} eps/task={matrix.n_episodes_per_task} "
          f"families={matrix.n_families} levels={matrix.n_levels} "
          f"seeds={dict(matrix.seeds_per_condition)} train_runs={n_train_runs}")
    print(format_report(proj))

    if not proj["fits"]:
        n_max = max_episodes_per_task(
            matrix, sec_per_episode=args.sec_per_episode,
            n_train_runs=n_train_runs, train_hours_per_run=args.train_hours_per_run,
            cap_days=args.cap_days,
        )
        print(f"  -> to fit, drop to <= {n_max} episodes/task (or cut tasks/levels/seeds).")
    return 0 if proj["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())
