"""Project the rollout/eval GPU-day budget and gate the < 5-day rental cap.

Reads the eval config for the matrix dimensions (task-SELECTION semantics: instances per
(family, level) cell + clean); you supply the measured per-task-trial wall-clock (from
``scripts/smoke_timing.py``) and per-train-run hours. Exits 0 if the matrix fits the cap, 1
otherwise -- so it can gate opening the rental in a script.

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
    max_instances_per_cell,
    project_budget,
)

REPO = Path(__file__).resolve().parent.parent


def _load_yaml(rel: str) -> dict:
    return yaml.safe_load((REPO / rel).read_text())


def add_matrix_args(p: argparse.ArgumentParser) -> None:
    """Matrix-dimension args shared by estimate_budget and smoke_timing (read by build_matrix)."""
    p.add_argument("--eval-config", default="configs/eval/default.yaml")
    p.add_argument("--include-d", action="store_true", help="Include STRETCH condition D.")
    p.add_argument("--instances-per-cell", type=int, default=None,
                   help="Override eval config instances_per_cell.")
    p.add_argument("--clean-instances", type=int, default=None,
                   help="Override eval config clean_instances.")
    p.add_argument("--n-families", type=int, default=None,
                   help="Override #families (default: trained_families + held_out_families).")
    p.add_argument("--n-levels", type=int, default=None,
                   help="Override #levels (default: len(levels)).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project GPU-day budget for the eval matrix.")
    p.add_argument("--sec-per-episode", type=float, required=True,
                   help="Measured serial per-task-trial wall-clock (s); use smoke_timing.py.")
    p.add_argument("--n-train-runs", type=int, default=None,
                   help="Default: (B,C[,D]) x comparison seeds.")
    p.add_argument("--train-hours-per-run", type=float, default=0.0)
    p.add_argument("--cap-days", type=float, default=DEFAULT_CAP_DAYS)
    add_matrix_args(p)
    return p


def build_matrix(args: argparse.Namespace) -> tuple[RolloutMatrix, int]:
    """Build the matrix from the eval config (+ overrides). Returns ``(matrix, default_train_runs)``."""
    eval_cfg = _load_yaml(args.eval_config)
    trained = eval_cfg.get("trained_families") or []
    held_out = eval_cfg.get("held_out_families") or []
    levels = eval_cfg.get("levels") or [2, 4]

    n_families = args.n_families if args.n_families is not None else (len(trained) + len(held_out))
    n_levels = args.n_levels if args.n_levels is not None else len(levels)
    instances = (args.instances_per_cell if args.instances_per_cell is not None
                 else eval_cfg.get("instances_per_cell", 20))
    clean = (args.clean_instances if args.clean_instances is not None
             else eval_cfg.get("clean_instances", 0))
    seeds_cmp = len(eval_cfg.get("seeds_comparison", [0, 1, 2]))
    seeds_col = len(eval_cfg.get("seeds_collapse", [0]))

    seeds = {"A": seeds_col, "B": seeds_cmp, "C": seeds_cmp}
    if args.include_d:
        seeds["D"] = seeds_cmp
    default_train_runs = (3 if args.include_d else 2) * seeds_cmp  # B,C[,D] x seeds

    # Language-conditioning probe adds extra instruction passes on the base model (Workstream 6).
    probe = eval_cfg.get("language_probe") or {}
    extra_units = int(probe.get("instances", 0)) * int(probe.get("variants", 0)) \
        if probe.get("enabled") else 0

    matrix = RolloutMatrix(
        instances_per_cell=instances, n_families=max(1, n_families),
        n_levels=max(1, n_levels), seeds_per_condition=seeds, clean_instances=clean,
        extra_units=extra_units,
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
    print(f"matrix: instances/cell={matrix.instances_per_cell} clean={matrix.clean_instances} "
          f"families={matrix.n_families} levels={matrix.n_levels} cells={matrix.n_cells()} "
          f"seeds={dict(matrix.seeds_per_condition)} probe_units={matrix.extra_units} "
          f"train_runs={n_train_runs}")
    print(format_report(proj))

    if not proj["fits"]:
        n_max = max_instances_per_cell(
            matrix, sec_per_episode=args.sec_per_episode,
            n_train_runs=n_train_runs, train_hours_per_run=args.train_hours_per_run,
            cap_days=args.cap_days,
        )
        print(f"  -> to fit, drop to <= {n_max} instances/cell (or cut families/levels/seeds).")
    return 0 if proj["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())
