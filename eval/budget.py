"""GPU-day budget estimation for the LIBERO-Plus task-instance eval matrix (rental planning).

Pure Python / numpy; off-GPU. Confirms the matrix fits the **< 5-day** GPU cap BEFORE renting,
and sizes ``instances_per_cell`` once a per-task-trial wall-clock has been measured by a smoke test.

**Task-SELECTION semantics.** The eval unit is a pre-built LIBERO-Plus task instance run **once**
(``num_trials_per_task = 1``). Per ``(category, level)`` cell we evaluate ``instances_per_cell``
task IDs; conditions A/B/C are run on the **same** task-ID set (paired by ``task_id``). So::

    total_units = total_seeds * (clean_instances + n_cells * instances_per_cell)
    n_cells     = n_families * n_levels

``sec_per_episode`` is the serial wall-clock per task trial (one unit). A "seed" for B/C is an
independent LoRA training run evaluated on the same task-ID set (training-init variance); the
collapse curve (A) needs only one seed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SECONDS_PER_DAY",
    "DEFAULT_CAP_DAYS",
    "RolloutMatrix",
    "project_budget",
    "max_instances_per_cell",
    "summarize_episode_times",
    "format_report",
]

SECONDS_PER_DAY = 86_400.0
DEFAULT_CAP_DAYS = 5.0


@dataclass(frozen=True)
class RolloutMatrix:
    """The task-instance eval matrix for budget estimation.

    Args:
        instances_per_cell: pre-built task IDs sampled per ``(category, level)`` cell.
        n_families: number of perturbation families evaluated (trained + held-out).
        n_levels: number of difficulty levels per family.
        seeds_per_condition: ``{condition: seed_count}``, e.g. ``{"A": 1, "B": 3, "C": 3}``
            (A = collapse curve, 1 seed; B/C = comparison, 3 LoRA-training seeds each).
        clean_instances: clean/original LIBERO task instances evaluated per seed (0 = skip clean).
        extra_units: flat additional eval units not tied to the per-seed matrix -- e.g. the
            language-conditioning probe's extra instruction passes over the base model (Workstream 6).
    """

    instances_per_cell: int
    n_families: int
    n_levels: int
    seeds_per_condition: Mapping[str, int]
    clean_instances: int = 0
    extra_units: int = 0

    def __post_init__(self) -> None:
        for name in ("instances_per_cell", "n_families", "n_levels", "clean_instances",
                     "extra_units"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not self.seeds_per_condition:
            raise ValueError("seeds_per_condition must list at least one condition")
        if any(s < 0 for s in self.seeds_per_condition.values()):
            raise ValueError("seed counts must be >= 0")

    def n_cells(self) -> int:
        """Number of ``(family, level)`` cells = ``n_families * n_levels``."""
        return self.n_families * self.n_levels

    def total_seeds(self) -> int:
        return sum(self.seeds_per_condition.values())

    def units_per_seed(self) -> int:
        """Eval units (task trials) per seed: clean + perturbed cells."""
        return self.clean_instances + self.n_cells() * self.instances_per_cell

    def total_units(self) -> int:
        return self.units_per_seed() * self.total_seeds() + self.extra_units


def project_budget(
    matrix: RolloutMatrix,
    *,
    sec_per_episode: float,
    n_train_runs: int = 0,
    train_hours_per_run: float = 0.0,
    cap_days: float = DEFAULT_CAP_DAYS,
) -> dict:
    """Project total GPU-days for ``matrix`` and whether it fits ``cap_days``.

    Args:
        matrix: the task-instance eval matrix.
        sec_per_episode: measured (or assumed) serial wall-clock per task trial, in seconds.
        n_train_runs: number of LoRA training runs (e.g. conditions B,C x seeds).
        train_hours_per_run: measured (or assumed) wall-clock per training run, in hours.
        cap_days: the hard GPU-rental cap (default 5 days).

    Returns:
        Dict with unit counts and eval/train/total GPU-days plus a ``fits`` bool.
    """
    if sec_per_episode < 0 or train_hours_per_run < 0 or n_train_runs < 0:
        raise ValueError("timings and run counts must be non-negative")

    total_units = matrix.total_units()
    eval_seconds = total_units * sec_per_episode
    train_seconds = n_train_runs * train_hours_per_run * 3600.0
    total_days = (eval_seconds + train_seconds) / SECONDS_PER_DAY
    return {
        "total_units": total_units,
        "units_per_seed": matrix.units_per_seed(),
        "n_cells": matrix.n_cells(),
        "total_seeds": matrix.total_seeds(),
        "eval_days": eval_seconds / SECONDS_PER_DAY,
        "train_days": train_seconds / SECONDS_PER_DAY,
        "total_days": total_days,
        "cap_days": cap_days,
        "fits": total_days <= cap_days,
    }


def max_instances_per_cell(
    matrix: RolloutMatrix,
    *,
    sec_per_episode: float,
    n_train_runs: int = 0,
    train_hours_per_run: float = 0.0,
    cap_days: float = DEFAULT_CAP_DAYS,
) -> int:
    """Largest ``instances_per_cell`` (all else fixed) that still fits ``cap_days``.

    Accounts for ``clean_instances`` and the seed/cell multipliers. Returns 0 if even one
    instance per cell would exceed the cap (or training alone does).
    """
    if sec_per_episode <= 0:
        raise ValueError("sec_per_episode must be > 0 to back-solve instances")
    train_seconds = n_train_runs * train_hours_per_run * 3600.0
    eval_seconds_available = cap_days * SECONDS_PER_DAY - train_seconds
    if eval_seconds_available <= 0:
        return 0
    units_allowed = eval_seconds_available / sec_per_episode
    if matrix.total_seeds() == 0 or matrix.n_cells() == 0:
        return 0
    per_seed_allowed = units_allowed / matrix.total_seeds()
    x = (per_seed_allowed - matrix.clean_instances) / matrix.n_cells()
    return max(0, math.floor(x))


def summarize_episode_times(durations: Sequence[float]) -> dict:
    """Summary stats (mean/median/p95/min/max) of per-task-trial wall-clock seconds.

    Used by the smoke-timing harness to feed :func:`project_budget` a robust
    ``sec_per_episode`` (p95 is a safe choice for a serial-card budget).
    """
    arr = np.asarray(list(durations), dtype=float)
    if arr.size == 0:
        raise ValueError("need at least one episode duration")
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def format_report(projection: Mapping[str, float]) -> str:
    """Human-readable one-block summary of a :func:`project_budget` result."""
    verdict = "FITS" if projection["fits"] else "EXCEEDS"
    return (
        f"Rollout/eval budget projection\n"
        f"  units: {projection['total_units']:,} "
        f"({projection['units_per_seed']:,}/seed x {projection['total_seeds']} seeds, "
        f"{projection['n_cells']} cells)\n"
        f"  eval : {projection['eval_days']:.2f} GPU-days\n"
        f"  train: {projection['train_days']:.2f} GPU-days\n"
        f"  total: {projection['total_days']:.2f} / {projection['cap_days']:.1f} "
        f"GPU-days  -> {verdict}"
    )
