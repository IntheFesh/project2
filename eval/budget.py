"""GPU-day budget estimation for the rollout/eval matrix (rental planning).

Pure Python / numpy; off-GPU. Used to confirm the planned matrix fits the **< 5-day** GPU
cap BEFORE opening the rental (build-plan Phase 3 gate), and to size episodes/tasks once a
per-episode wall-clock has been measured by a smoke test.

Rollouts are serial on a single card, so total wall-clock ~= total_episodes * sec_per_episode
(+ training time). The headline B/C(/D) comparison uses multiple seeds; the collapse curve
(condition A) needs only one.
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
    "max_episodes_per_task",
    "summarize_episode_times",
    "format_report",
]

SECONDS_PER_DAY = 86_400.0
DEFAULT_CAP_DAYS = 5.0


@dataclass(frozen=True)
class RolloutMatrix:
    """The rollout/eval matrix dimensions for budget estimation.

    ``seeds_per_condition`` maps a condition label to its seed count, e.g.
    ``{"A": 1, "B": 3, "C": 3}`` (collapse curve A = 1 seed; comparison B/C = 3 seeds).
    Every condition is assumed to evaluate the same settings (clean + families x levels).
    """

    n_tasks: int
    n_episodes_per_task: int
    n_families: int
    n_levels: int
    seeds_per_condition: Mapping[str, int]
    include_clean: bool = True

    def __post_init__(self) -> None:
        for name in ("n_tasks", "n_episodes_per_task", "n_families", "n_levels"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not self.seeds_per_condition:
            raise ValueError("seeds_per_condition must list at least one condition")
        if any(s < 0 for s in self.seeds_per_condition.values()):
            raise ValueError("seed counts must be >= 0")

    def n_settings(self) -> int:
        """Number of eval settings per seed: clean (optional) + families x levels."""
        return (1 if self.include_clean else 0) + self.n_families * self.n_levels

    def total_seeds(self) -> int:
        return sum(self.seeds_per_condition.values())

    def episodes_per_seed(self) -> int:
        return self.n_settings() * self.n_tasks * self.n_episodes_per_task

    def total_episodes(self) -> int:
        return self.episodes_per_seed() * self.total_seeds()


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
        matrix: the rollout/eval matrix.
        sec_per_episode: measured (or assumed) serial per-episode wall-clock in seconds.
        n_train_runs: number of LoRA training runs (e.g. conditions B,C x seeds).
        train_hours_per_run: measured (or assumed) wall-clock per training run, in hours.
        cap_days: the hard GPU-rental cap (default 5 days).

    Returns:
        Dict with episode counts and eval/train/total GPU-days plus a ``fits`` bool.
    """
    if sec_per_episode < 0 or train_hours_per_run < 0 or n_train_runs < 0:
        raise ValueError("timings and run counts must be non-negative")

    total_episodes = matrix.total_episodes()
    eval_seconds = total_episodes * sec_per_episode
    train_seconds = n_train_runs * train_hours_per_run * 3600.0
    total_seconds = eval_seconds + train_seconds
    total_days = total_seconds / SECONDS_PER_DAY
    return {
        "total_episodes": total_episodes,
        "episodes_per_seed": matrix.episodes_per_seed(),
        "n_settings": matrix.n_settings(),
        "total_seeds": matrix.total_seeds(),
        "eval_days": eval_seconds / SECONDS_PER_DAY,
        "train_days": train_seconds / SECONDS_PER_DAY,
        "total_days": total_days,
        "cap_days": cap_days,
        "fits": total_days <= cap_days,
    }


def max_episodes_per_task(
    matrix: RolloutMatrix,
    *,
    sec_per_episode: float,
    n_train_runs: int = 0,
    train_hours_per_run: float = 0.0,
    cap_days: float = DEFAULT_CAP_DAYS,
) -> int:
    """Largest ``n_episodes_per_task`` (all else fixed) that still fits ``cap_days``.

    Returns 0 if even one episode per task would exceed the cap (or training alone does).
    """
    if sec_per_episode <= 0:
        raise ValueError("sec_per_episode must be > 0 to back-solve episodes")
    train_seconds = n_train_runs * train_hours_per_run * 3600.0
    eval_seconds_available = cap_days * SECONDS_PER_DAY - train_seconds
    if eval_seconds_available <= 0:
        return 0
    episodes_allowed = eval_seconds_available / sec_per_episode
    per_task_denom = matrix.total_seeds() * matrix.n_settings() * matrix.n_tasks
    if per_task_denom == 0:
        return 0
    return max(0, math.floor(episodes_allowed / per_task_denom))


def summarize_episode_times(durations: Sequence[float]) -> dict:
    """Summary stats (mean/median/p95/min/max) of per-episode wall-clock seconds.

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
        f"  episodes: {projection['total_episodes']:,} "
        f"({projection['episodes_per_seed']:,}/seed x {projection['total_seeds']} seeds, "
        f"{projection['n_settings']} settings)\n"
        f"  eval : {projection['eval_days']:.2f} GPU-days\n"
        f"  train: {projection['train_days']:.2f} GPU-days\n"
        f"  total: {projection['total_days']:.2f} / {projection['cap_days']:.1f} "
        f"GPU-days  -> {verdict}"
    )
