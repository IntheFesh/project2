"""Bootstrap confidence intervals for success rates.

Phase 5 (the differentiator). Off-GPU; numpy only. Percentile bootstrap 95% CIs over
per-episode success outcomes (>= 10,000 resamples for the final tables).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["bootstrap_ci"]


def bootstrap_ci(
    outcomes: Sequence[bool | int | float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the success rate.

    Args:
        outcomes: per-episode 0/1 (or bool) success outcomes.
        n_resamples: number of bootstrap resamples (>= 10,000 for the final tables).
        alpha: two-sided significance level (0.05 -> 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        ``(point_estimate, lo, hi)`` where ``point_estimate`` is the observed success rate
        and ``[lo, hi]`` is the ``1 - alpha`` percentile interval.

    Raises:
        ValueError: if ``outcomes`` is empty, ``n_resamples`` < 1, or ``alpha`` not in (0, 1).
    """
    x = np.asarray(outcomes, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("outcomes must be a non-empty 1-D sequence of 0/1 outcomes")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    point = float(x.mean())
    n = x.size
    rng = np.random.default_rng(seed)
    # Vectorised resampling: (n_resamples, n) indices drawn with replacement.
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return point, lo, hi
