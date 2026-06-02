"""Bootstrap confidence intervals for success rates.

Phase 5 (the differentiator). Off-GPU; pure numpy. Implements >= 10,000-resample
percentile 95% CIs over per-episode success outcomes (BCa is a TODO).
"""

from __future__ import annotations

from collections.abc import Sequence


def bootstrap_ci(
    outcomes: Sequence[bool | int | float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(point_estimate, lo, hi)`` for the success rate at ``1 - alpha`` coverage.

    Args:
        outcomes: per-episode 0/1 (or bool) success outcomes.
        n_resamples: number of bootstrap resamples (>= 10,000 for the final tables).
        alpha: two-sided significance level (0.05 -> 95% CI).
        seed: RNG seed for reproducibility.
    """
    raise NotImplementedError("Phase 5: implement >=10k-resample percentile bootstrap CI.")
