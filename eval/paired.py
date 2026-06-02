"""Paired comparison at fixed init states.

Phase 5 (the differentiator). Conditions are evaluated on *identical* init states, so
each episode is matched across conditions. The paired analysis is what makes the small
(~5-10pp) B-vs-C method gap statistically detectable. Off-GPU; numpy/scipy.
"""

from __future__ import annotations

from collections.abc import Sequence


def mcnemar(a_success: Sequence[bool | int], b_success: Sequence[bool | int]) -> dict:
    """McNemar's test on matched per-episode successes for two conditions.

    Args:
        a_success: per-episode success for condition A (e.g. C), at fixed init states.
        b_success: per-episode success for condition B, *same* init states / order.

    Returns:
        Dict with discordant counts (``b01``, ``b10``), the statistic, and a p-value.
    """
    raise NotImplementedError("Phase 5: implement McNemar's test on matched episodes.")


def paired_bootstrap_delta(
    a_success: Sequence[bool | int],
    b_success: Sequence[bool | int],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for ``SR_a - SR_b`` (Delta_method), resampling matched episodes.

    Returns ``(delta, lo, hi)`` for the paired success-rate difference.
    """
    raise NotImplementedError("Phase 5: implement paired bootstrap over matched episodes.")
