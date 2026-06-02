"""Paired comparison at fixed init states.

Phase 5 (the differentiator). Conditions are evaluated on *identical* init states, so each
episode is matched across conditions. The paired analysis is what makes the small (~5-10pp)
B-vs-C method gap statistically detectable. Off-GPU; numpy/scipy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

__all__ = ["mcnemar", "paired_bootstrap_delta"]


def _as_matched_pair(
    a_success: Sequence[bool | int], b_success: Sequence[bool | int]
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a_success, dtype=int)
    b = np.asarray(b_success, dtype=int)
    if a.shape != b.shape:
        raise ValueError("a_success and b_success must be matched (same length / order)")
    if a.size == 0:
        raise ValueError("need at least one matched episode")
    return a, b


def mcnemar(
    a_success: Sequence[bool | int],
    b_success: Sequence[bool | int],
    *,
    exact_threshold: int = 25,
) -> dict:
    """McNemar's test on matched per-episode successes for two conditions.

    Uses the exact binomial test on the discordant pairs when there are few of them
    (``< exact_threshold``), otherwise the continuity-corrected chi-square statistic.

    Args:
        a_success: per-episode success for condition A (e.g. C), at fixed init states.
        b_success: per-episode success for condition B, on the *same* init states / order.
        exact_threshold: discordant-pair count below which the exact binomial test is used.

    Returns:
        Dict with ``a_only`` (#A-success, B-fail), ``b_only`` (#A-fail, B-success),
        ``n_discordant``, ``statistic``, ``pvalue`` and ``method``.
    """
    a, b = _as_matched_pair(a_success, b_success)
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    n_discordant = a_only + b_only

    if n_discordant == 0:
        return {
            "a_only": a_only, "b_only": b_only, "n_discordant": 0,
            "statistic": 0.0, "pvalue": 1.0, "method": "none (no discordant pairs)",
        }
    if n_discordant < exact_threshold:
        res = stats.binomtest(min(a_only, b_only), n_discordant, 0.5, alternative="two-sided")
        return {
            "a_only": a_only, "b_only": b_only, "n_discordant": n_discordant,
            "statistic": float(min(a_only, b_only)), "pvalue": float(res.pvalue),
            "method": "exact binomial",
        }
    statistic = (abs(a_only - b_only) - 1.0) ** 2 / n_discordant
    pvalue = float(stats.chi2.sf(statistic, df=1))
    return {
        "a_only": a_only, "b_only": b_only, "n_discordant": n_discordant,
        "statistic": float(statistic), "pvalue": pvalue,
        "method": "chi2 (continuity-corrected)",
    }


def paired_bootstrap_delta(
    a_success: Sequence[bool | int],
    b_success: Sequence[bool | int],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for ``SR_a - SR_b`` (Delta_method), resampling matched episodes.

    The same resampled episode indices are applied to both conditions, preserving the
    pairing induced by fixed init states.

    Returns:
        ``(delta, lo, hi)`` for the paired success-rate difference at ``1 - alpha`` coverage.
    """
    a, b = _as_matched_pair(a_success, b_success)
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    a = a.astype(float)
    b = b.astype(float)
    delta = float(a.mean() - b.mean())
    n = a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.quantile(diffs, alpha / 2.0))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    return delta, lo, hi
