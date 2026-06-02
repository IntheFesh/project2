"""Holm-Bonferroni multiple-comparison correction.

Phase 5 (the differentiator). We test the method gap across several perturbation families,
so family-level p-values are corrected for multiplicity. Off-GPU; pure Python.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["holm_bonferroni"]


def holm_bonferroni(
    pvalues: Sequence[float] | Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> list[tuple]:
    """Apply the Holm-Bonferroni step-down correction.

    Adjusted p-values use the standard step-down form
    ``p_adj(k) = max_{j<=k} min(1, (m - j + 1) * p(j))`` over the ascending sort, and a
    hypothesis is rejected iff its adjusted p-value is ``<= alpha`` (which enforces the
    "stop at the first non-rejection" rule via monotonicity).

    Args:
        pvalues: raw p-values, either a sequence or a mapping ``{name: p}``.
        alpha: family-wise error rate.

    Returns:
        Per-hypothesis tuples preserving the input order/keys:
        ``(name, p_raw, p_adjusted, reject)`` if a mapping was given, else
        ``(p_raw, p_adjusted, reject)``.

    Raises:
        ValueError: if any p-value is outside ``[0, 1]`` or ``alpha`` not in (0, 1).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    keyed = isinstance(pvalues, Mapping)
    if keyed:
        names = list(pvalues.keys())
        raw = [float(pvalues[k]) for k in names]
    else:
        names = None
        raw = [float(p) for p in pvalues]

    for p in raw:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p-values must be in [0, 1], got {p!r}")

    m = len(raw)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: raw[i])  # ascending by p
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        factor = m - rank  # = m - (rank+1) + 1, the Holm step-down weight
        running_max = max(running_max, min(1.0, raw[i] * factor))
        adjusted[i] = running_max

    out: list[tuple] = []
    for i in range(m):
        reject = adjusted[i] <= alpha
        if keyed:
            out.append((names[i], raw[i], adjusted[i], reject))
        else:
            out.append((raw[i], adjusted[i], reject))
    return out
