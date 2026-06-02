"""Holm-Bonferroni multiple-comparison correction.

Phase 5 (the differentiator). We test the method gap across several perturbation families,
so family-level p-values are corrected for multiplicity. Off-GPU; pure Python / numpy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def holm_bonferroni(
    pvalues: Sequence[float] | Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> list[tuple[str, float, float, bool]] | list[tuple[float, float, bool]]:
    """Apply the Holm-Bonferroni step-down correction.

    Args:
        pvalues: raw p-values, optionally keyed by family name.
        alpha: family-wise error rate.

    Returns:
        Per-hypothesis ``(name?, p_raw, p_adjusted, reject)`` preserving input keys/order.
    """
    raise NotImplementedError("Phase 5: implement Holm-Bonferroni step-down correction.")
