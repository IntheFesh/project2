"""Success-rate and robustness / recovery metrics.

Pure-Python (no numpy) so these run off-GPU with zero heavy dependencies. Confidence
intervals and paired significance live in ``bootstrap.py`` / ``paired.py`` / ``holm.py``
(Phase 5); this module holds the point-estimate arithmetic that defines the project's
metric contracts.

Definitions (see README "Metrics"). For a model ``M`` and the base model ``A``:

* ``SR``        success rate = successes / episodes.
* ``delta_robust``  collapse magnitude = ``SR_clean - SR_perturbed`` for a fixed model;
  larger means a bigger collapse.
* ``recovery``  fraction of the *base* model's lost performance that intervention ``M``
  restores on a perturbed family:
  ``(SR_M_pert - SR_A_pert) / (SR_A_clean - SR_A_pert)``. ``1.0`` means restored to the
  base model's clean level; values can exceed ``1.0``. The zero-collapse denominator is
  guarded (returns NaN).
* ``delta_method``  headline statistic = ``SR_C_pert - SR_B_pert`` (computed *paired* at
  fixed init states in Phase 5; this module only does the point estimate).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["success_rate", "delta_robust", "recovery", "delta_method", "generalization_gap"]

_RATE_TOL = 1e-9


def _check_rate(value: float, name: str) -> float:
    """Validate that ``value`` is a success rate in ``[0, 1]`` (within tolerance)."""
    if not (-_RATE_TOL <= value <= 1.0 + _RATE_TOL):
        raise ValueError(f"{name} must be a success rate in [0, 1], got {value!r}")
    return min(1.0, max(0.0, value))


def success_rate(outcomes: Sequence[bool | int | float]) -> float:
    """Success rate = mean of per-episode success outcomes.

    Args:
        outcomes: per-episode outcomes, each truthy/1.0 for success and falsy/0.0 for
            failure (e.g. a list of bools or 0/1 ints).

    Returns:
        Fraction of successful episodes in ``[0, 1]``.

    Raises:
        ValueError: if ``outcomes`` is empty.
    """
    n = len(outcomes)
    if n == 0:
        raise ValueError("success_rate requires at least one episode outcome")
    return sum(float(bool(o)) for o in outcomes) / n


def delta_robust(sr_clean: float, sr_perturbed: float) -> float:
    """Collapse magnitude: ``SR_clean - SR_perturbed`` for a fixed model.

    Positive values indicate a performance drop (collapse) under perturbation.
    """
    sr_clean = _check_rate(sr_clean, "sr_clean")
    sr_perturbed = _check_rate(sr_perturbed, "sr_perturbed")
    return sr_clean - sr_perturbed


def recovery(
    sr_model_perturbed: float,
    sr_base_perturbed: float,
    sr_base_clean: float,
) -> float:
    """Fraction of the base model's collapse that an intervention restores.

    ``(SR_M_pert - SR_A_pert) / (SR_A_clean - SR_A_pert)`` where ``A`` is the base model.
    Returns ``NaN`` when the base did not collapse (denominator ~ 0), since "recovery" is
    undefined with nothing to recover.
    """
    sr_model_perturbed = _check_rate(sr_model_perturbed, "sr_model_perturbed")
    sr_base_perturbed = _check_rate(sr_base_perturbed, "sr_base_perturbed")
    sr_base_clean = _check_rate(sr_base_clean, "sr_base_clean")
    base_collapse = sr_base_clean - sr_base_perturbed
    if abs(base_collapse) < _RATE_TOL:
        return math.nan
    return (sr_model_perturbed - sr_base_perturbed) / base_collapse


def delta_method(sr_a: float, sr_b: float) -> float:
    """Headline method gap: ``sr_a - sr_b`` (e.g. condition C minus condition B).

    The point estimate only; the paired significance test lives in ``paired.py`` (Phase 5).
    """
    sr_a = _check_rate(sr_a, "sr_a")
    sr_b = _check_rate(sr_b, "sr_b")
    return sr_a - sr_b


def generalization_gap(recovery_in_dist: float, recovery_held_out: float) -> float:
    """Cross-family generalization gap: ``recovery_in_dist - recovery_held_out``.

    Both inputs are :func:`recovery` values (fractions of the base collapse restored) for the
    *same* intervention, computed on **in-distribution** augmented families vs a **held-out**
    family never seen during augmentation. A large positive gap means the intervention helps
    in-dist far more than held-out (it overfits its augmentation); ``~0`` means it generalizes.

    These are recovery ratios, not success rates, so no ``[0, 1]`` validation is applied (they may
    be negative or exceed 1). Returns ``NaN`` if either recovery is ``NaN`` (undefined collapse),
    consistent with :func:`recovery`.
    """
    if math.isnan(recovery_in_dist) or math.isnan(recovery_held_out):
        return math.nan
    return recovery_in_dist - recovery_held_out
