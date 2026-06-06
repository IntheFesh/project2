"""Paired-statistics evaluation harness for VLA robustness -- the project's headline contribution.

A small, reusable protocol for comparing VLA training interventions under matched perturbations,
packaged as one public surface. It is the VLA-manipulation analogue of rliable (Agarwal et al.,
NeurIPS 2021, arXiv:2108.13264): report **effect sizes with uncertainty**, not bare point success
rates. See ``docs/EVALUATION.md`` for the full protocol.

The protocol
------------
1. **Point estimates** -- ``success_rate``; collapse ``delta_robust``; ``recovery``;
   ``delta_method`` (= ``SR_C − SR_B``); cross-family ``generalization_gap``.
2. **Uncertainty** -- ``bootstrap_ci`` (>= 10k-resample percentile 95% CI on success rates).
3. **Paired significance** -- conditions are evaluated on the **same task IDs**, so episodes are
   matched; ``mcnemar`` / ``paired_bootstrap_delta`` detect the small (~5-10pp) B-vs-C gap.
4. **Multiplicity** -- ``holm_bonferroni`` across perturbation families.
5. **Honest labelling** -- in-dist vs held-out (``classify_distribution`` in ``perturb``).

``build_report`` assembles all of the above from a per-trial rollout CSV; ``format_text`` renders it.
This module only **re-exports** the implementations from ``eval.metrics`` / ``eval.bootstrap`` /
``eval.paired`` / ``eval.holm`` / ``eval.stats.report`` -- no logic is duplicated.
"""

from __future__ import annotations

from eval.bootstrap import bootstrap_ci
from eval.holm import holm_bonferroni
from eval.metrics import (
    delta_method,
    delta_robust,
    generalization_gap,
    recovery,
    success_rate,
)
from eval.paired import mcnemar, paired_bootstrap_delta
from eval.stats.report import build_report, format_text, read_rows

__all__ = [
    "success_rate",
    "delta_robust",
    "recovery",
    "delta_method",
    "generalization_gap",
    "bootstrap_ci",
    "mcnemar",
    "paired_bootstrap_delta",
    "holm_bonferroni",
    "build_report",
    "format_text",
    "read_rows",
]
