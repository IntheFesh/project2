"""Mechanistic probe: language-conditioning sensitivity (Phase 6 / probe).

Motivated by the LIBERO-Plus / LIBERO-PRO finding that VLAs largely *ignore* language: we run the
**same task-ID set** under several instruction conditions and measure the change in success rate.

    correct    -- the task's real instruction
    blank      -- empty instruction
    shuffled   -- the same words, deterministically reordered
    mismatched -- another task's instruction

If ``SR_correct - SR_ablated ~= 0`` the policy is effectively a vision-action model (it ignores the
instruction). This is evidence about the *mechanism* of collapse, not a headline metric.

Instruction construction and the ΔSR / paired-stat computation are pure and unit-tested here; the
rollout under each instruction variant is the GPU seam (``_probe_rollout``). The paired stats reuse
``eval.paired`` (matched per task ID).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from eval.metrics import success_rate
from eval.stats.paired import mcnemar, paired_bootstrap_delta

INSTRUCTION_VARIANTS: tuple[str, ...] = ("correct", "blank", "shuffled", "mismatched")
ABLATED_VARIANTS: tuple[str, ...] = ("blank", "shuffled", "mismatched")


def blank_instruction(_instruction: str) -> str:
    """The empty instruction (ablates language entirely)."""
    return ""


def shuffle_instruction(instruction: str, *, seed: int = 0) -> str:
    """Deterministically reorder the words of ``instruction`` (same words, scrambled order).

    Single-word / empty instructions are returned unchanged.
    """
    words = instruction.split()
    if len(words) <= 1:
        return instruction
    rng = random.Random(seed)
    shuffled = words[:]
    for _ in range(10):  # avoid returning the original order when possible
        rng.shuffle(shuffled)
        if shuffled != words:
            break
    return " ".join(shuffled)


def mismatch_instruction(instruction: str, others: Sequence[str], *, seed: int = 0) -> str:
    """Pick a *different* task's instruction from ``others`` (returns input if none available)."""
    pool = [o for o in others if o != instruction]
    if not pool:
        return instruction
    return random.Random(seed).choice(pool)


def build_instruction_variants(
    instruction: str, others: Sequence[str] = (), *, seed: int = 0
) -> dict[str, str]:
    """Build all instruction variants for one task (correct/blank/shuffled/mismatched)."""
    return {
        "correct": instruction,
        "blank": blank_instruction(instruction),
        "shuffled": shuffle_instruction(instruction, seed=seed),
        "mismatched": mismatch_instruction(instruction, others, seed=seed),
    }


def language_sensitivity(sr_correct: float, sr_ablated: float) -> float:
    """Point estimate ``SR_correct - SR_ablated``: >0 => policy uses language; ~0 => ignores it."""
    for name, value in (("sr_correct", sr_correct), ("sr_ablated", sr_ablated)):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be a success rate in [0, 1], got {value!r}")
    return sr_correct - sr_ablated


def language_sensitivity_paired(
    correct_outcomes: Sequence[bool | int],
    ablated_outcomes: Sequence[bool | int],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired language sensitivity on matched per-task outcomes (correct vs an ablated variant).

    Returns point ΔSR, a paired-bootstrap 95% CI, the McNemar p-value, and the two SRs. Outcomes
    must be matched by task ID (same tasks, same order) so the comparison is paired.
    """
    delta, lo, hi = paired_bootstrap_delta(
        correct_outcomes, ablated_outcomes, n_resamples=n_resamples, seed=seed
    )
    mc = mcnemar(correct_outcomes, ablated_outcomes)
    return {
        "delta": delta,
        "ci_lo": lo,
        "ci_hi": hi,
        "pvalue": mc["pvalue"],
        "n": len(correct_outcomes),
        "sr_correct": success_rate(correct_outcomes),
        "sr_ablated": success_rate(ablated_outcomes),
    }


# ----------------------------------------------------------------- GPU seams --
def _probe_rollout(policy, task_uid: str, instruction: str, seed: int) -> int:
    """Roll out ``task_uid`` once under ``instruction`` and return success (0/1). GPU seam."""
    raise NotImplementedError(
        "Phase 6/probe: roll out the task under the given instruction variant; return success."
    )


def feature_shift_probe(*args, **kwargs):
    """(Optional, low priority) cosine distance of vision-encoder features, clean vs perturbed.

    Fragile -- it needs model-internal hooks into the vision encoder -- so it is only scaffolded
    here, not implemented. Do not over-invest.
    """
    raise NotImplementedError(
        "Optional probe (low priority): vision-feature cosine shift; needs encoder hooks."
    )
