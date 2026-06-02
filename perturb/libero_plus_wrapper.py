"""Thin wrapper over LIBERO-Plus (arXiv 2510.13626).

We do **not** reimplement perturbations. LIBERO-Plus is installed as a drop-in replacement
for ``libero`` (``pip install -e .``); this wrapper only *selects* a perturbation **family**
and a graded **level** (L1-L5), keeping our collapse numbers comparable to the published
benchmark and saving off-GPU implementation time.

The :class:`PerturbSpec` dataclass and string parsing are pure and validated here (Phase 0);
the actual perturbed-env construction is wired against LIBERO-Plus in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

CORE_FAMILIES: tuple[str, ...] = ("viewpoint", "lighting", "texture", "noise")
STRETCH_FAMILIES: tuple[str, ...] = ("layout", "robot_init")
ALL_FAMILIES: tuple[str, ...] = CORE_FAMILIES + STRETCH_FAMILIES

MIN_LEVEL = 1
MAX_LEVEL = 5


@dataclass(frozen=True)
class PerturbSpec:
    """A single LIBERO-Plus perturbation selection: a ``family`` at graded ``level`` (L1-L5)."""

    family: str
    level: int

    def __post_init__(self) -> None:
        if self.family not in ALL_FAMILIES:
            raise ValueError(
                f"unknown perturbation family {self.family!r}; expected one of {ALL_FAMILIES}"
            )
        if not (MIN_LEVEL <= self.level <= MAX_LEVEL):
            raise ValueError(
                f"level must be in [{MIN_LEVEL}, {MAX_LEVEL}] (L1-L5), got {self.level!r}"
            )

    @property
    def is_core(self) -> bool:
        """True if this family is part of the CORE suite (viewpoint/lighting/texture/noise)."""
        return self.family in CORE_FAMILIES

    def __str__(self) -> str:
        return f"{self.family}:{self.level}"


def parse_perturb_spec(text: str | None) -> PerturbSpec | None:
    """Parse a ``'family:level'`` string (e.g. ``'viewpoint:4'``); ``None``/empty -> clean (None)."""
    if text is None or text.strip() == "":
        return None
    if ":" not in text:
        raise ValueError(f"perturbation spec must be 'family:level', got {text!r}")
    family, _, level = text.partition(":")
    try:
        level_int = int(level)
    except ValueError as exc:
        raise ValueError(f"level must be an integer in {text!r}") from exc
    return PerturbSpec(family=family.strip(), level=level_int)


def make_perturbed_env(task: str, spec: PerturbSpec | None, **kwargs):
    """Return a LIBERO-Plus environment for ``task`` under ``spec`` (``None`` -> clean env).

    GPU/simulator runtime; wired against the installed LIBERO-Plus drop-in in Phase 2.
    """
    raise NotImplementedError(
        "Phase 2: construct the LIBERO-Plus env for the selected family/level."
    )
