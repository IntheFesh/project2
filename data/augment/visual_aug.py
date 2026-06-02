"""Visual augmentation generators aligned to LIBERO-Plus families (Phase 3).

Conditions differ only in augmentation:
    B (standard): generic torchvision augmentation (color jitter, blur, crop, ...).
    C (targeted): augmentation whose *family and magnitude* mirror the eval perturbations
        (viewpoint / lighting / texture / noise), RESample-style (arXiv 2510.17640).

torchvision transforms require the GPU extra, so transform *construction* is wired in
Phase 3; the family bookkeeping below is pure and validated now.
"""

from __future__ import annotations

from perturb.libero_plus_wrapper import CORE_FAMILIES

# Families for which condition C provides a magnitude-aligned augmentation generator.
TARGETED_AUG_FAMILIES: tuple[str, ...] = CORE_FAMILIES


def validate_aug_families(families: list[str]) -> list[str]:
    """Validate requested augmentation families against the supported targeted set."""
    unknown = [f for f in families if f not in TARGETED_AUG_FAMILIES]
    if unknown:
        raise ValueError(
            f"unsupported augmentation families {unknown}; expected subset of "
            f"{TARGETED_AUG_FAMILIES}"
        )
    return families


def build_transform(family: str, magnitude: float):
    """Build a torchvision transform for ``family`` at the given ``magnitude`` (condition C)."""
    raise NotImplementedError(
        "Phase 3: construct torchvision transforms with magnitudes aligned to LIBERO-Plus."
    )


def build_standard_transform():
    """Build the generic torchvision augmentation pipeline (condition B)."""
    raise NotImplementedError("Phase 3: construct the standard torchvision augmentation pipeline.")
