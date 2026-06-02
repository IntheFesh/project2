"""Visual augmentation generators aligned to LIBERO-Plus families (Phase 3).

Conditions differ only in augmentation:
    B (standard): generic torchvision augmentation (color jitter, blur, crop, ...).
    C (targeted): augmentation whose *family and magnitude* mirror the eval perturbations
        (viewpoint / lighting / texture / noise), RESample-style (arXiv 2510.17640).

The magnitude-alignment bookkeeping below is *our* augmentation design (tunable) and is
pure + tested now. The actual torchvision transforms require the GPU extra, so their
construction is the Phase-3 seam. Train-time augmentation and eval-time perturbation share a
*family* but live on *separate splits* (honesty guard: in-dist recovery != generalization).
"""

from __future__ import annotations

from perturb.libero_plus_wrapper import CORE_FAMILIES, level_to_fraction

# Families for which condition C provides a magnitude-aligned augmentation generator.
TARGETED_AUG_FAMILIES: tuple[str, ...] = CORE_FAMILIES

# Augmentation parameters at full magnitude (level 5), per family. These are OUR design
# choices (initial values; to be tuned in Phase 3), aligned in *family* to the eval
# perturbations. ``aligned_magnitude`` scales them linearly by level / MAX_LEVEL.
_FULL_MAGNITUDE_PARAMS: dict[str, dict[str, float]] = {
    "lighting": {"brightness": 0.4, "contrast": 0.4},
    "noise": {"gaussian_std": 0.08},
    "texture": {"saturation": 0.4, "blur_sigma": 1.5},
    "viewpoint": {"perspective_distortion": 0.3, "translate_frac": 0.10},
}


def validate_aug_families(families: list[str]) -> list[str]:
    """Validate requested augmentation families against the supported targeted set."""
    unknown = [f for f in families if f not in TARGETED_AUG_FAMILIES]
    if unknown:
        raise ValueError(
            f"unsupported augmentation families {unknown}; expected subset of "
            f"{TARGETED_AUG_FAMILIES}"
        )
    return families


def aligned_magnitude(family: str, level: int) -> dict[str, float]:
    """Augmentation parameters for ``family`` scaled to a LIBERO-Plus ``level`` (condition C).

    Linearly scales the full-magnitude params by ``level / MAX_LEVEL`` (see
    :func:`perturb.libero_plus_wrapper.level_to_fraction`). Pure; consumed by
    :func:`build_transform` (Phase 3).

    Raises:
        ValueError: if ``family`` is not a targeted family or ``level`` is out of range.
    """
    if family not in _FULL_MAGNITUDE_PARAMS:
        raise ValueError(
            f"no targeted augmentation for family {family!r}; expected one of "
            f"{tuple(_FULL_MAGNITUDE_PARAMS)}"
        )
    frac = level_to_fraction(level)
    return {key: value * frac for key, value in _FULL_MAGNITUDE_PARAMS[family].items()}


def build_transform(family: str, level: int):
    """Build a torchvision transform for ``family`` at ``level`` (condition C, Phase 3).

    Consumes :func:`aligned_magnitude`; requires the GPU extra (torchvision).
    """
    raise NotImplementedError(
        "Phase 3: construct a torchvision transform from aligned_magnitude(family, level)."
    )


def build_standard_transform():
    """Build the generic torchvision augmentation pipeline (condition B, Phase 3)."""
    raise NotImplementedError("Phase 3: construct the standard torchvision augmentation pipeline.")
