"""Visual augmentation generators aligned to LIBERO-Plus families (Phase 3).

Conditions differ ONLY in augmentation:
    B (standard): LeRobot's generic photometric pipeline (ColorJitter x4 + SharpnessJitter +
        RandomAffine) -- the off-the-shelf augmentation everyone uses.
    C (targeted): augmentation whose *family and magnitude* mirror the eval perturbations
        (viewpoint / lighting / texture / noise), RESample-style (arXiv 2510.17640).

Implementation (Phase 3, verified against lerobot/transforms/transforms.py 0.5.2):
``make_transform_from_config`` accepts ``type`` = any class under ``torchvision.transforms.v2``
(or ``"SharpnessJitter"``), with ``kwargs`` forwarded to its constructor. So both conditions are
expressed as LeRobot ``image_transforms.tfs`` dicts and driven through the *verified* ``lerobot-train``
CLI -- no hand-written training loop, no custom transform classes.

Fidelity is NOT uniform across families (honesty guard, see report/technical_report.md S8):
    lighting -> ColorJitter(brightness, contrast)        FAITHFUL (photometric == photometric)
    noise    -> GaussianNoise(sigma)                     FAITHFUL
    texture  -> ColorJitter(saturation) + GaussianBlur   PROXY (cannot swap textures in 2-D)
    viewpoint-> RandomPerspective + RandomAffine          PROXY (2-D warp != moved 3-D camera)
``layout`` is never augmented -> evaluated as held-out generalization.

``aligned_magnitude`` (pure, unit-tested) is the single source of truth for C's magnitudes;
the builders below map it onto LeRobot's tfs schema.
"""

from __future__ import annotations

from perturb.libero_plus_wrapper import CORE_FAMILIES, MAX_LEVEL, level_to_fraction

# Families for which condition C provides a magnitude-aligned augmentation generator.
TARGETED_AUG_FAMILIES: tuple[str, ...] = CORE_FAMILIES

# Augmentation parameters at full magnitude (level 5), per family. These are OUR design
# choices (initial values; tunable), aligned in *family* to the eval perturbations.
# ``aligned_magnitude`` scales them linearly by level / MAX_LEVEL.
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

    Linearly scales the full-magnitude params by ``level / MAX_LEVEL``. Pure; consumed by
    :func:`build_targeted_tfs`.

    Raises:
        ValueError: if ``family`` is not a targeted family.
    """
    if family not in _FULL_MAGNITUDE_PARAMS:
        raise ValueError(
            f"no targeted augmentation for family {family!r}; expected one of "
            f"{tuple(_FULL_MAGNITUDE_PARAMS)}"
        )
    frac = level_to_fraction(level)
    return {key: value * frac for key, value in _FULL_MAGNITUDE_PARAMS[family].items()}


# --------------------------------------------------------------------------- tfs builders --
# A "tf entry" is LeRobot's ImageTransformConfig as a plain dict: {weight, type, kwargs}.
# ``type`` is a torchvision.transforms.v2 class name (or "SharpnessJitter"); ``kwargs`` are
# forwarded to its constructor. train/train_lora.py serializes the dict to the
# ``--dataset.image_transforms.tfs`` CLI override.

def _cj_range(mag: float) -> list[float]:
    """ColorJitter factor range [1-mag, 1+mag] for a jitter amount ``mag``."""
    return [round(1.0 - mag, 4), round(1.0 + mag, 4)]


def build_standard_tfs() -> dict[str, dict]:
    """Condition B: LeRobot's default generic photometric pipeline (as explicit tfs)."""
    return {
        "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}},
        "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}},
        "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
        "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}},
        "sharpness": {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
        "affine": {"weight": 1.0, "type": "RandomAffine",
                   "kwargs": {"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}},
    }


def build_targeted_tfs(
    families: list[str] | tuple[str, ...] = CORE_FAMILIES,
    level: int = MAX_LEVEL,
) -> dict[str, dict]:
    """Condition C: family-aligned tfs at the given ``level`` (default full magnitude).

    Returns a LeRobot ``tfs`` dict. lighting/noise are faithful; texture/viewpoint are 2-D proxies.
    """
    families = validate_aug_families(list(families))
    tfs: dict[str, dict] = {}
    for fam in families:
        m = aligned_magnitude(fam, level)
        if fam == "lighting":
            tfs["c_lighting"] = {"weight": 1.0, "type": "ColorJitter",
                                 "kwargs": {"brightness": _cj_range(m["brightness"]),
                                            "contrast": _cj_range(m["contrast"])}}
        elif fam == "noise":
            tfs["c_noise"] = {"weight": 1.0, "type": "GaussianNoise",
                              "kwargs": {"sigma": round(m["gaussian_std"], 4), "clip": True}}
        elif fam == "texture":
            tfs["c_texture_sat"] = {"weight": 1.0, "type": "ColorJitter",
                                    "kwargs": {"saturation": _cj_range(m["saturation"])}}
            tfs["c_texture_blur"] = {"weight": 1.0, "type": "GaussianBlur",
                                     "kwargs": {"kernel_size": 5,
                                                "sigma": [0.1, round(m["blur_sigma"], 4)]}}
        elif fam == "viewpoint":
            tfs["c_view_persp"] = {"weight": 1.0, "type": "RandomPerspective",
                                   "kwargs": {"distortion_scale": round(m["perspective_distortion"], 4),
                                              "p": 1.0}}
            tfs["c_view_affine"] = {"weight": 1.0, "type": "RandomAffine",
                                    "kwargs": {"degrees": 0.0,
                                               "translate": [round(m["translate_frac"], 4)] * 2}}
    return tfs


# --------------------------------------------------------------------------- seam shims --
def build_standard_transform() -> dict[str, dict]:
    """Condition B augmentation as a LeRobot tfs dict (Phase 3 seam, now implemented)."""
    return build_standard_tfs()


def build_transform(family: str, level: int = MAX_LEVEL) -> dict[str, dict]:
    """Condition C augmentation for a single ``family`` as a LeRobot tfs dict (Phase 3 seam)."""
    return build_targeted_tfs([family], level=level)
