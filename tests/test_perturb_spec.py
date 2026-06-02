"""Tests for the LIBERO-Plus perturbation selector (Phase 0)."""

import pytest

from perturb.libero_plus_wrapper import (
    CORE_FAMILIES,
    PerturbSpec,
    parse_perturb_spec,
)


def test_valid_spec_str_and_is_core() -> None:
    spec = PerturbSpec("viewpoint", 4)
    assert spec.is_core
    assert str(spec) == "viewpoint:4"


def test_invalid_family_rejected() -> None:
    with pytest.raises(ValueError):
        PerturbSpec("blur", 3)


@pytest.mark.parametrize("level", [0, 6, -1])
def test_invalid_level_rejected(level: int) -> None:
    with pytest.raises(ValueError):
        PerturbSpec("noise", level)


def test_parse_none_or_empty_is_clean() -> None:
    assert parse_perturb_spec(None) is None
    assert parse_perturb_spec("") is None
    assert parse_perturb_spec("   ") is None


def test_parse_roundtrip() -> None:
    assert parse_perturb_spec("lighting:2") == PerturbSpec("lighting", 2)


def test_parse_missing_level_rejected() -> None:
    with pytest.raises(ValueError):
        parse_perturb_spec("viewpoint")


def test_parse_non_integer_level_rejected() -> None:
    with pytest.raises(ValueError):
        parse_perturb_spec("viewpoint:high")


def test_core_families_are_expected() -> None:
    assert set(CORE_FAMILIES) == {"viewpoint", "lighting", "texture", "noise"}
