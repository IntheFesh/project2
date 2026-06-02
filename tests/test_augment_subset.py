"""Tests for level->magnitude alignment, in-dist/held-out tagging, and subset selection."""

import pytest
import yaml

from data.augment.visual_aug import aligned_magnitude, validate_aug_families
from data.prepare_libero_subset import select_subset, write_task_subset
from perturb.libero_plus_wrapper import (
    PerturbSpec,
    classify_distribution,
    level_to_fraction,
)


# --------------------------- level_to_fraction ---------------------------

@pytest.mark.parametrize("level,frac", [(1, 0.2), (2, 0.4), (5, 1.0)])
def test_level_to_fraction(level: int, frac: float) -> None:
    assert level_to_fraction(level) == pytest.approx(frac)


def test_level_to_fraction_out_of_range() -> None:
    with pytest.raises(ValueError):
        level_to_fraction(0)
    with pytest.raises(ValueError):
        level_to_fraction(6)


# --------------------------- classify_distribution (honesty guard) ---------------------------

def test_classify_clean() -> None:
    assert classify_distribution(None, ["viewpoint"]) == "clean"


def test_classify_in_dist_vs_held_out() -> None:
    trained = ["viewpoint", "lighting"]
    assert classify_distribution(PerturbSpec("viewpoint", 4), trained) == "in_dist"
    assert classify_distribution(PerturbSpec("texture", 4), trained) == "held_out"


# --------------------------- aligned_magnitude ---------------------------

def test_aligned_magnitude_full_and_scaled() -> None:
    assert aligned_magnitude("lighting", 5) == pytest.approx({"brightness": 0.4, "contrast": 0.4})
    assert aligned_magnitude("lighting", 1) == pytest.approx({"brightness": 0.08, "contrast": 0.08})
    assert aligned_magnitude("noise", 5) == pytest.approx({"gaussian_std": 0.08})


def test_aligned_magnitude_is_monotonic_in_level() -> None:
    lo = aligned_magnitude("viewpoint", 2)["perspective_distortion"]
    hi = aligned_magnitude("viewpoint", 4)["perspective_distortion"]
    assert hi > lo


def test_aligned_magnitude_unknown_family() -> None:
    with pytest.raises(ValueError):
        aligned_magnitude("layout", 3)  # STRETCH family has no targeted aug


def test_validate_aug_families() -> None:
    assert validate_aug_families(["viewpoint", "noise"]) == ["viewpoint", "noise"]
    with pytest.raises(ValueError):
        validate_aug_families(["viewpoint", "bogus"])


# --------------------------- select_subset / write_task_subset ---------------------------

def test_select_subset_first_n_per_suite() -> None:
    tasks = {"a": ["a0", "a1", "a2"], "b": ["b0", "b1"]}
    assert select_subset(tasks, 2) == ["a0", "a1", "b0", "b1"]
    assert select_subset(tasks, 0) == []
    assert select_subset(tasks, 5) == ["a0", "a1", "a2", "b0", "b1"]  # clamps to available


def test_select_subset_negative_raises() -> None:
    with pytest.raises(ValueError):
        select_subset({"a": ["a0"]}, -1)


def test_write_task_subset_roundtrip(tmp_path) -> None:
    out = write_task_subset(tmp_path / "sub" / "task_subset.yaml", ["a0", "b1"])
    assert out.exists()
    loaded = yaml.safe_load(out.read_text())
    assert loaded == {"task_subset": ["a0", "b1"]}
