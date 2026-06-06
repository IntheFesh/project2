"""Tests for the LIBERO-Plus task-selection layer (family<->category, parse, group, select)."""

from pathlib import Path

import pytest

from data.prepare_libero_subset import enumerate_libero_tasks, load_task_classification
from perturb.libero_plus_constants import (
    CATEGORY_TO_FAMILY,
    FAMILY_TO_CATEGORY,
    VERIFIED_CATEGORIES,
)
from perturb.libero_plus_wrapper import (
    category_for_family,
    family_for_category,
    group_by_category_level,
    parse_task_classification,
    parse_task_uid,
    select_perturbed_tasks,
)

FIXTURE = Path(__file__).parent / "fixtures" / "task_classification.json"


def _tasks():
    return parse_task_classification(load_task_classification(FIXTURE))


# --------------------------- family <-> category mapping ---------------------------

def test_family_category_roundtrip() -> None:
    for family, category in FAMILY_TO_CATEGORY.items():
        assert category_for_family(family) == category
        assert family_for_category(category) == family
        assert CATEGORY_TO_FAMILY[category] == family


def test_verified_categories_present() -> None:
    assert {"Background Textures", "Robot Initial States"} <= set(FAMILY_TO_CATEGORY.values())
    assert VERIFIED_CATEGORIES <= set(FAMILY_TO_CATEGORY.values())


def test_unknown_family_and_unmapped_category() -> None:
    with pytest.raises(ValueError):
        category_for_family("bogus")
    assert family_for_category("Language Instructions") is None  # intentionally unmapped


# --------------------------- parse + uid ---------------------------

def test_parse_and_uid_and_family() -> None:
    tasks = _tasks()
    assert len(tasks) == 7
    t0 = next(t for t in tasks if t.suite == "libero_spatial" and t.id == 0)
    assert t0.uid == "libero_spatial:0"
    assert t0.category == "Camera Viewpoints"
    assert t0.family == "viewpoint"
    assert parse_task_uid("libero_spatial:0") == ("libero_spatial", 0)


def test_parse_malformed_raises() -> None:
    with pytest.raises(ValueError):
        parse_task_classification({"s": [{"id": 1, "name": "x"}]})  # missing category/level


def test_parse_task_uid_bad() -> None:
    with pytest.raises(ValueError):
        parse_task_uid("nocolon")
    with pytest.raises(ValueError):
        parse_task_uid("suite:notint")


# --------------------------- group + select ---------------------------

def test_group_by_category_level() -> None:
    groups = group_by_category_level(_tasks())
    assert len(groups[("Camera Viewpoints", 1)]) == 3
    assert len(groups[("Background Textures", 1)]) == 1
    assert len(groups[("Sensor Noise", 5)]) == 1


def test_select_filters_and_clamps() -> None:
    tasks = _tasks()
    all_vp1 = select_perturbed_tasks(tasks, "viewpoint", 1, 99, seed=0)
    assert all_vp1 == ["libero_object:0", "libero_spatial:0", "libero_spatial:1"]
    sel = select_perturbed_tasks(tasks, "viewpoint", 1, 2, seed=0)
    assert len(sel) == 2 and set(sel) <= set(all_vp1)


def test_select_is_deterministic() -> None:
    tasks = _tasks()
    assert select_perturbed_tasks(tasks, "viewpoint", 1, 2, seed=3) == select_perturbed_tasks(
        tasks, "viewpoint", 1, 2, seed=3
    )


def test_select_empty_cell_returns_empty() -> None:
    # "lighting" -> "Light Conditions" is absent from the fixture.
    assert select_perturbed_tasks(_tasks(), "lighting", 1, 5, seed=0) == []


def test_select_validation() -> None:
    tasks = _tasks()
    with pytest.raises(ValueError):
        select_perturbed_tasks(tasks, "bogus", 1, 1)
    with pytest.raises(ValueError):
        select_perturbed_tasks(tasks, "viewpoint", 6, 1)
    with pytest.raises(ValueError):
        select_perturbed_tasks(tasks, "viewpoint", 1, -1)


# --------------------------- enumerate (compose) ---------------------------

def test_enumerate_libero_tasks_from_path() -> None:
    groups = enumerate_libero_tasks(FIXTURE)
    assert ("Camera Viewpoints", 1) in groups
    assert len(groups[("Camera Viewpoints", 1)]) == 3
    assert len(groups[("Robot Initial States", 2)]) == 1
