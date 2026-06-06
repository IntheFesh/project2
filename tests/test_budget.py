"""Tests for the GPU-day budget estimator (task-instance semantics)."""

import pytest

from eval.budget import (
    RolloutMatrix,
    max_instances_per_cell,
    project_budget,
    summarize_episode_times,
)


def _matrix(instances_per_cell: int = 20) -> RolloutMatrix:
    # 4 families x 2 levels = 8 cells; clean=20; A=1, B/C=3 seeds (7 total).
    return RolloutMatrix(
        instances_per_cell=instances_per_cell,
        n_families=4,
        n_levels=2,
        seeds_per_condition={"A": 1, "B": 3, "C": 3},
        clean_instances=20,
    )


def test_matrix_unit_counts() -> None:
    m = _matrix()
    assert m.n_cells() == 8                      # 4 * 2
    assert m.units_per_seed() == 20 + 8 * 20     # clean + cells*instances = 180
    assert m.total_seeds() == 7
    assert m.total_units() == 180 * 7            # 1260


def test_project_budget_fits() -> None:
    proj = project_budget(_matrix(), sec_per_episode=30.0, n_train_runs=6, train_hours_per_run=4.0)
    assert proj["total_units"] == 1260
    assert proj["eval_days"] == pytest.approx(1260 * 30 / 86400)  # 0.4375
    assert proj["train_days"] == pytest.approx(1.0)
    assert proj["total_days"] == pytest.approx(1.4375)
    assert proj["fits"] is True


def test_project_budget_can_exceed_cap() -> None:
    proj = project_budget(
        _matrix(instances_per_cell=300), sec_per_episode=30.0,
        n_train_runs=6, train_hours_per_run=4.0,
    )
    assert proj["fits"] is False
    assert proj["total_days"] > 5.0


def test_max_instances_back_solves_boundary() -> None:
    m = _matrix()
    # eval budget = 5d - 6*4h = 345600s; /30 = 11520 units; /7 seeds = 1645.71; -20 clean;
    # /8 cells = 203.21 -> 203.
    n = max_instances_per_cell(
        m, sec_per_episode=30.0, n_train_runs=6, train_hours_per_run=4.0, cap_days=5.0
    )
    assert n == 203
    assert project_budget(
        _matrix(203), sec_per_episode=30.0, n_train_runs=6, train_hours_per_run=4.0
    )["fits"] is True
    assert project_budget(
        _matrix(204), sec_per_episode=30.0, n_train_runs=6, train_hours_per_run=4.0
    )["fits"] is False


def test_max_instances_zero_when_training_alone_exceeds_cap() -> None:
    # 40 runs x 4h = 160h = 6.67 days > 5-day cap before any rollout.
    assert max_instances_per_cell(
        _matrix(), sec_per_episode=30.0, n_train_runs=40, train_hours_per_run=4.0, cap_days=5.0
    ) == 0


def test_summarize_episode_times() -> None:
    s = summarize_episode_times([10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(25.0)
    assert s["median"] == pytest.approx(25.0)
    assert s["min"] == 10.0 and s["max"] == 40.0
    with pytest.raises(ValueError):
        summarize_episode_times([])


def test_extra_units_added_for_probe() -> None:
    m = RolloutMatrix(
        instances_per_cell=20, n_families=4, n_levels=2,
        seeds_per_condition={"A": 1, "B": 3, "C": 3}, clean_instances=20, extra_units=80,
    )
    assert m.total_units() == 180 * 7 + 80  # per-seed matrix + flat probe units


def test_matrix_validation() -> None:
    with pytest.raises(ValueError):
        RolloutMatrix(1, 1, 1, seeds_per_condition={})
    with pytest.raises(ValueError):
        RolloutMatrix(-1, 1, 1, seeds_per_condition={"A": 1})
    with pytest.raises(ValueError):
        RolloutMatrix(1, 1, 1, seeds_per_condition={"A": 1}, extra_units=-5)
