"""Tests for the GPU-day budget estimator (rental planning)."""

import pytest

from eval.budget import (
    RolloutMatrix,
    max_episodes_per_task,
    project_budget,
    summarize_episode_times,
)


def _matrix(n_episodes_per_task: int = 20) -> RolloutMatrix:
    # 8 tasks, 4 families x 2 levels + clean = 9 settings; A=1 seed, B/C=3 seeds (7 total).
    return RolloutMatrix(
        n_tasks=8,
        n_episodes_per_task=n_episodes_per_task,
        n_families=4,
        n_levels=2,
        seeds_per_condition={"A": 1, "B": 3, "C": 3},
        include_clean=True,
    )


def test_matrix_episode_counts() -> None:
    m = _matrix()
    assert m.n_settings() == 9          # 1 clean + 4*2
    assert m.episodes_per_seed() == 9 * 8 * 20
    assert m.total_seeds() == 7
    assert m.total_episodes() == 1440 * 7  # 10080


def test_project_budget_fits_at_4_5_days() -> None:
    proj = project_budget(
        _matrix(), sec_per_episode=30.0, n_train_runs=6, train_hours_per_run=4.0
    )
    assert proj["total_episodes"] == 10080
    assert proj["eval_days"] == pytest.approx(302400 / 86400)   # 3.5
    assert proj["train_days"] == pytest.approx(86400 / 86400)   # 1.0
    assert proj["total_days"] == pytest.approx(4.5)
    assert proj["fits"] is True


def test_project_budget_can_exceed_cap() -> None:
    proj = project_budget(
        _matrix(n_episodes_per_task=40), sec_per_episode=30.0,
        n_train_runs=6, train_hours_per_run=4.0,
    )
    assert proj["fits"] is False
    assert proj["total_days"] > 5.0


def test_max_episodes_back_solves_boundary() -> None:
    m = _matrix()
    n = max_episodes_per_task(
        m, sec_per_episode=30.0, n_train_runs=6, train_hours_per_run=4.0, cap_days=5.0
    )
    assert n == 22
    # The boundary holds: 22 ep/task fits, 23 does not.
    assert project_budget(
        _matrix(22), sec_per_episode=30.0, n_train_runs=6, train_hours_per_run=4.0
    )["fits"] is True
    assert project_budget(
        _matrix(23), sec_per_episode=30.0, n_train_runs=6, train_hours_per_run=4.0
    )["fits"] is False


def test_max_episodes_zero_when_training_alone_exceeds_cap() -> None:
    m = _matrix()
    # 40 runs x 4h = 160h = 6.67 days > 5-day cap before any rollout.
    assert max_episodes_per_task(
        m, sec_per_episode=30.0, n_train_runs=40, train_hours_per_run=4.0, cap_days=5.0
    ) == 0


def test_summarize_episode_times() -> None:
    s = summarize_episode_times([10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(25.0)
    assert s["median"] == pytest.approx(25.0)
    assert s["min"] == 10.0 and s["max"] == 40.0
    with pytest.raises(ValueError):
        summarize_episode_times([])


def test_matrix_validation() -> None:
    with pytest.raises(ValueError):
        RolloutMatrix(1, 1, 1, 1, seeds_per_condition={})
    with pytest.raises(ValueError):
        RolloutMatrix(-1, 1, 1, 1, seeds_per_condition={"A": 1})
