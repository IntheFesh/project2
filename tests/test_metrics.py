"""Tests for the point-estimate metrics (Phase 0)."""

import math

import pytest

from eval.metrics import delta_method, delta_robust, recovery, success_rate


def test_success_rate_basic() -> None:
    assert success_rate([1, 0, 1, 1]) == pytest.approx(0.75)
    assert success_rate([True, False]) == pytest.approx(0.5)


def test_success_rate_empty_raises() -> None:
    with pytest.raises(ValueError):
        success_rate([])


def test_delta_robust_is_collapse_magnitude() -> None:
    assert delta_robust(0.9, 0.4) == pytest.approx(0.5)


def test_recovery_partial() -> None:
    # base clean 0.9, base pert 0.4 (collapse 0.5); model pert 0.65 -> recovered 0.25/0.5 = 0.5
    assert recovery(0.65, 0.4, 0.9) == pytest.approx(0.5)


def test_recovery_full_and_overshoot() -> None:
    assert recovery(0.9, 0.4, 0.9) == pytest.approx(1.0)
    assert recovery(1.0, 0.4, 0.9) == pytest.approx(1.2)


def test_recovery_zero_collapse_is_nan() -> None:
    assert math.isnan(recovery(0.8, 0.8, 0.8))


def test_delta_method() -> None:
    assert delta_method(0.7, 0.6) == pytest.approx(0.1)


def test_rate_validation_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        delta_robust(1.5, 0.2)
    with pytest.raises(ValueError):
        recovery(0.5, -0.1, 0.9)
