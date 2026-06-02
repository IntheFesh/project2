"""Tests for the Phase 5 statistics: bootstrap CI, paired tests, Holm-Bonferroni."""

import pytest

from eval.bootstrap import bootstrap_ci
from eval.holm import holm_bonferroni
from eval.paired import mcnemar, paired_bootstrap_delta


def _matched(a_only: int, b_only: int, both: int = 0, neither: int = 0):
    """Build matched (a, b) success vectors from a 2x2 paired count table."""
    a = [1] * a_only + [0] * b_only + [1] * both + [0] * neither
    b = [0] * a_only + [1] * b_only + [1] * both + [0] * neither
    return a, b


# --------------------------- bootstrap_ci ---------------------------

def test_bootstrap_point_equals_mean_and_brackets() -> None:
    point, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 0], n_resamples=2000, seed=0)
    assert point == pytest.approx(0.625)
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_all_success_is_degenerate() -> None:
    point, lo, hi = bootstrap_ci([1, 1, 1, 1], n_resamples=1000, seed=1)
    assert point == 1.0 and lo == 1.0 and hi == 1.0


def test_bootstrap_is_reproducible() -> None:
    data = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    assert bootstrap_ci(data, n_resamples=1000, seed=42) == bootstrap_ci(
        data, n_resamples=1000, seed=42
    )


def test_bootstrap_input_validation() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1, 0], alpha=1.5)


# --------------------------- mcnemar ---------------------------

def test_mcnemar_no_discordant_pairs() -> None:
    a, b = _matched(0, 0, both=10, neither=5)
    res = mcnemar(a, b)
    assert res["n_discordant"] == 0
    assert res["pvalue"] == 1.0


def test_mcnemar_exact_path_significant() -> None:
    a, b = _matched(a_only=8, b_only=0, both=20)  # C wins 8, B wins 0
    res = mcnemar(a, b)
    assert res["method"] == "exact binomial"
    assert res["a_only"] == 8 and res["b_only"] == 0
    assert res["pvalue"] < 0.05


def test_mcnemar_chi2_path_for_many_discordant() -> None:
    a, b = _matched(a_only=30, b_only=10)
    res = mcnemar(a, b)
    assert res["method"].startswith("chi2")
    assert res["n_discordant"] == 40
    assert res["pvalue"] < 0.05


def test_mcnemar_is_symmetric_in_pvalue() -> None:
    a, b = _matched(a_only=12, b_only=3, both=5)
    r1 = mcnemar(a, b)
    r2 = mcnemar(b, a)
    assert r1["pvalue"] == pytest.approx(r2["pvalue"])
    assert r1["a_only"] == r2["b_only"] and r1["b_only"] == r2["a_only"]


def test_mcnemar_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mcnemar([1, 0, 1], [1, 0])


# --------------------------- paired_bootstrap_delta ---------------------------

def test_paired_delta_equals_mean_difference() -> None:
    a = [1, 1, 1, 0, 1, 1]   # SR 5/6
    b = [1, 0, 1, 0, 0, 1]   # SR 3/6
    delta, lo, hi = paired_bootstrap_delta(a, b, n_resamples=2000, seed=0)
    assert delta == pytest.approx(5 / 6 - 3 / 6)
    assert lo <= delta <= hi


def test_paired_delta_reproducible() -> None:
    a, b = [1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1]
    assert paired_bootstrap_delta(a, b, n_resamples=1500, seed=7) == paired_bootstrap_delta(
        a, b, n_resamples=1500, seed=7
    )


# --------------------------- holm_bonferroni ---------------------------

def test_holm_known_example() -> None:
    res = holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    raw = [r[0] for r in res]
    adj = [r[1] for r in res]
    reject = [r[2] for r in res]
    assert raw == [0.01, 0.04, 0.03]
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert reject == [True, False, False]


def test_holm_preserves_mapping_keys_and_order() -> None:
    res = holm_bonferroni({"viewpoint": 0.001, "lighting": 0.2, "texture": 0.02}, alpha=0.05)
    names = [r[0] for r in res]
    assert names == ["viewpoint", "lighting", "texture"]
    by_name = {r[0]: r for r in res}
    assert by_name["viewpoint"][3] is True
    assert by_name["lighting"][3] is False


def test_holm_none_rejected_when_all_large() -> None:
    res = holm_bonferroni([0.3, 0.5, 0.9], alpha=0.05)
    assert all(r[2] is False for r in res)


def test_holm_rejects_validation() -> None:
    with pytest.raises(ValueError):
        holm_bonferroni([0.1, 1.5])
    with pytest.raises(ValueError):
        holm_bonferroni([0.1], alpha=0.0)
    assert holm_bonferroni([]) == []
