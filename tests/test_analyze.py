"""Tests for the packaged stats harness: facade re-exports + report builder + analyze CLI."""

import json
from pathlib import Path

import pytest

from eval.stats import build_report, format_text, read_rows
from eval.stats.report import matched_pairs, method_gap_by_family, per_family_sr
from scripts.analyze_results import _json_safe
from scripts.analyze_results import main as analyze_main

CSV = Path(__file__).parent / "fixtures" / "rollout_sample.csv"


def _rows():
    return read_rows(CSV)


def test_read_rows_count_and_types() -> None:
    rows = _rows()
    assert len(rows) == 26
    assert isinstance(rows[0]["success"], int) and isinstance(rows[0]["level"], int)


def test_facade_reexports_public_api() -> None:
    from eval import stats

    for name in ("success_rate", "bootstrap_ci", "mcnemar", "paired_bootstrap_delta",
                 "holm_bonferroni", "delta_robust", "recovery", "delta_method",
                 "generalization_gap", "build_report", "format_text", "read_rows"):
        assert hasattr(stats, name), name


def test_per_family_sr() -> None:
    pf = per_family_sr(_rows(), "C", ["viewpoint", "layout"], n_resamples=200, seed=0)
    assert pf["viewpoint"]["sr"] == pytest.approx(0.75)
    assert pf["layout"]["sr"] == pytest.approx(0.25)
    assert pf["viewpoint"]["ci_lo"] <= 0.75 <= pf["viewpoint"]["ci_hi"]


def test_method_gap_and_holm() -> None:
    g = method_gap_by_family(
        _rows(), ["viewpoint", "layout"], cond_c="C", cond_b="B", n_resamples=200, seed=0
    )
    assert g["viewpoint"]["delta_method"] == pytest.approx(0.25)  # SR_C - SR_B = 0.75 - 0.5
    assert "holm_p" in g["viewpoint"] and "reject" in g["viewpoint"]


def test_matched_pairs_alignment() -> None:
    c_out, b_out = matched_pairs(_rows(), "C", "B", "viewpoint")
    assert len(c_out) == len(b_out) == 4


def test_build_report_values() -> None:
    rep = build_report(_rows(), trained_families=["viewpoint"], n_resamples=200, seed=0)
    assert rep["recovery_C"]["viewpoint"] == pytest.approx(0.75)
    assert rep["recovery_C"]["layout"] == pytest.approx(0.25)
    assert rep["generalization_gap"]["gap"] == pytest.approx(0.5)  # 0.75 in-dist - 0.25 held-out
    assert rep["held_out_families"] == ["layout"]
    assert rep["trained_families"] == ["viewpoint"]
    text = format_text(rep)
    assert "viewpoint" in text and "Generalization gap" in text


def test_cli_writes_text_and_json(tmp_path) -> None:
    out = tmp_path / "report"
    rc = analyze_main(
        ["--csv", str(CSV), "--trained-families", "viewpoint", "--n-resamples", "200",
         "--out", str(out)]
    )
    assert rc == 0
    assert out.with_suffix(".txt").exists()
    data = json.loads(out.with_suffix(".json").read_text())
    assert data["recovery_C"]["viewpoint"] == pytest.approx(0.75)


def test_json_safe_converts_nan() -> None:
    assert _json_safe(float("nan")) is None
    assert _json_safe({"a": float("nan"), "b": [1.0, float("nan")]}) == {"a": None, "b": [1.0, None]}


# --------------------------- Probe 2: language conditioning ---------------------------

PROBE_CSV = Path(__file__).parent / "fixtures" / "rollout_probe_sample.csv"


def test_language_probe_inline_rows() -> None:
    rows = [
        {"condition": "A", "task_id": "t1", "family": "viewpoint", "level": 2, "seed": 0,
         "success": 1, "instruction": "correct"},
        {"condition": "A", "task_id": "t2", "family": "viewpoint", "level": 2, "seed": 0,
         "success": 1, "instruction": "correct"},
        {"condition": "A", "task_id": "t1", "family": "viewpoint", "level": 2, "seed": 0,
         "success": 0, "instruction": "blank"},
        {"condition": "A", "task_id": "t2", "family": "viewpoint", "level": 2, "seed": 0,
         "success": 0, "instruction": "blank"},
    ]
    rep = build_report(rows, trained_families=["viewpoint"], n_resamples=200, seed=0)
    blank = rep["language_probe"]["A"]["blank"]
    assert blank["sr_correct"] == pytest.approx(1.0)
    assert blank["sr_ablated"] == pytest.approx(0.0)
    assert blank["delta"] == pytest.approx(1.0)  # SR_correct - SR_ablated
    assert blank["n"] == 2
    assert "Language-conditioning probe" in format_text(rep)


def test_no_language_probe_without_instruction_column() -> None:
    # The standard rollout fixture has no instruction column -> no language_probe key.
    rep = build_report(read_rows(CSV), trained_families=["viewpoint"], n_resamples=200, seed=0)
    assert "language_probe" not in rep


def test_cli_surfaces_language_probe(tmp_path) -> None:
    out = tmp_path / "probe_report"
    rc = analyze_main(
        ["--csv", str(PROBE_CSV), "--trained-families", "viewpoint", "--n-resamples", "200",
         "--out", str(out)]
    )
    assert rc == 0
    text = out.with_suffix(".txt").read_text()
    assert "Language-conditioning probe" in text
    data = json.loads(out.with_suffix(".json").read_text())
    assert "language_probe" in data and "A" in data["language_probe"]
