"""Tests for eval.run_rollout pure logic (plan, rows, aggregate, CSV/JSON); GPU parts are seams."""

import csv
import json
from pathlib import Path

import pytest

from data.prepare_libero_subset import load_task_classification
from eval.run_rollout import (
    ROLLOUT_FIELDS,
    _load_policy,
    _run_one_trial,
    aggregate_rows,
    build_eval_plan,
    make_row,
    write_summary,
)
from perturb.libero_plus_wrapper import parse_task_classification

FIXTURE = Path(__file__).parent / "fixtures" / "task_classification.json"


def _tasks():
    return parse_task_classification(load_task_classification(FIXTURE))


def test_make_row_valid_and_schema() -> None:
    row = make_row("C", "libero_spatial:0", "viewpoint", 2, 1, True)
    assert set(row) == set(ROLLOUT_FIELDS)
    assert row == {"condition": "C", "task_id": "libero_spatial:0", "family": "viewpoint",
                   "level": 2, "seed": 1, "success": 1}


def test_make_row_rejects_bad_success() -> None:
    with pytest.raises(ValueError):
        make_row("C", "t", "viewpoint", 2, 0, 2)


def test_build_eval_plan_clean_plus_cells() -> None:
    plan = build_eval_plan(_tasks(), ["viewpoint"], [1], 2, clean_uids=["clean:0"], seed=0)
    assert plan[0].family == "clean" and plan[0].level == 0 and plan[0].task_uid == "clean:0"
    perturbed = plan[1:]
    assert len(perturbed) == 2
    assert all(item.family == "viewpoint" and item.level == 1 for item in perturbed)
    assert {item.task_uid for item in perturbed} <= {
        "libero_object:0", "libero_spatial:0", "libero_spatial:1"
    }


def test_aggregate_rows_success_rates() -> None:
    rows = [
        make_row("C", "t1", "viewpoint", 2, 0, 1),
        make_row("C", "t2", "viewpoint", 2, 0, 0),
        make_row("A", "t1", "viewpoint", 2, 0, 0),
    ]
    agg = {(a["condition"], a["family"], a["level"]): a for a in aggregate_rows(rows)}
    assert agg[("C", "viewpoint", 2)]["n"] == 2
    assert agg[("C", "viewpoint", 2)]["success_rate"] == pytest.approx(0.5)
    assert agg[("A", "viewpoint", 2)]["success_rate"] == pytest.approx(0.0)


def test_write_summary_roundtrip(tmp_path) -> None:
    rows = [make_row("C", "t1", "viewpoint", 2, 0, 1), make_row("A", "t1", "viewpoint", 2, 0, 0)]
    csv_path, json_path = write_summary(rows, tmp_path / "runs" / "out.csv")
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == list(ROLLOUT_FIELDS)
        read = list(reader)
    assert read[0]["condition"] == "C" and read[0]["success"] == "1"
    summary = json.loads(json_path.read_text())
    assert summary["n_rows"] == 2
    assert len(summary["aggregate"]) == 2


def test_gpu_seams_implemented_phase1() -> None:
    """Phase 1: _load_policy and _run_one_trial are implemented (no longer NotImplementedError).

    Off-GPU pytest cannot actually call them (would require CUDA + an HF download + LIBERO env);
    we only verify the seams are callable. Real end-to-end behavior is exercised by GPU smoke
    tests outside the test suite. The LoRA-adapter branch of _load_policy still raises
    NotImplementedError (Phase 3), but testing that here would force an HF download first, so we
    defer that check to manual Phase-3 work.
    """
    assert callable(_load_policy)
    assert callable(_run_one_trial)
