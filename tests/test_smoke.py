"""Phase 0 smoke test: the scaffold imports cleanly and basic arithmetic holds.

Off-GPU; no heavy deps required (modules avoid importing torch/lerobot at module scope).
"""

import importlib

import pytest

MODULES = [
    "eval.metrics",
    "eval.bootstrap",
    "eval.paired",
    "eval.holm",
    "eval.run_rollout",
    "eval.record_demo",
    "perturb.libero_plus_wrapper",
    "data.augment.visual_aug",
    "train.train_lora",
    "train.feature_mod",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Every scaffold module imports without requiring torch/lerobot/a GPU."""
    importlib.import_module(module_name)


def test_trivial() -> None:
    assert 1 + 1 == 2
