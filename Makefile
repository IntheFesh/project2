# VLA-Collapse-Recover -- common commands (run `make help`).
#
# Off-GPU targets (setup, stats, test, lint) need only `make setup`.
# GPU targets (smoke, train-B, train-C, eval) require the rented RTX 5090 stack
# (uv sync --extra gpu + cu128 torch + LeRobot from source); see docs/REPRODUCING.md.

PY := uv run python

.DEFAULT_GOAL := help
.PHONY: help setup smoke train-B train-C eval stats test lint clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Off-GPU: create the core+dev env (uv sync). On the rental add: uv sync --extra gpu
	uv sync

smoke:  ## GPU: 60-step LoRA smoke to validate train/aug/CLI wiring before scaling
	$(PY) -m train.train_lora --condition C --smoke-test

train-B:  ## GPU: train condition B (LoRA + standard/default augmentation)
	$(PY) -m train.train_lora --condition B

train-C:  ## GPU: train condition C (LoRA + perturbation-targeted aug; 4 CORE families)
	$(PY) -m train.train_lora --condition C

eval:  ## GPU: Phase 4 recovery rollouts (A layout + B/C all cells) -> phase4_recovery.csv
	$(PY) -m eval.runners.phase4_recovery

stats:  ## Off-GPU: regenerate phase5_summary.{md,json} + the recovery heatmap from the CSVs
	$(PY) -m eval.runners.phase5_stats
	$(PY) analysis/make_heatmap.py

test:  ## Off-GPU: run the pure-logic test suite (no GPU / no lerobot needed)
	uv run pytest

lint:  ## Off-GPU: ruff lint
	uv run ruff check .

clean:  ## Remove caches (keeps .venv and analysis/runs)
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
