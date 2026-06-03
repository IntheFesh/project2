#!/usr/bin/env bash
# =============================================================================
# VLA-Collapse-Recover — one-click setup / test / verify / deploy.
#
#   ./run.sh            off-GPU one-click: setup + test + verify + next steps
#   ./run.sh setup      create the lightweight core+dev env (uv sync)
#   ./run.sh test       ruff lint + pytest
#   ./run.sh verify     scripts/verify_env.py (PASSes only on the rented sm_120 card)
#   ./run.sh budget     project the GPU-day budget (override: SEC_PER_EP=.. TRAIN_H=..)
#   ./run.sh smoke ...   smoke-time rollouts -> budget (pass-through flags)
#   ./run.sh gpu        RENTAL: install cu128 torch + GPU stack + LeRobot (+LIBERO-Plus)
#   ./run.sh deploy     RENTAL one-click: setup + gpu + test + verify
#   ./run.sh clean      remove .venv and caches
#
# The default (off-GPU) NEVER triggers a multi-GB download. The GPU stack is only
# installed by the explicit, user-initiated `gpu` / `deploy` commands.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else B=""; G=""; Y=""; R=""; N=""; fi
log()  { printf "%s==> %s%s\n" "$B" "$*" "$N"; }
ok()   { printf "%sOK%s %s\n" "$G" "$N" "$*"; }
warn() { printf "%s!! %s%s\n" "$Y" "$*" "$N"; }
die()  { printf "%sxx %s%s\n" "$R" "$*" "$N" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
gpu_present() { have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; }
require_uv() {
  have uv || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh   then re-run."
}

cmd_setup() {
  require_uv
  log "uv sync — lightweight core + dev (off-GPU)"
  uv sync
  ok "environment ready in .venv"
}

cmd_test() {
  require_uv
  log "ruff lint"; uv run ruff check .
  log "pytest";    uv run pytest -q
  ok "tests + lint passed"
}

cmd_verify() {
  require_uv
  log "verify_env.py — sm_120 / cu128 / torch>=2.7 / LeRobot (PASSes only on the rented 5090)"
  uv run python scripts/verify_env.py || warn "env not GPU-ready (expected when off-GPU)."
}

cmd_budget() {
  require_uv
  log "GPU-day budget projection (SEC_PER_EP=${SEC_PER_EP:-30}s, TRAIN_H=${TRAIN_H:-4}h/run)"
  uv run python -m scripts.estimate_budget \
    --sec-per-episode "${SEC_PER_EP:-30}" --train-hours-per-run "${TRAIN_H:-4}" "$@" || true
}

cmd_smoke() {
  require_uv
  log "smoke-timing -> budget projection"
  uv run python -m scripts.smoke_timing "$@"
}

cmd_gpu() {
  require_uv
  if gpu_present; then ok "NVIDIA GPU detected"; else warn "no GPU detected — installing cu128 stack anyway (as requested)."; fi
  log "cu128 PyTorch (requirements-gpu.txt) — large download"
  uv pip install -r requirements-gpu.txt
  log "GPU-adjacent PyPI stack (uv sync --extra gpu)"
  uv sync --extra gpu
  log "LeRobot (SmolVLA / pi0.5) from source"
  uv pip install "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot"
  if [ -d third_party/LIBERO-Plus ]; then
    log "LIBERO-Plus (editable drop-in for libero)"; uv pip install -e third_party/LIBERO-Plus
  else
    warn "third_party/LIBERO-Plus not found. Clone it, then: uv pip install -e third_party/LIBERO-Plus"
  fi
  cmd_verify
  warn "Rollout/training entry points (Phases 1-4,6) are scaffolded with NotImplementedError"
  warn "seams — implement per the build plan before scaling. Off-GPU stats/budget run today."
}

cmd_clean() {
  log "removing .venv + caches"
  rm -rf .venv .pytest_cache .ruff_cache
  find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  ok "cleaned"
}

cmd_all() {
  cmd_setup; cmd_test; cmd_verify
  echo
  log "Next steps"
  if gpu_present; then
    printf "  GPU detected — deploy the full stack with: %s./run.sh deploy%s\n" "$B" "$N"
  else
    printf "  Off-GPU dev box. On the rented RTX 5090: %s./run.sh deploy%s\n" "$B" "$N"
  fi
  printf "  Gate the matrix under the 5-day cap:  %s./run.sh budget%s\n" "$B" "$N"
  printf "  See the tutorial in %sREADME.md%s.\n" "$B" "$N"
}

usage() { sed -n '/^# ===/,/^# ===/p' "$0" | sed 's/^# \{0,1\}//'; }

case "${1:-all}" in
  all)            cmd_all ;;
  setup)          cmd_setup ;;
  test)           cmd_test ;;
  verify)         cmd_verify ;;
  budget)  shift; cmd_budget "$@" ;;
  smoke)   shift; cmd_smoke "$@" ;;
  gpu)            cmd_gpu ;;
  deploy)         cmd_setup; cmd_gpu; cmd_test ;;
  clean)          cmd_clean ;;
  -h|--help|help) usage ;;
  *)              die "unknown command: $1   (try ./run.sh help)" ;;
esac
