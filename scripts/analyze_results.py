"""Analyze a per-trial rollout CSV into the full paired-statistics report (text + JSON).

Consumes the canonical schema ``condition, task_id, family, level, seed, success`` produced by
``eval/run_rollout.py`` and emits the report defined in ``docs/EVALUATION.md``: per-family SR with
bootstrap 95% CI, the headline paired ``Δ_method (C−B)`` with paired-bootstrap CI + McNemar (matched
by task), Holm–Bonferroni across families, Recovery, the **held-out generalization gap** (Probe 1),
and an rliable-style aggregate. If the CSV also carries an optional ``instruction`` column, the
**language-conditioning probe** (Probe 2; paired ΔSR per ablated variant) is added automatically.
The two probes are the headline (see ``docs/PROBES.md``); the statistics are supporting infra. Fully off-GPU.

    uv run python -m scripts.analyze_results --csv analysis/runs/all.csv --out analysis/report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

from eval.stats import build_report, format_text, read_rows

REPO = Path(__file__).resolve().parent.parent


def _trained_from_config(eval_config: str) -> list[str]:
    cfg = yaml.safe_load((REPO / eval_config).read_text())
    return cfg.get("trained_families") or []


def _json_safe(obj):
    """Recursively convert NaN floats to ``None`` so the JSON is valid for other tools."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze a rollout CSV into a paired-statistics report.")
    p.add_argument("--csv", required=True, help="Per-trial rollout CSV (run_rollout schema).")
    p.add_argument("--eval-config", default="configs/eval/default.yaml")
    p.add_argument("--trained-families", nargs="*", default=None,
                   help="In-dist families (default: trained_families from the eval config).")
    p.add_argument("--cond-base", default="A")
    p.add_argument("--cond-b", default="B")
    p.add_argument("--cond-c", default="C")
    p.add_argument("--n-resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Output basename; writes <out>.txt and <out>.json.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_rows(args.csv)
    trained = (args.trained_families if args.trained_families is not None
               else _trained_from_config(args.eval_config))
    report = build_report(
        rows, trained_families=trained, cond_base=args.cond_base,
        cond_b=args.cond_b, cond_c=args.cond_c, n_resamples=args.n_resamples, seed=args.seed,
    )
    text = format_text(report)
    print(text)
    if args.out:
        base = Path(args.out)
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".txt").write_text(text + "\n")
        base.with_suffix(".json").write_text(json.dumps(_json_safe(report), indent=2))
        print(f"\nwrote {base.with_suffix('.txt')} and {base.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
