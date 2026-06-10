"""Build the full paired-statistics evaluation report from per-trial rollout rows.

Pure (numpy/scipy); off-GPU and unit-tested. Consumes the canonical rollout CSV schema
``condition, task_id, family, level, seed, success`` (see ``eval/run_rollout.py``) and produces a
JSON-able report + a human-readable rendering. All statistics are re-used from
``eval.metrics`` / ``eval.bootstrap`` / ``eval.paired`` / ``eval.holm`` -- no logic is duplicated.

Protocol (see ``docs/EVALUATION.md``): per-family SR with bootstrap 95% CI; the headline paired
``Δ_method = SR_C − SR_B`` with a paired-bootstrap CI + McNemar, **matched by ``(task_id, level,
seed)``**; Holm–Bonferroni across families; Recovery vs the base model; the in-dist/held-out
generalization gap; and an rliable-style aggregate (IQM of per-cell SRs + bootstrap CI).
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from eval.stats.bootstrap import bootstrap_ci
from eval.stats.holm import holm_bonferroni
from eval.metrics import generalization_gap, recovery, success_rate
from eval.stats.paired import mcnemar, paired_bootstrap_delta
from eval.probe import ABLATED_VARIANTS, language_sensitivity_paired

CLEAN_FAMILY = "clean"
_NON_PERTURBED = {CLEAN_FAMILY, "unknown"}


# ------------------------------------------------------------------- io / helpers --
def read_rows(path: str | Path) -> list[dict]:
    """Read a per-trial rollout CSV into typed row dicts."""
    rows: list[dict] = []
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "condition": r["condition"], "task_id": r["task_id"], "family": r["family"],
                "level": int(r["level"]), "seed": int(r["seed"]), "success": int(r["success"]),
                # Optional probe column: present only in language-probe CSVs (None otherwise).
                "instruction": r.get("instruction"),
            })
    return rows


def has_instructions(rows: Sequence[Mapping]) -> bool:
    """True if any row carries a (non-empty) ``instruction`` -- i.e. it is a language-probe CSV."""
    return any(r.get("instruction") for r in rows)


def perturbed_families(rows: Sequence[Mapping]) -> list[str]:
    """Sorted perturbed family names present (excludes clean/unknown)."""
    return sorted({r["family"] for r in rows} - _NON_PERTURBED)


def _successes(rows: Sequence[Mapping], condition: str, family: str | None) -> list[int]:
    return [
        int(r["success"]) for r in rows
        if r["condition"] == condition and (family is None or r["family"] == family)
    ]


def _sr(rows: Sequence[Mapping], condition: str, family: str | None) -> float:
    succ = _successes(rows, condition, family)
    return success_rate(succ) if succ else math.nan


def _pooled_sr(rows: Sequence[Mapping], condition: str, families: Sequence[str]) -> float:
    succ = [int(r["success"]) for r in rows
            if r["condition"] == condition and r["family"] in set(families)]
    return success_rate(succ) if succ else math.nan


def _safe_recovery(sr_model: float, sr_base: float, sr_base_clean: float) -> float:
    if any(math.isnan(x) for x in (sr_model, sr_base, sr_base_clean)):
        return math.nan
    return recovery(sr_model, sr_base, sr_base_clean)


def matched_pairs(
    rows: Sequence[Mapping], cond_a: str, cond_b: str, family: str | None = None
) -> tuple[list[int], list[int]]:
    """Aligned per-trial outcomes for two conditions, matched on ``(task_id, level, seed)``."""
    def index(cond: str) -> dict[tuple[str, int, int], int]:
        return {
            (r["task_id"], r["level"], r["seed"]): int(r["success"])
            for r in rows if r["condition"] == cond and (family is None or r["family"] == family)
        }

    a_idx, b_idx = index(cond_a), index(cond_b)
    keys = sorted(set(a_idx) & set(b_idx))
    return [a_idx[k] for k in keys], [b_idx[k] for k in keys]


def _instr_index(rows: Sequence[Mapping], condition: str, instruction: str) -> dict:
    return {
        (r["task_id"], r["level"], r["seed"]): int(r["success"])
        for r in rows if r["condition"] == condition and r.get("instruction") == instruction
    }


def language_probe_report(rows: Sequence[Mapping], *, n_resamples: int, seed: int) -> dict:
    """Per-condition language sensitivity: paired ΔSR (correct vs each ablated instruction).

    Matched within a condition on ``(task_id, level, seed)``; only conditions carrying a ``correct``
    instruction and at least one ablated variant are included. Returns
    ``{condition: {variant: language_sensitivity_paired(...)}}``.
    """
    out: dict[str, dict] = {}
    for cond in sorted({r["condition"] for r in rows if r.get("instruction")}):
        correct = _instr_index(rows, cond, "correct")
        if not correct:
            continue
        variants: dict[str, dict] = {}
        for variant in ABLATED_VARIANTS:
            ablated = _instr_index(rows, cond, variant)
            keys = sorted(set(correct) & set(ablated))
            if not keys:
                continue
            variants[variant] = language_sensitivity_paired(
                [correct[k] for k in keys], [ablated[k] for k in keys],
                n_resamples=n_resamples, seed=seed,
            )
        if variants:
            out[cond] = variants
    return out


def _iqm(values: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50% (rliable-style robust aggregate)."""
    xs = sorted(values)
    if not xs:
        return math.nan
    lo = len(xs) // 4
    middle = xs[lo: len(xs) - lo] or xs
    return sum(middle) / len(middle)


# ------------------------------------------------------------------- report pieces --
def per_family_sr(
    rows: Sequence[Mapping], condition: str, families: Sequence[str], *,
    n_resamples: int, seed: int,
) -> dict[str, dict]:
    """Per-family success rate + bootstrap 95% CI for one condition."""
    out: dict[str, dict] = {}
    for fam in families:
        succ = _successes(rows, condition, fam)
        if not succ:
            continue
        point, lo, hi = bootstrap_ci(succ, n_resamples=n_resamples, seed=seed)
        out[fam] = {"sr": point, "ci_lo": lo, "ci_hi": hi, "n": len(succ)}
    return out


def method_gap_by_family(
    rows: Sequence[Mapping], families: Sequence[str], *,
    cond_c: str, cond_b: str, n_resamples: int, seed: int,
) -> dict[str, dict]:
    """Headline paired ``Δ_method = SR_C − SR_B`` per family (paired-bootstrap CI + McNemar)."""
    out: dict[str, dict] = {}
    for fam in families:
        c_out, b_out = matched_pairs(rows, cond_c, cond_b, fam)
        if not c_out:
            continue
        delta, lo, hi = paired_bootstrap_delta(c_out, b_out, n_resamples=n_resamples, seed=seed)
        mc = mcnemar(c_out, b_out)
        out[fam] = {
            "delta_method": delta, "ci_lo": lo, "ci_hi": hi,
            "mcnemar_p": mc["pvalue"], "n_pairs": len(c_out),
        }
    # Holm-Bonferroni across families on the McNemar p-values.
    if out:
        pvals = {fam: out[fam]["mcnemar_p"] for fam in out}
        for fam, _raw, adj, reject in holm_bonferroni(pvals):
            out[fam]["holm_p"] = adj
            out[fam]["reject"] = reject
    return out


def aggregate_summary(
    rows: Sequence[Mapping], condition: str, *, n_resamples: int, seed: int
) -> dict:
    """rliable-style aggregate for a condition over perturbed families (pooled SR + IQM of cells)."""
    pooled = [int(r["success"]) for r in rows
              if r["condition"] == condition and r["family"] not in _NON_PERTURBED]
    cells: dict[tuple[str, int], list[int]] = defaultdict(list)
    for r in rows:
        if r["condition"] == condition and r["family"] not in _NON_PERTURBED:
            cells[(r["family"], r["level"])].append(int(r["success"]))
    cell_srs = [success_rate(v) for v in cells.values()]
    if not pooled:
        return {"sr": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "n": 0,
                "iqm_cell_sr": math.nan, "n_cells": 0}
    point, lo, hi = bootstrap_ci(pooled, n_resamples=n_resamples, seed=seed)
    return {"sr": point, "ci_lo": lo, "ci_hi": hi, "n": len(pooled),
            "iqm_cell_sr": _iqm(cell_srs), "n_cells": len(cell_srs)}


def build_report(
    rows: Sequence[Mapping], *,
    trained_families: Sequence[str],
    cond_base: str = "A", cond_b: str = "B", cond_c: str = "C",
    n_resamples: int = 10_000, seed: int = 0,
) -> dict:
    """Assemble the full evaluation report dict from per-trial ``rows``."""
    families = perturbed_families(rows)
    trained = [f for f in trained_families if f in families]
    held_out = [f for f in families if f not in set(trained_families)]
    conditions = sorted({r["condition"] for r in rows})

    per_family = {
        cond: per_family_sr(rows, cond, families, n_resamples=n_resamples, seed=seed)
        for cond in conditions
    }
    method_gap = method_gap_by_family(
        rows, families, cond_c=cond_c, cond_b=cond_b, n_resamples=n_resamples, seed=seed
    )
    sr_base_clean = _sr(rows, cond_base, CLEAN_FAMILY)
    recovery_c = {
        fam: _safe_recovery(_sr(rows, cond_c, fam), _sr(rows, cond_base, fam), sr_base_clean)
        for fam in families
    }
    rec_in = _safe_recovery(
        _pooled_sr(rows, cond_c, trained), _pooled_sr(rows, cond_base, trained), sr_base_clean
    )
    rec_held = _safe_recovery(
        _pooled_sr(rows, cond_c, held_out), _pooled_sr(rows, cond_base, held_out), sr_base_clean
    ) if held_out else math.nan
    aggregate = {
        cond: aggregate_summary(rows, cond, n_resamples=n_resamples, seed=seed)
        for cond in conditions
    }
    report = {
        "conditions": conditions,
        "families": families,
        "trained_families": trained,
        "held_out_families": held_out,
        "per_family_sr": per_family,
        "method_gap_C_minus_B": method_gap,
        "recovery_C": recovery_c,
        "generalization_gap": {
            "recovery_in_dist": rec_in,
            "recovery_held_out": rec_held,
            "gap": generalization_gap(rec_in, rec_held),
        },
        "aggregate": aggregate,
        "clean_sr_base": sr_base_clean,
    }
    # Probe 2 (language conditioning): only when the CSV carries an `instruction` column.
    if has_instructions(rows):
        report["language_probe"] = language_probe_report(rows, n_resamples=n_resamples, seed=seed)
    return report


def _fmt(x: float) -> str:
    return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.3f}"


def format_text(report: Mapping) -> str:
    """Render a :func:`build_report` dict as a human-readable text block."""
    lines = ["=== VLA-Collapse-Recover evaluation report ==="]
    lines.append(f"conditions={report['conditions']} families={report['families']}")
    lines.append(f"trained(in-dist)={report['trained_families']} held-out={report['held_out_families']}")
    lines.append("")
    lines.append("Headline paired Δ_method (C − B), matched by (task_id, level, seed):")
    for fam, g in report["method_gap_C_minus_B"].items():
        lines.append(
            f"  {fam:<12} Δ={_fmt(g['delta_method'])} "
            f"[{_fmt(g['ci_lo'])}, {_fmt(g['ci_hi'])}]  McNemar p={_fmt(g['mcnemar_p'])} "
            f"Holm p={_fmt(g.get('holm_p', float('nan')))} "
            f"{'REJECT' if g.get('reject') else 'keep'} (n={g['n_pairs']})"
        )
    gg = report["generalization_gap"]
    lines.append("")
    lines.append(
        f"Generalization gap = Recovery_in_dist({_fmt(gg['recovery_in_dist'])}) − "
        f"Recovery_held_out({_fmt(gg['recovery_held_out'])}) = {_fmt(gg['gap'])}"
    )
    lines.append("")
    lines.append("Recovery_C by family:")
    for fam, rec in report["recovery_C"].items():
        lines.append(f"  {fam:<12} {_fmt(rec)}")
    lines.append("")
    lines.append("Aggregate (rliable-style: pooled SR + 95% CI, IQM of per-cell SRs):")
    for cond, a in report["aggregate"].items():
        lines.append(
            f"  {cond}: SR={_fmt(a['sr'])} [{_fmt(a['ci_lo'])}, {_fmt(a['ci_hi'])}] "
            f"IQM={_fmt(a['iqm_cell_sr'])} (n={a['n']}, cells={a['n_cells']})"
        )
    if report.get("language_probe"):
        lines.append("")
        lines.append("Language-conditioning probe (paired ΔSR = SR_correct − SR_ablated):")
        for cond, variants in report["language_probe"].items():
            for variant, r in variants.items():
                lines.append(
                    f"  {cond} correct→{variant:<10} ΔSR={_fmt(r['delta'])} "
                    f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] McNemar p={_fmt(r['pvalue'])} "
                    f"(n={r['n']})"
                )
    return "\n".join(lines)
