"""Phase 5 -- paired-statistics report over Phase 2 + Phase 4 per-episode CSVs.

Reads:
  analysis/runs/phase2_collapse.csv  -- condition A: clean + core4 perturbations.
  analysis/runs/phase4_recovery.csv  -- A_layout + B (all cells) + C (all cells).

Produces three pre-registered statistical tests over matched (family, level, task_id,
episode_index) episodes:

  H1 (LoRA + standard augmentation helps robustness, A vs B):
        SR_B > SR_A pooled across all perturbed cells (clean excluded). One McNemar test
        + one paired-bootstrap CI for the pooled delta. No multiplicity correction.

  H2 (Targeted augmentation vs standard, B vs C, in-distribution families only):
        For each in-distribution family f in {lighting, noise, texture}: SR_C(f) > SR_B(f)
        pooled across L2+L4. Three McNemar tests; Holm-Bonferroni over the family.

  H3 (Held-out generalization, A vs B on layout): one McNemar test + paired-bootstrap CI.
        layout never appeared in any augmentation, so this isolates LoRA's task-representation
        benefit from augmentation matching.

Off-GPU (numpy/scipy/pandas only). Writes a Markdown summary table + a per-family JSON for
downstream documentation (docs/PROBES.md, README results table).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from eval.bootstrap import bootstrap_ci
from eval.holm import holm_bonferroni
from eval.paired import mcnemar, paired_bootstrap_delta

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE2_CSV = REPO_ROOT / "analysis" / "runs" / "phase2_collapse.csv"
PHASE4_CSV = REPO_ROOT / "analysis" / "runs" / "phase4_recovery.csv"
OUT_MD = REPO_ROOT / "analysis" / "runs" / "phase5_summary.md"
OUT_JSON = REPO_ROOT / "analysis" / "runs" / "phase5_summary.json"

IN_DIST_FAMILIES = ("lighting", "noise", "texture")  # families C trained against
HELDOUT_FAMILY = "layout"                            # never augmented
ALL_PERT_FAMILIES = ("viewpoint", "lighting", "texture", "noise", "layout")
LEVELS = (2, 4)


def _load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _index_by_cell(rows: list[dict]) -> dict:
    """{(condition, family, level): {task_id -> [success per episode in input order]}}."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["condition"], r["family"], int(r["level"]))
        out[key][int(r["task_id"])].append(int(r["success"]))
    return out


def _matched_pair(cells_X, cells_Y, family, level):
    """Per-episode matched success arrays X vs Y on shared task_ids; preserves episode order."""
    cx = cells_X.get(("A" if False else "X", family, level), {})  # placeholder
    cx = cells_X[(cells_X._cond, family, level)] if hasattr(cells_X, "_cond") else None
    raise RuntimeError("use _matched_pair_v2")


def _matched_arrays(idx, cond_X, cond_Y, family, level):
    """Aligned per-episode 0/1 arrays for two conditions on the same (family, level).

    Pairs by task_id and episode index. If the two conditions evaluated different task_ids
    (e.g. noise cells used 12 tasks for A, 6 for B/C), the intersection is used and reported.
    """
    cx = idx.get((cond_X, family, level), {})
    cy = idx.get((cond_Y, family, level), {})
    shared = sorted(set(cx) & set(cy))
    a, b = [], []
    for tid in shared:
        xs, ys = cx[tid], cy[tid]
        m = min(len(xs), len(ys))
        a.extend(xs[:m]); b.extend(ys[:m])
    return a, b, shared


def _sr(arr): return 100.0 * sum(arr) / len(arr) if arr else float("nan")


def _fmt_pct_ci(p, lo, hi): return f"{100*p:5.1f}% [{100*lo:5.1f}, {100*hi:5.1f}]"


def _fmt_delta(d, lo, hi): return f"{100*d:+5.1f}pp [{100*lo:+5.1f}, {100*hi:+5.1f}]"


# -------------------------- Test runners ----------------------------------------------------

def _pool_pairs(idx, cond_X, cond_Y, families, levels):
    a_all, b_all, cells = [], [], []
    for fam in families:
        for lvl in levels:
            a, b, shared = _matched_arrays(idx, cond_X, cond_Y, fam, lvl)
            if not a:
                continue
            a_all.extend(a); b_all.extend(b)
            cells.append((fam, lvl, len(shared), len(a)))
    return a_all, b_all, cells


def test_H1_lora_helps(idx):
    """A vs B pooled across all perturbed families (clean excluded)."""
    a, b, cells = _pool_pairs(idx, "A", "B", ALL_PERT_FAMILIES, LEVELS)
    mc = mcnemar(b, a)  # b=treatment, a=control; A_only counts A-success-only
    delta, lo, hi = paired_bootstrap_delta(b, a)
    return {
        "name": "H1: SR_B > SR_A pooled across perturbations",
        "n_episodes": len(a),
        "n_cells": len(cells),
        "sr_A": _sr(a), "sr_B": _sr(b),
        "delta_pp": 100 * (sum(b)/len(b) - sum(a)/len(a)),
        "delta_ci_pp": (100*lo, 100*hi),
        "mcnemar": mc,
    }


def test_H2_targeted_vs_standard(idx):
    """Per-family B vs C on in-dist families; Holm-Bonferroni over the 3 families."""
    per_family = {}
    pvals = {}
    for fam in IN_DIST_FAMILIES:
        a, b, cells = _pool_pairs(idx, "B", "C", [fam], LEVELS)
        if not a:
            continue
        mc = mcnemar(b, a)  # b=C, a=B; so a_only=C-only, b_only=B-only
        delta, lo, hi = paired_bootstrap_delta(b, a)
        per_family[fam] = {
            "n_episodes": len(a),
            "sr_B": _sr(a), "sr_C": _sr(b),
            "delta_method_pp": 100 * (sum(b)/len(b) - sum(a)/len(a)),
            "delta_ci_pp": (100*lo, 100*hi),
            "mcnemar": mc,
        }
        pvals[fam] = mc["pvalue"]
    if pvals:
        adj = {name: (p_raw, p_adj, reject)
               for name, p_raw, p_adj, reject in holm_bonferroni(pvals)}
        for fam in per_family:
            p_raw, p_adj, reject = adj[fam]
            per_family[fam]["p_raw"] = p_raw
            per_family[fam]["p_holm"] = p_adj
            per_family[fam]["reject_holm"] = reject
    return {"name": "H2: SR_C > SR_B per in-dist family (Holm corrected)", "per_family": per_family}


def test_H3_heldout_lora(idx):
    """A vs B on held-out layout."""
    a, b, _ = _pool_pairs(idx, "A", "B", [HELDOUT_FAMILY], LEVELS)
    if not a:
        return {"name": "H3", "skipped": "no layout pairs available"}
    mc = mcnemar(b, a)
    delta, lo, hi = paired_bootstrap_delta(b, a)
    return {
        "name": "H3: SR_B > SR_A on held-out layout (LoRA task-rep benefit)",
        "n_episodes": len(a),
        "sr_A": _sr(a), "sr_B": _sr(b),
        "delta_pp": 100 * (sum(b)/len(b) - sum(a)/len(a)),
        "delta_ci_pp": (100*lo, 100*hi),
        "mcnemar": mc,
    }


# -------------------------- Cell-level SR table ---------------------------------------------

def cell_sr_table(idx):
    """{condition: {(family,level): (n, sr, lo, hi)}}; bootstrap CI per cell."""
    out: dict = defaultdict(dict)
    for (cond, fam, lvl), by_tid in idx.items():
        outcomes = [s for arr in by_tid.values() for s in arr]
        if not outcomes:
            continue
        sr, lo, hi = bootstrap_ci(outcomes)
        out[cond][(fam, lvl)] = (len(outcomes), sr, lo, hi)
    return out


def main() -> None:
    rows = _load_csv(PHASE2_CSV) + _load_csv(PHASE4_CSV)
    idx = _index_by_cell(rows)
    table = cell_sr_table(idx)
    h1 = test_H1_lora_helps(idx)
    h2 = test_H2_targeted_vs_standard(idx)
    h3 = test_H3_heldout_lora(idx)

    # ---- Markdown summary ----
    md = ["# Phase 5 -- Paired-statistics summary\n",
          "Generated from `analysis/runs/phase2_collapse.csv` + `analysis/runs/phase4_recovery.csv`.",
          "All CIs are 95% percentile bootstrap (10,000 resamples). Significance: McNemar exact "
          "(< 25 discordant pairs) or chi2 continuity-corrected.\n",
          "## Cell-level success rates (A / B / C)\n",
          "| family | level | A | B | C |",
          "|---|---|---|---|---|"]
    fam_order = ("clean", "viewpoint", "lighting", "texture", "noise", "layout")
    seen = set()
    for fam in fam_order:
        for lvl in ((0,) if fam == "clean" else LEVELS):
            row = [fam, str(lvl)]
            for cond in ("A", "B", "C"):
                cell = table.get(cond, {}).get((fam, lvl))
                row.append(_fmt_pct_ci(*cell[1:]) + f" (n={cell[0]})" if cell else "--")
            md.append("| " + " | ".join(row) + " |")
            seen.add((fam, lvl))

    md.append("\n## H1 -- LoRA + standard augmentation lifts robustness (A vs B, pooled)\n")
    md.append(f"- Pooled over {h1['n_cells']} perturbed cells, n={h1['n_episodes']} matched episodes.")
    md.append(f"- SR_A = {h1['sr_A']:.1f}%, SR_B = {h1['sr_B']:.1f}%")
    md.append(f"- delta = {h1['delta_pp']:+.1f}pp, 95% CI {h1['delta_ci_pp'][0]:+.1f} to {h1['delta_ci_pp'][1]:+.1f}")
    md.append(f"- McNemar: {h1['mcnemar']['method']}, p = {h1['mcnemar']['pvalue']:.4g} "
              f"(B-only wins={h1['mcnemar']['b_only']}, A-only wins={h1['mcnemar']['a_only']})")

    md.append("\n## H2 -- Targeted augmentation per in-dist family (B vs C, Holm corrected)\n")
    md.append("| family | n | SR_B | SR_C | delta (pp) [95% CI] | p_raw | p_Holm | reject @ 0.05 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for fam, r in h2["per_family"].items():
        md.append(f"| {fam} | {r['n_episodes']} | {r['sr_B']:.1f}% | {r['sr_C']:.1f}% | "
                  f"{r['delta_method_pp']:+.1f} [{r['delta_ci_pp'][0]:+.1f}, {r['delta_ci_pp'][1]:+.1f}] | "
                  f"{r['p_raw']:.3g} | {r['p_holm']:.3g} | {'YES' if r['reject_holm'] else 'no'} |")

    md.append("\n## H3 -- Held-out generalization on layout (A vs B)\n")
    if "skipped" in h3:
        md.append(f"_{h3['skipped']}_")
    else:
        md.append(f"- n={h3['n_episodes']} matched episodes (layout L2+L4 pooled)")
        md.append(f"- SR_A = {h3['sr_A']:.1f}%, SR_B = {h3['sr_B']:.1f}%")
        md.append(f"- delta = {h3['delta_pp']:+.1f}pp, 95% CI {h3['delta_ci_pp'][0]:+.1f} to {h3['delta_ci_pp'][1]:+.1f}")
        md.append(f"- McNemar: {h3['mcnemar']['method']}, p = {h3['mcnemar']['pvalue']:.4g}")

    md.append("\n## Headline numbers (point estimates for README/PROBES tables)\n")
    clean_a = table["A"].get(("clean", 0))
    if clean_a:
        md.append(f"- SR_A clean = {clean_a[1]*100:.1f}% (n={clean_a[0]})")
    md.append(f"- Delta_robust pooled = {h1['delta_pp']:+.1f}pp on A vs B (proxy for LoRA's net robustness gain)")
    md.append(f"- Held-out (layout) gain = {h3['delta_pp']:+.1f}pp on A vs B (LoRA's task-rep transfer)")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n")
    OUT_JSON.write_text(json.dumps(
        {"cell_table": {c: {f"{fam}_L{lvl}": v for (fam, lvl), v in cells.items()}
                        for c, cells in table.items()},
         "H1": h1, "H2": h2, "H3": h3}, indent=2, default=float))
    print(f"[Phase5] wrote {OUT_MD}")
    print(f"[Phase5] wrote {OUT_JSON}")
    print()
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
