"""Generate the recovery-curve heatmap for README / one-pager.

Reads phase2_collapse.csv + phase4_recovery.csv, produces a per-condition x per-family-level
success-rate heatmap with annotated values. Saved to analysis/runs/recovery_heatmap.{png,svg}.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CSVS = [REPO_ROOT / "analysis/runs/phase2_collapse.csv",
        REPO_ROOT / "analysis/runs/phase4_recovery.csv"]
OUT_BASE = REPO_ROOT / "analysis/runs/recovery_heatmap"

COL_ORDER = [("clean", 0), ("viewpoint", 2), ("viewpoint", 4),
             ("lighting", 2), ("lighting", 4), ("texture", 2), ("texture", 4),
             ("noise", 2), ("noise", 4), ("layout", 2), ("layout", 4)]
COL_LABELS = ["clean", "view L2", "view L4", "light L2", "light L4",
              "tex L2", "tex L4", "noise L2", "noise L4", "layout L2", "layout L4"]
ROW_ORDER = ["A", "B", "C"]
ROW_LABELS = [
    "A: SmolVLA (base)",
    "B: + LoRA + standard aug",
    "C: + LoRA + targeted aug",
]


def main():
    agg = defaultdict(lambda: [0, 0])  # (cond, fam, lvl) -> [succ, n]
    for csv_path in CSVS:
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                k = (r["condition"], r["family"], int(r["level"]))
                agg[k][0] += int(r["success"])
                agg[k][1] += 1

    M = np.full((len(ROW_ORDER), len(COL_ORDER)), np.nan)
    N = np.full_like(M, 0)
    for i, cond in enumerate(ROW_ORDER):
        for j, key in enumerate(COL_ORDER):
            s, n = agg.get((cond, *key), (0, 0))
            if n:
                M[i, j] = 100 * s / n
                N[i, j] = n

    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    cmap = plt.cm.RdYlGn  # red (bad) -> green (good)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(COL_LABELS)))
    ax.set_xticklabels(COL_LABELS, rotation=20, ha="right")
    ax.set_yticks(range(len(ROW_LABELS)))
    ax.set_yticklabels(ROW_LABELS)
    ax.set_xlabel("perturbation family x level")
    ax.set_title("Recovery heatmap -- success rate (%), LIBERO-Plus spatial suite",
                 loc="left", fontsize=11)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "--" if np.isnan(v) else f"{v:.0f}"
            color = "white" if (np.isnan(v) or v < 30 or v > 70) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("Success rate (%)")
    plt.tight_layout()
    for ext in ("png", "svg"):
        out = OUT_BASE.with_suffix(f".{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
