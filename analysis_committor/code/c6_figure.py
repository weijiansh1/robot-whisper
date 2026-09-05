"""C6 — 主图：内部分叉 vs 身体分叉 vs 命运可读性。"""
import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent.parent
FEATS = [("late_flow_volatility", "V^late", "#c0392b"),
         ("route_acceleration", "A^route", "#e67e22"),
         ("gate_entropy", "gate entropy", "#2980b9"),
         ("token_consensus", "token consensus", "#16a085")]


def load(corpus):
    return json.load(open(OUT / "out" / f"c3_fate_curve_{corpus}.json"))


fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex="col")
for col, corpus in enumerate(("main16x32", "grid50x8")):
    rows = load(corpus)
    q = np.array([r["q"] for r in rows])
    keep = q <= 28
    q = q[keep]
    dph = np.array([r["d_phys"] for r in rows])[keep]
    drt = np.array([r["route_sib_over_between"] for r in rows])[keep]
    ncell = np.array([r["n_cells"] for r in rows])[keep]

    ax = axes[0, col]
    ax.plot(q, dph * 100, "o-", color="#34495e", ms=3, lw=1.6, label="body fork  D_phys (cm)")
    ax.set_ylabel("sibling physical separation (cm)")
    ax.axhline(0, color="k", lw=.6)
    ax2 = ax.twinx()
    ax2.plot(q, drt, "s--", color="#8e44ad", ms=3, lw=1.4,
             label="internal fork  D_route / between-state")
    ax2.set_ylabel("routing sibling sep. / between-init-state sep.", color="#8e44ad")
    ax2.tick_params(axis="y", colors="#8e44ad")
    ax.set_title(f"{corpus}: at q=0 bodies identical, routing already {drt[0]*100:.0f}% apart")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

    ax = axes[1, col]
    for f, lab, c in FEATS:
        v = np.array([r[f"auc_{f}"] for r in rows])[keep]
        ax.plot(q, v, "-", color=c, lw=1.5, label=lab)
    ax.axhline(.5, color="k", lw=.8, ls=":")
    ax.axvspan(-0.5, 5.5, color="#bdc3c7", alpha=.28, lw=0)
    ax.text(2.4, .70, "bodies not yet apart\nfate unreadable", ha="center", fontsize=8, color="#555")
    ax.set_ylim(.25, .78)
    ax.set_xlabel("within-episode query index q")
    ax.set_ylabel("within-cell AUC (predict this sibling fails)")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    axt = ax.twinx()
    axt.plot(q, ncell, color="#95a5a6", lw=.8, alpha=.7)
    axt.set_ylabel("# sensitive cells", color="#95a5a6", fontsize=8)
    axt.tick_params(axis="y", colors="#95a5a6", labelsize=7)

fig.suptitle("Siblings from one initial state: the computation forks first; fate becomes readable only after the bodies separate", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .96])
p = OUT / "figs" / "fate_determination.png"
fig.savefig(p, dpi=150)
print("wrote", p)
