"""Figure for the generalisation batch: native success and rescue/damage of the two pause arms per benchmark."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/swj/data/libero-runtime/simulations/control-scale")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 9.5,
                     "axes.titleweight": "semibold", "axes.titlelocation": "left", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5})


def title(ax, main, sub):
    ax.set_title(main, pad=9 + 11 * (sub.count("\n") + 1))
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", linespacing=1.25)


rep = json.loads((OUT / "report.json").read_text())
benches = [b for b in ("libero10", "pro-swap", "plus") if b in rep]
labels = {"libero10": "LIBERO-10\n(init 10-29)", "pro-swap": "Pro swap\n(init 0-19)", "plus": "Plus\n(7 cat. x 3)"}
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.8))
x = np.arange(len(benches))
# (a) native success rate
for xi, b in zip(x, benches):
    n, k = rep[b]["native_episodes"], rep[b]["native_successes"]
    ax1.bar(xi, k / max(n, 1), width=0.5, color=BLUE, edgecolor=SURFACE)
    ax1.annotate("%d/%d" % (k, n), (xi, k / max(n, 1)), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK2)
ax1.set_xticks(x, [labels[b] for b in benches]); ax1.set_ylim(0, 1); ax1.grid(axis="x", visible=False)
title(ax1, "(a) Native success rate on the new episodes", "same flow noise seed 42; these runs are the paired baselines")
ax1.set_ylabel("success rate")
# (b) rescue fraction per arm with CI
w = 0.36
for j, (arm, color, name) in enumerate((("hold16", BLUE, "hold16: recurrence-triggered pause"), ("hold16_random", ORANGE, "hold16_random: schedule-matched random pause"))):
    for xi, b in zip(x, benches):
        r = rep[b].get(arm)
        if not r or not r["failures"]: continue
        frac = r["rescued"] / r["failures"]; lo, hi = r["rescued_ci"]
        xx = xi + (j - 0.5) * w
        ax2.bar(xx, frac, width=w, color=color, edgecolor=SURFACE, label=name if xi == x[0] else None)
        ax2.plot([xx, xx], [lo, hi], color=INK, linewidth=1.2)
        ax2.annotate("%d/%d" % (r["rescued"], r["failures"]), (xx, hi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
ax2.set_xticks(x, [labels[b] for b in benches]); ax2.set_ylim(0, 0.35); ax2.grid(axis="x", visible=False); ax2.legend(loc="upper right")
title(ax2, "(b) Rescued fraction of native failures", "pause arms paired with native; whiskers = Wilson 95% CI")
ax2.set_ylabel("rescued / native failures")
# (c) damage fraction per arm
for j, (arm, color, name) in enumerate((("hold16", BLUE, "hold16"), ("hold16_random", ORANGE, "hold16_random"))):
    for xi, b in zip(x, benches):
        r = rep[b].get(arm)
        if not r or not r["successes"]: continue
        frac = r["damaged"] / r["successes"]; lo, hi = r["damaged_ci"]
        xx = xi + (j - 0.5) * w
        ax3.bar(xx, frac, width=w, color=color, edgecolor=SURFACE)
        ax3.plot([xx, xx], [lo, hi], color=INK, linewidth=1.2)
        ax3.annotate("%d/%d\ntrig %d" % (r["damaged"], r["successes"], r["triggered_successes"]), (xx, hi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
ax3.set_xticks(x, [labels[b] for b in benches]); ax3.set_ylim(0, 0.35); ax3.grid(axis="x", visible=False)
title(ax3, "(c) Damaged fraction of native successes", "labels: damaged / successes and how many successes were paused at all")
ax3.set_ylabel("damaged / native successes")
fig.suptitle("Generalisation of the recurrence-triggered pause: %d new episodes, %d runs" % (sum(rep[b]["native_episodes"] for b in benches), rep["runs"]["completed"]), x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.5)
fig.savefig(OUT / "control-scale.png", dpi=150)
print("saved", OUT / "control-scale.png")
