"""Figure for control batches: rescue/damage per arm with Wilson CIs, RR after interventions, trigger timing."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTS = [Path(p) for p in sys.argv[1:]] or [Path("/home/swj/data/libero-runtime/simulations/control-r1")]
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


arms, res = [], {}
for out in OUTS:
    a = json.loads((out / "analysis.json").read_text())
    for k, v in a.items():
        if k != "native":
            arms.append(k); res[k] = v
native = json.loads((OUTS[0] / "analysis.json").read_text())["native"]
ORDER = ["hold16", "withdraw", "withdraw8", "withdraw_early", "retrace", "escalate", "persist", "resample_escape", "resample_random", "reprompt", "noise2"]
arms = [a for a in ORDER if a in res] + [a for a in arms if a not in ORDER]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.25, 1, 1]})
y = np.arange(len(arms))[::-1]
for yi, a in zip(y, arms):
    r = res[a]
    rr, rc = r["rescued"] / max(r["failures"], 1), r["rescued_ci"]
    dr, dc = (r["damaged"] / max(r["successes_native"], 1)) if r["successes_native"] else np.nan, r["damaged_ci"]
    ax1.plot(rc, [yi + 0.15, yi + 0.15], color=BLUE, linewidth=2, solid_capstyle="round")
    ax1.plot([rr], [yi + 0.15], color=BLUE, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
    ax1.annotate("rescued %d/%d" % (r["rescued"], r["failures"]), (max(rc[1], rr) if np.isfinite(rc[1]) else rr, yi + 0.15), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    if np.isfinite(dr):
        ax1.plot(dc, [yi - 0.15, yi - 0.15], color=ORANGE, linewidth=2, solid_capstyle="round")
        ax1.plot([dr], [yi - 0.15], color=ORANGE, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
        ax1.annotate("damaged %d/%d" % (r["damaged"], r["successes_native"]), (max(dc[1], dr), yi - 0.15), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
ax1.set_yticks(y, arms); ax1.set_xlim(0, 0.32)
ax1.plot([], [], color=BLUE, marker="o", label="rescued / %d native failures" % (native["episodes"] - native["successes"]))
ax1.plot([], [], color=ORANGE, marker="o", label="damaged / %d native successes" % native["successes"])
ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8); ax1.grid(axis="y", visible=False)
title(ax1, "(a) Rescue and damage per arm", "paired with the native run (same flow noise); bars = Wilson 95%% CI\nnative %d/%d successes; damage measured only where shown" % (native["successes"], native["episodes"]))
ax1.set_xlabel("fraction")
# (b) RR after interventions
for yi, a in zip(y, arms):
    r = res[a]
    ax2.barh(yi, r["rr_after_mean"], height=0.36, color=BLUE, edgecolor=SURFACE)
    ax2.annotate("%.2f, below theta %.0f%%" % (r["rr_after_mean"], 100 * r["rr_drop_frac"]), (r["rr_after_mean"], yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
ax2.axvline(0.4, color=AXIS, linewidth=1); ax2.set_yticks(y, arms); ax2.set_xlim(0, 1.45); ax2.grid(axis="y", visible=False)
title(ax2, "(b) Recurrence after intervention", "mean RR over the 6 queries after each intervention\nlabel = mean, share of interventions with RR < 0.4 within 6 queries")
ax2.set_xlabel("recurrence rate")
# (c) trigger timing and cost
for yi, a in zip(y, arms):
    r = res[a]
    ax3.barh(yi, r["median_first_trigger_q"] or 0, height=0.36, color=AQUA, edgecolor=SURFACE)
    ax3.annotate("q%.0f, %.1f int., %.0f phys steps, %.0f cand." % (r["median_first_trigger_q"] or 0, r["interventions_per_triggered"], r["mean_physical_steps"], r["mean_candidate_calls"]),
                 (r["median_first_trigger_q"] or 0, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
ax3.set_yticks(y, arms); ax3.set_xlim(0, 110); ax3.grid(axis="y", visible=False)
title(ax3, "(c) Trigger timing and cost", "bar = median first-trigger query among triggered failures\nlabel: interventions per triggered run, physical steps, candidate calls (means)")
ax3.set_xlabel("query index (10 actions each)")
fig.suptitle("Recurrence-triggered control via parallel API requests: %s" % ", ".join(o.name for o in OUTS), x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
fig.tight_layout(rect=(0, 0.02, 1, 0.95), w_pad=1.2)
fig.savefig(OUTS[-1] / "control-results.png", dpi=150)
print("saved", OUTS[-1] / "control-results.png")
