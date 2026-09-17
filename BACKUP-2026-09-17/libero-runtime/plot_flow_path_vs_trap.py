"""Figure for the bending-vs-trap analysis (reads saved rows and trap-analysis.json)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/swj/data/libero-runtime/samples/flow-path-straightness-20260916")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.titlecolor": INK, "axes.titlesize": 9.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5,
})
LINE = dict(linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)


def title(ax, main, sub):
    ax.set_title(main, pad=9 + 11 * (sub.count("\n") + 1))
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", linespacing=1.25)


res = json.loads((OUT / "trap-analysis.json").read_text())
runs = list(np.load(OUT / "trap-run-rows.npy", allow_pickle=True))
chunks = list(np.load(OUT / "trap-chunk-rows.npy", allow_pickle=True))
ld_by = {}
for r in chunks:
    ld_by.setdefault(r["episode"], {})[r["q"]] = r["ld"]

fig = plt.figure(figsize=(13.5, 8.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1.25, 1])
ax_a = fig.add_subplot(gs[0, 0]); ax_c = fig.add_subplot(gs[0, 1]); ax_b = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])

# (a) L/D aligned at the end of stagnation runs
offsets = np.arange(-5, 3)
for label, color, name in (("ends_by_motion", BLUE, "stagnation ends, motion resumes"), ("persist", ORANGE, "stagnation persists to a failed end")):
    sel = [r for r in runs if (r["ends_by_motion"] if label == "ends_by_motion" else (r["ends_at_end"] and not r["success"]))]
    series = []
    for r in sel:
        row = []
        for o in offsets:
            q = r["end"] + o
            inside = (q >= r["start"]) and (q <= r["end"])
            after = (q > r["end"]) and label == "ends_by_motion"
            row.append(ld_by[r["episode"]].get(q, np.nan) if (inside or after) else np.nan)
        series.append(row)
    S = np.array(series, float)
    n = np.sum(np.isfinite(S), axis=0)
    mean = np.nanmean(S, axis=0)
    se = np.nanstd(S, axis=0) / np.sqrt(np.maximum(n, 1))
    ok = n >= 5
    ax_a.fill_between(offsets[ok], (mean - se)[ok], (mean + se)[ok], color=color, alpha=0.10, linewidth=0)
    ax_a.plot(offsets[ok], mean[ok], color=color, label="%s (n=%d runs)" % (name, len(sel)), **LINE)
ax_a.axvline(0, color=AXIS, linewidth=1)
ax_a.annotate("last stagnant chunk", (0, ax_a.get_ylim()[0]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED, va="bottom")
title(ax_a, "(a) Bending around the end of stagnation runs", "mean L/D per chunk, wash = standard error\noffsets > 0 exist only for runs where motion resumed")
ax_a.set_xlabel("chunk offset from the last stagnant chunk (10 steps each)")
ax_a.set_ylabel("path length / chord length")
ax_a.legend(loc="upper left")

# (b) AUROC forest
items = []
t1 = res["T1_stagnant_vs_moving"]
items.append(("stagnant vs moving chunk (all)", t1["auroc"], None, None))
t3 = res["T3_next_chunk"]
items.append(("stagnant now -> next chunk moves", t3["ld"]["auroc"], *t3["ld"]["ci95"]))
items.append(("  same, L/D relative to episode", t3["ld_relative_to_episode_moving_median"]["auroc"], *t3["ld_relative_to_episode_moving_median"]["ci95"]))
for k, lab in (("ends_by_motion__ld_last", "run end: motion resumes vs persists"), ("escape_then_success__ld_last", "run end: resumes and succeeds vs persists"), ("escape_sustained__ld_last", "run end: resumes 2+ chunks vs persists"), ("ends_by_motion__zld_last", "  same as first, within-episode z-score")):
    v = res["T2_bootstrap"].get(k)
    if v:
        items.append((lab, v["auroc"], *v["ci95"]))
eo = res["episode_outcome_per_episode"]
items.append(("episode median: failure vs success", eo["auroc_failure_vs_success"], None, None))
y = np.arange(len(items))[::-1]
for yi, (lab, a, lo, hi) in zip(y, items):
    if lo is not None:
        ax_b.plot([lo, hi], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round")
    ax_b.plot([a], [yi], color=BLUE, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
    ax_b.annotate("%.2f" % a, (a, yi), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK2)
ax_b.axvline(0.5, color=AXIS, linewidth=1)
ax_b.set_yticks(y, [i[0] for i in items])
ax_b.set_xlim(0.2, 1.0)
title(ax_b, "(b) Does more bending mean escape?  AUROC of L/D", "0.5 = no information\nbars = 95% CI from resampling episodes (where computed)")
ax_b.set_xlabel("AUROC (higher L/D -> the named positive class)")
ax_b.grid(axis="y", visible=False)

# (c) per-episode median L/D by outcome
ep = {}
for r in chunks:
    ep.setdefault(r["episode"], []).append((r["ld"], r["success"]))
rng = np.random.default_rng(0)
for i, (succ, color, name) in enumerate(((True, ORANGE, "success"), (False, BLUE, "failure"))):
    vals = np.array([np.median([v for v, _ in rows]) for rows in ep.values() if rows[0][1] == succ])
    ax_c.scatter(vals, i + rng.uniform(-0.2, 0.2, size=vals.size), s=26, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    med = np.median(vals)
    ax_c.plot([med, med], [i - 0.32, i + 0.32], color=INK, linewidth=1.5, zorder=4)
    ax_c.annotate("median %.4f (n=%d)" % (med, vals.size), (med, i + 0.38), fontsize=8, color=INK2, va="bottom")
ax_c.set_yticks([0, 1], ["success", "failure"])
ax_c.set_ylim(-0.6, 1.8)
title(ax_c, "(c) Per-episode median L/D by outcome", "170 topology-run episodes, whole-episode median\n(not a causal-prefix score); black tick = median")
ax_c.set_xlabel("median path length / chord length")
ax_c.grid(axis="y", visible=False)

anch = json.loads((OUT / "anchor-outcome-auroc.json").read_text())
anchors = np.arange(16, 21)
for key, color, name in (("ld_mean5", BLUE, "bending: mean L/D over the last 5 queries"), ("freeze_at", ORANGE, "v8.2 freeze score at the anchor"), ("v82_alarm_by", AQUA, "v8.2 alarm fired by the anchor")):
    for ds, ls in (("pooled", "-"),):
        v = [r["auroc"] for r in anch[key + "|" + ds]]
        ax_d.plot(anchors, v, color=color, linestyle=ls, label=name, **LINE)
        ax_d.annotate("%.2f" % v[-1], (anchors[-1], v[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
ax_d.axhline(0.5, color=AXIS, linewidth=1)
ax_d.set_ylim(0.4, 0.9)
ax_d.set_xticks(anchors)
title(ax_d, "(d) Causal-prefix outcome AUROC at fixed anchors, pooled", "failure vs success using only queries <= anchor\n%d fail / %d success at a=18; per-benchmark values in anchor-outcome-auroc.json" % (anch["ld_mean5|pooled"][2]["n_fail"], anch["ld_mean5|pooled"][2]["n_succ"]))
ax_d.set_xlabel("anchor query index (10 actions each)")
ax_d.set_ylabel("AUROC (failure scores higher)")
ax_d.legend(loc="upper right")

fig.suptitle("Denoising-path bending vs physical stagnation and escape (HiMoE-VLA, LIBERO-10 and LIBERO-Plus topology runs)",
             x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=1.5, h_pad=2.5)
fig.savefig(OUT / "bending-vs-trap.png", dpi=150)
fig.savefig(OUT / "bending-vs-trap.pdf")
print("saved", OUT / "bending-vs-trap.png")
