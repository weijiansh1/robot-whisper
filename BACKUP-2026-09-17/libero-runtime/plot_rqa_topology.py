"""Figure for the recurrence analysis: example recurrence plots, AUROC by anchor, beyond-magnitude tests."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

GRID = Path("/home/swj/data/moe-capture/topo-20260916/_grid/episodes")
OUT = Path("/home/swj/data/libero-runtime/samples/moe-recurrence-20260916")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID_C, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.titlecolor": INK, "axes.titlesize": 9.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5,
})
LINE = dict(linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)


def title(ax, main, sub):
    ax.set_title(main, pad=9 + 11 * (sub.count("\n") + 1))
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", linespacing=1.25)


res = json.loads((OUT / "rqa-results.json").read_text())
pq = np.load(OUT / "rqa-per-query.npz", allow_pickle=True)
names = list(pq["names"]); keys = list(pq["keys"]); succ = pq["success"]; stag_frac = pq["stag_frac"]
MAT = "input|back_path"
mi = names.index(MAT)
# global epsilon exactly as in the analysis (alpha 0.10, theiler 3, first 14 queries pooled)
vals = []
mats = {}
for k in keys:
    ds, tag = k.split("__", 1)
    z = np.load(GRID / (tag + ".npz"), allow_pickle=True)
    D = z["matrices"][mi].astype(float); mats[k] = D
    S = D[:14, :14]; J = np.arange(len(S)); vals.append(S[(J[None, :] - J[:, None]) >= 3])
eps = float(np.quantile(np.concatenate(vals), 0.10))

# pick examples: stagnant failure, moving failure, success (long enough), by stag_frac
fail_idx = np.flatnonzero(~succ); succ_idx = np.flatnonzero(succ)
Qs = np.array([mats[k].shape[0] for k in keys])
ex_stag = fail_idx[np.argmax(stag_frac[fail_idx])]
moving_fail = fail_idx[(stag_frac[fail_idx] < 0.15)]
ex_move = moving_fail[np.argsort(stag_frac[moving_fail])[len(moving_fail) // 2]]
long_succ = succ_idx[Qs[succ_idx] >= 30]
ex_succ = long_succ[np.argsort(Qs[long_succ])[len(long_succ) // 2]]

fig = plt.figure(figsize=(13.5, 9.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.0])
cmap = ListedColormap([SURFACE, BLUE])
for col, (idx, label) in enumerate(((ex_stag, "stagnant failure"), (ex_move, "moving failure"), (ex_succ, "success"))):
    ax = fig.add_subplot(gs[0, col])
    D = mats[keys[idx]]; R = (D <= eps).astype(float)
    ax.imshow(R, cmap=cmap, origin="lower", interpolation="nearest", vmin=0, vmax=1)
    ax.grid(False)
    F = pq["feat__" + MAT][idx]
    rr = np.nanmean(F[:, 0]); lam = np.nanmean(F[:, 2]); lmax = np.nanmax(F[:, 3])
    title(ax, "(%s) %s: %s" % ("abc"[col], label, keys[idx].split("__")[1]),
          ("%s, dot = distance <= eps (10%% quantile)\n" % MAT if col == 0 else "same representation and eps\n") + "stagnant %.0f%%, RR %.2f, LAM %.2f, LMAX %d" % (100 * stag_frac[idx], rr, lam, int(lmax)))
    ax.set_xlabel("query j"); ax.set_ylabel("query i")

# (d) AUROC by anchor
ax = fig.add_subplot(gs[1, 0])
anchors = [16, 20, 24, 32]
series = {
    "RR median (60 matrices)": ([res["E3_outcome_anchors"]["q%d" % a]["features"]["rr"]["median_pooled"] for a in anchors], BLUE, "-"),
    "RR best matrix": ([res["E3_outcome_anchors"]["q%d" % a]["features"]["rr"]["best_pooled"] for a in anchors], BLUE, ":"),
    "diameter median (oriented)": ([1 - res["E3_outcome_anchors"]["q%d" % a]["features"]["diam"]["median_pooled"] for a in anchors], ORANGE, "-"),
    "v8.2 freeze score": ([res["E3_outcome_anchors"]["q%d" % a]["features"]["v82_freeze"]["pooled"] for a in anchors], AQUA, "-"),
}
for lab, (v, c, ls) in series.items():
    ax.plot(anchors, v, color=c, linestyle=ls, label=lab, **LINE)
ax.axhline(0.5, color=AXIS, linewidth=1)
ax.set_xticks(anchors, ["q16\n84/86", "q20\n84/79", "q24\n84/51", "q32\n84/5"])
ax.set_ylim(0.4, 1.0)
title(ax, "(d) Causal outcome AUROC at anchors (pooled)", "failure scores higher; x labels give n fail / n success; q32 has only 5 successes")
ax.set_xlabel("anchor query (10 actions each)"); ax.set_ylabel("AUROC")
ax.legend(loc="lower right", fontsize=7.5, handlelength=1.6)

# (e) beyond magnitude at q20: diameter-matched and moving-failure split
ax = fig.add_subplot(gs[1, 1])
e5 = res["E5_beyond_magnitude"]["q20"]
labels = ["RR", "LMAX", "LAM", "diameter"]
keys5 = ["rr", "lmax", "lam", "diam"]
x = np.arange(len(labels)); wdt = 0.36
m1 = [e5[k]["diam_matched_median"] for k in keys5]; m2 = [e5[k]["moving_fail_vs_succ_median"] for k in keys5]
ax.bar(x - wdt / 2, m1, wdt, color=BLUE, label="diameter-matched (within tertiles)", edgecolor=SURFACE, linewidth=1)
ax.bar(x + wdt / 2, m2, wdt, color=ORANGE, label="moving failures vs successes (n %d vs %d)" % (e5["n_moving_fail"], e5["n_succ"]), edgecolor=SURFACE, linewidth=1)
for xi, (a1, a2) in enumerate(zip(m1, m2)):
    ax.annotate("%.2f" % a1, (xi - wdt / 2, a1), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.annotate("%.2f" % a2, (xi + wdt / 2, a2), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
bl = e5["baselines_moving_fail_vs_succ"]
ax.axhline(bl["freeze"], color=AQUA, linewidth=1.5, label="v8.2 freeze, moving failures vs successes: %.2f" % bl["freeze"])
ax.axhline(0.5, color=AXIS, linewidth=1)
ax.set_xticks(x, labels); ax.set_ylim(0.4, 1.0)
title(ax, "(e) Beyond magnitude at q20 (median of 60 matrices)", "left: AUROC after matching on window diameter; right: physically moving failures only")
ax.set_ylabel("AUROC"); ax.legend(loc="upper left", fontsize=8); ax.grid(axis="x", visible=False)

# (f) robustness across thresholds (E4): RR and LMAX q24 medians per setting
ax = fig.add_subplot(gs[1, 2])
e4 = res["E4_robustness"]
lbl = ["%s a=%.2f w=%d" % (r["setting"]["eps"][:3], r["setting"]["alpha"], r["setting"]["theiler"]) for r in e4]
y = np.arange(len(e4))[::-1]
ax.plot([r["rr"]["q24_median"] for r in e4], y, color=BLUE, linestyle="none", marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label="RR, q24 median of 12 matrices")
ax.plot([r["lmax"]["q24_median"] for r in e4], y, color=ORANGE, linestyle="none", marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label="LMAX, q24 median")
ax.plot([r["ret"]["q24_max"] for r in e4], y, color=AQUA, linestyle="none", marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label="true-return RET, q24 best")
ax.axvline(0.5, color=AXIS, linewidth=1)
ax.set_yticks(y, lbl); ax.set_xlim(0.3, 1.0)
title(ax, "(f) Robustness over settings", "glo = pooled early-window quantile\npre = episode's own causal quantile")
ax.set_xlabel("outcome AUROC at q24 (failure higher)"); ax.legend(loc="lower left", fontsize=8); ax.grid(axis="y", visible=False)

fig.suptitle("Recurrence in MoE state space: failures sit in a compact recurrent regime (RR / LAM / LMAX), true loops (RET) are absent",
             x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=1.5, h_pad=2.5)
fig.savefig(OUT / "rqa-topology.png", dpi=150); fig.savefig(OUT / "rqa-topology.pdf")
print("saved", OUT / "rqa-topology.png", "examples:", keys[ex_stag], keys[ex_move], keys[ex_succ], "eps %.4f" % eps)
