"""Figure for the denoising-path straightness probe (reads the saved npz paths, no model calls)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RT = Path("/home/swj/data/libero-runtime")
sys.path.insert(0, str(RT))
from probe_flow_path_straightness import straightness, per_token_ratio  # noqa: E402

OUT = RT / "samples/flow-path-straightness-20260916"
SERIES = [  # fixed categorical order: slot 1 blue, slot 2 orange
    ("episode-libero-10-swap-task00-seed7-20260916T090439Z", "Pro swap task00 (failed, 52 queries)", "#2a78d6"),
    ("episode-libero-10-task08-seed7-20260916T110257Z", "LIBERO-10 task08 (success, 40 queries)", "#eb6834"),
]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.titlecolor": INK, "axes.titlesize": 9.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5,
})
LINE = dict(linewidth=2, solid_joinstyle="round", solid_capstyle="round", marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)


def title(ax, main, sub):
    n = sub.count("\n") + 1
    ax.set_title(main, pad=9 + 11 * n)
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", linespacing=1.25)


def load(name):
    z = np.load(OUT / (name + ".npz"))
    dims = np.flatnonzero(z["data_mask"])
    rows = [straightness(z["X"][q], z["V"][q], dims) for q in range(z["X"].shape[0])]
    tok = np.stack([per_token_ratio(z["X"][q], dims) for q in range(z["X"].shape[0])])
    return z, dims, rows, tok


data = {name: load(name) for name, _, _ in SERIES}
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), gridspec_kw={"height_ratios": [1, 0.72]})
(ax_cos, ax_perp, ax_ld), (ax_pca1, ax_pca2, ax_tok) = axes
steps = np.arange(10)

# (a) direction change between consecutive Euler steps
for name, label, color in SERIES:
    cc = np.array([r["cos_consec"] for r in data[name][2]])
    m = cc.mean(axis=0)
    ax_cos.fill_between(steps[:-1] + 0.5, cc.min(axis=0), cc.max(axis=0), color=color, alpha=0.10, linewidth=0)
    ax_cos.plot(steps[:-1] + 0.5, m, color=color, label=label, **LINE)
    ax_cos.annotate("%.4f" % m[-1], (8.5, m[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
title(ax_cos, "(a) Turn between consecutive Euler steps", "cos(v_k, v_k+1), mean per transition; wash = min-max over queries")
ax_cos.set_xlabel("transition k -> k+1  (t = 1 is noise, t = 0 is the action)")
ax_cos.set_ylabel("cosine (1 = no turn)")
ax_cos.set_xticks(steps[:-1] + 0.5, ["%d-%d" % (k, k + 1) for k in steps[:-1]])
ax_cos.set_ylim(0.9, 1.001)
ax_cos.legend(loc="lower left")

# (b) perpendicular deviation from the chord, by path point
for name, label, color in SERIES:
    pp = np.array([r["perp_over_chord"] for r in data[name][2]])
    m = pp.mean(axis=0)
    ax_perp.fill_between(np.arange(11), pp.min(axis=0), pp.max(axis=0), color=color, alpha=0.10, linewidth=0)
    ax_perp.plot(np.arange(11), m, color=color, label=label, **LINE)
    j = int(m.argmax())
    ax_perp.annotate("peak %.3f" % m[j], (j, m[j]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK2)
title(ax_perp, "(b) Distance from the straight chord", "perpendicular distance / chord length per path point, mean; wash = min-max")
ax_perp.set_xlabel("path point k  (x_0 = noise, x_10 = final action)")
ax_perp.set_ylabel("perpendicular distance / chord length")
ax_perp.set_xticks(np.arange(11))
ax_perp.set_ylim(0, None)

# (c) path length / chord per query
for i, (name, label, color) in enumerate(SERIES):
    ld = np.array([r["path_over_chord"] for r in data[name][2]])
    rng = np.random.default_rng(0)
    y = i + rng.uniform(-0.18, 0.18, size=ld.size)
    ax_ld.scatter(ld, y, s=28, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=3, label=label)
    med = np.median(ld)
    ax_ld.plot([med, med], [i - 0.3, i + 0.3], color=INK, linewidth=1.5, zorder=4)
    ax_ld.annotate("median %.4f, max %.3f" % (med, ld.max()), (med, i + 0.36), ha="left", va="bottom", fontsize=8, color=INK2)
ax_ld.axvline(1.0, color=AXIS, linewidth=1)
ax_ld.annotate("1.0 = straight line", (1.0, -0.5), xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED, va="center")
title(ax_ld, "(c) Path length / chord, one dot per query", "real 7 action dims x 10 tokens; black tick = median")
ax_ld.set_yticks([0, 1], ["Pro swap\n(failed)", "LIBERO-10\n(success)"])
ax_ld.set_ylim(-0.6, 1.8)
ax_ld.set_xlabel("path length / straight-line distance")
ax_ld.grid(axis="y", visible=False)

# (d)/(e) PCA projection of the median-curvature query in each episode
for ax, (name, label, color) in zip((ax_pca1, ax_pca2), SERIES):
    z, dims, rows, _ = data[name]
    ld = np.array([r["path_over_chord"] for r in rows])
    q = int(np.argsort(ld)[len(ld) // 2])
    P = z["X"][q][:, :, dims].reshape(11, -1)
    C = P - P.mean(axis=0)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    proj = C @ Vt[:2].T
    var2 = float((S[:2] ** 2).sum() / (S ** 2).sum())
    ax.plot(proj[[0, -1], 0], proj[[0, -1], 1], color=AXIS, linewidth=1, zorder=1)
    ax.plot(proj[:, 0], proj[:, 1], color=color, zorder=3, **LINE)
    ax.annotate("x_0 noise (t=1)", proj[0], xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK2)
    ax.annotate("x_10 action (t=0)", proj[-1], xytext=(6, -10), textcoords="offset points", fontsize=8, color=INK2)
    var1 = float(S[0] ** 2 / (S ** 2).sum())
    title(ax, "(%s) %s: median-curvature query, PCA plane" % ("d" if ax is ax_pca1 else "e", label.split(" (")[0]),
          "query %d, L/D %.4f, equal axis scale, chord in gray\nPC1 carries %.2f%% of path variance, PC1+PC2 %.3f%%" % (q, ld[q], 100 * var1, 100 * var2))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")

# (f) per-token curvature
for name, label, color in SERIES:
    tok = data[name][3]
    m = np.median(tok, axis=0)
    ax_tok.fill_between(np.arange(10), np.percentile(tok, 25, axis=0), np.percentile(tok, 75, axis=0), color=color, alpha=0.10, linewidth=0)
    ax_tok.plot(np.arange(10), m, color=color, label=label, **LINE)
    ax_tok.annotate("%.4f" % m[-1], (9, m[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
ax_tok.axhline(1.0, color=AXIS, linewidth=1)
title(ax_tok, "(f) Path length / chord, per action token", "median over queries; wash = interquartile range")
ax_tok.set_xlabel("action token index within the 10-step chunk")
ax_tok.set_ylabel("path length / chord length")
ax_tok.set_xticks(np.arange(10))
ax_tok.set_ylim(0.995, None)

fig.suptitle("HiMoE-VLA flow-matching denoising path within one inference (10 Euler steps, dt = -0.1, bf16 velocity): near-straight, bending in the last steps",
             x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.0, w_pad=1.5)
fig.savefig(OUT / "straightness.png", dpi=150)
fig.savefig(OUT / "straightness.pdf")
for name, label, color in SERIES:
    z, dims, rows, _ = data[name]
    v1, v2, ang = [], [], []
    for q in range(z["X"].shape[0]):
        P = z["X"][q][:, :, dims].reshape(11, -1)
        C = P - P.mean(axis=0)
        S = np.linalg.svd(C, compute_uv=False)
        v1.append(S[0] ** 2 / (S ** 2).sum()); v2.append((S[:2] ** 2).sum() / (S ** 2).sum())
        ang.append(rows[q]["angle_first_last_deg"])
    print("%s: PC1 var median %.4f min %.4f | PC1+PC2 var median %.5f min %.5f | angle(v0,v9) median %.1f max %.1f" % (
        label, np.median(v1), np.min(v1), np.median(v2), np.min(v2), np.median(ang), np.max(ang)))
print("saved", OUT / "straightness.png")
