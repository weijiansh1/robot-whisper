#!/usr/bin/env python3
"""E8 图（英文标签）。色彩按 dataviz 参考调色板：diverging blue<->red（灰中点）、
sequential blue、categorical slot 1..5。数据来自 run_e8e9.py 的落盘中间量。"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = pathlib.Path("/home/jovyan/work/himoe-vla/analysis_moe_phenotype/E8_portrait_silent")
AXES = ("inst", "comm", "cons", "stick", "flow")
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#eceae4"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_br", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32423"])
SEQ_BAD = "#e6e5e0"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": "#c9c8c0", "font.size": 8})


DIV.set_bad(SEQ_BAD)
SEQ.set_bad(SEQ_BAD)


def bits(s, n_ax):
    return [(s >> b) & 1 for b in range(n_ax)]


def order_states(n_states):
    """按 (inst, cons) 分块排序：switch/desync -> (1,1) -> (0,0) -> flatten/shared。"""
    n_ax = 5 if n_states == 32 else 4
    blocks = [(1, 0), (1, 1), (0, 0), (0, 1)]
    key = []
    for s in range(n_states):
        b = bits(s, n_ax)
        key.append((blocks.index((b[0], b[2])), b[1], b[3], b[4] if n_ax == 5 else 0, s))
    order = [k[-1] for k in sorted(key)]
    sizes = [sum(1 for s in order if (bits(s, n_ax)[0], bits(s, n_ax)[2]) == bl) for bl in blocks]
    return order, sizes, n_ax


def state_ticklabels(order, n_ax):
    return ["".join(str(x) for x in bits(s, n_ax)) for s in order]


BAND_NOTE = ("State-bit tick labels: blue = switch/desync band (inst=1, cons=0), "
             "green = flatten/shared band (inst=0, cons=1), grey = mixed quadrants.")


def colour_band_ticks(ax, order, n_ax):
    for s, tl in zip(order, ax.get_yticklabels()):
        b = bits(s, n_ax)
        tl.set_color(CAT[0] if (b[0], b[2]) == (1, 0) else
                     CAT[2] if (b[0], b[2]) == (0, 1) else "#7a7975")


# ---------------------------------------------------------------- fig 1
def fig1():
    occ = pd.read_csv(OUT / "macrostate_occupancy.csv")
    ns = int(json.loads((OUT / "summary.json").read_text())["e8"]["primary_n_states"])
    occ = occ[occ["n_states"] == ns]
    cells = sorted(occ["cell"].unique())
    order, sizes, n_ax = order_states(ns)
    D = np.full((ns, len(cells)), np.nan)
    P = np.full((ns, len(cells)), np.nan)
    for j, c in enumerate(cells):
        sub = occ[occ["cell"] == c].set_index("state")
        for i, s in enumerate(order):
            D[i, j] = sub.loc[s, "d_occ"]
            P[i, j] = sub.loc[s, "p_self_fail"] - sub.loc[s, "p_self_succ"]
    short = [c.replace("grid50x8|libero_", "grid ").replace("main16x32|libero_", "main ")
             for c in cells]

    fig, axs = plt.subplots(1, 2, figsize=(9.4, 7.4), sharey=True)
    for ax, M, ttl, vm in ((axs[0], D * 100, "A  Occupancy difference (failure - success), pp", 12.0),
                           (axs[1], P, "B  Dwell difference $P_{ii}$(fail) - $P_{ii}$(succ)", 0.5)):
        im = ax.imshow(M, cmap=DIV, vmin=-vm, vmax=vm, aspect="auto", origin="lower",
                       interpolation="nearest")
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(ttl, fontsize=9, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        for y in np.cumsum(sizes)[:-1]:
            ax.axhline(y - 0.5, color="#9a9992", lw=0.9)
        ax.set_yticks(range(ns))
        ax.set_yticklabels(state_ticklabels(order, n_ax), fontsize=6.4, family="monospace")
    axs[0].set_ylabel("macro-state bits [inst, comm, cons, stick, flow]", fontsize=8.5)
    colour_band_ticks(axs[0], order, n_ax)
    fig.text(0.055, 0.012, "Grey cells in B: fewer than 20 outgoing transitions on one side "
             "(dwell undefined).  " + BAND_NOTE, fontsize=7.2, color=INK2)
    fig.suptitle("E8  Macro-state occupancy and dwell, failure vs success (32 states, "
                 "group-internal median split)", fontsize=10, x=0.012, ha="left")
    fig.tight_layout(rect=(0.055, 0.035, 1, 0.955))
    fig.savefig(OUT / "fig1_macrostate_occupancy.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig2():
    z = np.load(OUT / "event_bundles.npz")
    meta = json.loads((OUT / "_bundles_meta.json").read_text())
    rel = z["rel"]
    ns = z["loop_fail_occ"].shape[0]
    order, sizes, n_ax = order_states(ns)
    fig, axs = plt.subplots(2, 3, figsize=(13.0, 8.2))

    for ax, key, ttl in ((axs[0, 0], "loop_fail", "A  Loop onset, failed episodes (n=%d)" % meta["loop_fail"]["n_events"]),
                         (axs[0, 1], "static_fail", "B  Static onset, failed episodes (n=%d)" % meta["static_fail"]["n_events"])):
        M = z[key + "_occ"][order] * 100
        im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=14, aspect="auto", origin="lower",
                       extent=(rel[0] - .5, rel[-1] + .5, -0.5, ns - 0.5),
                       interpolation="nearest")
        ax.axvline(0, color="#e34948", lw=1.2)
        for y in np.cumsum(sizes)[:-1]:
            ax.axhline(y - 0.5, color="#9a9992", lw=0.9)
        ax.set_yticks(range(ns))
        ax.set_yticklabels(state_ticklabels(order, n_ax), fontsize=6.0, family="monospace")
        ax.set_xlabel("queries relative to onset", fontsize=8)
        ax.set_title(ttl, fontsize=9, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("state occupancy (%)", fontsize=7)
        cb.ax.tick_params(labelsize=7)
    axs[0, 0].set_ylabel("macro-state bits [inst, comm, cons, stick, flow]", fontsize=8)
    colour_band_ticks(axs[0, 0], order, n_ax)
    colour_band_ticks(axs[0, 1], order, n_ax)
    fig.text(0.045, 0.012, BAND_NOTE, fontsize=7.2, color=INK2)

    ax = axs[0, 2]
    for i, (key, lab) in enumerate((("loop_fail", "loop (failed)"), ("static_fail", "static (failed)"))):
        m = meta[key]
        y = np.array(m["band_switch_desync"]) - np.array(m["band_flatten_shared"])
        ax.plot(rel, y, color=CAT[i], lw=2.0, marker="o", ms=4.5, label=lab)
        ax.annotate(lab, (rel[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    color=CAT[i], fontsize=7.5, va="center")
    ax.axhline(0, color="#9a9992", lw=0.8, ls="--")
    ax.axvline(0, color="#e34948", lw=1.2)
    ax.grid(color=GRID, lw=0.6)
    ax.set_title("C  Band polarity: switch/desync minus flatten/shared", fontsize=9, loc="left")
    ax.set_xlabel("queries relative to onset", fontsize=8)
    ax.set_ylabel("occupancy difference", fontsize=8)
    ax.set_xlim(rel[0] - 0.5, rel[-1] + 2.6)

    names = {"inst": "instability", "comm": "commitment", "cons": "consensus",
             "stick": "stickiness", "flow": "flow shape"}
    for ax, key, ttl in ((axs[1, 0], "loop_fail", "D  Axis means, loop onset (failed)"),
                         (axs[1, 1], "static_fail", "E  Axis means, static onset (failed)")):
        lab_ok = AXES if key == "loop_fail" else ("stick",)
        for i, a in enumerate(AXES):
            y = meta[key]["axis_mean"][a]
            ax.plot(rel, y, color=CAT[i], lw=1.9, marker="o", ms=3.6, label=names[a])
            if a in lab_ok:
                ax.annotate(names[a], (rel[-1], y[-1]), xytext=(4, 0),
                            textcoords="offset points", color=CAT[i], fontsize=7, va="center")
        ax.axhline(0, color="#9a9992", lw=0.8, ls="--")
        ax.axvline(0, color="#e34948", lw=1.2)
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(ttl, fontsize=9, loc="left")
        ax.set_xlabel("queries relative to onset", fontsize=8)
        ax.set_ylabel("group-internal z (MAD units)", fontsize=8)
        ax.set_xlim(rel[0] - 0.5, rel[-1] + 3.4)
    axs[1, 0].legend(fontsize=7, frameon=False, ncol=2, loc="upper left")

    ax = axs[1, 2]
    for i, (key, a, lab) in enumerate((("loop_fail", "stick", "loop failed - stickiness"),
                                       ("loop_succ", "stick", "loop succeeded - stickiness"),
                                       ("loop_succ", "cons", "loop succeeded - consensus"))):
        y = meta[key]["axis_mean"][a]
        ax.plot(rel, y, color=CAT[i], lw=2.0, marker="o", ms=4.0, label=lab)
    ax.axhline(0, color="#9a9992", lw=0.8, ls="--")
    ax.axvline(0, color="#e34948", lw=1.2)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("F  Post-onset divergence, loop failed vs succeeded", fontsize=9, loc="left")
    ax.set_xlabel("queries relative to onset", fontsize=8)
    ax.set_ylabel("group-internal z (MAD units)", fontsize=8)
    fig.suptitle("E8  Event-aligned macro-state bundles (onset +/- 6 queries; bits ordered by "
                 "instability x consensus band)", fontsize=10, x=0.012, ha="left")
    fig.tight_layout(rect=(0.01, 0.03, 1, 0.955))
    fig.savefig(OUT / "fig2_event_bundles.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig3():
    c = np.load(OUT / "_portrait_cache.npz")
    valid = c["valid"]
    inst, cons, succ = c["inst"], c["cons"], c["success"]
    bnd = {"Success (all valid rows)": valid & (succ == 1),
           "Failure (all valid rows)": valid & (succ == 0)}
    mk = np.zeros(len(valid), bool)
    mk[c["loop_pre_rows"]] = True
    bnd["Loop pre-onset (-6..-1)"] = mk & valid
    mk2 = np.zeros(len(valid), bool)
    mk2[c["static_pre_rows"]] = True
    bnd["Static pre-onset (-6..-1)"] = mk2 & valid

    xr, yr = (-3.0, 6.0), (-4.0, 4.0)
    nb = 40
    xe = np.linspace(*xr, nb + 1)
    ye = np.linspace(*yr, nb + 1)
    H, shown = {}, {}
    for k, m in bnd.items():
        x, y = inst[m], cons[m]
        inw = (x >= xr[0]) & (x <= xr[1]) & (y >= yr[0]) & (y <= yr[1])
        h, _, _ = np.histogram2d(x[inw], y[inw], bins=[xe, ye])
        H[k] = (h / max(h.sum(), 1)).T * 100.0
        shown[k] = (int(inw.sum()), int(m.sum()))

    fig, axs = plt.subplots(2, 3, figsize=(12.4, 7.4))
    vmax = max(np.percentile(v[v > 0], 99.5) for v in H.values())
    for ax, k in zip(axs.ravel()[:4], bnd):
        im = ax.imshow(H[k], cmap=SEQ, vmin=0, vmax=vmax, origin="lower", aspect="auto",
                       extent=(*xr, *yr), interpolation="nearest")
        ax.set_title(f"{k}  (n={shown[k][0]:,} of {shown[k][1]:,} rows in view)",
                     fontsize=8.5, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("% of rows", fontsize=7)
        cb.ax.tick_params(labelsize=7)
    d = H["Failure (all valid rows)"] - H["Success (all valid rows)"]
    v = np.abs(d).max() * 0.75
    im = axs[1, 1].imshow(d, cmap=DIV, vmin=-v, vmax=v, origin="lower", aspect="auto",
                          extent=(*xr, *yr), interpolation="nearest")
    axs[1, 1].set_title("Failure minus success density", fontsize=8.5, loc="left")
    cb = fig.colorbar(im, ax=axs[1, 1], fraction=0.045, pad=0.02)
    cb.set_label("pp of rows", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    ax = axs[1, 2]
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    from scipy.ndimage import gaussian_filter
    for i, k in enumerate(("Success (all valid rows)", "Loop pre-onset (-6..-1)",
                           "Static pre-onset (-6..-1)")):
        G = gaussian_filter(H[k], 1.3)
        lv = np.percentile(G[G > 0], [70, 90])
        ax.contour(xc, yc, G, levels=lv, colors=CAT[i], linewidths=[1.2, 2.2])
        ax.plot([], [], color=CAT[i], lw=2.0, label=k)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("Overlay: 70th / 90th smoothed-density contours", fontsize=8.5, loc="left")
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)

    for ax in axs.ravel():
        ax.axhline(0, color="#9a9992", lw=0.8, ls="--")
        ax.axvline(0, color="#9a9992", lw=0.8, ls="--")
        ax.set_xlabel("instability  z(V+A)", fontsize=8)
        ax.set_ylabel("consensus  z(C - D_tok)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("E8  Phase portrait: instability x consensus (group-internal z, MAD units; "
                 "other three axes marginalised)", fontsize=10, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "fig3_phase_portrait.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("figures written")
