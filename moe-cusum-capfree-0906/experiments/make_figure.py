"""Figure 1: the cap-free TP-vs-FP frontier at lead >= 4 on external_8b,
plus the lead-time trade.  One set of axes for every arm."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capfree_common as C  # noqa: E402

for p in ("/home/jovyan/.fonts/NotoSansSC.ttf",):
    if Path(p).exists():
        fm.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": ["Noto Sans SC", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 9,
    "axes.edgecolor": "#8a8987", "axes.linewidth": 0.8,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
})
# Validated categorical slots 1-5 (light mode), assigned in fixed order.
# The null is a control rather than a series, so it takes a recessive neutral
# instead of a sixth hue - which also keeps aqua and green off the same axes.
SLOT = {"cusum": "#2a78d6", "kofm": "#eb6834", "single": "#1baf7a",
        "v7": "#eda100", "baseline": "#e87ba4", "null": "#8a8987"}
LABEL = {"cusum": "CUSUM", "kofm": "K-of-M 确认", "single": "单 chunk 阈值",
         "v7": "v7_guard（锚点）", "baseline": "无 cap 基线",
         "null": "零对照（白噪声，同一流程）"}


def pareto(sub, lead=4):
    p = sub[[f"fp_lead{lead}", f"tp_lead{lead}"]].sort_values(
        f"fp_lead{lead}").to_numpy(float)
    xs, ys, b = [], [], -1
    for fp, tp in p:
        if tp > b:
            b = tp
            xs.append(max(fp, 0.6))
            ys.append(tp)
    return np.array(xs), np.array(ys)


def main() -> None:
    fr = pd.read_csv(C.RESULTS / "frontier_all.csv")
    ops = pd.read_csv(C.RESULTS / "frozen_operating_points.csv")
    e = fr[fr.cohort == "external_8b"]
    oe = ops[ops.cohort == "external_8b"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.6),
                                 gridspec_kw={"width_ratios": [1.45, 1]})

    band = ax.axvspan(80, 320, color="#eda100", alpha=0.10, lw=0, zorder=0,
                      label="目标工作区 FP∈[80,320]")

    for fam in ("cusum", "kofm", "single"):
        sub = e[e.family == fam]              # union of both frozen selections
        xs, ys = pareto(sub)
        ax.plot(xs, ys, "-", lw=2, color=SLOT[fam], label=LABEL[fam],
                zorder=5, solid_capstyle="round")
    xs, ys = pareto(e[e.family == "baseline"])
    ax.step(xs, ys, where="post", lw=2, color=SLOT["baseline"],
            label=LABEL["baseline"], zorder=4)
    nu = e[e["mode"].astype(str).str.startswith("white_noise")]
    xs, ys = pareto(nu)
    ax.plot(xs, ys, ls=(0, (5, 3)), lw=1.6, color=SLOT["null"],
            label=LABEL["null"], zorder=3)

    arrow = dict(arrowstyle="-", lw=0.8, color="#52514e",
                 shrinkA=1, shrinkB=4)
    v7 = oe[oe.arm == "v7_guard"].iloc[0]
    ax.plot([v7.fp_lead4], [v7.tp_lead4], "*", ms=18, color=SLOT["v7"],
            mec="#fcfcfb", mew=1.4, zorder=9, label=LABEL["v7"])
    ax.annotate(f"v7_guard 锚点\n{int(v7.tp_lead4)}/{int(v7.fp_lead4)}",
                (v7.fp_lead4, v7.tp_lead4), xytext=(1.15, 305),
                textcoords="data", fontsize=8.5, color="#0b0b0b",
                ha="left", va="center", arrowprops=arrow)
    cz = oe[oe.arm == "cusum|v7budget"].iloc[0]
    ax.plot([cz.fp_lead4], [cz.tp_lead4], "o", ms=9, color=SLOT["cusum"],
            mec="#fcfcfb", mew=1.5, zorder=9)
    ax.annotate(f"CUSUM 冻结工作点\n{int(cz.tp_lead4)}/{int(cz.fp_lead4)}",
                (cz.fp_lead4, cz.tp_lead4), xytext=(180, 520),
                textcoords="data", fontsize=8.5, color="#0b0b0b",
                ha="left", va="center", arrowprops=arrow)
    fp57 = e[(e.arm == "cusum|v7budget") & (e.fp_lead4 <= 57)]
    r57 = fp57.loc[fp57.tp_lead4.idxmax()]
    ax.plot([r57.fp_lead4], [r57.tp_lead4], "o", ms=8, mfc="#fcfcfb",
            mec=SLOT["cusum"], mew=2, zorder=9)
    ax.annotate(f"同 FP≤57 事后读点\n{int(r57.tp_lead4)}/{int(r57.fp_lead4)}",
                (r57.fp_lead4, r57.tp_lead4), xytext=(1.15, 520),
                textcoords="data", fontsize=8.5, color="#52514e",
                ha="left", va="center", arrowprops=arrow)

    ax.set_xscale("log")
    ax.set_xlim(0.6, 1200)
    ax.set_ylim(0, 600)
    ax.set_xlabel("误报 episode 数（提前量≥4；对数轴）")
    ax.set_ylabel("命中 risk episode 数（提前量≥4）")
    ax.set_title("external_8b：无 cap 协议下的 TP–FP 前沿（564 个 risk）",
                 fontsize=10, loc="left")
    ax.grid(True, which="major", color="#dedddb", lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    h, lb = ax.get_legend_handles_labels()
    order = [i for i in range(len(h)) if h[i] is not band] + [h.index(band)]
    ax.legend([h[i] for i in order], [lb[i] for i in order],
              loc="lower right", frameon=False, fontsize=8.5)

    # ---- panel B: lead-time trade at each frozen operating point ----
    arms = [("cusum|v7budget", "cusum"), ("kofm|v7budget", "kofm"),
            ("single|v7budget", "single"), ("v7_guard", "v7"),
            ("still_running_q37", "baseline")]
    leads = list(C.LEADS)
    for arm, slot in arms:
        r = oe[oe.arm == arm]
        if not len(r):
            continue
        r = r.iloc[0]
        ys = [r[f"tp_lead{b}"] for b in leads]
        bx.plot(leads, ys, "-o", lw=2, ms=6, color=SLOT[slot],
                mec="#fcfcfb", mew=1.2,
                label=f"{LABEL[slot]}  FP={int(r.fp_lead4)}")
    bx.axvline(4, color="#8a8987", lw=0.9, ls=":")
    bx.text(4.2, 22, "头条要求", fontsize=7.5, color="#52514e")
    bx.set_xticks(leads)
    bx.set_xlim(-0.6, 12.6)
    bx.set_ylim(0, 600)
    bx.set_xlabel("提前量要求（chunk）")
    bx.set_ylabel("命中 risk episode 数")
    bx.set_title("提前量的代价（各臂在各自冻结工作点上）", fontsize=10,
                 loc="left")
    bx.grid(True, color="#dedddb", lw=0.6)
    bx.set_axisbelow(True)
    bx.legend(loc="upper right", frameon=False, fontsize=8.5)

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(C.RESULTS / f"frontier_external.{ext}", dpi=200)
    print("wrote", C.RESULTS / "frontier_external.png")


if __name__ == "__main__":
    main()
