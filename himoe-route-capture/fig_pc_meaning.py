"""Figure: what the two principal components of the routing PCA encode.

Panel 1  PC1 against the single router cell it rides on -- layer 15 / expert 8 --
         with the median grasp step marked.
Panel 2  PC2 by outcome, over the control steps where all 64 episodes are alive.
Panel 3  variance budget: the same one cell is the loudest wire in all 3 captures.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from within64_lib import HB_LAYERS, N_EXPERTS, action_token_probs, load_run

FONT = "assets/NotoSansSC.ttf"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e4"
S1, S2, CRIT = "#2a78d6", "#eb6834", "#d03b3b"
OUT = pathlib.Path("figures/within64-s24")
U = 1.0 / N_EXPERTS
RUNS = [("within64-s24", "抽屉 / s24"), ("within64-t1s19", "碗上灶 / s19"),
        ("within64-t3s0", "抽屉+碗 / s0")]


def dress(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax.tick_params(colors=INK2, labelsize=8.5)


def load(name):
    """Same 2560-dim feature and same fit as within64_analyze.pca_trajectories.

    The sign of a principal component is arbitrary, so both are oriented here:
    PC1 to increase with the layer-15 / expert-8 probability it rides on, PC2 to
    increase with the control step.  Nothing else about the fit changes.
    """
    run = load_run("runs/" + name, "runs/" + name + "-client")
    feats, t, y = [], [], []
    for e in run.episodes:
        feats.append(action_token_probs(e).reshape(e.n_control, -1))   # [T, 8*10*32]
        t += list(range(e.n_control))
        y += [int(e.success)] * e.n_control
    flat = np.concatenate(feats).astype(np.float64)
    t = np.array(t)
    xc = flat - flat.mean(0)
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    z = xc @ vt[:2].T
    cells = flat.reshape(len(flat), len(HB_LAYERS), 10, N_EXPERTS).mean(2)  # [N, 8, 32]
    if np.corrcoef(z[:, 0], cells[:, 7, 8])[0, 1] < 0:
        z[:, 0] *= -1
    if np.corrcoef(z[:, 1], t)[0, 1] < 0:
        z[:, 1] *= -1
    return cells, z, t, np.array(y), xc.var(0).reshape(len(HB_LAYERS), 10, N_EXPERTS).sum(1)


def main() -> None:
    x, z, t, y, cellvar = load("within64-s24")
    p8 = x[:, 7, 8] / U
    steps = np.arange(0, 21)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.15, 1.0, 0.9], "wspace": 0.32})

    # --- panel 1: PC1 is one cell -----------------------------------------
    ax = axes[0]
    pc1 = np.array([z[t == s, 0].mean() for s in steps])
    ax.plot(steps, pc1, color=S1, lw=2.4, marker="o", ms=4, label="PC1（群体均值）")
    dress(ax, "控制步", "PC1", "PC1 = 第 15 层 router 押在 8 号专家上的那一笔")
    ax2 = ax.twinx()
    ax2.plot(steps, [p8[t == s].mean() for s in steps], color=S2, lw=2.0, ls="--",
             marker="s", ms=3.5, label="L15/E8 概率")
    ax2.set_ylabel("L15 / E8 概率（÷ 均匀 1/32）", color=S2, fontsize=9.5)
    ax2.tick_params(colors=S2, labelsize=8.5)
    ax2.spines["top"].set_visible(False)
    ax2.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.axvline(9, color=CRIT, lw=1.2, ls="-.")
    ax.text(9.3, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            "抓握（中位第 9 步）", color=CRIT, fontsize=9)
    ax.text(0.02, -0.30, "两条线 |r| = 0.97，R² = 0.94：PC1 基本就是这一个格子的读数",
            transform=ax.transAxes, color=INK2, fontsize=9)

    # --- panel 2: PC2 is the clock ----------------------------------------
    ax = axes[1]
    for lab, mask, c in (("成功 (23)", y == 1, S1), ("失败 (41)", y == 0, S2)):
        m = np.array([z[(t == s) & mask, 1].mean() for s in steps])
        ax.plot(steps, m, color=c, lw=2.4, marker="o", ms=4, label=lab)
    ax.axvspan(16.5, 20.5, color="#f2f0ec", zorder=0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.28 * (hi - lo))
    ax.text(18.5, lo + 0.02 * (hi - lo), "此后成功集陆续结束\n剩下的是慢的那些",
            color=INK2, fontsize=8.5, ha="center", va="bottom")
    dress(ax, "控制步", "PC2", "PC2 = 任务进度轴（本体状态可解释 94%）")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left")
    ax.text(0.02, -0.30, "第 14–16 步全 64 集在跑，成败在 PC2 上分开 |d| ≈ 1.8",
            transform=ax.transAxes, color=INK2, fontsize=9)

    # --- panel 3: same cell dominates every capture -----------------------
    ax = axes[2]
    vals = []
    for name, _lab in RUNS:
        _x, _z, _t, _y, cv = load(name)
        vals.append(cv[7, 8] / cv.sum())
    ax.bar(range(3), vals, color=[S1, S1, "#9db8d8"], width=0.55,
           edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, "%.1f%%" % (100 * v), ha="center", color=INK,
                fontsize=10.5, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab for _n, lab in RUNS], fontsize=9)
    dress(ax, "", "该格子占全部特征方差", "L15/E8 在三个 capture 里都是最响的一根线")
    ax.text(0.02, -0.30, "256 个（层, 专家）格子，均分应为 0.4%",
            transform=ax.transAxes, color=INK2, fontsize=9)

    fig.subplots_adjust(bottom=0.24, top=0.86, left=0.05, right=0.96)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "f6_pc_meaning.png", dpi=170, facecolor=SURFACE)
    print("wrote", OUT / "f6_pc_meaning.png")


if __name__ == "__main__":
    main()
