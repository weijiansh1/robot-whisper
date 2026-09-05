"""E5 图（英文标签）：φ̂ vs q 三束（success / failure / slow-success）+ d_healthy 对齐曲线。

调色（dataviz 参考盘前三槽，已验证 all-pairs CVD 安全）：
  success=#2a78d6(blue) failure=#eb6834(orange) slow success=#1baf7a(aqua, 需直接标注)。
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype/E5_phase"
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
C_SUC, C_FAIL, C_SLOW = "#2a78d6", "#eb6834", "#1baf7a"
TPTS = (20, 25, 30)
MIN_EP = 10  # 每束曲线每 q 至少在场集数


def bundle_curve(arr_row, first, nq, ep_idx, qmax):
    """束内逐绝对 q 的 mean 与 IQR（在场集 ≥ MIN_EP）。"""
    m = np.full((3, qmax), np.nan)
    for q in range(qmax):
        alive = [i for i in ep_idx if nq[i] > q]
        if len(alive) < MIN_EP:
            continue
        v = np.array([arr_row[first[i] + q] for i in alive], float)
        v = v[np.isfinite(v)]
        if len(v) < MIN_EP:
            continue
        m[0, q] = np.mean(v)
        m[1, q], m[2, q] = np.percentile(v, [25, 75])
    return m


def style_ax(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def draw(tag, title_suffix, fname):
    ph = np.load(os.path.join(OUT, f"phase_{tag}.npz"))
    lab = np.load(os.path.join(OUT, f"eplab_{tag}.npz"))
    ep, phi, dh = ph["ep"], ph["phi"], ph["dh"]
    eps, succ, nq, tert = lab["eps"], lab["succ"], lab["nq"], lab["tert"]
    first = np.searchsorted(ep, eps)
    idx_fail = np.flatnonzero(succ == 0)
    idx_slow = np.flatnonzero((succ == 1) & (tert == "slow"))
    idx_suc = np.flatnonzero((succ == 1) & (tert != "slow"))  # fast+mid，避免与慢束重叠
    qmax = int(nq.max())
    bundles = [("Success (fast/mid)", idx_suc, C_SUC),
               ("Failure", idx_fail, C_FAIL),
               ("Slow success (top tertile)", idx_slow, C_SLOW)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    fig.patch.set_facecolor(SURF)
    for ax, (arr, ylab, ttl) in zip(axes, [
            (phi, "Internal phase estimate  $\\hat{\\varphi}$",
             f"A  Internal phase vs control step — {title_suffix}"),
            (dh, "Distance to healthy manifold  $d_{healthy}$",
             f"B  Template distance vs control step — {title_suffix}")]):
        style_ax(ax, "Control step $q$ (absolute)", ylab, ttl)
        for t in TPTS:
            ax.axvline(t, color=BASE, lw=0.8, ls=(0, (4, 3)), zorder=1)
        if arr is phi:  # 成功侧真实归一化进度参考线
            ref = np.full(qmax, np.nan)
            sidx = np.flatnonzero(succ == 1)
            for q in range(qmax):
                al = [i for i in sidx if nq[i] > q]
                if len(al) >= MIN_EP:
                    ref[q] = np.mean([q / (nq[i] - 1) for i in al])
            ax.plot(np.arange(qmax), ref, color=MUTED, lw=1.0, ls=":", zorder=2)
            iref = np.flatnonzero(np.isfinite(ref))
            if len(iref):
                ax.annotate("true progress\n(success mean)", (iref[-1], ref[iref[-1]]),
                            xytext=(4, -14), textcoords="offset points",
                            color=MUTED, fontsize=8)
        for name, idx, col in bundles:
            if not len(idx):
                continue
            m = bundle_curve(arr, first, nq, idx, qmax)
            x = np.arange(qmax)
            ok = np.isfinite(m[0])
            ax.fill_between(x[ok], m[1][ok], m[2][ok], color=col, alpha=0.14, lw=0)
            ax.plot(x[ok], m[0][ok], color=col, lw=2.0, label=name, zorder=3,
                    solid_capstyle="round")
            xe = x[ok][-1]
            ax.annotate(name, (xe, m[0][ok][-1]), xytext=(5, 0),
                        textcoords="offset points", color=INK2, fontsize=8,
                        va="center")
        ax.margins(x=0.02)
        ax.set_xlim(0, qmax * 1.22)
        leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5,
                        labelcolor=INK2)
        for line in leg.get_lines():
            line.set_linewidth(3)
    axes[0].annotate("fixed test points t=20,25,30", (TPTS[0], axes[0].get_ylim()[1]),
                     xytext=(3, -10), textcoords="offset points", color=MUTED,
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    draw("main_S8", "SCENE8 main16x32", "fig1_phase_dhealthy_main.png")
    draw("grid_S8", "SCENE8 grid50x8 (replication)", "fig2_phase_dhealthy_grid.png")
