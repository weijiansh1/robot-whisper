"""Figures for the HiMoE-VLA wrist-layout routing comparison (50 + 50 episodes)."""

from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")
from analyze_routes import HB_LAYERS, load, load_per_layer, topk_set_jaccard, tv  # noqa: E402
from himoe_libero_bridge.routing import sparse_routes_to_dense  # noqa: E402

FONT = "/home/jovyan/work/himoe-route-capture/assets/NotoSansSC.ttf"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e8e7e4"
S1, S2, CRIT = "#2a78d6", "#eb6834", "#d03b3b"
DIVERGE = LinearSegmentedColormap.from_list("bl_gy_rd", ["#184f95", "#f0efec", "#d03b3b"])

OUT = pathlib.Path("/home/jovyan/work/himoe-route-capture/figures")
OUT.mkdir(exist_ok=True)


def style(ax, xlabel="", ylabel="", title="", sub=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    if title:
        ax.set_title(title, color=INK, fontsize=12.5, fontweight="bold", loc="left",
                     pad=20 if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def feats(eps, n):
    return np.array([load_per_layer(e["ids"][:n], e["w"][:n]).ravel() for e in eps])


def drift(ids, w, n):
    d = sparse_routes_to_dense(ids[:n], w[:n])
    d = d / d.sum(-1, keepdims=True)
    return float(tv(d[1:], d[:-1]).mean())


def fig1(right, left, n_pair=50):
    cross = np.zeros((8, n_pair))
    within = [[] for _ in range(8)]
    rng0 = np.random.default_rng(7)
    pairs = [tuple(p) for p in rng0.choice(len(right), (300, 2)) if p[0] != p[1]][:120]
    for li in range(8):
        for ep in range(n_pair):
            cross[li, ep] = topk_set_jaccard(right[ep]["ids"][0, :, li], left[ep]["ids"][0, :, li])
        for i, j in pairs:
            within[li].append(topk_set_jaccard(right[i]["ids"][0, :, li], right[j]["ids"][0, :, li]))

    xs = np.array([0, 1, 2, 3, 5.0, 6.0, 7.0, 8.0])
    fig, ax = plt.subplots(figsize=(8.8, 5.2), facecolor=SURFACE)
    style(ax, ylabel="top-4 专家集合 Jaccard",
          title="wrist 槽位错配对 HB-MoE 路由的扰动随层深增大",
          sub="第一次推理的配对对比（%d 对）：同初始状态、同 settle、同 flow noise，唯一差异是 wrist 图像在 left 还是 right 槽" % n_pair)
    rng = np.random.default_rng(0)
    for li in range(8):
        ax.scatter(np.full(n_pair, xs[li]) + rng.uniform(-0.15, 0.15, n_pair), cross[li],
                   s=20, color=S1, alpha=0.30, linewidths=0, zorder=3)
        w = np.array(within[li])
        ax.scatter(np.full(len(w), xs[li]) + rng.uniform(-0.15, 0.15, len(w)), w,
                   s=14, color=S2, alpha=0.22, linewidths=0, zorder=3)
    for blk in (slice(0, 4), slice(4, 8)):
        ax.plot(xs[blk], cross[blk].mean(1), "-o", color=S1, lw=2.2, ms=8, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.plot(xs[blk], [np.mean(w) for w in within[blk]], "-s", color=S2, lw=2.2, ms=7,
                zorder=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot([], [], "-o", color=S1, lw=2.2, ms=8, label="跨 layout（配对，同一集）")
    ax.plot([], [], "-s", color=S2, lw=2.2, ms=7, label="组内跨 episode（同 layout，不同初始状态）")

    for y, lab, dash in ((0.0742, "随机基线 0.074", (0, (5, 4))), (1.0, "完全相同 1.0", (0, (2, 3)))):
        ax.plot([-0.45, 3.75], [y, y], color=MUTED, lw=1.2, ls=dash, zorder=2)
        ax.plot([4.75, 8.45], [y, y], color=MUTED, lw=1.2, ls=dash, zorder=2)
        ax.text(4.25, y, lab, color=MUTED, fontsize=8.5, va="center", ha="center",
                bbox=dict(fc=SURFACE, ec="none", pad=1.5))
    ax.set_xticks(xs)
    ax.set_xticklabels([str(l) for l in HB_LAYERS])
    ax.set_xlabel("HB-MoE 层（6-11 层是 dense MLP，无路由）", color=INK2, fontsize=9.5)
    ax.set_xlim(-0.55, 8.55)
    ax.set_ylim(0, 1.22)
    ax.axvspan(-0.55, 4.25, color="#f6f5f2", zorder=0)
    ax.text(1.85, 1.16, "早层  2-5", color=INK2, fontsize=10.5, ha="center", fontweight="bold")
    ax.text(6.5, 1.16, "晚层  12-15", color=INK2, fontsize=10.5, ha="center", fontweight="bold")
    ax.text(6.5, 0.46, "晚层跨 layout 相似度已掉到\n「无关 episode」水平（两线重合）\n= 路由被 wrist 输入完全打散",
            color=INK2, fontsize=9, ha="center", va="center", linespacing=1.5)
    ax.text(1.75, 0.44, "早层跨 layout 仍比\n换个初始状态更相似",
            color=INK2, fontsize=9, ha="center", va="center", linespacing=1.5)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(0.015, 0.80))
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_depth_gradient.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig2(right, left, n):
    g = [
        ("checkpoint-right\n50/50 成功", [drift(e["ids"], e["w"], n) for e in right], S1, "o", True),
        ("released-left\n成功 (14)", [drift(e["ids"], e["w"], n) for e in left if e["summary"]["success"]], S2, "o", True),
        ("released-left\n失败 (36)", [drift(e["ids"], e["w"], n) for e in left if not e["summary"]["success"]], S2, "X", False),
    ]
    p_out = mannwhitneyu(g[1][1], g[2][1]).pvalue
    fig, ax = plt.subplots(figsize=(8.0, 5.2), facecolor=SURFACE)
    style(ax, ylabel="相邻控制步的路由 TV（漂移量）",
          title="路由漂移量完全分辨 layout，却分不出成功与失败",
          sub="100 集，每集统一取前 %d 个控制步（消除长度混淆：失败集平均 270 步 vs 成功集 121 步）" % n)
    rng = np.random.default_rng(1)
    for gi, (lab, vals, c, mk, filled) in enumerate(g):
        x = np.full(len(vals), gi) + rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(x, vals, s=68, marker=mk, facecolor=c if filled else "none",
                   edgecolor=c if not filled else SURFACE,
                   linewidths=1.6 if not filled else 0.8, alpha=0.85, zorder=4)
        ax.plot([gi - 0.26, gi + 0.26], [np.mean(vals)] * 2, color=INK, lw=2.4, zorder=5)
        ax.text(gi + 0.31, np.mean(vals), "%.3f" % np.mean(vals), color=INK,
                fontsize=10, va="center", fontweight="bold")
    rmin, lmax = min(g[0][1]), max(g[1][1] + g[2][1])
    for y in (rmin, lmax):
        ax.plot([-0.35, 2.6], [y, y], color=CRIT, lw=0.9, ls=(0, (4, 4)), alpha=0.5, zorder=2)
    ax.annotate("", xy=(0.45, rmin), xytext=(0.45, lmax),
                arrowprops=dict(arrowstyle="<->", color=CRIT, lw=1.6))
    ax.text(0.52, (rmin + lmax) / 2,
            "100 集零重叠\n最低的 right = %.3f\n最高的 left  = %.3f" % (rmin, lmax),
            color=CRIT, fontsize=9.5, va="center", linespacing=1.5)
    ax.plot([1, 2], [0.617, 0.617], color=MUTED, lw=1.2)
    ax.text(1.5, 0.6095, "两组重叠  Mann-Whitney p = %.2f" % p_out,
            color=INK2, fontsize=9.5, ha="center")
    ax.set_xticks(range(3))
    ax.set_xticklabels([x[0] for x in g], fontsize=9.5)
    ax.set_xlim(-0.45, 2.75)
    ax.set_ylim(0.600, 0.775)
    ax.text(2.05, 0.766, "实心 = 成功     空心叉 = 失败", color=INK2, fontsize=9, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_drift.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig3():
    rows = [
        ("从路由预测 wrist layout", 1.000, 0.0033, S1, "n=100"),
        ("从路由预测成功/失败\n（released-left 组内）", 0.794, 0.0033, S2, "n=50，14 成功"),
        ("对照：仅用 robot state\n预测成功/失败", 0.623, 0.0800, MUTED, "n=50"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.0), facecolor=SURFACE)
    style(ax, xlabel="留一交叉验证的平衡准确率",
          title="路由里确实带着 layout 和结果的信息",
          sub="线性探针 + 标签置换零分布；特征 = 每集前 11 个控制步的专家负载 [8 层 × 32 专家]")
    ys = np.arange(len(rows))[::-1]
    for y, (lab, acc, p, c, note) in zip(ys, rows):
        ax.barh(y, acc - 0.5, left=0.5, height=0.40, color=c, zorder=3,
                edgecolor=SURFACE, linewidth=2)
        ax.text(acc + 0.012, y, "%.3f" % acc, va="center", color=INK,
                fontsize=11, fontweight="bold")
        ax.text(acc + 0.078, y, ("p = %.4f" % p) if p < 0.01 else ("p = %.2f  (n.s.)" % p),
                va="center", color=CRIT if p < 0.05 else MUTED, fontsize=9.5)
        ax.text(0.505, y + 0.30, note, va="bottom", color=MUTED, fontsize=8.5)
    ax.axvline(0.5, color=INK2, lw=1.4, zorder=4)
    ax.text(0.5, len(rows) - 0.45, "随机水平 0.5", color=INK2, fontsize=9,
            ha="center", va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlim(0.47, 1.19)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_decoding.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig4(right, left, n):
    lr = np.mean([load_per_layer(e["ids"][:n], e["w"][:n]) for e in right], axis=0)
    ll = np.mean([load_per_layer(e["ids"][:n], e["w"][:n]) for e in left], axis=0)
    d = (ll - lr) * 100
    lim = np.abs(d).max()
    fig, ax = plt.subplots(figsize=(10.2, 3.9), facecolor=SURFACE)
    im = ax.imshow(d, cmap=DIVERGE, vmin=-lim, vmax=lim, aspect="auto")
    style(ax, xlabel="HB-MoE 专家编号（每层 32 个）", ylabel="HB-MoE 层",
          title="负载变化摊在全部 32 个专家上，没有专家被废弃",
          sub="released-left 减 checkpoint-right 的专家负载占比（百分点），各 50 集")
    ax.grid(False)
    ax.set_yticks(range(8)); ax.set_yticklabels(HB_LAYERS)
    ax.set_xticks(range(0, 32, 2)); ax.set_xticklabels(range(0, 32, 2))
    ax.set_xticks(np.arange(-0.5, 32, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=9, length=3)
    cb.set_label("right 更多  ←→  left 更多   (百分点)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_load_delta.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig5(right, left, n):
    X = np.vstack([feats(right, n), feats(left, n)])
    Z = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    ok = np.array([e["summary"]["success"] for e in right + left])
    is_left = np.array([False] * len(right) + [True] * len(left))
    fig, ax = plt.subplots(figsize=(7.8, 5.6), facecolor=SURFACE)
    style(ax, xlabel="PC1", ylabel="PC2",
          title="每集一个点：两种 layout 在路由空间里完全分开",
          sub="每集前 %d 个控制步的专家负载 [8×32] 做 PCA；成功/失败只用形状标注，未参与降维" % n)
    for mask, c, mk, filled, lab in (
        (~is_left & ok, S1, "o", True, "checkpoint-right 成功 (50)"),
        (is_left & ok, S2, "o", True, "released-left 成功 (14)"),
        (is_left & ~ok, S2, "X", False, "released-left 失败 (36)"),
    ):
        ax.scatter(Z[mask, 0], Z[mask, 1], s=95, marker=mk,
                   facecolor=c if filled else "none", edgecolor=c if not filled else SURFACE,
                   linewidths=1.8 if not filled else 0.9, alpha=0.85, zorder=4, label=lab)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(0.0, 1.0))
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(0.02, 0.03, "layout 的线性探针留一准确率 = 1.000（100/100）",
            transform=ax.transAxes, color=CRIT, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_pca.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main():
    right = load("/home/jovyan/work/himoe-route-capture/runs/right50")
    left = load("/home/jovyan/work/himoe-route-capture/runs/left50")
    n = min(e["ids"].shape[0] for e in right + left)
    fig1(right, left)
    fig2(right, left, n)
    fig3()
    fig4(right, left, n)
    fig5(right, left, n)
    for p in sorted(OUT.glob("*.png")):
        print(p.name, p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
