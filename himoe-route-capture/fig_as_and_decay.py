"""Fig 7: AS-MoE is frozen on LIBERO.  Fig 8: within-scene drift decay."""

from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")
sys.path.insert(0, "/home/jovyan/work/himoe-route-capture")
from analyze_routes import load, tv  # noqa: E402
from himoe_libero_bridge.routing import sparse_routes_to_dense  # noqa: E402
from make_figures import CRIT, INK, INK2, MUTED, OUT, S1, S2, SURFACE, style  # noqa: E402

FONT = "/home/jovyan/work/himoe-route-capture/assets/NotoSansSC.ttf"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

MIXED = [1, 5, 9, 19, 24, 31]


def fig7():
    d = json.load(open("/home/jovyan/work/himoe-route-capture/as_routing.json"))
    layers = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURFACE)
    style(ax, ylabel="softmax 概率",
          title="AS-MoE 在 LIBERO 上是一个常量路由器",
          sub="输入只有 data_mask = [1]×7 + [0]×17（LIBERO 的动作维度掩码），全程不变，"
              "所以 4 层各自恒选同一个专家")
    w = 0.24
    xs = np.arange(len(layers))
    for e in range(3):
        vals = [d[str(l)]["fp32"][e] for l in layers]
        sel = [d[str(l)]["sel_bf16"] == e for l in layers]
        ax.bar(xs + (e - 1) * w, vals, width=w * 0.88,
               color=[S1 if s else "#c6d9f2" for s in sel],
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for x, v, s in zip(xs + (e - 1) * w, vals, sel):
            ax.text(x, v + 0.012, "%.3f" % v, ha="center", color=INK if s else MUTED,
                    fontsize=8.5, fontweight="bold" if s else "normal")
    ax.axhline(1 / 3, color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax.text(3.55, 1 / 3 + 0.008, "均匀 1/3", color=MUTED, fontsize=8.5, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(["layer %d" % l for l in layers], fontsize=10)
    ax.set_xlabel("AS-MoE 层（每层 3 个专家，Top-1）", color=INK2, fontsize=9.5)
    ax.set_ylim(0, 0.68)
    ax.text(0.5, 0.63,
            "深蓝 = 实际选中的专家",
            color=INK2, fontsize=9.5, ha="center")
    ax.annotate("三个专家精确并列\nbf16 下 logits 全是 0.25，\ntopk 按索引取 0；fp32 会取 2\n→ 选谁由舍入决定",
                xy=(1, 0.345), xytext=(1.62, 0.50), color=CRIT, fontsize=9,
                linespacing=1.5, ha="left",
                arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.5))
    fig.tight_layout()
    fig.savefig(OUT / "fig7_as_frozen.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig8():
    eps = load("/home/jovyan/work/himoe-route-capture/runs/within")
    eps = [e for e in eps if e["summary"]["init_state_id"] in MIXED]
    curves = {True: [], False: []}
    for e in eps:
        d = sparse_routes_to_dense(e["ids"], e["w"])
        d = d / d.sum(-1, keepdims=True)
        curves[e["summary"]["success"]].append(tv(d[1:], d[:-1]).mean(axis=(1, 2, 3)))
    overlap = min(min(len(x) for x in curves[True]), min(len(x) for x in curves[False]))

    fig, ax = plt.subplots(figsize=(9.0, 4.9), facecolor=SURFACE)
    style(ax, xlabel="控制步", ylabel="相邻控制步的路由 TV",
          title="漂移量分不出成败——失败尾巴的衰减是截尾造成的",
          sub="6 个混合 init state 的 60 集：成功和失败来自同一批场景。重叠窗口内两条线重合；"
              "之后没有成功回合可比")
    m_all = max(len(x) for x in curves[False])
    ax.axvspan(overlap + 0.5, m_all + 0.5, color="#f2f1ee", zorder=0)
    for ok, c, lab in ((False, MUTED, "失败 (23 集)"), (True, S2, "成功 (37 集)")):
        arr = np.full((len(curves[ok]), max(len(x) for x in curves[ok])), np.nan)
        for i, x in enumerate(curves[ok]):
            arr[i, :len(x)] = x
        mu, sd = np.nanmean(arr, 0), np.nanstd(arr, 0)
        x = np.arange(1, arr.shape[1] + 1)
        ax.plot(x, mu, color=c, lw=2.4, zorder=5, label=lab)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0, zorder=3)
    ax.axvline(overlap + 0.5, color=CRIT, lw=1.3, ls=(0, (4, 3)), zorder=6)
    ax.text(overlap + 1.0, 0.795, "第 %d 步之后\n成功回合已全部结束\n（无可比样本）" % overlap,
            color=CRIT, fontsize=9, va="top", linespacing=1.5)
    ax.annotate("", xy=(1, 0.55), xytext=(overlap + 0.5, 0.55),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text((overlap + 1) / 2, 0.535,
            "重叠窗口：成功 %.4f vs 失败 %.4f，Mann-Whitney p = 0.20\n"
            "按 state 配对 Wilcoxon p = 0.69（6 个场景里只有 4 个同号）" % (0.6643, 0.6577),
            color=INK2, fontsize=9, ha="center", va="top", linespacing=1.5)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_xlim(0.5, m_all + 0.5)
    ax.set_ylim(0.46, 0.83)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_drift_decay.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    fig7()
    fig8()
    print("saved fig7, fig8")
