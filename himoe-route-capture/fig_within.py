import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
sys.path.insert(0,"/home/jovyan/work/himoe-libero-wrist-fix/src"); sys.path.insert(0,".")
from analyze_routes import load, load_per_layer
from make_figures import style, SURFACE, INK, INK2, MUTED, S1, S2, CRIT, OUT
FONT="/home/jovyan/work/himoe-route-capture/assets/NotoSansSC.ttf"; fm.fontManager.addfont(FONT)
plt.rcParams["font.family"]=fm.FontProperties(fname=FONT).get_name(); plt.rcParams["axes.unicode_minus"]=False

sm=json.load(open("runs/within/summaries.json"))
states=sorted({e["init_state_id"] for e in sm})
rate={s: (sum(e["success"] for e in sm if e["init_state_id"]==s), 10) for s in states}

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), facecolor=SURFACE,
                         gridspec_kw={"width_ratios":[1.15,1]})
ax=axes[0]
style(ax, ylabel="10 次重复中的成功次数",
      title="同一场景，只换 flow noise，成败就会翻转",
      sub="每个 init state 跑 10 次，环境 seed / 初始状态 / 模型全部固定，唯一变量是采样噪声")
xs=np.arange(len(states))
cols=[MUTED if rate[s][0] in (0,10) else S2 for s in states]
ax.bar(xs, [rate[s][0] for s in states], color=cols, width=0.62, zorder=3,
       edgecolor=SURFACE, linewidth=2)
for x,s in zip(xs,states):
    c = MUTED if rate[s][0] in (0, 10) else INK
    ax.text(x, rate[s][0]+0.25, "%d/10" % rate[s][0], ha="center", color=c, fontsize=10,
            fontweight="bold" if rate[s][0] == 0 else "normal")
ax.set_xticks(xs); ax.set_xticklabels(["s%d"%s for s in states])
ax.set_xlabel("LIBERO-Goal task 0 的初始状态", color=INK2, fontsize=9.5)
ax.set_ylim(0,11.4); ax.set_yticks(range(0,11,2))
ax.text(0.5, 10.6, "无柱 = 10 次全失败，场景本身决定    橙柱 = 混合，噪声决定成败", color=INK2, fontsize=9.5)

ax=axes[1]
style(ax, xlabel="留一交叉验证的平衡准确率",
      title="扣掉场景之后，路由仍然能区分成败",
      sub="60 集来自 6 个混合 state；置换检验在每个 state 内部打乱标签")
rows=[("不扣场景\n（会混进场景难度）",0.859,None,MUTED),
      ("按 state 去均值后\n（场景被完全扣除）",0.710,0.0060,S2)]
ys=np.arange(len(rows))[::-1]
for y,(lab,acc,p,c) in zip(ys,rows):
    ax.barh(y, acc-0.5, left=0.5, height=0.36, color=c, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.text(acc+0.006, y, "%.3f"%acc, va="center", color=INK, fontsize=11, fontweight="bold")
    if p: ax.text(acc+0.045, y, "p = %.4f"%p, va="center", color=CRIT, fontsize=9.5)
ax.axvline(0.5, color=INK2, lw=1.4, zorder=4)
ax.text(0.5, 1.55, "随机水平 0.5", color=INK2, fontsize=9, ha="center")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xlim(0.47,0.99); ax.set_ylim(-0.55,1.75); ax.grid(axis="y", visible=False)
ax.annotate("", xy=(0.710,-0.34), xytext=(0.859,-0.34),
            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.3))
ax.text(0.785,-0.47,"这段是场景难度贡献的", color=INK2, fontsize=9, ha="center")
fig.tight_layout()
fig.savefig(OUT/"fig6_within_state.png", dpi=170, facecolor=SURFACE)
print("saved")
