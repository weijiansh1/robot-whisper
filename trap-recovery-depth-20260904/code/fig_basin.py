"""假说示意：报警点分叉 vs 谷底分叉。**不是实测数据。**"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

F = "/home/jovyan/work/himoe-vla/himoe-route-capture/assets/NotoSansSC.ttf"
fm.fontManager.addfont(F)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=F).get_name()]
plt.rcParams["axes.unicode_minus"] = False

ESCAPE, STUCK, TRUNK, ALARM = "#2A9D8F", "#B9BEC4", "#202124", "#E8710A"
rng = np.random.default_rng(7)

def terrain(x):
    """背景地形：左侧高地 → 下坡 → 谷 → 右侧出口坡。"""
    return (0.95 * np.exp(-((x - 6) / 14) ** 2)
            - 0.55 * np.exp(-((x - 42) / 9) ** 2) + 0.42)

fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
X = np.linspace(0, 62, 700)
ax.fill_between(X, terrain(X) - 0.9, terrain(X), color="#EDEFF2", zorder=0)
ax.plot(X, terrain(X), color="#C7CCD1", lw=1.6, zorder=1)

# 主轨迹：滑进谷，再也没出来
xt = np.linspace(0, 52, 260)
yt = terrain(xt) + 0.055 * np.exp(-((xt - 20) / 16) ** 2) * np.sin(xt / 2.2)
ax.plot(xt, yt, color=TRUNK, lw=2.6, zorder=6,
        label="原轨迹（不干预）— 跑满 52 步仍失败")
ax.plot(52, yt[-1], "x", color=TRUNK, ms=11, mew=2.6, zorder=7)

def fan(x0, n_escape, k=8, reach=62.0):
    """从 x0 分叉 k 条，其中 n_escape 条爬出右侧。"""
    y0 = terrain(np.array([x0]))[0]
    for i in range(k):
        esc = i < n_escape
        xs = np.linspace(x0, reach if esc else min(x0 + 17, 56), 90)
        t = (xs - x0) / max(reach - x0, 1e-6)
        if esc:                      # 爬出去
            ys = y0 + (1.05 - y0) * t ** 1.5 + rng.normal(0, .012, xs.size).cumsum() * .05
        else:                        # 在谷里打转
            ys = y0 - 0.06 * t + rng.normal(0, .015, xs.size).cumsum() * .06
        ax.plot(xs, ys, color=ESCAPE if esc else STUCK, lw=1.5 if esc else 1.1,
                alpha=.95 if esc else .75, zorder=5 if esc else 3)
    ax.plot([x0], [y0], "o", color=ALARM, ms=9, mec="white", mew=1.6, zorder=8)

fan(29, 7)          # 报警点：8 条里 7 条逃出
fan(45, 1)          # 谷底：8 条里 1 条

ax.axhline(1.05, color=ESCAPE, ls="--", lw=1.1, alpha=.6)
ax.text(61.4, 1.075, "逃出 / 任务成功", color=ESCAPE, ha="right", fontsize=11)

ax.annotate("报警点 q29\n重抽 8 条噪声 → **7/8 逃出**", xy=(29, terrain(np.array([29]))[0]),
            xytext=(15.5, 1.16), fontsize=11.5, color=ALARM, ha="center",
            arrowprops=dict(arrowstyle="->", color=ALARM, lw=1.5))
ax.annotate("谷底 q45\n重抽 8 条噪声 → **1/8 逃出**", xy=(45, terrain(np.array([45]))[0]),
            xytext=(50.5, 0.03), fontsize=11.5, color=ALARM, ha="center",
            arrowprops=dict(arrowstyle="->", color=ALARM, lw=1.5))
ax.text(42, -0.30, "谷 = trap", fontsize=12, color="#5F6368", ha="center", style="italic")

ax.set_xlim(0, 62); ax.set_ylim(-0.42, 1.30)
ax.set_xlabel("控制步 (query)", fontsize=12)
ax.set_ylabel("离逃出还差多少（示意）", fontsize=12)
ax.set_title("假说：越晚干预越难走出 trap —— 这正是本实验要检验的，不是已知结论",
             fontsize=13, pad=12)
ax.legend(loc="upper left", frameon=False, fontsize=10.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.5, -0.155, "示意图，非实测数据。数字为假设值。",
        transform=ax.transAxes, ha="center", fontsize=9.5, color="#80868B")
fig.tight_layout()
fig.savefig("/home/jovyan/work/himoe-vla/trap-recovery-depth-20260904/figs/basin_hypothesis.png",
            bbox_inches="tight", facecolor="white")
print("saved figs/basin_hypothesis.png")
