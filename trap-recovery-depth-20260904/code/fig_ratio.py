"""真实数据：检测器 r(t) 轨迹 + 各分叉点的实测逃出比例。"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

F = "/home/jovyan/work/himoe-vla/himoe-route-capture/assets/NotoSansSC.ttf"
fm.fontManager.addfont(F)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=F).get_name()]
plt.rcParams["axes.unicode_minus"] = False
OK, BAD, LINE, AL = "#2A9D8F", "#C1443B", "#202124", "#E8710A"

R = "runs/smoke_alarm_i3w1"
t = json.load(open(f"{R}/trunk.json")); m = json.load(open(f"{R}/manifest.json"))
r = np.array([np.nan if v is None else v for v in t["ratios"]], float)
q = np.arange(len(r)); THETA, W0 = 0.95, 8

arms = [(m["control_query"], m["control"]["successes"], m["control"]["n"], "报警前\n(对照)")]
arms.append((m["fork_query"], m["triggered"]["successes"], m["triggered"]["n"], "报警点"))
for k in sorted(m["delayed"], key=lambda x: int(x)):
    d = m["delayed"][k]
    arms.append((d["query"], d["successes"], d["n"], f"报警后 {k}"))

fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=150)
ax.axhspan(r[~np.isnan(r)].min() - .06, THETA, color="#FBE9E7", zorder=0)
ax.axhline(THETA, color=BAD, ls="--", lw=1.3, zorder=2)
ax.text(51.6, THETA + .006, f"θ = {THETA}", color=BAD, ha="right", fontsize=10.5)
ax.axhline(1.0, color="#9AA0A6", ls=":", lw=1.1, zorder=1)
ax.text(51.6, 1.004, "健康水平 r = 1", color="#5F6368", ha="right", fontsize=10)

ax.axvspan(0, 2 * W0, color="#F1F3F4", zorder=0)
ax.text(8, 1.20, "历史不足 2·W0=16\n报警器不可读",
        ha="center", fontsize=9.5, color="#80868B", linespacing=1.3)

ax.plot(q, r, color=LINE, lw=2.4, zorder=5, label="r(t) — 路由换手率 ÷ 自身开头 8 步")
ax.plot(q, r, ".", color=LINE, ms=4, zorder=6)

for x, s, n, lab in arms:
    frac = s / n
    col = OK if frac > .5 else (BAD if frac == 0 else AL)
    y = r[x] if x < len(r) and not np.isnan(r[x]) else np.nanmin(r)
    ax.plot([x, x], [y, 1.135], color=col, lw=1.1, alpha=.5, zorder=3)
    ax.plot([x], [y], "o", color=col, ms=10, mec="white", mew=1.8, zorder=8)
    ax.text(x, 1.15, f"{s}/{n}", color=col, ha="center", fontsize=13, weight="bold")
    ax.text(x, 1.205, lab, color="#3C4043", ha="center", fontsize=9.5, linespacing=1.3)

PHYS = 42
ax.axvline(PHYS, color="#7B1FA2", ls="-.", lw=1.4, alpha=.8, zorder=2)
ax.text(PHYS + .6, 0.52, "物理 loop onset q42\n（要特权物体位姿才算得出）",
        color="#7B1FA2", fontsize=9.5, linespacing=1.35)
ax.annotate("", xy=(29, 0.50), xytext=(42, 0.50),
            arrowprops=dict(arrowstyle="<->", color="#7B1FA2", lw=1.3))
ax.text(35.5, 0.475, "报警早 13 步", color="#7B1FA2", ha="center", fontsize=10)
ax.annotate("报警：连续 3 步 r<θ", xy=(m["fork_query"], r[m["fork_query"]]),
            xytext=(m["fork_query"] - 12, 0.68), fontsize=11, color=AL,
            arrowprops=dict(arrowstyle="->", color=AL, lw=1.4))
ax.plot(len(r) - 1, r[-1], "x", color=LINE, ms=11, mew=2.6, zorder=8)
ax.text(len(r) - 1.4, r[-1] - .035, "原轨迹跑满 52 步\n未成功", ha="right",
        fontsize=9.5, color="#3C4043", linespacing=1.3)

ax.set_xlim(0, 52.5); ax.set_ylim(np.nanmin(r) - .07, 1.27)
ax.set_xlabel("控制步 (query)", fontsize=12)
ax.set_ylabel("检测器 r(t)", fontsize=12)
ax.set_title("真实数据：一条失败 rollout 的检测器轨迹，与各时点重抽噪声的逃出比例\n"
             f"libero_10 t08 · paper-right · init 3 · 冒烟 K=2（样本极小，仅示形）",
             fontsize=12.5, pad=14)
ax.legend(loc="lower right", frameon=False, fontsize=10.5,
          bbox_to_anchor=(1.0, -0.015))
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.tight_layout()
fig.savefig("figs/ratio_real.png", bbox_inches="tight", facecolor="white")
print("saved figs/ratio_real.png   alarm =", m["fork_query"], " arms =", [(a[0], f"{a[1]}/{a[2]}") for a in arms])
