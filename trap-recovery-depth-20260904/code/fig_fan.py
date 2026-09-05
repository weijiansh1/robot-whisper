"""真实数据：r(t) 轨迹 + 各时点重抽 K 条的实测结局（分支形状为示意，比例与步数为实测）。"""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
F = "/home/jovyan/work/himoe-vla/himoe-route-capture/assets/NotoSansSC.ttf"
fm.fontManager.addfont(F)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=F).get_name()]
plt.rcParams["axes.unicode_minus"] = False
OK, BAD, LINE, AL, GREY = "#2A9D8F", "#C1443B", "#202124", "#E8710A", "#B9BEC4"

R = "runs/smoke_budget"
t = json.load(open(f"{R}/trunk.json")); m = json.load(open(f"{R}/manifest.json"))
r = np.array([np.nan if v is None else v for v in t["ratios"]], float)
ESC = 1.20                                     # 逃出线（画在 r 之上，仅作视觉锚）

arms = [("control", m["control_query"], "报警前\n(对照)"),
        ("triggered", m["fork_query"], "报警点"),
        ("delayed_+4", m["fork_query"] + 4, "报警后 +4"),
        ("delayed_+8", m["fork_query"] + 8, "报警后 +8")]

fig, ax = plt.subplots(figsize=(12.4, 6.2), dpi=150)
ax.axhline(0.95, color=BAD, ls="--", lw=1.2); ax.text(1, .955, "θ = 0.95", color=BAD, fontsize=10)
ax.axhline(1.0, color="#9AA0A6", ls=":", lw=1.0)
ax.axhline(ESC, color=OK, ls="--", lw=1.2)
ax.text(66, ESC + .012, "逃出（任务成功）", color=OK, ha="right", fontsize=11)
ax.plot(np.arange(len(r)), r, color=LINE, lw=2.4, zorder=6)
ax.plot(len(r) - 1, r[-1], "x", color=LINE, ms=11, mew=2.6, zorder=7)
ax.text(len(r) + .8, r[-1], "原轨迹\n跑满未成功", fontsize=9.5, color="#3C4043",
        va="center", linespacing=1.3)

rng = np.random.default_rng(3)
for arm, q, lab in arms:
    fs = sorted(glob.glob(f"{R}/{arm}/candidate_*.json"))
    ss = [json.load(open(f)) for f in fs]
    n, k = sum(s["success"] for s in ss), len(ss)
    y0 = r[q]
    for j, s in enumerate(ss):
        steps = s["inference_calls"]
        xs = np.linspace(q, q + steps, 40)
        tt = (xs - q) / max(steps, 1)
        if s["success"]:
            ys = y0 + (ESC - y0) * tt ** 1.4
        else:
            ys = y0 - (0.10 + .05 * j) * tt + rng.normal(0, .006, xs.size).cumsum() * .35
        ax.plot(xs, ys, color=OK if s["success"] else GREY,
                lw=1.9 if s["success"] else 1.2, alpha=.95 if s["success"] else .8,
                zorder=5 if s["success"] else 3)
        ax.plot(xs[-1], ys[-1], "o" if s["success"] else "|",
                color=OK if s["success"] else GREY, ms=5 if s["success"] else 7, zorder=5)
    col = OK if n else BAD
    ax.plot([q], [y0], "o", color=AL, ms=10, mec="white", mew=1.8, zorder=9)
    ax.plot([q, q], [y0, 1.30], color=AL, lw=.9, alpha=.35, zorder=2)
    ax.text(q, 1.315, f"{n}/{k}", color=col, ha="center", fontsize=15, weight="bold")
    ax.text(q, 1.375, lab, color="#3C4043", ha="center", fontsize=10, linespacing=1.3)

ax.annotate("报警：连续 3 步 r<θ", xy=(m["fork_query"], r[m["fork_query"]]),
            xytext=(m["fork_query"] - 17, .70), fontsize=11, color=AL,
            arrowprops=dict(arrowstyle="->", color=AL, lw=1.4))
ax.text(.5, -.145, "纵轴 r(t) 与各分叉点的成功比例、每条分支的步数均为实测；"
                   "分支曲线的形状是示意（分支未存路由，算不出自己的 r）。",
        transform=ax.transAxes, ha="center", fontsize=9.5, color="#80868B")
ax.set_xlim(0, 67); ax.set_ylim(np.nanmin(r) - .06, 1.44)
ax.set_xlabel("控制步 (query)", fontsize=12); ax.set_ylabel("检测器 r(t)", fontsize=12)
ax.set_title("各时点重抽噪声 K=3 的实测结局 —— 每臂等预算 20 步\n"
             "libero_10 t08 · paper-right · init 0 · **仅 1 条 trunk，统计上不成立**",
             fontsize=12.5, pad=26)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig("figs/fan_real.png", bbox_inches="tight", facecolor="white")
print("saved figs/fan_real.png")
for a, q, _ in arms:
    ss = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/{a}/candidate_*.json"))]
    print(f"  {a:12} q{q:<3} {sum(s['success'] for s in ss)}/{len(ss)}  步数={[s['inference_calls'] for s in ss]}")
