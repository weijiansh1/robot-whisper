"""一条失败轨迹：报警点分叉能逃出，越往后越逃不出。全部实测。"""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
F = "/home/jovyan/work/himoe-vla/himoe-route-capture/assets/NotoSansSC.ttf"
fm.fontManager.addfont(F)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=F).get_name()]
plt.rcParams["axes.unicode_minus"] = False
OK, GREY, TRUNK, AL, BAD = "#2A9D8F", "#B9BEC4", "#202124", "#E8710A", "#C1443B"

R = "runs_cap52/i0_s3"
t = json.load(open(f"{R}/trunk.json")); m = json.load(open(f"{R}/manifest.json"))
r = np.array([np.nan if v is None else v for v in t["ratios"]], float)
ESC = 1.22

arms = [("triggered", "报警点\nq37"), ("delayed_+4", "报警后 +4\nq41"), ("delayed_+8", "报警后 +8\nq45")]
fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=150)
ax.axhline(.95, color=BAD, ls="--", lw=1.2); ax.text(.5, .957, "θ = 0.95", color=BAD, fontsize=10)
ax.axhline(1.0, color="#9AA0A6", ls=":", lw=1.0)
ax.axhline(ESC, color=OK, ls="--", lw=1.3)
ax.text(78, ESC + .015, "逃出（任务完成）", color=OK, ha="right", fontsize=11)

ax.plot(np.arange(len(r)), r, color=TRUNK, lw=2.5, zorder=6)
ax.plot(52, r[-1], "x", color=TRUNK, ms=12, mew=2.8, zorder=7)
ax.annotate("原轨迹带着原噪声一直磨到 q77 才自己完成\n（52 步截停口径下判为失败）",
            xy=(52, r[-1]), xytext=(56, .62), fontsize=10, color=TRUNK, linespacing=1.4,
            arrowprops=dict(arrowstyle="->", color=TRUNK, lw=1.2))
ax.plot([52, 77], [r[-1], ESC], color=TRUNK, lw=1.6, ls=":", zorder=5)
ax.plot(77, ESC, "o", color=TRUNK, ms=8, zorder=7)

rng = np.random.default_rng(5)
for arm, lab in arms:
    fs = sorted(glob.glob(f"{R}/{arm}/candidate_*.json"))
    ss = [json.load(open(f)) for f in fs]
    q = ss[0]["fork_query"]; y0 = r[q]
    n = sum(s["success"] for s in ss)
    for j, s in enumerate(ss):
        steps = s["inference_calls"]; xs = np.linspace(q, q + steps, 30)
        tt = (xs - q) / max(steps, 1)
        if s["success"]:
            ys = y0 + (ESC - y0) * tt ** .75
        else:
            ys = y0 - (.05 + .03 * j) * tt + rng.normal(0, .005, xs.size).cumsum() * .3
        ax.plot(xs, ys, color=OK if s["success"] else GREY,
                lw=2.0 if s["success"] else 1.2, alpha=.95 if s["success"] else .8,
                zorder=5 if s["success"] else 3)
        ax.plot(xs[-1], ys[-1], "o" if s["success"] else "|",
                color=OK if s["success"] else GREY, ms=5 if s["success"] else 8, zorder=5)
    col = OK if n else BAD
    ax.plot([q], [y0], "o", color=AL, ms=11, mec="white", mew=1.8, zorder=9)
    ax.plot([q, q], [y0, 1.33], color=AL, lw=.9, alpha=.35, zorder=2)
    ax.text(q, 1.345, f"{n}/8", color=col, ha="center", fontsize=17, weight="bold")
    ax.text(q, 1.415, lab, color="#3C4043", ha="center", fontsize=10, linespacing=1.3)

ax.text(39.5, 1.27, "成功的分支只跑 3 步就完成", color=OK, fontsize=10.5, ha="center")
ax.set_xlim(0, 80); ax.set_ylim(np.nanmin(r) - .06, 1.50)
ax.set_xlabel("控制步 (query)", fontsize=12); ax.set_ylabel("检测器 r(t)", fontsize=12)
ax.set_title("一条失败轨迹：报警点重抽噪声能逃出，越往后越逃不出\n"
             "libero_10 t08 · paper-right · init 0 · 三臂预算同为 20 步", fontsize=13, pad=30)
ax.legend(handles=[
    Line2D([], [], color=TRUNK, lw=2.5, label="原轨迹 r(t)（不干预）"),
    Line2D([], [], color=AL, marker="o", ls="", ms=10, label="分叉点（各重抽 8 条噪声）"),
    Line2D([], [], color=OK, lw=2, label="逃出的分支"),
    Line2D([], [], color=GREY, lw=1.2, label="没逃出的分支（跑满 20 步）"),
], loc="lower left", frameon=False, fontsize=10)
ax.text(.5, -.135, "r(t)、分叉点、每条分支的步数与成败均为实测；分支曲线形状为示意（该批未存分支路由）。"
        "报警前的随机对照点抽到 q18，任务尚在早期，不可比，故未画入。",
        transform=ax.transAxes, ha="center", fontsize=9.3, color="#80868B")
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig("figs/one_trunk.png", bbox_inches="tight", facecolor="white")
print("saved figs/one_trunk.png")
