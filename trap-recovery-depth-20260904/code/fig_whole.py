"""一条完整轨迹 + 四个分叉口 + 分支按红绿。全部实测（含分支路由）。"""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
F = "/home/jovyan/work/himoe-vla/himoe-route-capture/assets/NotoSansSC.ttf"
fm.fontManager.addfont(F); plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=F).get_name()]
plt.rcParams["axes.unicode_minus"] = False
GREEN, RED, TRUNK, AL = "#12855F", "#D93025", "#202124", "#E8710A"
DEN, DEEP, W0 = 9, slice(4, 8), 8

def jac(p, c):
    a, b = p[DEN, DEEP], c[DEN, DEEP]
    return float(np.mean([1.0 - len(set(a[l,s].tolist()) & set(b[l,s].tolist()))
                          / max(len(set(a[l,s].tolist()) | set(b[l,s].tolist())), 1)
                          for l in range(a.shape[0]) for s in range(a.shape[1])]))

R = "runs_routed/i0_s3"
tj = json.load(open(f"{R}/trunk.json")); m = json.load(open(f"{R}/manifest.json"))
td = np.array(tj["distances"], float); base = td[:W0].mean(); a0 = m["fork_query"]
FORKS = [("control", m["control_query"], "报警前"), ("triggered", a0, "报警点"),
         ("delayed_+4", a0+4, "报警后+4"), ("delayed_+8", a0+8, "报警后+8")]

fig, ax = plt.subplots(figsize=(15, 7.4), dpi=150)
tx, ty = np.arange(1, len(td)+1), td/base
ax.plot(tx, ty, color=TRUNK, lw=3.0, zorder=8, solid_capstyle="round")
ax.plot(tx[-1], ty[-1], "X", color=TRUNK, ms=13, mew=0, zorder=9)
ax.axhline(1.0, color="#9AA0A6", ls=":", lw=1.1, zorder=1)
ax.text(.6, 1.005, "开头的正常换手水平", color="#5F6368", fontsize=10)

for arm, q, lab in FORKS:
    fs = sorted(glob.glob(f"{R}/{arm}/candidate_*.npz"))
    ok = 0
    for f in fs:
        z = np.load(f); s = json.load(open(f.replace(".npz", ".json")))
        ids = z["routing_expert_ids"]
        if len(ids) < 2: continue
        d = np.array([jac(ids[i-1], ids[i]) for i in range(1, len(ids))])/base
        xs = np.arange(q+1, q+1+len(d))
        # 从 trunk 上引一小段，视觉上真的"分叉"出来
        xs = np.r_[q, xs]; d = np.r_[ty[q-1], d]
        good = s["success"]; ok += good
        ax.plot(xs, d, color=GREEN if good else RED, lw=2.6 if good else 1.1,
                alpha=1.0 if good else .32, zorder=7 if good else 3,
                solid_capstyle="round")
        if good:
            ax.plot(xs[-1], d[-1], "o", color=GREEN, ms=7, zorder=8)
    ax.axvline(q, color=AL, lw=1.3, alpha=.55, ls="--", zorder=2)
    ax.plot([q], [ty[q-1]], "o", color=AL, ms=13, mec="white", mew=2.0, zorder=10)
    col = GREEN if ok else RED
    ax.text(q, 1.30, "%d/8" % ok, color=col, ha="center", fontsize=19, weight="bold")
    ax.text(q, 1.245, "逃出", color=col, ha="center", fontsize=10)
    ax.text(q, 1.355, "%s\nq%d" % (lab, q), color="#3C4043", ha="center",
            fontsize=11.5, linespacing=1.35)

ax.annotate("原轨迹不干预：一直磨到 q77 才自己完成",
            xy=(tx[-1], ty[-1]), xytext=(56, .72), fontsize=11, color=TRUNK,
            arrowprops=dict(arrowstyle="->", color=TRUNK, lw=1.4))
ax.set_xlim(0, 70); ax.set_ylim(.33, 1.45)
ax.set_xlabel("控制步 (query)", fontsize=13)
ax.set_ylabel("MoE 路由换手率  d(t) / 开头基线", fontsize=13)
ax.set_title("一条失败轨迹，在四个时刻各重抽 8 条噪声 —— libero_10 t08 · init 0\n"
             "绿 = 逃出（任务完成）   红 = 没逃出   黑 = 原轨迹", fontsize=13.5, pad=52)
ax.legend(handles=[
    Line2D([], [], color=TRUNK, lw=3, label="原轨迹（不干预，跑满判失败）"),
    Line2D([], [], color=AL, marker="o", ls="", ms=11, label="分叉口（各重抽 8 条噪声）"),
    Line2D([], [], color=GREEN, lw=2.6, label="逃出的分支（只跑 3 步就完成）"),
    Line2D([], [], color=RED, lw=1.1, alpha=.5, label="没逃出的分支（跑满 20 步）"),
], loc="lower left", frameon=False, fontsize=10.5)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig("figs/whole.png", bbox_inches="tight", facecolor="white")
print("saved figs/whole.png")
