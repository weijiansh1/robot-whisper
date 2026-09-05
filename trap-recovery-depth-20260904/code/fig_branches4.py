"""四个分叉点各一格，每格只画该点的 8 条分支：逃出 vs 没逃出。"""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
F = "/home/jovyan/work/himoe-vla/himoe-route-capture/assets/NotoSansSC.ttf"
fm.fontManager.addfont(F)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=F).get_name()]
plt.rcParams["axes.unicode_minus"] = False
OK, NO, TRUNK = "#2A9D8F", "#C1443B", "#202124"
DEN, DEEP, W0 = 9, slice(4, 8), 8

def jac(p, c):
    a, b = p[DEN, DEEP], c[DEN, DEEP]
    return float(np.mean([1.0 - len(set(a[l,s].tolist()) & set(b[l,s].tolist()))
                          / max(len(set(a[l,s].tolist()) | set(b[l,s].tolist())), 1)
                          for l in range(a.shape[0]) for s in range(a.shape[1])]))

R = "runs_routed/i0_s3"
tj = json.load(open(f"{R}/trunk.json")); m = json.load(open(f"{R}/manifest.json"))
td = np.array(tj["distances"], float); base = td[:W0].mean()
a0 = m["fork_query"]
PANELS = [("control", m["control_query"], "报警前 q%d（对照）" % m["control_query"]),
          ("triggered", a0, "报警点 q%d" % a0),
          ("delayed_+4", a0+4, "报警后 +4  q%d" % (a0+4)),
          ("delayed_+8", a0+8, "报警后 +8  q%d" % (a0+8))]

fig, axes = plt.subplots(1, 4, figsize=(19, 5.4), dpi=150, sharey=True)
for ax, (arm, q, title) in zip(axes, PANELS):
    # trunk 在同一区间的换手率作参照
    seg = td[q:q+21] / base
    ax.plot(np.arange(1, len(seg)+1), seg, color=TRUNK, lw=2.0, alpha=.9, zorder=4)
    nok = 0
    for f in sorted(glob.glob(f"{R}/{arm}/candidate_*.npz")):
        z = np.load(f); s = json.load(open(f.replace(".npz", ".json")))
        ids = z["routing_expert_ids"]
        if len(ids) < 2: continue
        d = np.array([jac(ids[i-1], ids[i]) for i in range(1, len(ids))]) / base
        x = np.arange(1, len(d)+1)
        good = s["success"]; nok += good
        ax.plot(x, d, color=OK if good else NO, lw=2.1 if good else 1.1,
                ls="-" if good else (0, (3, 2)), alpha=.95 if good else .55,
                zorder=6 if good else 3)
        ax.text(x[-1]+.4, d[-1], "✓%d步" % s["inference_calls"] if good else "×",
                color=OK if good else NO, fontsize=8.5 if good else 9,
                va="center", zorder=7)
    ax.axhline(1.0, color="#9AA0A6", ls=":", lw=1.0)
    ax.set_title("%s\n%d/8 逃出" % (title, nok), fontsize=12,
                 color=OK if nok else NO, pad=8)
    ax.set_xlabel("分叉后第几步", fontsize=11)
    ax.set_xlim(.5, 21.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axes[0].set_ylabel("换手率 d(t) / 该轨迹开头基线", fontsize=11.5)
axes[0].legend(handles=[
    Line2D([], [], color=TRUNK, lw=2.0, label="原轨迹（不干预）"),
    Line2D([], [], color=OK, lw=2.1, label="逃出的分支"),
    Line2D([], [], color=NO, lw=1.1, ls=(0, (3, 2)), label="没逃出的分支"),
], loc="lower left", frameon=False, fontsize=9.5)
fig.suptitle("同一条失败轨迹的四个分叉点，每格 8 条分支 — libero_10 t08 · init 0 · 分支路由实测",
             fontsize=13.5, y=1.03)
fig.tight_layout(); fig.savefig("figs/branches4.png", bbox_inches="tight", facecolor="white")
print("saved figs/branches4.png")
