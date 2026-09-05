"""分支自己的路由动态：用触发报警的同一个检测器，逐字同算。"""
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
DENOISE, DEEP, W0 = 9, slice(4, 8), 8

def jac(prev, cur):                       # 与 triggered_fork_collect.jaccard_distance 相同
    a, b = prev[DENOISE, DEEP], cur[DENOISE, DEEP]
    sc = []
    for l in range(a.shape[0]):
        for s in range(a.shape[1]):
            sa, sb = set(a[l, s].tolist()), set(b[l, s].tolist())
            sc.append(1.0 - len(sa & sb) / max(len(sa | sb), 1))
    return float(np.mean(sc))

R = "runs_routed/i0_s3"
t = json.load(open(f"{R}/trunk.json")); m = json.load(open(f"{R}/manifest.json"))
dist = np.array(t["distances"], float)
base = dist[:W0].mean()
ratios = np.array([np.nan if v is None else v for v in t["ratios"]], float)
ARMS = [("triggered", m["fork_query"]), ("delayed_+4", m["fork_query"]+4),
        ("delayed_+8", m["fork_query"]+8)]

fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.6, 8.6), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.15]})
# 上：trunk 的 r(t)（报警就是基于它）
ax.plot(np.arange(len(ratios)), ratios, color=TRUNK, lw=2.4, zorder=5)
ax.axhline(.95, color=BAD, ls="--", lw=1.2); ax.axhline(1.0, color="#9AA0A6", ls=":", lw=1.0)
ax.text(1, .957, "θ = 0.95", color=BAD, fontsize=10)
for arm, q in ARMS:
    ax.plot([q], [ratios[q]], "o", color=AL, ms=10, mec="white", mew=1.6, zorder=8)
ax.set_ylabel("trunk 的 r(t)\n窗口 W=4", fontsize=11.5, linespacing=1.4)
ax.set_title("同一条轨迹：分叉出来的分支，路由动态是什么样的\n"
             "libero_10 t08 · paper-right · init 0 · 检测器与报警器逐字相同", fontsize=13, pad=14)

# 下：瞬时 d / baseline —— trunk 与每条分支同尺度
bx.plot(np.arange(1, len(dist)+1), dist/base, color=TRUNK, lw=2.2, zorder=6)
bx.axhline(1.0, color="#9AA0A6", ls=":", lw=1.0)
n_ok = n_all = 0
for arm, q in ARMS:
    for f in sorted(glob.glob(f"{R}/{arm}/candidate_*.npz")):
        z = np.load(f); s = json.load(open(f.replace(".npz", ".json")))
        if "routing_expert_ids" not in z.files: continue
        ids = z["routing_expert_ids"]
        d = np.array([jac(ids[i-1], ids[i]) for i in range(1, len(ids))], float)
        if not len(d): continue
        n_all += 1; n_ok += s["success"]
        bx.plot(np.arange(q+2, q+2+len(d)), d/base,
                color=OK if s["success"] else GREY, lw=1.9 if s["success"] else 1.0,
                alpha=.95 if s["success"] else .55, zorder=5 if s["success"] else 3)
    bx.axvline(q, color=AL, lw=1.0, alpha=.45, zorder=2)
    bx.text(q, bx.get_ylim()[1]*.97, " q%d" % q, color=AL, fontsize=9.5, va="top")
bx.set_xlabel("控制步 (query)", fontsize=12)
bx.set_ylabel("瞬时换手率 d(t) / 自身开头基线\n（不做窗口平均，故 3 步的分支也画得出）",
              fontsize=11.5, linespacing=1.4)
bx.legend(handles=[
    Line2D([], [], color=TRUNK, lw=2.2, label="原轨迹 trunk"),
    Line2D([], [], color=OK, lw=1.9, label="逃出的分支"),
    Line2D([], [], color=GREY, lw=1.0, label="没逃出的分支"),
], loc="upper left", frameon=False, fontsize=10)
for A in (ax, bx):
    for s_ in ("top", "right"): A.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig("figs/branch_routes.png", bbox_inches="tight", facecolor="white")
print("saved figs/branch_routes.png   分支 %d 条，其中逃出 %d" % (n_all, n_ok))
