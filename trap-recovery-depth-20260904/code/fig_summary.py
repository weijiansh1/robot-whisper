"""汇总图：trunk 的 r(t)（绝对 + 对齐两种坐标）与各臂救回率。"""
import json, glob, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from math import comb, sqrt
F = "/home/jovyan/work/himoe-vla/himoe-route-capture/assets/NotoSansSC.ttf"
fm.fontManager.addfont(F)
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=F).get_name()]
plt.rcParams["axes.unicode_minus"] = False
OK, BAD, TRUNK, AL, BLUE = "#2A9D8F", "#C1443B", "#5F6368", "#E8710A", "#1A73E8"
CAP = 52

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0, c-h), min(1, c+h)

def fisher(a, b, c, d):
    n = a+b+c+d; r1 = a+b; c1 = a+c
    f = lambda x: comb(r1, x)*comb(n-r1, c1-x)/comb(n, c1)
    lo, hi = max(0, c1-(n-r1)), min(r1, c1); p0 = f(a)
    return sum(f(x) for x in range(lo, hi+1) if f(x) <= p0+1e-12)

runs = []
for m in sorted(glob.glob(sys.argv[1] if len(sys.argv)>1 else "runs/i*_s*/manifest.json")):
    d = json.load(open(m))
    if d.get("status") == "complete":
        runs.append((m.rsplit("/", 2)[1], d, json.load(open(m.replace("manifest", "trunk")))))
if not runs: sys.exit("没有可用 trunk")

OUTPNG = sys.argv[2] if len(sys.argv)>2 else "figs/summary.png"
ARMS = [("control", "报警前\n(对照)"), ("triggered", "报警点"), ("delayed_+4", "报警后 +4")]
agg = {a: [0, 0] for a, _ in ARMS}
for _, d, _t in runs:
    for a, key in (("control", "control"), ("triggered", "triggered")):
        agg[a][0] += d[key]["successes"]; agg[a][1] += d[key]["n"]
    for k, v in d.get("delayed", {}).items():
        if "delayed_"+k in agg:
            agg["delayed_"+k][0] += v["successes"]; agg["delayed_"+k][1] += v["n"]

fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), dpi=150,
                         gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
ax, bx, cx = axes

def draw(axis, aligned):
    for name, d, t in runs:
        r = np.array([np.nan if v is None else v for v in t["ratios"]], float)
        a = d["fork_query"]
        x = np.arange(len(r)) - (a if aligned else 0)
        axis.plot(x, r, color=TRUNK, lw=1.1, alpha=.5, zorder=3)
        axis.plot(x[-1], r[-1], "x", color=TRUNK, ms=7, mew=1.8, zorder=4)
        axis.plot(a - (a if aligned else 0), r[a], "o", color=AL, ms=7,
                  mec="white", mew=1.2, zorder=6)
    axis.axhline(.95, color=BAD, ls="--", lw=1.2, zorder=2)
    axis.axhline(1.0, color="#9AA0A6", ls=":", lw=1.0, zorder=2)

draw(ax, False)
ax.set_title("(a) 绝对坐标：报警散在 q%d–q%d" %
             (min(d["fork_query"] for _, d, _ in runs),
              max(d["fork_query"] for _, d, _ in runs)), fontsize=11.5)
ax.set_xlabel("控制步 (query)", fontsize=11.5); ax.set_ylabel("检测器 r(t)", fontsize=11.5)

draw(bx, True)
bx.axvline(0, color=AL, lw=1.6, zorder=5); bx.axvline(4, color=BLUE, ls="--", lw=1.4, zorder=5)
tails = sorted(CAP - d["fork_query"] for _, d, _ in runs)
bx.axvspan(tails[0], max(tails), color="#FDECEA", zorder=0)
bx.text((tails[0]+max(tails))/2, bx.get_ylim()[0]+.02,
        "此后 trunk 逐条到 cap\n条数由 %d 递减到 1" % len(runs),
        ha="center", fontsize=9, color=BAD, linespacing=1.3)
bx.set_title("(b) 对齐报警点（用于看报警前后的共同形状）", fontsize=11.5)
bx.set_xlabel("相对报警点的控制步", fontsize=11.5)
for a_, lab, col in [(0, "报警", AL), (4, "+4", BLUE)]:
    bx.text(a_, bx.get_ylim()[1], " "+lab, color=col, fontsize=10.5, va="top")

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color=TRUNK, lw=1.1, alpha=.6, label="原轨迹 trunk（%d 条，按设计全部失败）" % len(runs)),
    Line2D([], [], color=AL, marker="o", ls="", ms=7, label="报警点（该处及其后 +4 处各分叉 K 条）"),
    Line2D([], [], color=TRUNK, marker="x", ls="", ms=7, label="trunk 到 cap，未成功"),
    Line2D([], [], color=BAD, ls="--", lw=1.2, label="θ = 0.95"),
], loc="lower left", frameon=False, fontsize=9)

for i, (a_, lab) in enumerate(ARMS):
    k, n = agg[a_]; p = k/n if n else 0; lo, hi = wilson(k, n)
    col = OK if p > 0 else BAD
    cx.bar(i, p, .55, color=col, alpha=.85)
    cx.plot([i, i], [lo, hi], color="#3C4043", lw=2)
    cx.text(i, hi+.03, "%d/%d" % (k, n), ha="center", fontsize=13, weight="bold", color=col)
cx.set_xticks(range(len(ARMS))); cx.set_xticklabels([l for _, l in ARMS], fontsize=11, linespacing=1.3)
cx.set_ylabel("等预算 16 步内逃出的比例", fontsize=11.5); cx.set_ylim(0, 1.08)
kc, nc = agg["control"]; kt, nt = agg["triggered"]; kd, nd = agg["delayed_+4"]
cx.set_title("(c) 分支结局（竖线 = Wilson 95%%）\n报警点vs对照 p=%.3f · 报警点vs+4 p=%.3f"
             % (fisher(kc, nc-kc, kt, nt-kt), fisher(kt, nt-kt, kd, nd-kd)), fontsize=11)
for A in axes:
    for s in ("top", "right"): A.spines[s].set_visible(False)
fig.suptitle("报警之后多久干预还救得回来 — libero_10 t08 · paper-right · %d 条失败 trunk" % len(runs),
             fontsize=13.5, y=1.02)
fig.text(.5, -.035, "(a)(b) 画的都是 trunk 自己的 r(t)——它们按设计全部失败；成功/失败的区分在分支上，"
                    "分支的 r(t) 待有路由且有成功的批次跑出后补入 (c) 左侧。",
         ha="center", fontsize=9.5, color="#80868B")
fig.tight_layout(); fig.savefig(OUTPNG, bbox_inches="tight", facecolor="white")
print("saved  trunk 数 =", len(runs))
for a_, _ in ARMS: print("  %-12s %d/%d" % (a_, *agg[a_]))
