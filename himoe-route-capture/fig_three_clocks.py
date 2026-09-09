"""Figure: the three clocks that decide whether routing-based pruning is possible.

A branch-and-prune method needs three things to happen in order, and the interesting
question is not whether any one of them happens but whether their windows overlap.

  t_B  branching   when do candidates drawn from one state actually land somewhere
                   different?  Measured against the fork's own reproduction floor,
                   because a spread smaller than that is not a spread.
  t_R  routing     when does a prune that uses the routing start retaining successes
                   above chance?  Plotted as z against the random-prune null so that
                   budgets are comparable.
  t_C  commitment  when can the outcome no longer be changed by future noise?  A row
                   is one fixed prefix run under eight different future streams; while
                   rows come back mixed, the outcome is still open.

Pruning is only actionable where t_B <= t_R < t_C.  If t_R lands past t_C the routing is
reading a decided outcome, which is still worth having as a monitor but is not selection.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

FONT = "assets/NotoSansSC.ttf"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e4"
ROUTE, STATE, ACTION, DENS, CRIT = "#2a78d6", "#7a7a76", "#eb6834", "#8e44ad", "#d03b3b"
OUT = pathlib.Path("figures/three-clocks")
BUDGET = 16


def dress(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=INK2, labelsize=8)


def panel_branching(ax):
    """Candidate landing spread against the fork's reproduction floor."""
    path = pathlib.Path("analysis/fork-fidelity/fidelity.json")
    if not path.exists():
        ax.text(0.5, 0.5, "no fidelity probe", ha="center", transform=ax.transAxes)
        return
    rows = json.loads(path.read_text())
    ph = [r["phase"] for r in rows]
    spread = [r["soft"]["qpos_spread"]["mean"] for r in rows]
    smax = [r["soft"]["qpos_spread"]["max"] for r in rows]
    floor = [max(r["hard"]["final_drift_max_abs"], 1e-16) for r in rows]
    ax.semilogy(ph, spread, "o-", color=ROUTE, lw=1.8, ms=5, label="候选落点离散（均值）")
    ax.semilogy(ph, smax, "^--", color=ACTION, lw=1.2, ms=5, label="候选落点离散（最大）")
    ax.semilogy(ph, floor, "s-", color=CRIT, lw=1.6, ms=5, label="分叉复现底（硬恢复）")
    ax.set_ylim(1e-17, 1e-1)
    dress(ax, "控制步", "qpos 距离", "① t_B：候选什么时候真的分开")
    ax.legend(frameon=False, fontsize=7.5, loc="center right")


def panel_routing(ax, bundles):
    """Enrichment z at a fixed budget, per control step, per bundle."""
    arms = [("route/supervised", ROUTE, "路由（监督）"),
            ("state/supervised", STATE, "世界状态（监督）"),
            ("route/density", DENS, "路由密度（RAD，无标签）")]
    for arm, colour, label in arms:
        xs, ys = [], []
        for tag in bundles:
            d = json.loads(pathlib.Path("analysis/prune-curve/%s.json" % tag).read_text())
            if arm not in d["arms"]:
                continue
            for c in d["arms"][arm]["cells"]:
                if c["budget"] == BUDGET:
                    xs.append(c["step"])
                    ys.append(c["z_exact"])
        if not xs:
            continue
        xs, ys = np.array(xs), np.array(ys)
        # one line per arm pooled over bundles: the median across bundles at each step,
        # which is what a claim about "the window" has to survive
        steps = np.unique(xs)
        med = np.array([np.median(ys[xs == s]) for s in steps])
        ax.plot(steps, med, "o-", color=colour, lw=1.8, ms=4, label=label)
    ax.axhline(0, color=INK2, lw=0.9)
    for y in (-2, 2):
        ax.axhline(y, color=GRID, lw=1.0, ls="--")
    dress(ax, "控制步", "Enrichment@%d 的 z" % BUDGET,
          "② t_R：减枝什么时候开始留住成功（4 个 bundle 的中位数）")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")


def panel_commitment(ax):
    """Mixed rows and per-row continuation success against the branch point."""
    path = pathlib.Path("runs/prefix-commitment-k6-s24-mig2g-client/summaries.json")
    if not path.exists():
        ax.text(0.5, 0.5, "no commitment grid", ha="center", transform=ax.transAxes)
        return
    cells = json.loads(path.read_text())
    by_k: dict = {}
    for c in cells:
        by_k.setdefault(c["branch_controls"], {}).setdefault(
            c["prefix_noise_seed"], []).append(bool(c["success"]))
    ks = sorted(by_k)
    for k in ks:
        rates = [np.mean(v) for v in by_k[k].values() if len(v) >= 4]
        ax.scatter([k] * len(rates), rates, s=26, color=ROUTE, alpha=0.55,
                   edgecolor="none", zorder=3)
    full = [[v for v in by_k[k].values() if len(v) >= 4] for k in ks]
    mixed = [np.mean([0 < sum(v) < len(v) for v in f]) if f else np.nan for f in full]
    ax.plot(ks, mixed, "s-", color=CRIT, lw=1.8, ms=5, label="混合行比例（结局仍可翻）")
    ax.set_ylim(-0.05, 1.05)
    n = sum(len(f) for f in full)
    dress(ax, "分叉控制步 k", "比例 / 每行成功率",
          "③ t_C：固定前 k 步，只换未来噪声（%d 行）" % n)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")


def main() -> int:
    bundles = [p.stem for p in sorted(pathlib.Path("analysis/prune-curve").glob("*.json"))]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), facecolor=SURFACE)
    panel_branching(axes[0])
    panel_routing(axes[1], bundles)
    panel_commitment(axes[2])
    fig.suptitle("路由减枝需要的三只时钟：分歧 → 可减枝 → 结局锁死",
                 color=INK, fontsize=12, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("three_clocks.%s" % ext), dpi=200, facecolor=SURFACE)
    print("wrote", OUT / "three_clocks.png", "from bundles:", ", ".join(bundles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
