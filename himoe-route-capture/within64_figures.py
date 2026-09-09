"""Static figures for the 64-draw single-scene capture.

Everything here is drawn from one init state with 64 flow-noise draws, so the
scene is constant by construction and no per-scene centring is needed.

Two display conventions, used consistently:

  * "raw" panels plot the router softmax against the uniform line 1/32.  They
    look flat because the router *is* flat -- that is the finding, not a
    rendering problem.
  * "residual" panels plot p - mean_over_all_sites(p) per expert, which removes
    each expert's constant preference and leaves the input-driven part.  The
    colour scale is stated on every such panel so the amplification is explicit.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from within64_lib import (
    HB_LAYERS,
    N_EXPERTS,
    Run,
    action_token_probs,
    load_run,
    onehot_sets,
    state_token_probs,
)

FONT = pathlib.Path(__file__).parent / "assets" / "NotoSansSC.ttf"
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

OK_C = "#1b7f47"
BAD_C = "#b8342a"
UNIFORM = 1.0 / N_EXPERTS


def _stack_padded(run: Run, fn) -> tuple[np.ndarray, np.ndarray]:
    """[E, Tmax, ...] with a [E, Tmax] validity mask."""
    arrs = [fn(e) for e in run.episodes]
    t_max = max(a.shape[0] for a in arrs)
    out = np.full((len(arrs), t_max) + arrs[0].shape[1:], np.nan, dtype=np.float32)
    mask = np.zeros((len(arrs), t_max), dtype=bool)
    for i, a in enumerate(arrs):
        out[i, : a.shape[0]] = a
        mask[i, : a.shape[0]] = True
    return out, mask


def fig_flat_router(run: Run, out: pathlib.Path) -> None:
    """The router distribution itself, raw, at control step 0 across all 64 draws."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), width_ratios=[1.35, 1])

    p0 = np.stack([action_token_probs(e)[0] for e in run.episodes])  # [E, 8, 10, 32]
    per_expert = p0.mean(axis=(0, 2))  # [8, 32]

    ax = axes[0]
    for li, layer in enumerate(HB_LAYERS):
        ax.plot(np.arange(N_EXPERTS), per_expert[li], lw=1.0, alpha=0.85,
                label=f"L{layer}")
    ax.axhline(UNIFORM, color="k", ls="--", lw=1.2)
    ax.annotate("均匀 1/32", xy=(31, UNIFORM), xytext=(24.5, UNIFORM * 1.14),
                fontsize=8, color="k")
    ax.set_xlabel("专家编号")
    ax.set_ylabel("router 概率")
    ax.set_title("控制步 0，64 次噪声平均：每个专家的 router 概率", fontsize=10)
    ax.set_ylim(0, max(UNIFORM * 2.0, per_expert.max() * 1.12))
    ax.legend(fontsize=7, ncol=4, loc="upper left", frameon=False)

    ax = axes[1]
    ent = np.stack([e.entropy.mean(axis=(0, 2, 3)) for e in run.episodes])  # [E, 8]
    ln32 = np.log(N_EXPERTS)
    ax.boxplot([ent[:, li] / ln32 for li in range(len(HB_LAYERS))],
               tick_labels=[str(l) for l in HB_LAYERS], widths=0.6,
               medianprops=dict(color="#333"), flierprops=dict(ms=2))
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xlabel("HB 层")
    ax.set_ylabel("熵 / ln32")
    ax.set_ylim(0.975, 1.002)
    ax.set_title("router 熵占满值的比例（每集一个点）", fontsize=10)

    fig.suptitle(
        "路由几乎是均匀的：有效专家数 30+ / 32。后续所有'变化'都发生在这个近乎平坦的分布上",
        fontsize=10.5, y=1.04,
    )
    fig.savefig(out / "f1_router_is_flat.png")
    plt.close(fig)


def fig_state_over_time(run: Run, out: pathlib.Path, layer_idx: int = 0) -> None:
    """Expert x control-step residual heatmaps for one success and one failure."""
    ok = next(e for e in run.episodes if e.success)
    bad = next(e for e in run.episodes if not e.success)

    # centre within the layer being plotted.  Pooling the mean across all 8 layers
    # would leave each layer's own expert preferences in the residual as constant
    # horizontal bands, which read as structure but never change with time.
    base = np.concatenate(
        [action_token_probs(e)[:, layer_idx].reshape(-1, N_EXPERTS) for e in run.episodes]
    ).mean(0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=False)
    vmax = 0.0
    panels = []
    for e in (ok, bad):
        m = action_token_probs(e)[:, layer_idx].mean(axis=1) - base  # [T, 32]
        panels.append(m)
        vmax = max(vmax, np.abs(m).max())

    for ax, e, m in zip(axes, (ok, bad), panels):
        im = ax.imshow(m.T, aspect="auto", origin="lower", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                       extent=(-0.5, m.shape[0] - 0.5, -0.5, N_EXPERTS - 0.5))
        tag = "成功" if e.success else "失败"
        colour = OK_C if e.success else BAD_C
        ax.set_ylabel("专家编号")
        ax.set_title(
            f"{tag}  seed={e.flow_noise_seed}  {e.action_steps} 步  "
            f"（HB 层 {HB_LAYERS[layer_idx]}，动作 token 与去噪步平均）",
            fontsize=9.5, color=colour,
        )
        fig.colorbar(im, ax=ax, pad=0.01, label="p − 全局均值")
    axes[-1].set_xlabel("控制步")
    fig.suptitle(
        f"同一初始状态、同一场景，只有 flow 噪声不同：路由残差随控制步的变化"
        f"（色标 ±{vmax:.4f}，均匀值 1/32 = {UNIFORM:.4f}）",
        fontsize=10.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "f2_state_over_time.png")
    plt.close(fig)


def fig_ensemble_spread(run: Run, out: pathlib.Path) -> None:
    """How the 64 draws fan apart in routing space, starting from an identical obs.

    Successful episodes stop as soon as the drawer opens, so past the shortest
    episode the surviving population is increasingly failure-only.  Any curve
    plotted there is comparing different sets of episodes at each step, not
    tracking one set through time.  The constant-population window is drawn
    solid; everything after it is dashed and shaded, and the separation panel is
    cut off entirely once either group falls below MIN_GROUP episodes.
    """
    MIN_GROUP = 5
    feats, mask = _stack_padded(
        run, lambda e: action_token_probs(e).reshape(e.n_control, -1)
    )
    t_max = feats.shape[1]
    y = np.array([e.success for e in run.episodes])
    t_all = int(mask.all(axis=0).sum())  # last step where every episode is alive

    spread = np.full(t_max, np.nan)
    sep = np.full(t_max, np.nan)
    for t in range(t_max):
        rows = feats[:, t][mask[:, t]]
        if len(rows) < 4:
            continue
        centred = rows - rows.mean(0)
        within = float(np.linalg.norm(centred, axis=1).mean())
        spread[t] = within
        lab = y[mask[:, t]]
        if lab.sum() < MIN_GROUP or (~lab).sum() < MIN_GROUP:
            continue
        gap = rows[lab].mean(0) - rows[~lab].mean(0)
        sep[t] = float(np.linalg.norm(gap) / (within + 1e-12))

    def _split_plot(ax, values, colour, label=None):
        x = np.arange(len(values))
        ax.plot(x[:t_all], values[:t_all], color=colour, lw=1.8, label=label)
        ax.plot(x[t_all - 1:], values[t_all - 1:], color=colour, lw=1.4, ls="--",
                alpha=0.75)
        ax.axvspan(t_all - 0.5, len(values) - 0.5, color="#bbbbbb", alpha=0.22, lw=0)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))

    ax = axes[0]
    _split_plot(ax, spread, "#2b5d8a")
    ax.set_xlabel("控制步")
    ax.set_ylabel("到当步集合均值的平均距离")
    ax.set_title("64 条轨迹在路由空间中的散开程度", fontsize=10)
    ax.annotate("步 0：观测逐字节相同，\n差异只来自 flow 噪声",
                xy=(0, spread[0]), xytext=(t_max * 0.13, spread[0] + 0.35 * np.nanmax(spread)),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_ylim(0, np.nanmax(spread) * 1.35)

    ax = axes[1]
    n_alive = mask.sum(0)
    ax.plot(n_alive, color="#555", lw=1.6, label="仍在运行")
    ax.plot((mask & y[:, None]).sum(0), color=OK_C, lw=1.6, label="其中最终成功")
    ax.plot((mask & ~y[:, None]).sum(0), color=BAD_C, lw=1.6, label="其中最终失败")
    ax.axvspan(t_all - 0.5, t_max - 0.5, color="#bbbbbb", alpha=0.22, lw=0)
    ax.set_xlabel("控制步")
    ax.set_ylabel("集数")
    ax.set_title("成功的集提前结束——长度几乎就是标签", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    valid = ~np.isnan(sep)
    ax.plot(np.arange(t_max)[valid], sep[valid], color="#8a2b6d", lw=1.8)
    ax.axvspan(t_all - 0.5, t_max - 0.5, color="#bbbbbb", alpha=0.22, lw=0)
    ax.set_xlim(-0.5, t_max - 0.5)
    ax.set_xlabel("控制步")
    ax.set_ylabel("类间距 / 类内散度")
    ax.set_title(f"成败两组路由中心的分离度\n（任一组不足 {MIN_GROUP} 集即不画）", fontsize=10)
    ax.axhline(0, color="k", lw=0.6)

    fig.suptitle(
        f"同一场景、64 次不同 flow 噪声。灰底 = 第 {t_all} 步之后陆续有集结束，"
        f"该区间的曲线在比较不同的集合，不能当作时间趋势读",
        fontsize=10.5, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out / "f3_ensemble_spread.png")
    plt.close(fig)


def fig_piano_roll(run: Run, out: pathlib.Path, layer_idx: int = 0,
                   denoise: int = 5, token: int = 5) -> None:
    """Which 4 experts are selected, step by step, for one success and one failure."""
    ok = next(e for e in run.episodes if e.success)
    bad = next(e for e in run.episodes if not e.success)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.0))
    for ax, e in zip(axes, (ok, bad)):
        oh = onehot_sets(e)[:, layer_idx, denoise, token]  # [T, 32]
        ys, xs = np.nonzero(oh.T)
        colour = OK_C if e.success else BAD_C
        ax.scatter(xs, ys, s=22, marker="s", color=colour, alpha=0.9)
        ax.set_ylim(-1, N_EXPERTS)
        ax.set_ylabel("专家编号")
        tag = "成功" if e.success else "失败"
        ax.set_title(f"{tag}  seed={e.flow_noise_seed}", fontsize=9.5, color=colour)
        ax.grid(alpha=0.15, lw=0.5)
    axes[-1].set_xlabel("控制步")
    fig.suptitle(
        f"top-4 选择的翻搅（HB 层 {HB_LAYERS[layer_idx]}，去噪步 {denoise}，token {token}）："
        f"每列 4 个方块，几乎每步都换一批",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out / "f4_piano_roll.png")
    plt.close(fig)


def fig_denoise_axis(run: Run, out: pathlib.Path) -> None:
    """Routing change along the flow integration axis, which the report never plotted."""
    p = np.stack([action_token_probs(e)[0] for e in run.episodes])  # [E, 8, 10, 32]
    base = p.mean(axis=(0, 2), keepdims=True)
    resid = np.abs(p - base).mean(axis=(0, 3))  # [8, 10]

    onehots = [onehot_sets(e) for e in run.episodes]
    jac = np.zeros((len(HB_LAYERS), 9))
    for li in range(len(HB_LAYERS)):
        vals = []
        for oh in onehots:
            a = oh[0, li, :-1]
            b = oh[0, li, 1:]
            inter = (a & b).sum(-1).astype(np.float64)
            vals.append((inter / (8 - inter)).mean(axis=-1))
        jac[li] = np.mean(vals, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    for li, layer in enumerate(HB_LAYERS):
        ax.plot(np.arange(10), resid[li], marker="o", ms=3, lw=1.3, label=f"L{layer}")
    ax.set_xlabel("去噪步 (flow ODE 积分, t=0 → 1)")
    ax.set_ylabel("|p − 该层均值| 平均")
    ax.set_title("路由随去噪步的变化幅度", fontsize=10)
    ax.legend(fontsize=7, ncol=4, frameon=False)

    ax = axes[1]
    for li, layer in enumerate(HB_LAYERS):
        ax.plot(np.arange(9) + 0.5, jac[li], marker="o", ms=3, lw=1.3, label=f"L{layer}")
    ax.axhline(0.0667, color="k", ls="--", lw=1.0)
    ax.annotate("随机 4/32 基线", xy=(4, 0.0667), xytext=(3.4, 0.10), fontsize=8)
    ax.set_xlabel("相邻去噪步之间")
    ax.set_ylabel("top-4 集合 Jaccard")
    ax.set_title("同一次推理内部，选择也在换", fontsize=10)

    fig.suptitle(
        "被忽略的第三条轴：同一个控制步内，10 个去噪步之间路由本身就在变（控制步 0，64 集平均）",
        fontsize=10.5, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out / "f5_denoise_axis.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = load_run(args.server_dir, args.client_dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("episodes=%d success=%d" % (len(run.episodes), run.n_success))

    fig_flat_router(run, out)
    fig_state_over_time(run, out)
    fig_ensemble_spread(run, out)
    fig_piano_roll(run, out)
    fig_denoise_axis(run, out)
    print("wrote figures to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
