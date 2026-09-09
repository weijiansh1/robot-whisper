"""Animate the MoE routing state over control steps.

Frame = one control step.  Three linked panels:

  A/B  the 32-expert router residual for one successful and one failed draw,
       shown as percent deviation from that expert's own average share, so the
       colour scale is interpretable without knowing that 1/32 = 0.03125.
  C    all 64 draws projected onto a common 2-D basis, coloured by final
       outcome.  Every point starts from the same observation; the fan-out is
       caused by the flow noise alone.

Episodes end at different times.  A finished episode is drawn as a hollow
marker frozen at its last state rather than deleted, so the eye does not read a
vanishing point as movement.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from within64_lib import HB_LAYERS, N_EXPERTS, Run, action_token_probs, load_run

FONT = pathlib.Path(__file__).parent / "assets" / "NotoSansSC.ttf"
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

OK_C = "#1b7f47"
BAD_C = "#b8342a"


def build(run: Run, layer_idx: int) -> dict:
    feats, coords, lengths = [], [], []
    for e in run.episodes:
        f = action_token_probs(e).reshape(e.n_control, -1)
        feats.append(f)
        lengths.append(e.n_control)
    x = np.concatenate(feats)
    mu = x.mean(0)
    _u, s, vt = np.linalg.svd(x - mu, full_matrices=False)
    basis = vt[:2]
    var = (s**2) / (s**2).sum()
    for f in feats:
        coords.append((f - mu) @ basis.T)

    # each expert's own average share at this layer, pooled over every episode and
    # control step; the bars below are deviations from it
    per_expert_mean = np.concatenate(
        [action_token_probs(e)[:, layer_idx].mean(axis=1) for e in run.episodes]
    ).mean(0)
    return {
        "coords": coords,
        "lengths": np.array(lengths),
        "explained": var[:2],
        "per_expert_mean": per_expert_mean,
    }


def residual_pct(e, layer_idx: int, per_expert_mean: np.ndarray) -> np.ndarray:
    p = action_token_probs(e)[:, layer_idx].mean(axis=1)  # [T, 32]
    return 100.0 * (p - per_expert_mean) / per_expert_mean


def make(run: Run, out: pathlib.Path, layer_idx: int = 0, fps: int = 4) -> None:
    b = build(run, layer_idx)
    ok = next(e for e in run.episodes if e.success)
    bad = next(e for e in run.episodes if not e.success)
    r_ok = residual_pct(ok, layer_idx, b["per_expert_mean"])
    r_bad = residual_pct(bad, layer_idx, b["per_expert_mean"])
    lim = float(np.percentile(np.abs(np.concatenate([r_ok, r_bad])), 99.5))

    coords = b["coords"]
    lengths = b["lengths"]
    success = np.array([e.success for e in run.episodes])
    t_max = int(lengths.max())
    all_xy = np.concatenate(coords)
    pad = 0.06 * (all_xy.max(0) - all_xy.min(0))
    lo, hi = all_xy.min(0) - pad, all_xy.max(0) + pad

    fig = plt.figure(figsize=(13.2, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.62, wspace=0.22,
                          bottom=0.16, top=0.86, left=0.06, right=0.98)
    ax_ok = fig.add_subplot(gs[0, 0])
    ax_bad = fig.add_subplot(gs[1, 0])
    ax_pca = fig.add_subplot(gs[:, 1])

    bars_ok = ax_ok.bar(np.arange(N_EXPERTS), np.zeros(N_EXPERTS), color=OK_C)
    bars_bad = ax_bad.bar(np.arange(N_EXPERTS), np.zeros(N_EXPERTS), color=BAD_C)
    for ax, e, tag in ((ax_ok, ok, "成功"), (ax_bad, bad, "失败")):
        ax.set_ylim(-lim, lim)
        ax.set_xlim(-1, N_EXPERTS)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("偏离自身均值 (%)", fontsize=7.5, labelpad=2)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{tag}  seed={e.flow_noise_seed}  共 {e.n_control} 个控制步",
                     fontsize=9, color=OK_C if e.success else BAD_C)
    ax_bad.set_xlabel("专家编号", fontsize=8)

    n_ep = len(coords)
    ax_pca.set_xlim(lo[0], hi[0])
    ax_pca.set_ylim(lo[1], hi[1])
    ax_pca.set_xlabel(f"PC1 ({100 * b['explained'][0]:.1f}% 方差)", fontsize=8)
    ax_pca.set_ylabel(f"PC2 ({100 * b['explained'][1]:.1f}% 方差)", fontsize=8)
    ax_pca.tick_params(labelsize=7)
    ax_pca.set_title(f"{n_ep} 次 flow 噪声抽样的路由状态\n（投影基底在全部控制步上拟合）",
                     fontsize=9)
    live = ax_pca.scatter([], [], s=34, edgecolors="none")
    done = ax_pca.scatter([], [], s=26, facecolors="none", linewidths=1.0)
    handles = [
        ax_pca.scatter([], [], s=34, color=OK_C, label="最终成功"),
        ax_pca.scatter([], [], s=34, color=BAD_C, label="最终失败"),
        ax_pca.scatter([], [], s=26, facecolors="none", edgecolors="#777",
                       label="已结束，冻结在最后状态"),
    ]
    # a figure-level legend: an in-axes one would sit on top of the points, which
    # move to a different corner at every control step
    fig.legend(handles=handles, fontsize=8.5, frameon=False, loc="lower center",
               ncol=3, columnspacing=2.2, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.005))

    title = fig.suptitle("", fontsize=11)

    def draw(t: int):
        i_ok = min(t, len(r_ok) - 1)
        i_bad = min(t, len(r_bad) - 1)
        for bar, v in zip(bars_ok, r_ok[i_ok]):
            bar.set_height(v)
        for bar, v in zip(bars_bad, r_bad[i_bad]):
            bar.set_height(v)
        # a finished showcase episode holds its last frame; say so rather than
        # letting a frozen bar chart read as "nothing is changing"
        ax_ok.patch.set_facecolor("#f2f2f2" if t >= len(r_ok) else "white")
        ax_bad.patch.set_facecolor("#f2f2f2" if t >= len(r_bad) else "white")

        live_xy, live_c, done_xy, done_c = [], [], [], []
        for k, c in enumerate(coords):
            if t < lengths[k]:
                live_xy.append(c[t])
                live_c.append(OK_C if success[k] else BAD_C)
            else:
                done_xy.append(c[-1])
                done_c.append(OK_C if success[k] else BAD_C)
        live.set_offsets(np.array(live_xy).reshape(-1, 2))
        live.set_color(live_c if live_c else "none")
        done.set_offsets(np.array(done_xy).reshape(-1, 2))
        done.set_edgecolors(done_c if done_c else "none")

        n_live = len(live_xy)
        title.set_text(
            f"HB 层 {HB_LAYERS[layer_idx]}｜控制步 {t}／{t_max - 1}｜"
            f"仍在运行 {n_live}/{len(coords)} 集"
            + ("｜观测在此刻逐字节相同" if t == 0 else "")
        )
        return [*bars_ok, *bars_bad, live, done, title]

    anim = animation.FuncAnimation(fig, draw, frames=t_max, interval=1000 // fps,
                                   blit=False)
    mp4 = out / "routing_evolution.mp4"
    anim.save(str(mp4), writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    print("wrote", mp4)
    gif = out / "routing_evolution.gif"
    anim.save(str(gif), writer=animation.PillowWriter(fps=fps))
    print("wrote", gif)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-dir", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer-idx", type=int, default=0)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    run = load_run(args.server_dir, args.client_dir, allow_partial=args.allow_partial)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    make(run, out, args.layer_idx, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
