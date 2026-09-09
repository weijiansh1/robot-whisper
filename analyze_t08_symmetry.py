#!/usr/bin/env python3
"""The MoE routing state along the architecture's mirror axis.  libero_10/t08.

HiMoE places its experts symmetrically (himoe.py:378-383, 18 layers):

    AS-MoE   layers 0,1        <->  16,17      3 experts, top-1, routed on the
                                               24-dim data_mask
    HB-MoE   layers 2,3,4,5    <->  12,13,14,15  32 experts, top-4 + 1 shared,
                                               routed on the 1024-dim hidden
    dense    layers 6..11

so the mirror pairs are 0-17, 1-16, 2-15, 3-14, 4-13, 5-12.  The placement is
symmetric by construction; whether the *behaviour* is symmetric is a question
about the trained weights, and it is not the same question.

Asked here, for the front block against the back block:
  1. how selective is each layer, and does the mirror pair match?
  2. do mirrored layers route to the same experts?
  3. where does each layer's variance come from?
  4. how much of the load profile is scene-specific?
  5. what do the four AS layers do?

Writes five figures to fig/.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

HUB = pathlib.Path(__file__).resolve().parent / "VLA_MUI_HUB"
FIG = pathlib.Path(__file__).resolve().parent / "fig"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
STEPS = [0, 5, 13, 20, 26, 34]
HB = [2, 3, 4, 5, 12, 13, 14, 15]        # axis-1 order of the hb_* arrays
AS = [0, 1, 16, 17]
MIRROR = [(0, 7), (1, 6), (2, 5), (3, 4)]  # (2,15) (3,14) (4,13) (5,12)


def main() -> int:
    FIG.mkdir(exist_ok=True)
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    scene = np.array([s["init_state_id"] for s in S])
    y = np.array([s["success"] for s in S], bool)
    off = np.concatenate([[0], np.cumsum([s["inference_calls"] for s in S])[:-1]])

    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    n, T = len(S), len(STEPS)
    P = np.empty((n, T, 8, 10, 11, 32), np.float32)
    E = np.empty((n, T, 8, 10, 11), np.float32)
    A = np.empty((n, T, 4, 3), np.float32)
    Aid = np.empty((n, T, 4), np.uint8)
    for i in range(n):
        r = off[i] + np.array(STEPS)
        P[i] = np.asarray(z["hb_router_probs"].oindex[r, :, :, :, :], np.float32)
        E[i] = np.asarray(z["hb_entropy"].oindex[r, :, :, :], np.float32)
        A[i] = np.asarray(z["as_probs"].oindex[r, :, :], np.float32)
        Aid[i] = z["as_expert_ids"].oindex[r, :]
        if i % 128 == 0:
            print("  read %d/%d" % (i, n), flush=True)

    load = P.mean((0, 1, 3, 4))                 # [8, 32] mean router probability
    ent_l = E.mean((0, 1, 3, 4))                # [8]
    ent_tok = E.mean((0, 1, 3))                 # [8, 11]
    UNIF = 1.0 / 32

    # ---- 1. selectivity ---------------------------------------------------
    print("\n=== 1. selectivity, front block vs its mirror ===")
    print("front            back            entropy f/b        max load f/b      "
          "  L1 from uniform f/b")
    for a, b in MIRROR:
        la, lb = load[a], load[b]
        print("layer %2d  <->  layer %2d       %.4f / %.4f    %.4f / %.4f    "
              "%.4f / %.4f"
              % (HB[a], HB[b], ent_l[a], ent_l[b], la.max(), lb.max(),
                 np.abs(la - UNIF).sum(), np.abs(lb - UNIF).sum()))
    print("  (max entropy = %.4f, uniform load = %.4f)" % (np.log(32), UNIF))
    print("  front block mean entropy %.4f   back block %.4f   gap %+.4f"
          % (ent_l[:4].mean(), ent_l[4:].mean(), ent_l[4:].mean() - ent_l[:4].mean()))
    print("  front block L1-from-uniform %.4f  back %.4f  ratio %.1fx"
          % (np.abs(load[:4] - UNIF).sum(1).mean(), np.abs(load[4:] - UNIF).sum(1).mean(),
             np.abs(load[:4] - UNIF).sum(1).mean() / np.abs(load[4:] - UNIF).sum(1).mean()))

    # ---- 2. do mirrored layers pick the same experts? ---------------------
    print("\n=== 2. expert-identity agreement between layers ===")
    C = np.corrcoef(load)
    print("correlation of the mean load profile, layer x layer:")
    print("        " + "  ".join("L%-4d" % l for l in HB))
    for i, l in enumerate(HB):
        print("  L%-4d " % l + "  ".join("%+.2f" % v for v in C[i]))
    mir = np.mean([C[a, b] for a, b in MIRROR])
    within_f = np.mean([C[i, j] for i in range(4) for j in range(4) if i < j])
    within_b = np.mean([C[i, j] for i in range(4, 8) for j in range(4, 8) if i < j])
    cross = np.mean([C[i, j] for i in range(4) for j in range(4, 8)
                     if (i, j) not in MIRROR])
    print("  mirror pairs %.3f | within front %.3f | within back %.3f | "
          "cross non-mirror %.3f" % (mir, within_f, within_b, cross))

    # top-4 experts by load, per layer
    print("\n  four heaviest experts per layer (id:load):")
    for i, l in enumerate(HB):
        o = np.argsort(-load[i])[:4]
        print("    L%-3d " % l + "  ".join("%2d:%.4f" % (e, load[i, e]) for e in o))

    # ---- 3. variance origin per layer -------------------------------------
    print("\n=== 3. where each layer's variance comes from ===")
    tot = P.var((0, 1, 3, 4)).mean(-1)
    v = {"denoise": P.mean(4).var(3).mean((0, 1, 3)),
         "suffix": P.mean(3).var(3).mean((0, 1, 3)),
         "ctrl step": P.mean((3, 4)).var(1).mean((0, 2)),
         "episode": P.mean((1, 3, 4)).var(0).mean(-1)}
    print("layer   total var   " + "  ".join("%-10s" % k for k in v))
    for i, l in enumerate(HB):
        print("  %2d   %.3e   " % (l, tot[i])
              + "  ".join("%9.1f%%" % (100 * v[k][i] / tot[i]) for k in v))

    # ---- 4. scene specificity ---------------------------------------------
    print("\n=== 4. how scene-specific is each layer's load profile? ===")
    prof = np.stack([P[scene == s].mean((0, 1, 3, 4)) for s in np.unique(scene)])
    # One-way ANOVA per layer, done at a *fixed* control step: pooling steps
    # into the within-scene term puts the control-step drift in the denominator
    # and the scene effect in the numerator, which is not the comparison asked.
    Pm = P.mean((3, 4))                                   # [ep, step, layer, expert]
    scenes = np.unique(scene)
    print("layer   " + "  ".join("F @ step%-3d" % t for t in STEPS))
    for i, l in enumerate(HB):
        row = []
        for ti in range(len(STEPS)):
            ep = Pm[:, ti, i]                             # [ep, expert]
            mu = np.stack([ep[scene == s].mean(0) for s in scenes])
            between = mu.var(0, ddof=1).mean()
            within = np.mean([ep[scene == s].var(0, ddof=1).mean() for s in scenes])
            row.append(between / within)
        print("  L%-3d  " % l + "  ".join("%10.1f " % f for f in row))
    print("  F = (spread of the 16 scene means) / (spread of the 32 draws "
          "inside a scene), same control step")

    # ---- 5. AS-MoE --------------------------------------------------------
    print("\n=== 5. AS-MoE, the 3-expert top-1 layers ===")
    for j, l in enumerate(AS):
        ids, cnt = np.unique(Aid[:, :, j], return_counts=True)
        print("  layer %2d: expert %s chosen %d/%d times, probs %s"
              % (l, ids.tolist(), cnt.max(), Aid[:, :, j].size,
                 np.array2string(A[:, :, j].mean((0, 1)), precision=6)))
    print("  distinct 4-tuples over %d sites: %d"
          % (Aid.reshape(-1, 4).shape[0], len(np.unique(Aid.reshape(-1, 4), axis=0))))

    # ================= figures =============================================
    lab = ["L%d" % l for l in HB]

    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(load, aspect="auto", cmap="magma")
    ax.set_yticks(range(8), lab)
    ax.set_xticks(range(0, 32, 2))
    ax.set_xlabel("expert id"); ax.set_ylabel("HB-MoE layer")
    ax.axhline(3.5, color="cyan", lw=2)
    ax.text(32.6, 1.5, "front\n2-5", color="cyan", va="center", fontsize=9)
    ax.text(32.6, 5.5, "back\n12-15", color="cyan", va="center", fontsize=9)
    fig.colorbar(im, label="mean router probability", pad=0.07)
    ax.set_title("Expert load, t08, 512 episodes x 6 control steps "
                 "(uniform = %.4f)" % UNIF)
    fig.tight_layout(); fig.savefig(FIG / "sym1_load.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(range(8), ent_l, "o-", color="k")
    axes[0].axhline(np.log(32), ls="--", c="r", label="max = ln32")
    for a, b in MIRROR:
        axes[0].plot([a, b], [ent_l[a], ent_l[b]], ":", c="tab:blue", lw=1)
    axes[0].set_xticks(range(8), lab); axes[0].set_ylabel("router entropy (nats)")
    axes[0].set_title("selectivity by layer\n(dotted = mirror pair)"); axes[0].legend()

    for i in range(8):
        axes[1].plot(range(11), ent_tok[i], "o-", ms=3,
                     color=plt.cm.coolwarm(i / 7), label=lab[i])
    axes[1].set_xticks(range(11), ["state"] + ["a%d" % k for k in range(1, 11)],
                       rotation=45, fontsize=7)
    axes[1].set_ylabel("router entropy"); axes[1].legend(fontsize=6, ncol=2)
    axes[1].set_title("entropy by suffix token\n(blue = front, red = back)")

    im = axes[2].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_xticks(range(8), lab, rotation=45); axes[2].set_yticks(range(8), lab)
    for a, b in MIRROR:
        axes[2].add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=False,
                                        ec="lime", lw=2))
        axes[2].add_patch(plt.Rectangle((a - .5, b - .5), 1, 1, fill=False,
                                        ec="lime", lw=2))
    fig.colorbar(im, ax=axes[2], label="corr of load profile")
    axes[2].set_title("do layers route alike?\n(green = mirror pair)")
    fig.tight_layout(); fig.savefig(FIG / "sym2_selectivity.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    bot = np.zeros(8)
    for k, c in zip(v, ["tab:orange", "tab:blue", "tab:green", "tab:red"]):
        frac = 100 * v[k] / tot
        ax.bar(range(8), frac, bottom=bot, label=k, color=c)
        bot += frac
    ax.set_xticks(range(8), lab); ax.set_ylabel("% of total router variance")
    ax.axvline(3.5, color="k", ls="--")
    ax.legend(); ax.set_title("what moves each router (front block | back block)")
    fig.tight_layout(); fig.savefig(FIG / "sym3_variance.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
    for i in range(8):
        a = axes[i // 4 if i < 4 else 1, i % 4]
        for s in np.unique(scene):
            a.plot(prof[list(np.unique(scene)).index(s), i] - UNIF, lw=.7,
                   color=plt.cm.viridis(list(np.unique(scene)).index(s) / 15))
        a.axhline(0, c="k", lw=.5); a.set_title(lab[i], fontsize=9)
    axes[1, 0].set_xlabel("expert id"); axes[0, 0].set_ylabel("load - uniform")
    axes[1, 0].set_ylabel("load - uniform")
    fig.suptitle("per-scene expert load, one line per initial state "
                 "(top row = front 2-5, bottom = back 12-15)")
    fig.tight_layout(); fig.savefig(FIG / "sym4_scene.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = A.mean((0, 1))
    x = np.arange(4)
    for e in range(3):
        ax.bar(x + (e - 1) * .26, w[:, e], .25, label="expert %d" % e)
    ax.set_xticks(x, ["L%d" % l for l in AS]); ax.axhline(1 / 3, ls="--", c="k", lw=.8)
    ax.set_ylabel("mean router probability"); ax.legend()
    ax.set_title("AS-MoE: 3 experts, top-1, routed on the 24-dim data_mask")
    fig.tight_layout(); fig.savefig(FIG / "sym5_as.png", dpi=130); plt.close(fig)

    print("\nfigures in %s" % FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
