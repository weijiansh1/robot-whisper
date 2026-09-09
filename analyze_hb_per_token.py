#!/usr/bin/env python3
"""HB routing token by token: all 11 suffix positions, not state-vs-the-rest.

`embed_suffix` (moevla.py:276-313) appends the state embedding first and the ten
action-chunk embeddings after it, so suffix index 0 is the state token and 1..10
are the chunk steps t..t+9.  Everything so far has lumped 1..10 together, which
is a guess that they behave alike.  They have near-identical entropy (3.453-3.455
against the state token's 3.207), but equal entropy does not mean the same
experts -- two tokens can each concentrate on a different set and look identical
in every scalar summary.

So the question that scalar summaries cannot answer: does the router *partition*
the suffix, giving each position its own experts, or do all eleven draw from one
pool with only the state token standing out?  The token axis carries 68-94% of
the router's variance, and this decides what that variance is made of.

  1. per (layer, token): effective number of experts, P(the most-used expert is
     in the top-4), and which expert that is
  2. token x token: cosine between the mean load profiles, after removing the
     uniform 1/32 -- a partition shows up as an off-diagonal near zero, a shared
     pool as an off-diagonal near one
  3. any structure along the chunk index a1..a10, which is future time
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

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
FIG = HERE / "fig"
RUN_ID = "right-16x32"
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
N_EXPERT, TOP_K, CHUNK = 32, 4, 2048
TOK = ["state"] + ["a%d" % k for k in range(1, 11)]


def eff_n(c, axis=-1):
    p = c / np.maximum(c.sum(axis, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log(np.where(p > 0, p, 1))).sum(axis)
    return np.exp(h)


def gather(run: pathlib.Path):
    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    n = z["hb_expert_ids"].shape[0]
    cnt = np.zeros((8, 11, N_EXPERT), np.int64)
    psum = np.zeros((8, 11, N_EXPERT), np.float64)
    psq = np.zeros((8, 11, N_EXPERT), np.float64)
    esum = np.zeros((8, 11), np.float64)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        ids = np.asarray(z["hb_expert_ids"][a:b]).astype(np.int64)
        for L in range(8):
            for s in range(11):
                cnt[L, s] += np.bincount(ids[:, L, :, s].ravel(), minlength=N_EXPERT)
        pr = np.asarray(z["hb_router_probs"][a:b], np.float64)
        psum += pr.sum((0, 2))
        psq += (pr ** 2).sum((0, 2))
        esum += np.asarray(z["hb_entropy"][a:b], np.float64).sum((0, 2))
    m = psum / (n * 10)
    sd = np.sqrt(np.maximum(psq / (n * 10) - m ** 2, 0.0))
    return n, cnt, m, esum / (n * 10), sd


def main() -> int:
    FIG.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE / "himoe-route-capture"))
    import corpus_layout as cl

    targets = [("libero_10", "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"),
               ("libero_goal", "open_the_middle_drawer_of_the_cabinet")]
    for suite, task in targets:
        run = HUB / "cache" / cl.HUB_MODEL / cl.HUB_DIR[suite] / task / RUN_ID
        if not (run / "server/routes.zarr").exists():
            continue
        n, cnt, P, E, SD = gather(run)
        n_tok = n * 10
        print("\n=== %s / %s  (%d control steps) ===" % (suite, task[:44], n))

        eff = eff_n(cnt)                                # [8,11]
        pmax = TOP_K * cnt.max(-1) / cnt.sum(-1)        # P(top-1 expert in top-4)
        who = cnt.argmax(-1)

        print("  effective number of experts, per layer x token (uniform 32.0):")
        print("        " + " ".join("%6s" % t for t in TOK))
        for i, L in enumerate(HB_LAYER):
            print("  L%-4d " % L + " ".join("%6.1f" % v for v in eff[i]))
        print("\n  P(the most-used expert is in the top-4)  (uniform 0.125):")
        print("        " + " ".join("%6s" % t for t in TOK))
        for i, L in enumerate(HB_LAYER):
            print("  L%-4d " % L + " ".join("%6.3f" % v for v in pmax[i]))
        print("\n  which expert that is:")
        print("        " + " ".join("%6s" % t for t in TOK))
        for i, L in enumerate(HB_LAYER):
            print("  L%-4d " % L + " ".join("%6d" % v for v in who[i]))
        print("\n  router entropy (max ln32 = %.4f):" % np.log(32))
        print("        " + " ".join("%6s" % t for t in TOK))
        for i, L in enumerate(HB_LAYER):
            print("  L%-4d " % L + " ".join("%6.3f" % v for v in E[i]))

        # ---- big-but-variable vs small-but-steady --------------------------
        # a10 reaches a higher selection rate than the state token with a tenth
        # of its L1 distance from uniform, so mean magnitude and per-step
        # reliability are not the same axis.  For the expert with the highest
        # mean probability at each (layer, token), put the two side by side.
        star = P.argmax(-1)
        li, si = np.meshgrid(np.arange(8), np.arange(11), indexing="ij")
        mu, sg = P[li, si, star], SD[li, si, star]
        print("\n  the single highest-mean expert at each (layer, token):")
        print("        " + " ".join("%13s" % t for t in TOK))
        for i, L in enumerate(HB_LAYER):
            print("  L%-4d " % L + " ".join(
                "%5.3f+-%5.3f" % (mu[i, j], sg[i, j]) for j in range(11)))
        print("  its coefficient of variation sd/mean (low = a steady bias):")
        print("        " + " ".join("%6s" % t for t in TOK))
        for i, L in enumerate(HB_LAYER):
            print("  L%-4d " % L + " ".join("%6.2f" % (sg[i, j] / mu[i, j])
                                            for j in range(11)))

        # ---- token x token profile similarity -----------------------------
        D = P - 1.0 / N_EXPERT                          # [8,11,32] deviation
        # Same metric as the front/back figure, but per token rather than with
        # a1..a10 averaged first.  Averaging them cancels their structure if they
        # point in different directions, which is exactly what the cosine below
        # shows they do -- so the earlier "action tokens are flat" number was an
        # artefact of the average, not a property of any single token.
        l1 = np.abs(D).sum(-1)                          # [8,11]
        print("\n  L1 distance of the load profile from uniform, per token:")
        print("        " + " ".join("%6s" % t for t in TOK) + "   mean(a1..a10)"
              "  |mean of profiles|")
        for i, L in enumerate(HB_LAYER):
            avg = np.abs(D[i, 1:].mean(0)).sum()
            print("  L%-4d " % L + " ".join("%6.3f" % v for v in l1[i])
                  + "      %6.3f         %6.3f" % (l1[i, 1:].mean(), avg))
        Dn = D / np.linalg.norm(D, axis=-1, keepdims=True)
        C = np.einsum("lse,lte->lst", Dn, Dn)           # [8,11,11]
        off = ~np.eye(11, dtype=bool)
        a_off = off.copy(); a_off[0] = False; a_off[:, 0] = False
        print("\n  cosine between token load profiles (uniform removed):")
        print("     all 11 tokens, off-diagonal mean: front %.3f   back %.3f"
              % (C[:4][:, off].mean(), C[4:][:, off].mean()))
        print("     the ten action tokens only:       front %.3f   back %.3f"
              % (C[:4][:, a_off].mean(), C[4:][:, a_off].mean()))
        print("     state token vs each action token: front %.3f   back %.3f"
              % (C[:4, 0, 1:].mean(), C[4:, 0, 1:].mean()))

        # ---- structure along the chunk -----------------------------------
        lag = np.zeros(10)
        for d in range(1, 10):
            pairs = [(i, i + d) for i in range(1, 11 - d)]
            lag[d] = np.mean([C[:, i, j].mean() for i, j in pairs])
        print("  action-token cosine by chunk distance d:")
        print("     " + "  ".join("d=%d %.3f" % (d, lag[d]) for d in range(1, 10)))

        if suite != "libero_10":
            continue
        # ================= figure =====================================
        fig, ax = plt.subplots(2, 2, figsize=(15, 8))
        lab = ["L%d" % L for L in HB_LAYER]

        im = ax[0, 0].imshow(eff, cmap="viridis", aspect="auto", vmin=10, vmax=32)
        ax[0, 0].set_xticks(range(11), TOK, rotation=45)
        ax[0, 0].set_yticks(range(8), lab); ax[0, 0].axhline(3.5, c="w", lw=2)
        for i in range(8):
            for j in range(11):
                ax[0, 0].text(j, i, "%.0f" % eff[i, j], ha="center", va="center",
                              fontsize=7, color="w")
        fig.colorbar(im, ax=ax[0, 0])
        ax[0, 0].set_title("A  effective number of experts (32 = spreads over all)",
                           fontsize=10)

        im = ax[0, 1].imshow(pmax, cmap="magma", aspect="auto", vmin=.125, vmax=.9)
        ax[0, 1].set_xticks(range(11), TOK, rotation=45)
        ax[0, 1].set_yticks(range(8), lab); ax[0, 1].axhline(3.5, c="c", lw=2)
        for i in range(8):
            for j in range(11):
                ax[0, 1].text(j, i, "%.2f" % pmax[i, j], ha="center", va="center",
                              fontsize=7, color="c")
        fig.colorbar(im, ax=ax[0, 1])
        ax[0, 1].set_title("B  P(most-used expert in top-4)   uniform = 0.125",
                           fontsize=10)

        for k, (sl, ttl) in enumerate(((slice(0, 4), "front L2-5"),
                                       (slice(4, 8), "back L12-15"))):
            a = ax[1, k]
            im = a.imshow(C[sl].mean(0), cmap="RdBu_r", vmin=-1, vmax=1)
            a.set_xticks(range(11), TOK, rotation=45); a.set_yticks(range(11), TOK)
            fig.colorbar(im, ax=a)
            a.set_title("%s  token x token load-profile cosine"
                        % ("C" if k == 0 else "D") + "\n" + ttl, fontsize=10)

        fig.suptitle("HB-MoE routing per suffix token, libero_10/t08, "
                     "512 episodes x 22883 control steps", fontsize=12)
        fig.tight_layout(); fig.savefig(FIG / "hb1_per_token.png", dpi=130)
        print("\n  wrote %s" % (FIG / "hb1_per_token.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
