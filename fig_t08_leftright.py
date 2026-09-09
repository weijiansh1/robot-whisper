#!/usr/bin/env python3
"""One figure for the front/back asymmetry, plus the number behind each panel.

The earlier pass showed the front HB block (layers 2-5) is 2.1x further from a
uniform expert load than its mirror (12-15), and the per-token curve suggested
the whole gap sits on the state token.  This separates the two token groups
explicitly, so the claim is measured rather than read off a plot.
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
HB = [2, 3, 4, 5, 12, 13, 14, 15]
UNIF = 1 / 32


def main() -> int:
    FIG.mkdir(exist_ok=True)
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    scene = np.array([s["init_state_id"] for s in S])
    off = np.concatenate([[0], np.cumsum([s["inference_calls"] for s in S])[:-1]])
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    n = len(S)
    P = np.empty((n, len(STEPS), 8, 10, 11, 32), np.float32)
    E = np.empty((n, len(STEPS), 8, 10, 11), np.float32)
    for i in range(n):
        r = off[i] + np.array(STEPS)
        P[i] = np.asarray(z["hb_router_probs"].oindex[r, :, :, :, :], np.float32)
        E[i] = np.asarray(z["hb_entropy"].oindex[r, :, :, :], np.float32)
        if i % 128 == 0:
            print("  read %d/%d" % (i, n), flush=True)

    st, ac = P[..., 0, :], P[..., 1:, :].mean(4)          # [n, T, 8, 10, 32]
    l1_st = np.abs(st.mean((0, 1, 3)) - UNIF).sum(-1)
    l1_ac = np.abs(ac.mean((0, 1, 3)) - UNIF).sum(-1)
    e_st, e_ac = E[..., 0].mean((0, 1, 3)), E[..., 1:].mean((0, 1, 3, 4))

    print("\nlayer  entropy state / action     L1-from-uniform state / action")
    for i, l in enumerate(HB):
        print("  %2d    %.4f / %.4f          %.4f / %.4f"
              % (l, e_st[i], e_ac[i], l1_st[i], l1_ac[i]))
    print("front block: state %.4f  action %.4f   (ratio %.1fx)"
          % (l1_st[:4].mean(), l1_ac[:4].mean(), l1_st[:4].mean() / l1_ac[:4].mean()))
    print("back  block: state %.4f  action %.4f   (ratio %.1fx)"
          % (l1_st[4:].mean(), l1_ac[4:].mean(), l1_st[4:].mean() / l1_ac[4:].mean()))
    print("state-token selectivity, front / back = %.1fx ; action-token %.1fx"
          % (l1_st[:4].mean() / l1_st[4:].mean(), l1_ac[:4].mean() / l1_ac[4:].mean()))

    Pm = P.mean((3, 4))
    scenes = np.unique(scene)
    F0 = []
    for i in range(8):
        ep = Pm[:, 0, i]
        mu = np.stack([ep[scene == s].mean(0) for s in scenes])
        F0.append(mu.var(0, ddof=1).mean()
                  / np.mean([ep[scene == s].var(0, ddof=1).mean() for s in scenes]))

    lab = ["L%d" % l for l in HB]
    x = np.arange(8)
    col = ["tab:blue"] * 4 + ["tab:red"] * 4
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))

    a = ax[0, 0]
    a.bar(x - .2, e_st, .38, label="state token", color=col, alpha=.95)
    a.bar(x + .2, e_ac, .38, label="action tokens", color=col, alpha=.4)
    a.axhline(np.log(32), ls="--", c="k", lw=.9)
    a.text(7.4, np.log(32) - .004, "max = ln32", ha="right", fontsize=8)
    a.set_ylim(2.7, 3.49); a.set_xticks(x, lab); a.axvline(3.5, c="k", lw=.8)
    a.set_ylabel("router entropy (nats)"); a.legend(fontsize=8)
    a.set_title("A  the front block sharpens only the state token\n"
                "(blue = front 2-5, red = back 12-15)", fontsize=10)

    a = ax[0, 1]
    a.bar(x - .2, l1_st, .38, color=col, alpha=.95, label="state token")
    a.bar(x + .2, l1_ac, .38, color=col, alpha=.4, label="action tokens")
    a.set_xticks(x, lab); a.axvline(3.5, c="k", lw=.8); a.legend(fontsize=8)
    a.set_ylabel(r"$\sum_e |p_e - 1/32|$")
    a.set_title("B  distance of the expert load from uniform", fontsize=10)

    a = ax[1, 0]
    a.bar(x, F0, color=col)
    a.axhline(1, ls="--", c="k", lw=.9)
    a.text(7.4, 1.4, "F = 1: a different kitchen moves the router\n"
                     "no more than a different noise seed", ha="right", fontsize=8)
    a.set_xticks(x, lab); a.axvline(3.5, c="k", lw=.8)
    a.set_ylabel("F = between-scene / within-scene")
    a.set_title("C  which block listens to the observation?  (control step 0,\n"
                "where within-scene variation is pure flow noise)", fontsize=10)

    a = ax[1, 1]
    for i, ls in ((3, "-"), (7, "--")):
        a.plot(st.mean((0, 1, 3))[i] - UNIF, ls, color=col[i], lw=1.4,
               label="%s state token" % lab[i])
        a.plot(ac.mean((0, 1, 3))[i] - UNIF, ls, color=col[i], lw=1.4, alpha=.35,
               label="%s action tokens" % lab[i])
    a.axhline(0, c="k", lw=.6); a.set_xlabel("expert id")
    a.set_ylabel("mean load - uniform"); a.legend(fontsize=8)
    a.set_title("D  who gets the traffic: layer 5 (front) vs layer 15 (back)",
                fontsize=10)

    fig.suptitle("HiMoE HB-MoE is placed symmetrically (2-5 <-> 12-15) but does "
                 "not behave symmetrically -- t08, 512 episodes", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "sym6_leftright.png", dpi=130)
    print("\nwrote %s" % (FIG / "sym6_leftright.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
