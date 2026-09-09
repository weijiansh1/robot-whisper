#!/usr/bin/env python3
"""How does MoE routing differ between rollouts of the same task?

"Different rollout" is two things here and they behave nothing alike.  The 512
rollouts of a task are 16 initial states x 32 flow-noise draws, so:

    between-scene   a different initial state -- a different observation
    within-scene    a bit-identical observation at control step 0, differing
                    only in the seed the flow-matching sampler starts from

At step 0 the within-scene contrast is therefore a pure measurement of what the
*sampler's noise* does to routing, with the world held exactly fixed.  Later it
stops being pure, because by then the arms have physically diverged, and the
rate at which it stops being pure is itself the answer to the question.

Reported per control step, separately for the state token and the ten action
tokens, since those two carry different things (a gripper gate and a positional
code):

  1. top-4 set overlap between two rollouts
  2. cosine distance between their router probability vectors
  3. whether a rollout's routing identifies which scene it came from
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
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
STEPS = [0, 1, 2, 4, 6, 9, 13, 18, 24, 30, 34]
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
RNG = np.random.default_rng(7)


def overlap(a, b):
    """|top-4(a) ∩ top-4(b)| for two [..., 32] probability arrays."""
    ta = np.argsort(-a, -1)[..., :4]
    tb = np.argsort(-b, -1)[..., :4]
    ha = np.zeros(a.shape, bool)
    hb = np.zeros(b.shape, bool)
    np.put_along_axis(ha, ta, True, -1)
    np.put_along_axis(hb, tb, True, -1)
    return (ha & hb).sum(-1)


def main() -> int:
    FIG.mkdir(exist_ok=True)
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    scene = np.array([s["init_state_id"] for s in S])
    scenes = np.unique(scene)
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")

    print("512 rollouts = %d scenes x %d noise draws; all alive through step %d"
          % (len(scenes), (scene == scenes[0]).sum(), n_rows.min() - 1))

    res = {k: [] for k in ("t", "w_st", "b_st", "w_ac", "b_ac",
                           "wc_st", "bc_st", "wc_ac", "bc_ac", "ident")}
    for t in STEPS:
        rows = off + t
        # denoise iteration 0: the state token's routing is invariant across the
        # ten iterations anyway, and this keeps the action tokens on a slice
        # whose x_t has not yet been fed back through the intervention-free loop
        P = np.asarray(z["hb_router_probs"].oindex[rows, :, 0, :, :], np.float32)
        st = P[:, :, 0]                       # [512, 8, 32]
        ac = P[:, :, 1:].reshape(512, 8, -1)  # [512, 8, 10*32]

        wi, bi = [], []
        for s in scenes:
            idx = np.flatnonzero(scene == s)
            wi += [(idx[i], idx[j]) for i in range(len(idx))
                   for j in range(i + 1, len(idx))]
        pool = RNG.permutation(len(wi))[:4000]
        wi = [wi[i] for i in pool]
        for _ in range(len(wi)):
            a, b = RNG.integers(0, 512, 2)
            while scene[a] == scene[b]:
                b = RNG.integers(0, 512)
            bi.append((a, b))

        def stats(pairs, X, is_state):
            i = np.array([p[0] for p in pairs])
            j = np.array([p[1] for p in pairs])
            A, B = X[i], X[j]
            cos = (A * B).sum(-1) / (np.linalg.norm(A, axis=-1)
                                     * np.linalg.norm(B, axis=-1))
            if is_state:
                ov = overlap(A, B).mean()
            else:
                ov = overlap(A.reshape(len(i), 8, 10, 32),
                             B.reshape(len(j), 8, 10, 32)).mean()
            return ov, float(1 - cos.mean())

        w_st, wc_st = stats(wi, st, True)
        b_st, bc_st = stats(bi, st, True)
        w_ac, wc_ac = stats(wi, ac, False)
        b_ac, bc_ac = stats(bi, ac, False)

        # can the routing say which scene this rollout is in?  nearest centroid,
        # each rollout held out of its own scene's centroid
        F = np.concatenate([st, ac], -1).reshape(512, -1)
        F = F - F.mean(0)
        cent = np.stack([F[scene == s].mean(0) for s in scenes])
        hit = 0
        for k in range(512):
            si = list(scenes).index(scene[k])
            c = cent.copy()
            m = scene == scene[k]
            c[si] = F[m & (np.arange(512) != k)].mean(0)
            hit += int(np.argmin(((c - F[k]) ** 2).sum(1)) == si)
        ident = hit / 512

        for key, v in (("t", t), ("w_st", w_st), ("b_st", b_st),
                       ("w_ac", w_ac), ("b_ac", b_ac), ("wc_st", wc_st),
                       ("bc_st", bc_st), ("wc_ac", wc_ac), ("bc_ac", bc_ac),
                       ("ident", ident)):
            res[key].append(v)
        print("  step %2d   top-4 shared: state  within %.2f / between %.2f   "
              "| action  within %.2f / between %.2f   | scene id %.1f%%"
              % (t, w_st, b_st, w_ac, b_ac, 100 * ident), flush=True)

    print("\ncosine distance between two rollouts' router probabilities:")
    print("step   state within / between      action within / between")
    for i, t in enumerate(res["t"]):
        print("%4d   %.5f / %.5f          %.5f / %.5f"
              % (t, res["wc_st"][i], res["bc_st"][i],
                 res["wc_ac"][i], res["bc_ac"][i]))
    print("\nchance baseline for top-4 overlap between unrelated rollouts: 0.50 / 4")
    print("scene identification by chance: %.1f%%" % (100 / len(scenes)))

    t = np.array(res["t"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    a = ax[0]
    a.plot(t, res["w_st"], "o-", c="tab:red", label="state, same scene (noise only)")
    a.plot(t, res["b_st"], "o--", c="tab:red", alpha=.5, label="state, other scene")
    a.plot(t, res["w_ac"], "s-", c="tab:blue", label="action, same scene")
    a.plot(t, res["b_ac"], "s--", c="tab:blue", alpha=.5, label="action, other scene")
    a.axhline(0.5, ls=":", c="k"); a.text(24, .58, "chance 0.50", fontsize=8)
    a.set_xlabel("control step"); a.set_ylabel("top-4 experts shared (of 4)")
    a.legend(fontsize=7); a.set_title("A  how much two rollouts route alike", fontsize=10)

    a = ax[1]
    a.plot(t, res["wc_st"], "o-", c="tab:red", label="state, same scene")
    a.plot(t, res["bc_st"], "o--", c="tab:red", alpha=.5, label="state, other scene")
    a.plot(t, res["wc_ac"], "s-", c="tab:blue", label="action, same scene")
    a.plot(t, res["bc_ac"], "s--", c="tab:blue", alpha=.5, label="action, other scene")
    a.set_yscale("log"); a.set_xlabel("control step")
    a.set_ylabel("1 - cosine of the router probabilities")
    a.legend(fontsize=7); a.set_title("B  and by how much", fontsize=10)

    a = ax[2]
    a.plot(t, 100 * np.array(res["ident"]), "o-", c="k")
    a.axhline(100 / len(scenes), ls=":", c="k")
    a.text(24, 100 / len(scenes) + 3, "chance %.1f%%" % (100 / len(scenes)), fontsize=8)
    a.set_ylim(0, 105); a.set_xlabel("control step")
    a.set_ylabel("% of rollouts assigned to the right scene")
    a.set_title("C  does the routing say which scene you are in?", fontsize=10)

    fig.suptitle("MoE routing across the 512 rollouts of libero_10/t08 "
                 "(16 initial states x 32 flow-noise draws)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "hb3_rollouts.png", dpi=130)
    print("\nwrote %s" % (FIG / "hb3_rollouts.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
