#!/usr/bin/env python3
"""The state-token switch: which layers have one, and what turns it on.

At (layer 5, state token) the favourite expert's router probability is bimodal --
56% of draws sit under 0.05, which is the uniform 1/32, and 24.7% sit above 0.5
with an upper mode near 0.8.  That is not a noisy preference, it is a gate that
flips, and it is the only input-dependent routing left in the model: AS is a
config constant and the ten action tokens carry a fixed positional offset that
never leaves a 0.004-wide band.

So this asks what the gate is a gate *on*.

  1. which of the eight HB layers are bimodal at the state token at all
  2. is the switch a per-control-step quantity or does it move within the ten
     denoise iterations?  `embed_suffix` re-embeds the suffix every iteration but
     the state input is unchanged, so if the state token does not attend to the
     action tokens its routing should be constant across d -- measurable, not
     assumable
  3. how the on-rate is distributed over control steps, scenes and episodes
  4. do the layers switch together
  5. what predicts the switch: the 8-dim proprioception, the 47-dim world state,
     or the gripper alone.  Leave-one-scene-out so it cannot learn the scene.

The state layout is LIBERO's: eef position (3), eef orientation (3), gripper
qpos (2) -- so dims 6,7 are the fingers.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
UNIF = 1 / 32
CHUNK = 2048
ON, OFF = 0.5, 2 * UNIF


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    ep = np.repeat(np.arange(len(S)), n_rows)
    step = np.concatenate([np.arange(k) for k in n_rows])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])

    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    n = z["hb_router_probs"].shape[0]

    # pass 1: the favourite expert at (layer, state token)
    acc = np.zeros((8, 32))
    for a in range(0, n, CHUNK):
        acc += np.asarray(z["hb_router_probs"][a:min(a + CHUNK, n), :, :, 0],
                          np.float64).sum((0, 2))
    star = acc.argmax(-1)

    # pass 2: that expert's probability, per control step and denoise iteration
    p = np.empty((n, 8, 10), np.float32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        pr = np.asarray(z["hb_router_probs"][a:b, :, :, 0], np.float32)
        for i in range(8):
            p[a:b, i] = pr[:, i, :, star[i]]
        if a % (CHUNK * 4) == 0:
            print("  read %d/%d" % (a, n), flush=True)

    print("\n=== 1. which layers have a switch at the state token? ===")
    print("layer  favourite  mean    p50     on-rate(p>.5)  off-rate(p<2/32)  "
          "middle")
    for i, L in enumerate(HB_LAYER):
        x = p[:, i].ravel()
        print("  %2d      e%-2d     %.4f  %.4f      %5.1f%%          %5.1f%%      "
              "%5.1f%%" % (L, star[i], x.mean(), np.median(x),
                           100 * (x > ON).mean(), 100 * (x < OFF).mean(),
                           100 * ((x >= OFF) & (x <= ON)).mean()))
    print("  (uniform = %.4f; a bimodal layer has a small middle)" % UNIF)

    print("\n=== 2. does the switch move across the 10 denoise iterations? ===")
    for i, L in enumerate(HB_LAYER):
        within = p[:, i].std(1).mean()          # spread across d, same ctrl step
        between = p[:, i].mean(1).std()         # spread across ctrl steps
        flip = (p[:, i] > ON).any(1) & ~(p[:, i] > ON).all(1)
        print("  L%-3d  sd across d %.5f   sd across control steps %.5f   "
              "ratio %.4f   steps where d disagrees on on/off: %.1f%%"
              % (L, within, between, within / between, 100 * flip.mean()))

    on = p.mean(2) > ON                          # [n, 8] per control step
    print("\n=== 3. when is it on? ===")
    for i, L in enumerate(HB_LAYER):
        if on[:, i].mean() < 0.02:
            continue
        by_step = [on[step == t, i].mean() for t in range(0, 35)]
        print("  L%-3d on-rate %.3f overall | by control step 0,4,8,12,16,20,"
              "24,28,34: %s" % (L, on[:, i].mean(),
                                " ".join("%.2f" % by_step[t] for t in
                                         (0, 4, 8, 12, 16, 20, 24, 28, 34))))
        per_scene = np.array([on[np.isin(ep, np.flatnonzero(scene == s)), i].mean()
                              for s in np.unique(scene)])
        print("        by scene: min %.2f  max %.2f  sd %.3f"
              % (per_scene.min(), per_scene.max(), per_scene.std()))

    print("\n=== 4. do the layers switch together? ===")
    C = np.corrcoef(on.T.astype(float))
    print("        " + "  ".join("L%-4d" % L for L in HB_LAYER))
    for i, L in enumerate(HB_LAYER):
        print("  L%-4d " % L + "  ".join("%+.2f" % v for v in C[i]))

    print("\n=== 5. what predicts the switch? ===")
    prop = np.empty((n, 8), np.float32)
    sim = np.empty((n, 47), np.float32)
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    for i, s in enumerate(S):
        d = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)
        a, b = off[i], off[i] + n_rows[i]
        prop[a:b] = d["state"][:n_rows[i]]
        sim[a:b] = d["sim_state"][:n_rows[i]]

    scenes = np.unique(scene)
    blocks = {"proprio(8)": prop, "gripper only(2)": prop[:, 6:8],
              "eef pos(3)": prop[:, 0:3], "sim_state(47)": sim}
    for i, L in enumerate(HB_LAYER):
        if not (0.02 < on[:, i].mean() < 0.98):
            continue
        out = []
        for name, X in blocks.items():
            num = den = 0.0
            for held in scenes:
                tr = np.isin(ep, np.flatnonzero(scene != held))
                te = ~tr
                if not (0 < on[tr, i].sum() < tr.sum()):
                    continue
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=1.0, max_iter=2000).fit(
                    sc.transform(X[tr]), on[tr, i])
                s_ = clf.decision_function(sc.transform(X[te]))
                yy = on[te, i]
                n1, n0 = int(yy.sum()), int((~yy).sum())
                if not n1 or not n0:
                    continue
                r = rankdata(s_)
                num += r[yy].sum() - n1 * (n1 + 1) / 2.0
                den += n1 * n0
            out.append("%s %.3f" % (name, num / den if den else float("nan")))
        print("  L%-3d (on %.1f%%):  %s" % (L, 100 * on[:, i].mean(), "   ".join(out)))
    print("  leave-one-scene-out AUC; 0.5 = the switch is not a function of the "
          "physical state")
    np.savez("/tmp/hb_switch.npz", p=p, star=star, on=on, ep=ep, step=step,
             y=y, scene=scene)
    return 0


if __name__ == "__main__":
    sys.exit(main())
