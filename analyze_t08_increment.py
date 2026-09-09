#!/usr/bin/env python3
"""Two closing questions for libero_10/t08.

1. Increment.  Routing and the world state both separate the outcome from step
   13 on and land in the same band.  Does routing add anything *on top of* the
   world state, or is it redundant with it?  Fit the same leave-one-scene-out
   probe on sim_state, on routing, and on the concatenation, and difference the
   paired within-scene ranks -- paired, because all three see the same episodes.

2. The hard scenes.  s49 is 0/32 and contributes no discordant pair, so it is
   invisible to every test above.  Ask a different question of it: is its
   routing distinguishable from the routing of the easy scenes at all?  If the
   router is blind to a scene it fails 32 times out of 32, that is a stronger
   statement than any within-scene AUC.

The bootstrap resamples scenes, not episodes: the unit that would be redrawn if
the experiment were rerun is the initial state.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HUB = pathlib.Path(__file__).resolve().parent / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
T_MAX = 35
STEPS = [13, 20, 26, 30, 34]
RNG = np.random.default_rng(20260817)


def loso_scores(X, y, scene, mixed, C=0.1):
    """Held-out decision values for every episode in a mixed scene."""
    out = np.full(len(y), np.nan)
    for held in mixed:
        tr = np.isin(scene, [s for s in mixed if s != held])
        te = scene == held
        s = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(s.transform(X[tr]), y[tr])
        out[te] = clf.decision_function(s.transform(X[te]))
    return out


def pooled_auc(score, y, scene, scenes):
    num = den = 0.0
    for s in scenes:
        m = scene == s
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if not n1 or not n0:
            continue
        r = rankdata(score[m])
        num += r[y[m]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * n0
    return num / den if den else np.nan


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    mixed = [int(s) for s in np.unique(scene)
             if 0 < y[scene == s].sum() < (scene == s).sum()]
    keep = np.isin(scene, mixed)

    f = np.load("/tmp/t08_features.npz")
    rout = np.concatenate([f["ent"], f["mass"],
                           f["load"].reshape(len(S), T_MAX, 256)], -1)
    sim = np.empty((len(S), T_MAX, 47), np.float32)
    for i, s in enumerate(S):
        sim[i] = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                         allow_pickle=True)["sim_state"][:T_MAX]

    print("=== 1. does routing add anything over the world state? ===")
    print("step   sim_state   routing   sim+routing   delta(sim+r - sim)  95%% CI (scene bootstrap)")
    for t in STEPS:
        blocks = {"sim": sim[:, t], "rout": rout[:, t],
                  "both": np.concatenate([sim[:, t], rout[:, t]], -1)}
        sc = {k: loso_scores(v[keep], y[keep], scene[keep], mixed)
              for k, v in blocks.items()}
        a = {k: pooled_auc(v, y[keep], scene[keep], mixed) for k, v in sc.items()}
        boot = []
        for _ in range(2000):
            pick = RNG.choice(mixed, len(mixed), replace=True)
            boot.append(pooled_auc(sc["both"], y[keep], scene[keep], pick)
                        - pooled_auc(sc["sim"], y[keep], scene[keep], pick))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        print("%4d   %9.3f   %7.3f   %11.3f   %+17.3f   [%+.3f, %+.3f]"
              % (t, a["sim"], a["rout"], a["both"], a["both"] - a["sim"], lo, hi))

    print("\n=== 2. can the routing see the scene it always fails in? ===")
    # s49 is 0/32; s33 and s36 are 32/32.  Train scene-vs-rest on routing alone,
    # holding out draws (not scenes -- the question is about one scene's identity).
    from sklearn.model_selection import cross_val_score
    for target, label in ((49, "s49 (0/32)"), (3, "s3 (5/32)"),
                          (33, "s33 (32/32)"), (36, "s36 (32/32)")):
        accs = []
        for t in (0, 13, 34):
            X = rout[:, t]
            lab = (scene == target)
            s = StandardScaler().fit_transform(X)
            a = cross_val_score(LogisticRegression(C=0.1, max_iter=2000), s, lab,
                                cv=5, scoring="roc_auc")
            accs.append(a.mean())
        print("  %-14s routing identifies this scene vs the other 15: "
              "AUC step0 %.3f  step13 %.3f  step34 %.3f"
              % (label, *accs))

    print("\n=== 3. the two hard mixed scenes on their own ===")
    for t in (13, 26, 34):
        row = []
        for s in (3, 39, 10, 13, 26, 0, 42, 20, 7, 23, 29, 16, 46):
            m = scene == s
            sc = loso_scores(rout[keep, t], y[keep], scene[keep], mixed)
            full = np.full(len(y), np.nan)
            full[keep] = sc
            row.append((s, int(y[m].sum()), pooled_auc(full, y, scene, [s])))
        print("  step %2d: " % t + "  ".join("s%d(%d/32):%.2f" % r for r in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
