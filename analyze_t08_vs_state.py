#!/usr/bin/env python3
"""Does the routing know anything the physical state does not?  libero_10/t08.

The scalar scan (analyze_t08_outcome.py) found nothing before control step 13
and a large, growing separation after it.  A separation that only appears late
has an obvious innocent explanation: by step 30 a success is three quarters of
the way through the task and a failure is stuck, so *any* function of the
observation separates them.  That would make the routing a readout of task
phase, not a carrier of the outcome.

So the routing is put next to the things it could be reading out, at the same
control step, under the same protocol:

    sim_state   47-dim full MuJoCo state -- every object pose in the scene
    state        8-dim proprioception -- the arm alone
    action      70-dim chunk the policy emitted at this step (10 x 7)
    routing     16 scalars (entropy + top-4 mass per HB layer)
    load       256-dim mean router distribution (8 layers x 32 experts)

Protocol: leave-one-scene-out.  Train on 12 mixed scenes, score the held-out
one, pool the within-scene ranks over all 13 held-out scenes.  A model cannot
score by learning which scene it is looking at, because the scene it is scored
on was never in its training set, and the AUC is computed within that scene.

If routing >= sim_state the routing carries something the world state does not.
If routing <= sim_state it is a readout and the interesting window is wherever
that ordering flips, if anywhere.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HUB = pathlib.Path(__file__).resolve().parent / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
T_MAX = 35


def within_scene_auc(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U and the pair count, for pooling across scenes."""
    from scipy.stats import rankdata
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return 0.0, 0.0
    r = rankdata(score)
    return float(r[y].sum() - n1 * (n1 + 1) / 2.0), float(n1 * n0)


def loso_auc(X: np.ndarray, y: np.ndarray, scene: np.ndarray,
             mixed: list[int], C: float = 0.1) -> float:
    """Pooled within-scene AUC of a leave-one-scene-out logistic probe."""
    num = den = 0.0
    for held in mixed:
        tr = np.isin(scene, [s for s in mixed if s != held])
        te = scene == held
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        s = clf.decision_function(sc.transform(X[te]))
        u, d = within_scene_auc(s, y[te])
        num += u
        den += d
    return num / den


def main() -> int:
    S = json.loads((RUN / "client/summaries.json").read_text())
    S.sort(key=lambda s: s["episode_index"])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    mixed = [int(s) for s in np.unique(scene)
             if 0 < y[scene == s].sum() < (scene == s).sum()]
    keep = np.isin(scene, mixed)

    f = np.load("/tmp/t08_features.npz")
    ent, mass, load = f["ent"], f["mass"], f["load"]

    sim = np.empty((len(S), T_MAX, 47), np.float32)
    prop = np.empty((len(S), T_MAX, 8), np.float32)
    act = np.empty((len(S), T_MAX, 70), np.float32)
    for i, s in enumerate(S):
        # names are zero-padded to two digits, so %02d covers 0..511
        d = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)
        sim[i] = d["sim_state"][:T_MAX]
        prop[i] = d["state"][:T_MAX]
        act[i] = d["actions"][:T_MAX].reshape(T_MAX, 70)

    blocks = {
        "sim_state(47)": sim,
        "proprio(8)": prop,
        "action(70)": act,
        "routing(16)": np.concatenate([ent, mass], -1),
        "load(256)": load.reshape(len(S), T_MAX, 256),
        "rout+load(272)": np.concatenate(
            [ent, mass, load.reshape(len(S), T_MAX, 256)], -1),
    }
    names = list(blocks)

    print("leave-one-scene-out AUC, %d mixed scenes, %d episodes\n" % (len(mixed), keep.sum()))
    print("step  " + "  ".join("%-14s" % n for n in names))
    out = np.empty((T_MAX, len(names)))
    for t in range(T_MAX):
        for j, n in enumerate(names):
            out[t, j] = loso_auc(blocks[n][keep, t], y[keep], scene[keep], mixed)
        print("%4d  " % t + "  ".join("%14.3f" % v for v in out[t]))
        sys.stdout.flush()

    print("\nbest step per block:")
    for j, n in enumerate(names):
        t = int(np.argmax(out[:, j]))
        print("  %-16s AUC %.3f at step %2d   (mean over t<13: %.3f, t>=13: %.3f)"
              % (n, out[t, j], t, out[:13, j].mean(), out[13:, j].mean()))

    np.savez("/tmp/t08_loso.npz", out=out, names=np.array(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
