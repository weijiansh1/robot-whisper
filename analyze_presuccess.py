#!/usr/bin/env python3
"""Routing before the success step: does it see success coming?

Aligning on "the success step" is the trap.  Successes end when they succeed and
failures run to the horizon, so aligning backwards from each episode's own end
compares a success at absolute step 38 against a failure at 51, and any drift
with absolute step leaks in as an effect.  Aligning forwards from step 0 has the
mirror problem.

The framing that avoids both is a hazard: fix the absolute control step, keep
only the episodes still running at it, and split those by how soon they are
about to succeed.  Everything compared is then at the same step of the same
task, and "about to succeed" is the only thing that differs.

  1. at step t, among the still-running episodes, does the routing separate the
     ones that succeed at t+1 from the ones that do not?  And within 3, within 5?
  2. is any of that more than the proprioceptive state already says?  The gate
     at HB 2-5 is a gripper detector, so a signal that vanishes against proprio
     is the arm's phase being read out, not success being predicted.
  3. scene-stratified throughout, so a scene simply being easy cannot score.
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
MIN_PER_GROUP = 8


def sauc(x, y, scene):
    num = den = 0.0
    for s in np.unique(scene):
        m = scene == s
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if not n1 or not n0:
            continue
        r = rankdata(x[m])
        num += r[y[m]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * n0
    return (num / den, den) if den else (float("nan"), 0)


def loso(X, y, scene):
    """Leave-one-scene-out probe, pooled within-scene AUC."""
    num = den = 0.0
    for held in np.unique(scene):
        tr, te = scene != held, scene == held
        if not (0 < y[tr].sum() < tr.sum()) or not (0 < y[te].sum() < te.sum()):
            continue
        sc = StandardScaler().fit(X[tr])
        f = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        s = f.decision_function(sc.transform(X[te]))
        n1, n0 = int(y[te].sum()), int((~y[te]).sum())
        r = rankdata(s)
        num += r[y[te]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * n0
    return num / den if den else float("nan")


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    end = n_rows - 1                      # the step at which a success succeeds
    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")

    prop = np.full((len(S), n_rows.max(), 8), np.nan, np.float32)
    sim = np.full((len(S), n_rows.max(), 47), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)
        prop[i, :n_rows[i]] = d["state"][:n_rows[i]]
        sim[i, :n_rows[i]] = d["sim_state"][:n_rows[i]]

    print("success lengths: min %d  median %d  max %d;  every failure runs %d"
          % (end[y].min(), int(np.median(end[y])), end[y].max(), end[~y].max()))
    print("\nat each absolute control step, among the episodes STILL RUNNING:")
    print(" step  alive  succeed  ---- scene-stratified AUC of 'succeeds within k' ----")
    print("              at t+1   k=1 gate  k=1 routing  k=3 routing  k=5 routing  "
          "| k=3 sim_state  k=3 rout+sim")
    for t in range(20, int(end[y].max())):
        alive = n_rows > t
        if alive.sum() < 40:
            break
        idx = np.flatnonzero(alive)
        P = np.asarray(z["hb_router_probs"].oindex[off[idx] + t, :, 0, :, :], np.float32)
        st = P[:, :, 0].reshape(len(idx), -1)          # state token [n, 256]
        full = P.reshape(len(idx), -1)                  # all 11 tokens
        gate = st.max(1)                                # the gate's height
        sc = scene[idx]
        pr = prop[idx, t]
        sm = sim[idx, t]
        row = [t, int(alive.sum())]
        outs = []
        for k in (1, 3, 5):
            soon = y[idx] & (end[idx] - t <= k) & (end[idx] - t >= 0)
            if k == 1:
                row.append(int(soon.sum()))
            if soon.sum() < MIN_PER_GROUP:
                outs.append((float("nan"),) * (4 if k == 3 else 2))
                continue
            a_gate, _ = sauc(gate, soon, sc)
            a_rt = loso(st, soon, sc)
            if k == 3:
                # the 47-dim MuJoCo state is the control that matters: routing
                # beating 8-dim proprio could just mean it also sees the objects
                a_pr = loso(sm, soon, sc)
                a_both = loso(np.concatenate([st, sm], 1), soon, sc)
                outs.append((a_gate, a_rt, a_pr, a_both))
            else:
                outs.append((a_gate, a_rt))
        g1 = outs[0][0] if len(outs) > 0 else float("nan")
        r1 = outs[0][1] if len(outs) > 0 else float("nan")
        r3 = outs[1][1] if len(outs) > 1 else float("nan")
        p3 = outs[1][2] if len(outs) > 1 and len(outs[1]) > 2 else float("nan")
        b3 = outs[1][3] if len(outs) > 1 and len(outs[1]) > 3 else float("nan")
        r5 = outs[2][1] if len(outs) > 2 else float("nan")
        print("%5d  %5d  %6d    %.3f      %.3f        %.3f        %.3f        "
              "|  %.3f       %.3f"
              % (row[0], row[1], row[2], g1, r1, r3, r5, p3, b3))
    print("\n0.5 = the routing of an episode about to succeed is indistinguishable")
    print("from one that is not, at the same control step and the same scene.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
