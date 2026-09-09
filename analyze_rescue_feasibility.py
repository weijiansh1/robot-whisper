#!/usr/bin/env python3
"""Could an intervention rescue the failures, and could it ever be triggered?

The pin-off arm rescued 10 of base's 12 failures, so the rescue itself is not in
doubt.  Two things decide whether that is useful.

  1. Is the rescue a repair or a retry?  If the intervention decorrelates the
     outcome -- and it does, phi fell from 1.000 to 0.106 -- then applying it to
     a failure just resamples at the marginal rate, and 10 of 12 is exactly what
     a resample predicts.  The test: within a scene the 32 draws differ only in
     the flow-noise seed, so "how often does a different seed succeed where this
     one failed" is the free retry rate, and if the pin's rescue rate matches it,
     the pin bought nothing that a new seed would not.

  2. Can it be triggered?  A rescue needs a signal that says "this episode is
     going to fail" while it is still running.  The pre-success detector is the
     wrong one by construction: it fires on episodes that are already finishing.
     The right question is whether anything -- routing or the full MuJoCo state
     -- separates the still-running episodes that will eventually succeed from
     the ones that never will.
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
T05 = ("libero_spatial/pick_up_the_black_bowl_on_the_ramekin"
       "_and_place_it_on_the_plate")
T08 = ("libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove")


def load(task, arm):
    run = HUB / "cache/HiMoE-VLA" / task / arm
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    return run, S


def loso(X, y, scene):
    num = den = 0.0
    for held in np.unique(scene):
        tr, te = scene != held, scene == held
        if not (0 < y[tr].sum() < tr.sum()) or not (0 < y[te].sum() < te.sum()):
            continue
        sc = StandardScaler().fit(X[tr])
        f = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        s = f.decision_function(sc.transform(X[te]))
        n1 = int(y[te].sum())
        r = rankdata(s)
        num += r[y[te]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * int((~y[te]).sum())
    return num / den if den else float("nan")


def main() -> int:
    print("=== 1. is the rescue a repair or a retry? ===")
    for tag, task in (("spatial t05", T05), ("long t08", T08)):
        _, S = load(task, "right-16x32")
        y = np.array([s["success"] for s in S], bool)
        sc = np.array([s["init_state_id"] for s in S])
        # for every failed draw, how many of the other 31 seeds of that scene
        # succeeded?  that is what re-rolling the sampler buys, for free
        rates = []
        for k in np.flatnonzero(~y):
            sib = (sc == sc[k]) & (np.arange(len(S)) != k)
            rates.append(y[sib].mean())
        print("  %-12s base success %.1f%%   a failed draw's siblings succeed "
              "%.1f%% of the time  (n=%d failures)"
              % (tag, 100 * y.mean(), 100 * np.mean(rates), len(rates)))
    base = load(T05, "pin-base")[1]
    offa = load(T05, "pin-off")[1]
    kb = {(e["init_state_id"], e["flow_noise_seed"]): e["success"] for e in base}
    ko = {(e["init_state_id"], e["flow_noise_seed"]): e["success"] for e in offa}
    keys = sorted(set(kb) & set(ko))
    fails = [k for k in keys if not kb[k]]
    resc = sum(ko[k] for k in fails)
    print("  pin-off applied to base's failures: rescued %d/%d = %.1f%%"
          % (resc, len(fails), 100 * resc / len(fails)))
    print("  pin-off's own marginal success rate: %.1f%%"
          % (100 * np.mean([ko[k] for k in keys])))
    print("  -> if those two match, the pin is a resample, not a repair")

    print("\n=== 2. can it be triggered?  among the episodes STILL RUNNING at")
    print("       step t, does anything separate 'will eventually succeed' from")
    print("       'will never succeed'? ===")
    run, S = load(T08, "right-16x32")
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    d0 = np.load(run / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)["sim_state"]
    sim = np.full((len(S), n_rows.max(), d0.shape[1]), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)["sim_state"]
        sim[i, :n_rows[i]] = d[:n_rows[i]]

    print("  step  alive  of which will succeed   routing AUC   sim_state AUC")
    for t in (0, 8, 16, 24, 30, 34, 38, 42, 46, 50):
        alive = n_rows > t
        idx = np.flatnonzero(alive)
        yy = y[idx]
        if yy.sum() < 12 or (~yy).sum() < 12:
            print("  %4d  %5d   %4d   -- one group too small" % (t, alive.sum(), yy.sum()))
            continue
        P = np.asarray(z["hb_router_probs"].oindex[off[idx] + t, :, 0, :, :],
                       np.float32)
        st = P[:, :, 0].reshape(len(idx), -1)
        print("  %4d  %5d          %4d            %.3f          %.3f"
              % (t, alive.sum(), int(yy.sum()),
                 loso(st, yy, scene[idx]), loso(sim[idx, t], yy, scene[idx])))
    print("\n  0.5 = a doomed episode looks exactly like a healthy one, so there")
    print("  is no moment at which a rescue could be aimed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
