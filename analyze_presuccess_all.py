#!/usr/bin/env python3
"""The pre-success routing signal, on every task captured.

Same hazard framing as analyze_presuccess.py: fix the absolute control step, keep
the episodes still running at it, split them by how soon they are about to
succeed, and stratify on scene.  Run across all five tasks, which differ in
median length by more than 4x (10 to 45 control steps), so a lead of "5 steps"
means very different things -- half an episode on libero_spatial t05, a ninth of
one on libero_10 t08.  Lead is therefore reported both ways.

The control is the 47-dim MuJoCo state, not the 8-dim proprioception.  Against
proprio the routing looks like it wins; against the full physical state it does
not, and reporting the weaker control would overstate the result.
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
RUN_ID = "right-16x32"
MIN_SOON, MIN_ALIVE = 10, 60
LEAD = 3


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


def run(suite, task, hub_dir):
    run = HUB / "cache/HiMoE-VLA" / hub_dir / task / RUN_ID
    if not (run / "server/routes.zarr").exists():
        return None
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
    y = np.array([s["success"] for s in S], bool)
    scene = np.array([s["init_state_id"] for s in S])
    end = n_rows - 1
    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    # the MuJoCo state vector is task-specific: its length is set by how many
    # objects the scene has, so read the width rather than assuming t08's 47
    d0 = np.load(run / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                 allow_pickle=True)["sim_state"]
    sim_dim = int(d0.shape[1])
    sim = np.full((len(S), n_rows.max(), sim_dim), np.nan, np.float32)
    for i, s in enumerate(S):
        d = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                    allow_pickle=True)["sim_state"]
        sim[i, :n_rows[i]] = d[:n_rows[i]]
    med = float(np.median(end[y])) if y.any() else float("nan")

    rows = []
    for t in range(0, int(end[y].max()) if y.any() else 0):
        alive = n_rows > t
        if alive.sum() < MIN_ALIVE:
            break
        idx = np.flatnonzero(alive)
        soon = y[idx] & (end[idx] - t <= LEAD) & (end[idx] - t >= 0)
        if soon.sum() < MIN_SOON or (~soon).sum() < MIN_SOON:
            continue
        P = np.asarray(z["hb_router_probs"].oindex[off[idx] + t, :, 0, :, :],
                       np.float32)
        st = P[:, :, 0].reshape(len(idx), -1)
        rows.append((t, int(alive.sum()), int(soon.sum()),
                     loso(st, soon, scene[idx]),
                     loso(sim[idx, t], soon, scene[idx])))
    return dict(task=task, med=med, n=len(S), succ=float(y.mean()), rows=rows,
                sim_dim=sim_dim,
                maxend=int(end[y].max()) if y.any() else 0)


def main() -> int:
    sys.path.insert(0, str(HERE / "himoe-route-capture"))
    import corpus_layout as cl
    out = []
    for suite in ("libero_goal", "libero_spatial", "libero_10"):
        base = HUB / "cache/HiMoE-VLA" / cl.HUB_DIR[suite]
        for task in sorted(p.name for p in base.iterdir()):
            r = run(suite, task, cl.HUB_DIR[suite])
            if r:
                r["suite"] = suite
                out.append(r)

    for r in out:
        print("\n=== %s / %s ===" % (r["suite"], r["task"][:46]))
        print("  %d rollouts, %.1f%% success, median success ends at step %.0f"
              % (r["n"], 100 * r["succ"], r["med"]))
        if not r["rows"]:
            print("  no step has >= %d episodes in both groups" % MIN_SOON)
            continue
        print("  step  frac of median  alive  succeeding within %d   routing   "
              "sim_state(%d)  winner" % (LEAD, r["sim_dim"]))
        for t, al, so, a_r, a_s in r["rows"]:
            w = "routing" if a_r > a_s + .005 else ("state" if a_s > a_r + .005 else "tie")
            print("  %4d      %5.2f       %5d        %4d            %.3f      "
                  "%.3f        %s" % (t, t / r["med"], al, so, a_r, a_s, w))

    print("\n\n=== summary across all %d tasks ===" % len(out))
    print("task                          earliest step with routing AUC>=.75   "
          "lead (steps / fraction)   peak routing   peak sim_state   routing wins")
    for r in out:
        rr = [x for x in r["rows"] if x[3] >= .75]
        if not rr:
            print("  %-28s none" % r["task"][:28])
            continue
        t0 = rr[0][0]
        peak_r = max(x[3] for x in r["rows"])
        peak_s = max(x[4] for x in r["rows"])
        wins = sum(1 for x in r["rows"] if x[3] > x[4] + .005)
        print("  %-28s %4d                                %4.0f / %.2f of the "
              "episode      %.3f          %.3f           %d/%d"
              % (r["task"][:28], t0, r["med"] - t0, (r["med"] - t0) / r["med"],
                 peak_r, peak_s, wins, len(r["rows"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
