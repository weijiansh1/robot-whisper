#!/usr/bin/env python3
"""At which control step do a task's rollouts split into distinct routing states?

Different from "can routing predict the outcome".  That asked whether a split
lines up with success; this asks whether there is a split at all, when it first
appears, and whether it appears while every episode is still running.

For each control step the 512 rollouts give 512 routing vectors.  A 2-means
silhouette says how cleanly they fall into two groups -- near 0 is one blob,
above ~0.5 is two separated ones.  Then each candidate split is checked against
the three things it could be: the outcome, the gripper (the variable the
state-token gate is keyed on), and the scene.

"Before success" is easy to establish here: the shortest successful episode of
t08 takes 35 control steps and every failure runs the full 52, so anything
happening at t < 35 happens while all 512 rollouts are still going.
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
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
FIG = HERE / "fig"
RUNS = [("libero_10", "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", 35),
        ("libero_spatial",
         "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate", 0)]
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]


def auc(x, y, scene):
    """Scene-stratified AUC, so a split that is just scene identity scores 0.5."""
    num = den = 0.0
    for s in np.unique(scene):
        m = scene == s
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if not n1 or not n0:
            continue
        r = rankdata(x[m])
        num += r[y[m]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * n0
    return num / den if den else float("nan")


def main() -> int:
    FIG.mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE / "himoe-route-capture"))
    import corpus_layout as cl
    curves = {}
    for suite, task, _ in RUNS:
        run = HUB / "cache/HiMoE-VLA" / cl.HUB_DIR[suite] / task / "right-16x32"
        if not (run / "server/routes.zarr").exists():
            continue
        S = sorted(json.loads((run / "client/summaries.json").read_text()),
                   key=lambda s: s["episode_index"])
        n_rows = np.array([s["inference_calls"] for s in S])
        off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])
        y = np.array([s["success"] for s in S], bool)
        scene = np.array([s["init_state_id"] for s in S])
        T = int(n_rows.min())                      # all rollouts alive up to here
        short = int(n_rows[y].min()) if y.any() else T
        z = zarr.open(str(run / "server/routes.zarr"), mode="r")
        prop = np.empty((len(S), T, 8), np.float32)
        for i, s in enumerate(S):
            prop[i] = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                              allow_pickle=True)["state"][:T]
        gap = prop[:, :, 6] - prop[:, :, 7]

        print("\n=== %s / %s ===" % (suite, task[:44]))
        print("  512 rollouts; every one still running through control step %d; "
              "the shortest success ends at %d" % (T - 1, short))
        print("  step  silhouette(state)  silhouette(action)   split vs outcome  "
              "split vs gripper   split vs scene   outcome AUC")
        rows = []
        for t in range(0, T, max(1, T // 14)):
            P = np.asarray(z["hb_router_probs"].oindex[off + t, :, 0, :, :], np.float32)
            st = P[:, :, 0].reshape(len(S), -1)
            ac = P[:, :, 1:].reshape(len(S), -1)
            out = []
            for X in (st, ac):
                Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
                lab = KMeans(2, n_init=10, random_state=0).fit_predict(Xs)
                out.append((silhouette_score(Xs, lab), lab))
            (sil_st, lab_st), (sil_ac, _) = out
            # what is the state-token split?
            a_out = auc(lab_st.astype(float), y, scene)
            a_grip = auc(gap[:, t], lab_st.astype(bool), np.zeros(len(S), int))
            purity = max(np.mean([np.bincount(lab_st[scene == s], minlength=2).max()
                                  / (scene == s).sum() for s in np.unique(scene)]),
                         0.0)
            a_route = auc(st.mean(1), y, scene)
            rows.append((t, sil_st, sil_ac, a_out, a_grip, purity, a_route))
            print("  %4d      %.3f              %.3f            %.3f             "
                  "%.3f              %.3f            %.3f"
                  % (t, sil_st, sil_ac, a_out, a_grip, purity, a_route))
        curves[task] = (np.array(rows), T, short)

    fig, ax = plt.subplots(1, len(curves), figsize=(7 * len(curves), 4.2), squeeze=False)
    for k, (task, (r, T, short)) in enumerate(curves.items()):
        a = ax[0, k]
        a.plot(r[:, 0], r[:, 1], "o-", c="tab:red", label="split of the state token")
        a.plot(r[:, 0], r[:, 2], "s-", c="tab:blue", label="split of the action tokens")
        a.plot(r[:, 0], np.abs(r[:, 3] - .5) * 2, "^--", c="k",
               label="does the split track the outcome")
        a.plot(r[:, 0], np.abs(r[:, 4] - .5) * 2, "v--", c="tab:green",
               label="does the split track the gripper")
        if short < T:
            a.axvline(short, ls=":", c="tab:orange")
            a.text(short + .4, .9, "earliest success\nends here", fontsize=8,
                   color="tab:orange")
        a.set_ylim(0, 1.02); a.set_xlabel("control step")
        a.set_ylabel("silhouette / |2·AUC - 1|")
        a.legend(fontsize=7); a.set_title(task[:44], fontsize=9)
    fig.suptitle("When do a task's rollouts separate in routing space, and into what?",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "hb4_split.png", dpi=130)
    print("\nwrote %s" % (FIG / "hb4_split.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
