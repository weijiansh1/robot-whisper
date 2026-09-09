#!/usr/bin/env python3
"""Does the routing choice depend on the task?

Two layers to the question and they have different answers, so keep them apart.

AS-MoE is settled: its router reads `data_mask`, a per-dataset config constant,
so it cannot vary with the task at all -- confirmed as [2,0,0,1] over 2560
episodes of 5 tasks in 3 suites.

HB-MoE needs measuring, and the measurement has a built-in confound that cannot
be designed away with this corpus: each LIBERO task carries its own initial
states, so "a different task" is always also "a different scene".  What *can* be
asked is whether changing the task moves the routing more than changing the scene
within a task does.  If the two are the same size, the router is reading the
situation and the task label is adding nothing; if between-task is systematically
larger, something task-specific is in there -- the prompt is the obvious
candidate, since it reaches the state token through the prefix.

Compared at control step 0, where every rollout is at its initial state and none
has diverged, and within a checkpoint, since different suites are different
weights.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN_ID = "right-16x32"
PAIRS = [("Goal", "libero_goal", ["open_the_middle_drawer_of_the_cabinet",
                                  "open_the_top_drawer_and_put_the_bowl_inside"]),
         ("Spatial", "libero_spatial",
          ["pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
           "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"])]


def step0(run: pathlib.Path):
    S = sorted(json.loads((run / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    off = np.concatenate([[0], np.cumsum([s["inference_calls"] for s in S])[:-1]])
    z = zarr.open(str(run / "server/routes.zarr"), mode="r")
    P = np.asarray(z["hb_router_probs"].oindex[off, :, 0, :, :], np.float32)
    scene = np.array([s["init_state_id"] for s in S])
    return P, scene, S[0]["prompt"]


def main() -> int:
    sys.path.insert(0, str(HERE / "himoe-route-capture"))
    import corpus_layout as cl

    for ck, suite, tasks in PAIRS:
        got = []
        for t in tasks:
            run = HUB / "cache/HiMoE-VLA" / cl.HUB_DIR[suite] / t / RUN_ID
            if not (run / "server/routes.zarr").exists():
                break
            got.append((t,) + step0(run))
        if len(got) != 2:
            print("skip %s (need both tasks captured)" % ck)
            continue

        print("\n=== %s checkpoint, same weights, two tasks ===" % ck)
        for t, _, _, pr in got:
            print("   %-52s %r" % (t[:52], pr))

        # a scene's signature: the mean step-0 routing of its 32 noise draws
        sigs, labels = [], []
        for ti, (t, P, scene, _) in enumerate(got):
            for s in np.unique(scene):
                sigs.append(P[scene == s].mean(0))
                labels.append(ti)
        sigs = np.stack(sigs)                     # [32 scenes, 8, 11, 32]
        labels = np.array(labels)
        F = sigs.reshape(len(sigs), -1)
        F = F / np.linalg.norm(F, axis=1, keepdims=True)
        D = 1 - F @ F.T                           # cosine distance between scenes

        iu = np.triu_indices(len(F), 1)
        same = labels[iu[0]] == labels[iu[1]]
        d_in, d_out = D[iu][same], D[iu][~same]
        print("\n  cosine distance between two scenes' step-0 routing:")
        print("     same task     %.5f +- %.5f   (n=%d)" % (d_in.mean(), d_in.std(), len(d_in)))
        print("     across tasks  %.5f +- %.5f   (n=%d)" % (d_out.mean(), d_out.std(), len(d_out)))
        print("     ratio across/same = %.2fx" % (d_out.mean() / d_in.mean()))
        ov = ((d_out[:, None] > d_in[None, :]).mean())
        print("     P(a cross-task pair is further apart than a same-task pair) "
              "= %.3f   (0.5 = indistinguishable)" % ov)

        # split by which token, since they carry different things
        for name, sl in (("state token", slice(0, 1)), ("action tokens", slice(1, 11))):
            G = sigs[:, :, sl].reshape(len(sigs), -1)
            G = G / np.linalg.norm(G, axis=1, keepdims=True)
            E = 1 - G @ G.T
            a, b = E[iu][same].mean(), E[iu][~same].mean()
            print("     %-14s same task %.5f   across tasks %.5f   ratio %.2fx"
                  % (name, a, b, b / a))

        # can a single rollout be assigned to its task?  leave-one-scene-out so
        # it is the task being identified and not the scene it came from.
        hit = 0
        for k in range(len(F)):
            c = [F[(labels == j) & (np.arange(len(F)) != k)].mean(0) for j in (0, 1)]
            c = [v / np.linalg.norm(v) for v in c]
            hit += int(np.argmin([((v - F[k]) ** 2).sum() for v in c]) == labels[k])
        print("     scene assigned to the right task: %d/%d = %.1f%%  (chance 50%%)"
              % (hit, len(F), 100 * hit / len(F)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
