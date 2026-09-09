#!/usr/bin/env python3
"""Pull the slice of the 76 GB corpus these experiments actually touch.

Every script in this line of work reads exactly two things out of the routing
store -- the state token's top-4 expert ids and the full 32-way gate
distribution, at denoise step 0 -- plus the client-side per-episode record.  That
is a few tens of MB against 76 GB, so the bundle can carry it and be rerun
without the corpus.

Written per task as one npz.  Row layout is flat over (episode, control step) in
episode order, with `n_rows` giving the split points, matching what
`analyze_rules.load` reconstructs.  Probabilities go to float16: they are read
back only for the commitment regression, where the 3rd decimal is far below the
scene-bootstrap noise, and it halves the archive.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT = HERE / "bundle/data"
TASKS = [
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet"),
    ("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside"),
    ("libero_long", "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"),
    ("libero_spatial", "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"),
    ("libero_spatial", "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for suite, task in TASKS:
        run = HUB / suite / task / "right-16x32"
        dst = OUT / ("%s__%s.npz" % (suite, task))
        if dst.exists():
            print("skip %s" % dst.name, flush=True)
            continue
        S = sorted(json.loads((run / "client/summaries.json").read_text()),
                   key=lambda s: s["episode_index"])
        nr = np.array([s["inference_calls"] for s in S], np.int32)
        y = np.array([s["success"] for s in S], bool)
        scene = np.array([s["init_state_id"] for s in S], np.int32)
        tot = int(nr.sum())
        d0 = np.load(run / ("client/episode_%02d.npz" % S[0]["episode_index"]),
                     allow_pickle=True)
        ns = d0["sim_state"].shape[1]
        prop = np.zeros((tot, 8), np.float32)
        sim = np.zeros((tot, ns), np.float32)
        act = np.zeros((tot, 70), np.float32)
        o = 0
        for i, s in enumerate(S):
            d = np.load(run / ("client/episode_%02d.npz" % s["episode_index"]),
                        allow_pickle=True)
            k = int(nr[i])
            prop[o:o + k] = d["state"][:k]
            sim[o:o + k] = d["sim_state"][:k]
            act[o:o + k] = d["actions"][:k].reshape(k, -1)
            o += k
        z = zarr.open(str(run / "server/routes.zarr"), mode="r")
        assert z["hb_expert_ids"].shape[0] == tot, (z["hb_expert_ids"].shape, tot)
        ids = np.zeros((tot, 8, 4), np.uint8)
        prob = np.zeros((tot, 8, 32), np.float16)
        CH = 4096
        for a in range(0, tot, CH):
            b = min(a + CH, tot)
            ids[a:b] = np.asarray(z["hb_expert_ids"][a:b, :, 0, 0, :], np.uint8)
            prob[a:b] = np.asarray(z["hb_router_probs"][a:b, :, 0, 0, :], np.float16)
            print("  %s %d/%d" % (task[:28], b, tot), flush=True)
        np.savez_compressed(
            dst, n_rows=nr, success=y, scene=scene, proprio=prop,
            sim_state=sim, actions=act, state_token_top4=ids,
            state_token_probs=prob, suite=suite, task=task,
            hb_layers=np.array([2, 3, 4, 5, 12, 13, 14, 15]))
        print("wrote %s  %.1f MB" % (dst.name, dst.stat().st_size / 1e6), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
