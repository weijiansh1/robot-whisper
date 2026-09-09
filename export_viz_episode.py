#!/usr/bin/env python3
"""Export one rollout's complete MoE state for the 3D viewer.

One control step of this model is not one routing decision, it is 110 of them:
ten flow-matching iterations times eleven suffix tokens, each picking four of 32
experts at each of eight HB layers, plus one of three at each of four AS layers.
The viewer needs all of it, so the export is per (control step, denoise, layer,
token) and nothing is averaged.

Packed as base64 Uint8 arrays rather than JSON numbers: the ids fit in a byte
and the weights quantise to a byte without any visible loss, which keeps a
52-step episode near half a megabyte instead of twenty.

Picks a successful episode whose gripper closes and opens more than once, so the
layer-5 gate visibly fires and releases rather than sitting in one state.
"""

from __future__ import annotations

import base64
import json
import pathlib
import sys

import numpy as np
import zarr

HERE = pathlib.Path(__file__).resolve().parent
HUB = HERE / "VLA_MUI_HUB"
RUN = (HUB / "cache/HiMoE-VLA/libero_long"
       / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
AS_LAYER = [0, 1, 16, 17]
OUT = HERE / "viz" / "episode.json"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main() -> int:
    S = sorted(json.loads((RUN / "client/summaries.json").read_text()),
               key=lambda s: s["episode_index"])
    n_rows = np.array([s["inference_calls"] for s in S])
    off = np.concatenate([[0], np.cumsum(n_rows)[:-1]])

    # choose: a success, medium length, with at least two distinct grasp phases
    best, best_score = None, -1
    for i, s in enumerate(S):
        if not s["success"] or not (34 <= n_rows[i] <= 46):
            continue
        st = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                     allow_pickle=True)["state"][:n_rows[i]]
        gap = st[:, 6] - st[:, 7]
        closed = gap < 0.04
        # number of times the gripper transitions closed<->open
        trans = int(np.abs(np.diff(closed.astype(int))).sum())
        if trans > best_score:
            best, best_score = i, trans
    i = best
    s = S[i]
    T = int(n_rows[i])
    print("episode %d: scene %d, seed %d, %s, %d control steps, %d gripper "
          "transitions" % (s["episode_index"], s["init_state_id"],
                           s["flow_noise_seed"],
                           "success" if s["success"] else "failure", T, best_score))

    z = zarr.open(str(RUN / "server/routes.zarr"), mode="r")
    a, b = off[i], off[i] + T
    ids = np.asarray(z["hb_expert_ids"][a:b]).astype(np.uint8)       # [T,8,10,11,4]
    sel = np.asarray(z["hb_selected_prob"][a:b], np.float32)         # [T,8,10,11,4]
    ent = np.asarray(z["hb_entropy"][a:b], np.float32)               # [T,8,10,11]
    as_id = np.asarray(z["as_expert_ids"][a:b]).astype(np.uint8)     # [T,4]
    as_p = np.asarray(z["as_probs"][a:b], np.float32)                # [T,4,3]

    w = sel / sel.sum(-1, keepdims=True)                 # norm_topk_prob
    wq = np.clip(np.round(w * 255), 0, 255).astype(np.uint8)

    d = np.load(RUN / ("client/episode_%02d.npz" % s["episode_index"]),
                allow_pickle=True)
    st = d["state"][:T]
    gap = (st[:, 6] - st[:, 7]).astype(np.float32)
    act = d["actions"][:T].astype(np.float32)

    # the layer-5 gate: its favourite expert and that expert's probability
    P = np.asarray(z["hb_router_probs"][a:b, 3, :, 0, :], np.float32)  # [T,10,32]
    fav = int(P.mean((0, 1)).argmax())
    gate = P[:, 0, fav].astype(np.float32)

    OUT.parent.mkdir(exist_ok=True)
    doc = {
        "task": s["task_name"], "prompt": s["prompt"],
        "episode": int(s["episode_index"]), "scene": int(s["init_state_id"]),
        "seed": int(s["flow_noise_seed"]), "success": bool(s["success"]),
        "T": T, "n_denoise": 10, "n_token": 11,
        "hb_layers": HB_LAYER, "as_layers": AS_LAYER,
        "n_expert_hb": 32, "n_expert_as": 3, "top_k": 4,
        "gate_layer": 5, "gate_expert": fav,
        "ids_b64": b64(ids),                 # [T,8,10,11,4] uint8
        "w_b64": b64(wq),                    # [T,8,10,11,4] uint8, /255
        "ent": np.round(ent.mean(-1), 4).ravel().tolist(),   # [T,8,10] mean over token
        "as_id": as_id[0].tolist(),          # constant, verified elsewhere
        "as_p": np.round(as_p[0], 4).tolist(),
        "gripper": np.round(gap, 5).tolist(),
        "gate_p": np.round(gate, 4).tolist(),
        "action_norm": np.round(np.linalg.norm(act.reshape(T, -1), axis=1), 3).tolist(),
    }
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    print("wrote %s  (%.2f MB)" % (OUT, OUT.stat().st_size / 1e6))
    print("  gate: layer 5, expert %d; it is above 0.5 on %d of %d control steps"
          % (fav, int((gate > .5).sum()), T))
    return 0


if __name__ == "__main__":
    sys.exit(main())
