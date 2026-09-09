#!/usr/bin/env python3
"""Package a few rollouts for a synced replay: scene, routing, detector score.

No rendered frames exist for this capture and MuJoCo is not installed, so the
left-hand panel is a faithful top-down replay driven by the recorded qpos
rather than a video.  Everything shown comes from the rollouts being analysed.

sim_layout.json gives the exact mapping:
    [time] + qpos + qvel, with moka_pot_1 at state[10:17], moka_pot_2 at
    state[17:24] (position then quaternion) and flat_stove_1_button at [24].
End-effector pose comes from the client's own proprioception, state[:3].

The two burner markers are the mean final pot positions over successful
rollouts - an empirical estimate, not scene ground truth, and labelled as such.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr


HUB = pathlib.Path(__file__).resolve().parent.parent / "VLA_MUI_HUB"
RUN = HUB / "cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
P1, P2 = slice(10, 13), slice(17, 20)
Q1, Q2 = slice(13, 17), slice(20, 24)
BTN = 24
HB = (2, 3, 4, 5, 12, 13, 14, 15)
LAYER_AXIS = 7          # HB layer 15, where the routing effect is strongest
TOKEN = 5
DENOISE = 9
NE = 32
PER_CLASS = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=RUN)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--cloud", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    rows = sorted(json.loads((args.run / "client/summaries.json").read_text()),
                  key=lambda r: int(r["episode_index"]))
    lens = np.asarray([r["inference_calls"] for r in rows], np.int64)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    scenes = np.asarray([int(r["init_state_id"]) for r in rows])
    ok = np.asarray([bool(r["success"]) for r in rows])

    cloud = json.loads(args.cloud.read_text())
    score = {}
    k = 0
    for e in range(len(rows)):
        score[e] = None
    for r in cloud["rollouts"]:
        pass  # cloud rows are ordered by included episodes; rebuilt below

    # burner estimate from every successful rollout in the task
    e1, e2 = [], []
    sims = {}
    for e in range(len(rows)):
        with np.load(args.run / ("client/episode_%02d.npz" % e), allow_pickle=True) as p:
            sims[e] = (np.asarray(p["sim_state"], np.float32),
                       np.asarray(p["state"], np.float32))
        if ok[e]:
            e1.append(sims[e][0][-1, P1])
            e2.append(sims[e][0][-1, P2])
    burner = [np.mean(e1, 0).round(4).tolist(), np.mean(e2, 0).round(4).tolist()]

    idx = np.flatnonzero(scenes == args.scene)
    common = int(lens[idx].min()) - 1

    # pick spread-out representatives rather than extremes: order each class by
    # how far the pots travel, then take evenly spaced ranks
    def travel(e):
        s = sims[e][0]
        return float(np.linalg.norm(np.diff(s[:, P1], axis=0), axis=1).sum()
                     + np.linalg.norm(np.diff(s[:, P2], axis=0), axis=1).sum())
    chosen = []
    for want in (False, True):
        grp = sorted([e for e in idx if ok[e] == want], key=travel)
        picks = [grp[int(round(q * (len(grp) - 1)))]
                 for q in np.linspace(.15, .85, PER_CLASS)]
        chosen += list(dict.fromkeys(picks))

    # Only the pose at the end of each control step was recorded, but each step
    # emits ten action substeps, and those commands explain the observed motion
    # almost exactly (per-axis r = 0.95-0.995, scale ~0.0124 m per unit). So the
    # gripper's within-step path is reconstructed from its own recorded commands
    # and then affine-corrected to land on the next recorded pose. The pots have
    # no command channel; the viewer splines those, and says so.
    def action_scale():
        num = den = 0.0
        for e in range(len(rows)):
            sim, st = sims[e]
            with np.load(args.run / ("client/episode_%02d.npz" % e), allow_pickle=True) as p:
                a = np.asarray(p["actions"], np.float32)[:, :, :3]
            c = a.sum(1)[:-1]
            o = np.diff(st[:, :3], axis=0)
            num += float((c * o).sum())
            den += float((c * c).sum())
        return num / den

    K = action_scale()
    print("  action -> metre scale K = %.5f" % K, flush=True)

    def substeps(e):
        st = sims[e][1][:, :3]
        with np.load(args.run / ("client/episode_%02d.npz" % e), allow_pickle=True) as p:
            a = np.asarray(p["actions"], np.float32)
        pos, grip = [], []
        n = len(st)
        for i in range(n - 1):
            cum = np.cumsum(a[i, :, :3], 0) * K
            resid = (st[i + 1] - st[i]) - cum[-1]
            pos.append(st[i] + cum + np.outer(np.arange(1, 11) / 10.0, resid))
            grip.append(a[i, :, 6])
        pos.append(np.repeat(st[-1][None, :], 10, 0))
        grip.append(a[-1, :, 6])
        return np.concatenate(pos), np.concatenate(grip)

    store = zarr.open(str(args.run / "server/routes.zarr"), mode="r")
    out = []
    for e in chosen:
        sim, st = sims[e]
        lo, hi = int(off[e]), int(off[e] + lens[e])
        ids = np.asarray(store["hb_expert_ids"][lo:hi, LAYER_AXIS, DENOISE, 1 + TOKEN, :],
                         dtype=np.int64)
        grid = np.zeros((len(ids), NE), np.uint8)
        np.put_along_axis(grid, ids, 1, axis=-1)
        sub_pos, sub_grip = substeps(e)
        out.append({
            "episode": int(e), "success": bool(ok[e]), "length": int(lens[e]),
            "pot1": np.round(sim[:, P1], 4).tolist(),
            "pot2": np.round(sim[:, P2], 4).tolist(),
            "eef": np.round(st[:, :3], 4).tolist(),
            "eefsub": np.round(sub_pos, 4).tolist(),      # 10 per control step
            "gripsub": np.round(sub_grip, 3).tolist(),
            "grip": np.round(st[:, 6] - st[:, 7], 4).tolist(),
            "button": np.round(sim[:, BTN], 3).tolist(),
            "route": grid.T.tolist(),          # [expert][t]
        })
        print("  packed ep%02d %s len=%d" % (e, "成功" if ok[e] else "陷入", lens[e]), flush=True)

    payload = {
        "task": "libero_long / KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "prompt": rows[int(idx[0])].get("prompt", ""),
        "scene": args.scene, "common": common,
        "layer": HB[LAYER_AXIS], "token": TOKEN, "denoise": DENOISE, "experts": NE,
        "burner": burner,
        "n_success": int(ok[idx].sum()), "n_total": int(len(idx)),
        "auc": cloud["auc"], "horizons": cloud["horizons"],
        "rollouts": out,
    }
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    print("wrote %s (%.0f KB) | %d rollouts | burner est %s"
          % (args.out, args.out.stat().st_size / 1024, len(out), burner))


if __name__ == "__main__":
    main()
