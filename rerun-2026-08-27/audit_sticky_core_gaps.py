#!/usr/bin/env python3
"""Three gap checks on the sticky-core result.

Q1  Is route stickiness just a frozen router input?  A stuck robot sees an
    almost constant observation, so its HB hidden state barely moves and the
    Top-4 would repeat for trivial reasons.  Compare route occupancy with the
    matched input-persistence statistic and residualise one on the other.

Q2  Does stickiness lead or follow the behavioural trap?  Occupancy is realigned
    to each failed rollout's own onset instead of a fixed control step, which is
    far more powerful than the t7/t12 horizons.

Q3  Does it survive outside the single initial state the figure was built on?
    Occupancy AUC is recomputed inside every initial state of the task.

Every window stays inside each initial state's common cohort, so a successful
rollout that ends early can never thin the comparison group.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr
from scipy.spatial.distance import cdist


HUB = pathlib.Path(__file__).resolve().parent.parent / "VLA_MUI_HUB"
RUN = HUB / "cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
DEEP = (4, 5, 6, 7)
DENOISE = 9
TOKENS = slice(1, 11)
N_EXPERTS = 32
WINDOW = 10
POSE = np.r_[np.arange(10, 13), np.arange(17, 20)]
GOAL_RADIUS_M = 0.05
PROGRESS_EPS_M = 0.01


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    w = float((pos[:, None] > neg[None, :]).sum()) + 0.5 * float((pos[:, None] == neg[None, :]).sum())
    return w / (len(pos) * len(neg))


def trap_onset(distance):
    best = np.minimum.accumulate(distance)
    future = np.minimum.accumulate(distance[::-1])[::-1]
    ok = np.flatnonzero((best - future <= PROGRESS_EPS_M) & (best > GOAL_RADIUS_M))
    return int(ok[0]) if len(ok) else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=RUN)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    rows = sorted(json.loads((args.run / "client/summaries.json").read_text()),
                  key=lambda r: int(r["episode_index"]))
    lengths = np.asarray([r["inference_calls"] for r in rows], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    scenes = np.asarray([int(r["init_state_id"]) for r in rows])
    seeds = np.asarray([int(r["flow_noise_seed"]) for r in rows])
    success = np.asarray([bool(r["success"]) for r in rows])

    sims = []
    for e in range(len(rows)):
        with np.load(args.run / ("client/episode_%02d.npz" % e), allow_pickle=True) as p:
            sims.append(np.asarray(p["sim_state"], dtype=np.float32))
    terminal = [(scenes[i], seeds[i], sims[i][-1, POSE]) for i in range(len(rows)) if success[i]]

    routes = zarr.open(str(args.run / "server/routes.zarr"), mode="r")
    hidden = zarr.open(str(args.run / "server/hidden.zarr"), mode="r")

    occ, inp, onset = {}, {}, {}
    for e in range(len(rows)):
        lo, hi = int(offsets[e]), int(offsets[e] + lengths[e])
        ids = np.asarray(routes["hb_expert_ids"][lo:hi, :, DENOISE, TOKENS, :], dtype=np.int64)
        m = np.zeros(ids.shape[:-1] + (N_EXPERTS,), dtype=bool)
        np.put_along_axis(m, ids, True, axis=-1)                     # [T, layer, token, expert]
        T = len(m)
        o = np.full(T, np.nan)
        for t in range(WINDOW - 1, T):
            o[t] = m[t - WINDOW + 1: t + 1].mean(0).max(-1).mean(-1)[list(DEEP)].mean()
        occ[e] = o

        # matched control: how still is the router's own input over the same window
        h = np.asarray(hidden["hb_hidden"][lo:hi, list(DEEP), DENOISE, TOKENS, :], dtype=np.float32)
        h /= np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
        stay = np.full(T, np.nan)
        cos = np.sum(h[1:] * h[:-1], axis=-1).mean(axis=(1, 2))      # [T-1] input persistence
        for t in range(WINDOW - 1, T):
            stay[t] = cos[t - WINDOW + 1: t].mean()
        inp[e] = stay

        reference = np.stack([p for s, d, p in terminal if s != scenes[e] and d != seeds[e]])
        dist = cdist(sims[e][:, POSE], reference).min(axis=1)
        onset[e] = -1 if success[e] else trap_onset(dist)
        if (e + 1) % 64 == 0:
            print("  %d/%d" % (e + 1, len(rows)), flush=True)

    # ---------------- Q3: every initial state ----------------
    per_scene = []
    for scene in np.unique(scenes):
        idx = np.flatnonzero(scenes == scene)
        common = int(lengths[idx].min()) - 1
        S = np.array([occ[e][common] for e in idx if success[e]])
        F = np.array([occ[e][common] for e in idx if not success[e]])
        SI = np.array([inp[e][common] for e in idx if success[e]])
        FI = np.array([inp[e][common] for e in idx if not success[e]])
        per_scene.append({
            "scene": int(scene), "common": common,
            "n_success": int(len(S)), "n_stasis": int(len(F)),
            "occ_success": float(S.mean()) if len(S) else None,
            "occ_stasis": float(F.mean()) if len(F) else None,
            "occ_auc": auc(F, S),
            "input_auc": auc(FI, SI),
        })

    usable = [r for r in per_scene if r["n_success"] >= 3 and r["n_stasis"] >= 3]
    print("\nQ3  occupancy AUC inside each initial state (>=3 per class):")
    print(f"{'scene':>6} {'nS':>3} {'nF':>3} {'occ AUC':>8} {'input AUC':>10}")
    for r in usable:
        print(f"{r['scene']:>6} {r['n_success']:>3} {r['n_stasis']:>3} "
              f"{r['occ_auc']:>8.3f} {r['input_auc']:>10.3f}")
    occ_aucs = [r["occ_auc"] for r in usable]
    inp_aucs = [r["input_auc"] for r in usable]
    print(f"  mean occupancy AUC {np.mean(occ_aucs):.3f} over {len(usable)} states; "
          f"above chance in {sum(a > .5 for a in occ_aucs)}/{len(usable)}")
    print(f"  mean input-persistence AUC {np.mean(inp_aucs):.3f}; "
          f"above chance in {sum(a > .5 for a in inp_aucs)}/{len(usable)}")

    # ---------------- Q1: is it just a frozen input? ----------------
    pooled_occ, pooled_inp, pooled_lab, pooled_scene = [], [], [], []
    for r in usable:
        idx = np.flatnonzero(scenes == r["scene"])
        for e in idx:
            pooled_occ.append(occ[e][r["common"]]); pooled_inp.append(inp[e][r["common"]])
            pooled_lab.append(not success[e]); pooled_scene.append(r["scene"])
    pooled_occ = np.array(pooled_occ); pooled_inp = np.array(pooled_inp)
    pooled_lab = np.array(pooled_lab); pooled_scene = np.array(pooled_scene)

    corr = float(np.corrcoef(pooled_occ, pooled_inp)[0, 1])
    resid_auc, raw_auc = [], []
    for scene in np.unique(pooled_scene):
        k = pooled_scene == scene
        x, y, lab = pooled_inp[k], pooled_occ[k], pooled_lab[k]
        design = np.column_stack([np.ones(k.sum()), x])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        resid = y - design @ beta
        resid_auc.append(auc(resid[lab], resid[~lab]))
        raw_auc.append(auc(y[lab], y[~lab]))
    print(f"\nQ1  occupancy vs input-persistence correlation r = {corr:+.3f}")
    print(f"    occupancy AUC                       {np.mean(raw_auc):.3f}")
    print(f"    occupancy AUC after removing input  {np.mean(resid_auc):.3f} "
          f"(above chance in {sum(a > .5 for a in resid_auc)}/{len(resid_auc)} states)")

    # ---------------- Q2: onset-aligned timing ----------------
    deltas = list(range(-14, 9, 2))
    aligned = []
    rng = np.random.default_rng(20260827)
    for d in deltas:
        stas, ctrl = [], []
        for e in range(len(rows)):
            common = int(lengths[np.flatnonzero(scenes == scenes[e])].min()) - 1
            if not success[e] and onset[e] >= 0:
                t = onset[e] + d
                if WINDOW - 1 <= t <= common:
                    stas.append(occ[e][t])
            elif success[e]:
                peers = [onset[k] for k in np.flatnonzero(scenes == scenes[e])
                         if not success[k] and onset[k] >= 0]
                if peers:
                    t = int(rng.choice(peers)) + d
                    if WINDOW - 1 <= t <= common:
                        ctrl.append(occ[e][t])
        aligned.append({"delta": d, "n_stasis": len(stas), "n_control": len(ctrl),
                        "stasis": float(np.mean(stas)) if stas else None,
                        "control": float(np.mean(ctrl)) if ctrl else None,
                        "auc": auc(stas, ctrl) if stas and ctrl else None})
    print("\nQ2  occupancy realigned to each rollout's own trap onset "
          "(control = success rollouts at a peer's onset):")
    print(f"{'t-onset':>8} {'nF':>4} {'nS':>4} {'stasis':>8} {'control':>8} {'AUC':>7}")
    for a in aligned:
        if a["auc"] is None:
            continue
        print(f"{a['delta']:>+8} {a['n_stasis']:>4} {a['n_control']:>4} "
              f"{a['stasis']:>8.3f} {a['control']:>8.3f} {a['auc']:>7.3f}")

    args.out.write_text(json.dumps({
        "window": WINDOW, "deep_layers": [HB_LAYERS[a] for a in DEEP],
        "per_scene": per_scene, "usable_states": len(usable),
        "occupancy_auc_mean": float(np.mean(occ_aucs)),
        "input_auc_mean": float(np.mean(inp_aucs)),
        "occ_input_corr": corr,
        "residual_auc_mean": float(np.mean(resid_auc)),
        "raw_auc_mean": float(np.mean(raw_auc)),
        "onset_aligned": aligned,
    }, indent=2) + "\n")
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
