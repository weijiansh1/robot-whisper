#!/usr/bin/env python3
"""Form C, expanded: a specificity ladder for the internal sentinel.

Rungs (all existing captures, new pipeline, goal/t00 unless noted):
  L0 nuisance   corpus-t00 (2g, checkpoint-right)  vs slice-control-t00 (other
                slice, checkpoint-right)          -- pure MIG-slice effect;
                the sentinel SHOULD stay quiet here (known JS 0.005-0.019).
  L1 mask flip  hub-t00 (1g, paper-right)          vs slice-control-t00
                (checkpoint-right) -- wrist images are all-zero in BOTH arms;
                the only config difference is the mask bit (plus whatever L0
                measures).  A pixel check is blind by construction.  This is
                the bug that cost the project two weeks.
  L1b           hub-t00 vs corpus-t00 (mask bit + slice) -- redundancy check.
  L2 task shift hub goal t00 vs hub goal t03 (same checkpoint, same config).

Signals per episode, first inference only: routing occupancy 8x32 over action
tokens (mean of top-4 one-hot over denoise rounds/tokens), first action chunk
(70), step-0 proprio (8, expected quiet on L0/L1 -- same init states).
Scoring identical to audit_ood_sentinel.py (diagonal-whitened distance to a
reference-half mean; AUC single-inference; min n averaging for >=95% detection
at FPR<=5%).  OOD streams vs hub are restricted to the 16 shared init states.
Writes audit_ood_ladder.json.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import zarr
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).resolve().parent
RC = HERE / "himoe-route-capture"
HUB = HERE / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_goal"
RNG = np.random.default_rng(0)
N_BOOT = 4000
HUB_STATES = {0, 3, 7, 10, 13, 16, 20, 23, 26, 29, 33, 36, 39, 42, 46, 49}


def load_new_format(server_dir, client_dir):
    g = zarr.open_group(str(server_dir / "routes.zarr"), mode="r")
    ep = np.asarray(g["episode_id"][:])
    uniq, first = np.unique(ep, return_index=True)
    ids = np.asarray(g["hb_expert_ids"].oindex[np.sort(first), :, :, :, :])
    order = uniq[np.argsort(first)]
    S = json.loads((client_dir / "summaries.json").read_text())
    S = {int(s["episode_index"]): s for s in S}
    occ, act, prop, init = [], [], [], []
    for k, e in enumerate(order):
        a = ids[k][:, :, 1:, :]                      # action tokens only
        hot = np.zeros((8, 32), np.float32)
        for L in range(8):
            v, c = np.unique(a[L], return_counts=True)
            hot[L, v] = c
        hot /= hot.sum(1, keepdims=True)
        occ.append(hot.ravel())
        z = np.load(client_dir / ("episode_%02d.npz" % int(e)),
                    allow_pickle=True)
        act.append(z["actions"][0].reshape(-1))
        prop.append(z["state"][0])
        init.append(int(S[int(e)]["init_state_id"]))
    return (np.array(occ), np.array(act), np.array(prop), np.array(init))


def auc(neg, pos):
    x = np.concatenate([neg, pos]); r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def score_contrast(ref, ood):
    res = {}
    for label, R, O in (("routing_256", ref[0], ood[0]),
                        ("action_70", ref[1], ood[1]),
                        ("proprio_8", ref[2], ood[2])):
        idx = RNG.permutation(len(R))
        fit, hold = idx[::2], idx[1::2]
        mu, sd = R[fit].mean(0), R[fit].std(0) + 1e-6
        s_h = np.linalg.norm((R[hold] - mu) / sd, axis=1)
        s_o = np.linalg.norm((O - mu) / sd, axis=1)
        a = auc(s_h, s_o)
        n95 = None
        for n in range(1, 26):
            nb = np.array([RNG.choice(s_h, n).mean() for _ in range(N_BOOT)])
            pb = np.array([RNG.choice(s_o, n).mean() for _ in range(N_BOOT)])
            if float((pb > np.quantile(nb, 0.95)).mean()) >= 0.95:
                n95 = n
                break
        res[label] = {"auc": float(a), "n95": n95}
    return res


def main() -> int:
    corpus = load_new_format(
        RC / "corpus/libero30-right-v1/libero_goal"
        / "t00__open_the_middle_drawer_of_the_cabinet/server",
        RC / "corpus/libero30-right-v1/libero_goal"
        / "t00__open_the_middle_drawer_of_the_cabinet/client")
    slicec = load_new_format(
        RC / "corpus/slice-control-v1/libero_goal"
        / "t00__open_the_middle_drawer_of_the_cabinet/server",
        RC / "corpus/slice-control-v1/libero_goal"
        / "t00__open_the_middle_drawer_of_the_cabinet/client")
    hub00 = load_new_format(HUB / "open_the_middle_drawer_of_the_cabinet"
                            / "right-16x32/server",
                            HUB / "open_the_middle_drawer_of_the_cabinet"
                            / "right-16x32/client")
    hub03 = load_new_format(HUB / "open_the_top_drawer_and_put_the_bowl_inside"
                            / "right-16x32/server",
                            HUB / "open_the_top_drawer_and_put_the_bowl_inside"
                            / "right-16x32/client")

    def sub16(arm):
        m = np.isin(arm[3], list(HUB_STATES))
        return tuple(x[m] for x in arm)

    out = {}
    ladder = [
        ("L0_slice_nuisance", corpus, slicec),
        ("L1_mask_flip_same_slice", hub00, sub16(slicec)),
        ("L1b_mask_plus_slice", hub00, sub16(corpus)),
        ("L2_task_shift", hub00, hub03),
    ]
    print("rung                        signal        AUC     n@95%")
    for name, ref, ood in ladder:
        out[name] = score_contrast(ref, ood)
        for sig, r in out[name].items():
            print("%-27s %-12s %.3f   %s" % (name, sig, r["auc"], r["n95"]))
        print()

    (HERE / "audit_ood_ladder.json").write_text(json.dumps(out, indent=1))
    print("wrote audit_ood_ladder.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
