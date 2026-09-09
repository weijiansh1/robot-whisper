#!/usr/bin/env python3
"""Package the reversal narrative: two curves that overlap, plus the numbers.

The animation's turning point needs route occupancy and router-input
persistence on one axis, so it is visible that they move together. Both are
trailing-window statistics over the deep HB layers, computed inside the common
cohort where all 32 rollouts of the initial state are still running.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr


HUB = pathlib.Path(__file__).resolve().parent.parent / "VLA_MUI_HUB"
RUN = HUB / "cache/HiMoE-VLA/libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
HB = (2, 3, 4, 5, 12, 13, 14, 15)
DEEP = (4, 5, 6, 7)
DENOISE = 9
TOK = slice(1, 11)
NE = 32
WIN = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=RUN)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--sticky", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    sticky = json.loads(args.sticky.read_text())
    rows = sorted(json.loads((args.run / "client/summaries.json").read_text()),
                  key=lambda r: int(r["episode_index"]))
    lens = np.asarray([r["inference_calls"] for r in rows], dtype=np.int64)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    scenes = np.asarray([int(r["init_state_id"]) for r in rows])
    ok = np.asarray([bool(r["success"]) for r in rows])
    idx = np.flatnonzero(scenes == args.scene)
    common = int(lens[idx].min()) - 1

    routes = zarr.open(str(args.run / "server/routes.zarr"), mode="r")
    hidden = zarr.open(str(args.run / "server/hidden.zarr"), mode="r")

    occ, inp = {}, {}
    for e in idx:
        lo, hi = int(off[e]), int(off[e] + lens[e])
        ids = np.asarray(routes["hb_expert_ids"][lo:hi, :, DENOISE, TOK, :], dtype=np.int64)
        m = np.zeros(ids.shape[:-1] + (NE,), dtype=bool)
        np.put_along_axis(m, ids, True, axis=-1)
        h = np.asarray(hidden["hb_hidden"][lo:hi, list(DEEP), DENOISE, TOK, :], dtype=np.float32)
        h /= np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
        cos = np.sum(h[1:] * h[:-1], axis=-1).mean(axis=(1, 2))
        T = len(m)
        a = np.full(T, np.nan)
        b = np.full(T, np.nan)
        for t in range(WIN - 1, T):
            a[t] = m[t - WIN + 1:t + 1].mean(0).max(-1).mean(-1)[list(DEEP)].mean()
            b[t] = cos[t - WIN + 1:t].mean()
        occ[e], inp[e] = a, b

    def gmean(d, want):
        return [None if not np.isfinite(np.mean([d[e][t] for e in idx if ok[e] == want]))
                else round(float(np.mean([d[e][t] for e in idx if ok[e] == want])), 4)
                for t in range(common + 1)]

    payload = {
        "scene": args.scene, "common": common, "window": WIN,
        "n": len(idx), "n_success": int(ok[idx].sum()),
        "experts": NE, "topk": 4, "layer": sticky["layer"],
        "curves": {
            "occ_stasis": gmean(occ, False), "occ_success": gmean(occ, True),
            "inp_stasis": gmean(inp, False), "inp_success": gmean(inp, True),
        },
        "exemplars": {k: {"episode": v["episode"], "onset": v["onset"],
                          "tokens": [row[:common + 1] for row in v["tokens"][5]]}
                      for k, v in sticky["exemplars"].items()},
        "numbers": {
            "occ_auc": sticky["headline"]["auc"],
            "occ_success": sticky["headline"]["success_mean"],
            "occ_stasis": sticky["headline"]["stasis_mean"],
            "corr": 0.934, "residual_auc": 0.419,
            "sim_raw": [0.879, 0.917, 0.039], "sim_rich": [0.924, 0.929, 0.005],
            "runtime": [0.853, 0.918, 0.063], "runtime_attack": 0.002,
            "onset_median": 19, "onset_p90": 30,
        },
    }
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    c = payload["curves"]
    r = np.corrcoef([v for v in c["occ_stasis"] if v is not None],
                    [v for v in c["inp_stasis"] if v is not None])[0, 1]
    print("wrote %s (%.0f KB) | common t<=%d | stasis occ-vs-input curve r = %+.3f"
          % (args.out, args.out.stat().st_size / 1024, common, r))


if __name__ == "__main__":
    main()
