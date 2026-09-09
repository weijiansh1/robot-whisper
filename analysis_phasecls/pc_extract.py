"""Extract per-branch chunk-to-chunk Hellinger series for corpus A and B.

Writes  <out>/hell_<corpus>.npz  with one (T,8,2,11) float32 array per branch.
Loops over branches; peak RSS stays well under the 8 GB cgroup cap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pc_common import branch_hellinger_series  # noqa: E402

ROOT = Path("/home/jovyan/work/himoe-vla")
OUT = ROOT / "analysis_phasecls"
A_ROOT = ROOT / "himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
B_ROOT = (ROOT / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
          "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")


def extract(store_path: Path, keep_ids, id_of_row, out_path: Path):
    z = zarr.open_group(str(store_path), mode="r")
    probs = z["hb_router_probs"]
    order = np.argsort(id_of_row, kind="stable")
    bounds = {}
    ids_sorted = id_of_row[order]
    edges = np.r_[0, np.flatnonzero(np.diff(ids_sorted) != 0) + 1, len(ids_sorted)]
    for a, b in zip(edges[:-1], edges[1:]):
        bounds[int(ids_sorted[a])] = order[a:b]
    res = {}
    lens = {}
    for i, eid in enumerate(sorted(keep_ids)):
        rows = np.sort(bounds[eid])            # global row counter == query order
        if len(rows) < 2:
            continue
        blk = np.asarray(probs[rows[0]:rows[-1] + 1], dtype=np.float32)
        if len(blk) != len(rows):              # non-contiguous fallback
            blk = np.stack([np.asarray(probs[r], dtype=np.float32) for r in rows])
        res[str(eid)] = branch_hellinger_series(blk)
        lens[str(eid)] = len(rows)
        if i % 50 == 0:
            print(f"  {i}/{len(keep_ids)} eid={eid} n_q={len(rows)}", flush=True)
    np.savez_compressed(out_path, **res)
    json.dump(lens, open(str(out_path).replace(".npz", "_len.json"), "w"))
    print(f"wrote {out_path}  branches={len(res)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    if which in ("A", "both"):
        lab = pd.read_csv(A_ROOT / "analysis/candidate_physical_labels.csv")
        z = zarr.open_group(str(A_ROOT / "formal/server/routes.zarr"), mode="r")
        eid = np.asarray(z["episode_id"][:], dtype=np.int64)
        cs = np.asarray(z["control_step"][:], dtype=np.int64)
        assert np.array_equal(cs, np.arange(len(cs))), "control_step is not the global counter"
        extract(A_ROOT / "formal/server/routes.zarr",
                set(lab["episode_id"].astype(int)), eid, OUT / "hell_A.npz")

    if which in ("B", "both"):
        summ = json.load(open(B_ROOT / "client/summaries.json"))
        calls = np.array([s["inference_calls"] for s in summ], dtype=np.int64)
        z = zarr.open_group(str(B_ROOT / "server/routes.zarr"), mode="r")
        n = z["hb_router_probs"].shape[0]
        assert calls.sum() == n, f"offset mismatch {calls.sum()} vs {n}"
        starts = np.r_[0, np.cumsum(calls)[:-1]]
        eid = np.repeat(np.arange(len(summ)), calls)
        extract(B_ROOT / "server/routes.zarr", set(range(len(summ))), eid, OUT / "hell_B.npz")


if __name__ == "__main__":
    main()
