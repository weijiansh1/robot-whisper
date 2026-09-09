#!/usr/bin/env python3
"""Test D: recompute the 320-dim router descriptor on a COMMON ABSOLUTE query
window, so that episode length cannot enter the descriptor at all.

The audited pipeline resamples 10 anchors at relative phase 0.5..1.0, i.e. at
fractional query index phase*(T-1).  That resampling operator is a closed-form
function of T, and the terminal anchor is by construction the last query -- which
means "success terminal state" for successes and "timeout / stall state" for
failures.  Here we instead pin the anchors to FIXED INTEGER absolute query
indices inside each task's common cohort (t <= min episode length in that task),
where every rollout is still running.  No interpolation, no T dependence.

Windows produced:
  full  : 10 integer anchors spanning [0, Tmin-1]
  late  : 10 integer anchors spanning [floor((Tmin-1)/2), Tmin-1]
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr

sys.path.insert(0, "/home/jovyan/work/himoe-vla/himoe-route-capture")
from analyze_failure_routing_clusters import normalize_probabilities  # noqa: E402
from analyze_single_chunk_early_signal import router_features  # noqa: E402

CACHE = pathlib.Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA")
SRC = pathlib.Path(
    "/home/jovyan/work/himoe-vla/himoe-route-capture/analysis/all-outcome-routing-clusters"
)
OUT = pathlib.Path(
    "/home/jovyan/work/himoe-vla/himoe-route-capture/analysis/AUDIT-clustering-leakage-20260828"
)
ANCHORS = 10


def main() -> None:
    manifest = json.loads((SRC / "feature_cache_manifest.json").read_text())
    meta = {
        n: np.load(SRC / "feature_cache" / manifest["arrays"][n]["file"], allow_pickle=False)
        for n in ("meta_task", "meta_episode", "meta_episode_length", "meta_failure")
    }
    tasks, episodes = meta["meta_task"], meta["meta_episode"]
    order = {(str(t), int(e)): i for i, (t, e) in enumerate(zip(tasks, episodes))}

    full = np.zeros((len(tasks), ANCHORS, 320), dtype=np.float32)
    late = np.zeros((len(tasks), ANCHORS, 320), dtype=np.float32)
    windows = {}

    for task in sorted(set(map(str, tasks))):
        run = CACHE / task / "right-16x32"
        rows = sorted(json.loads((run / "client/summaries.json").read_text()),
                      key=lambda r: int(r["episode_index"]))
        lengths = np.asarray([int(r["inference_calls"]) for r in rows])
        offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        tmin = int(lengths.min())
        idx_full = np.unique(np.round(np.linspace(0, tmin - 1, ANCHORS)).astype(int))
        half = (tmin - 1) // 2
        idx_late = np.unique(np.round(np.linspace(half, tmin - 1, ANCHORS)).astype(int))
        # pad with repeats if Tmin < ANCHORS so the tensor stays (10, 320)
        idx_full = np.round(np.linspace(0, tmin - 1, ANCHORS)).astype(int)
        idx_late = np.round(np.linspace(half, tmin - 1, ANCHORS)).astype(int)
        windows[task] = dict(tmin=tmin, full=idx_full.tolist(), late=idx_late.tolist(),
                             distinct_full=int(len(np.unique(idx_full))),
                             distinct_late=int(len(np.unique(idx_late))))
        route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        probs = route["hb_router_probs"]
        ids_a = route["hb_expert_ids"]
        for r in rows:
            ep = int(r["episode_index"])
            start = int(offsets[ep])
            raw = np.asarray(probs[start:start + tmin], np.float32)
            ids = np.asarray(ids_a[start:start + tmin], np.uint8)
            normalized, _lo, _hi = normalize_probabilities(raw)
            tape = router_features(normalized.copy(), ids)  # (tmin, 320)
            i = order[(task, ep)]
            full[i] = tape[idx_full]
            late[i] = tape[idx_late]
        print("done", task, "tmin", tmin, "full idx", idx_full.tolist(), flush=True)

    np.savez_compressed(
        OUT / "common_prefix_tapes.npz",
        full=full, late=late, task=tasks, episode=episodes,
        episode_length=meta["meta_episode_length"], failure=meta["meta_failure"],
        windows=json.dumps(windows),
    )
    print(json.dumps(windows, indent=2))


if __name__ == "__main__":
    main()
