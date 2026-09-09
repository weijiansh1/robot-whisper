"""Extract two cross-query 2x2 route-change tables for all three cohorts.

Both are Hellinger distances, || sqrt(p) - sqrt(r) ||_2 / sqrt(2) in [0, 1],
and both compare consecutive queries.  They differ in *which* denoising steps
they put side by side:

  boundary   b[q] = H( p[q-1, layer, step 9, token],  p[q, layer, step 0, token] )
             the replan seam - the last step of one pass against the first step
             of the next.  This is the axis v7 and v8 never look at.

  withinchunk w[q] = H( p[q-1, layer, step 9, token],  p[q, layer, step 9, token] )
             adjacent-query mobility at a fixed step.  At step 9 this is
             bit-for-bit the frozen v4 `mobility` cache for the action tokens,
             and `state_mobility` for the state token.

Four cells each, by (layer block) x (token block):

    front = L2..L5  (raw layer indices 0..3)      state  = token 0
    back  = L12..L15 (raw layer indices 4..7)     action = tokens 1..10

Every cell is a plain mean over its layers and tokens.  Both tables are stored
at the *arrival* chunk q, never the departure chunk: the comparison is not
observable until query q exists, so indexing it at q-1 would hand the detector
one chunk of lookahead.  Chunk 0 of every episode is therefore NaN by
construction.  Nothing here uses the horizon cap or any label.

Why extract the within-chunk table here rather than read the cache: the cache
`moe-flow-semantics-0906/results/step_profiles/{cohort}_{state_,}mobility.npy`
exists only for development_main and external_8b.  legacy_main16x32 has none,
and legacy is the one cohort no version of this guard was ever fitted on.
`tests/test_withinchunk_matches_cache.py` checks this extraction against that
cache on the two cohorts that have it.

Independent of `moe-v8-0906/experiments/extract_flow_speed.py`, which reads only
action tokens and only differences *within* one pass.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "results"
V4 = ROOT / "moe-v4-0904/results/layerwise_mobility"
NEW_ROOT = ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
OLD_ROOT = ROOT / "VLA_MUI_HUB/cache/HiMoE-VLA"

COHORTS = {
    "development_main": {"cache": V4 / "main_reference.npz", "root": NEW_ROOT,
                         "run_id": None},
    "external_8b": {"cache": V4 / "external_8b.npz", "root": NEW_ROOT,
                    "run_id": None},
    "legacy_main16x32": {
        "cache": ROOT / "moe-v4-0904/results/cache16x32_v4/layerwise_mobility.npz",
        "root": OLD_ROOT, "run_id": "right-16x32"},
}

FRONT, BACK = slice(0, 4), slice(4, 8)
STATE, ACTION = slice(0, 1), slice(1, 11)
BASE_CELLS = ("front_state", "front_action", "back_state", "back_action")
CELLS = (BASE_CELLS + tuple(f"wc_{c}" for c in BASE_CELLS)
         + tuple(f"chord_{c}" for c in BASE_CELLS))
BATCH = 512
LAST_STEP, FIRST_STEP = 9, 0


def _normalise(raw: np.ndarray) -> np.ndarray:
    """float16 router probabilities -> clean simplex, float64.

    float64 and not float32: the state cell's low tail reaches H ~ 0.002 against
    a median of 0.169, and `sqrt(p) - sqrt(r)` there is a near-total
    cancellation.  In float32 that costs ~5e-3 relative on 0.45% of cells, all
    of them in the tail the state head is built to fire on.  float16 -> float64
    is exact, so this removes the only numerical question in the pipeline.
    """
    p = np.clip(raw.astype(np.float64), 0.0, None)
    return p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-300)


def _split(h: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}front_state": h[:, FRONT, STATE].mean(axis=(1, 2)),
        f"{prefix}front_action": h[:, FRONT, ACTION].mean(axis=(1, 2)),
        f"{prefix}back_state": h[:, BACK, STATE].mean(axis=(1, 2)),
        f"{prefix}back_action": h[:, BACK, ACTION].mean(axis=(1, 2)),
    }


def boundary_cells(block: np.ndarray) -> dict[str, np.ndarray]:
    """[q, 8, 10, 11, 32] for one contiguous run -> twelve [q-1] vectors.

    Three legs of one triangle in sqrt-space, all at chunk q:

        seam  = || A - C ||   A = p[q-1, step 9], C = p[q, step 0]
        step9 = || A - B ||   B = p[q, step 9]
        chord = || C - B ||   the within-query denoising displacement

    so the triangle inequality bounds how much of the seam can possibly be new
    relative to the adjacent-query mobility that v7 already measures.
    """
    if block.shape[0] < 2:
        return {c: np.zeros(0, np.float64) for c in CELLS}
    prev_last = np.sqrt(_normalise(block[:-1, :, LAST_STEP]))   # A, [q-1,8,11,32]
    this_first = np.sqrt(_normalise(block[1:, :, FIRST_STEP]))  # C
    this_last = np.sqrt(_normalise(block[1:, :, LAST_STEP]))    # B
    r2 = np.sqrt(2.0)
    return (_split(np.linalg.norm(prev_last - this_first, axis=-1) / r2, "")
            | _split(np.linalg.norm(prev_last - this_last, axis=-1) / r2, "wc_")
            | _split(np.linalg.norm(this_first - this_last, axis=-1) / r2,
                     "chord_"))


def _one_task(args):
    path, wanted, max_query = args
    group = zarr.open_group(str(path), mode="r")
    source = group["hb_router_probs"]
    if tuple(source.shape[1:]) != (8, 10, 11, 32):
        raise ValueError(f"unexpected shape at {path}: {source.shape}")
    raw_episode = np.asarray(group["episode_id"][:], dtype=int)
    starts = np.flatnonzero(np.r_[True, np.diff(raw_episode) != 0])
    ends = np.r_[starts[1:], len(raw_episode)]
    blocks = {int(raw_episode[s]): (int(s), int(e)) for s, e in zip(starts, ends)}

    out = {}
    for row, ep in wanted:
        if ep not in blocks:
            continue
        lo, hi = blocks[ep]
        acc = {c: np.full(max_query, np.nan, np.float64) for c in CELLS}
        # one query of overlap so the seam between read batches is not lost
        for start in range(lo, hi, BATCH):
            stop = min(start + BATCH + 1, hi)
            if stop - start < 2:
                break
            got = boundary_cells(np.asarray(source[start:stop]))
            a = start - lo + 1                      # arrival chunk, not departure
            b = min(a + got["front_state"].shape[0], max_query)
            if b > a:
                for c in CELLS:
                    acc[c][a:b] = got[c][: b - a]
        out[row] = acc
    return out


def build(cohort: str, workers: int) -> None:
    cfg = COHORTS[cohort]
    with np.load(cfg["cache"], allow_pickle=False) as archive:
        cache = {k: np.asarray(archive[k]) for k in archive.files}
    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    max_query = cache["valid"].shape[1]
    run_id = cfg["run_id"] or str(cache["run_id"])
    n = len(episodes)

    jobs = []
    for position, task in enumerate(task_names):
        rows = np.flatnonzero(task_index == position)
        path = cfg["root"] / task / run_id / "server/routes.zarr"
        if not path.is_dir():
            raise FileNotFoundError(path)
        jobs.append((path, [(int(r), int(episodes[r])) for r in rows], max_query))

    out = {c: np.full((n, max_query), np.nan, np.float64) for c in CELLS}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(_one_task, jobs):
            for row, acc in chunk.items():
                for c in CELLS:
                    out[c][row] = acc[c]
            done += 1
            print(f"  {cohort}: {done}/{len(jobs)} tasks", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{cohort}_boundary.npz", **out,
                        cell_names=np.array(CELLS))
    for c in CELLS:
        print(f"{cohort} {c:17s} finite {np.isfinite(out[c]).mean():.4f} "
              f"mean {np.nanmean(out[c]):.4f}", flush=True)


def main() -> None:
    workers = int(os.environ.get("BOUNDARY_WORKERS", "20"))
    names = sys.argv[1:] or list(COHORTS)
    for cohort in names:
        build(cohort, workers)


if __name__ == "__main__":
    main()
