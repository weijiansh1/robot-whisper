"""Extract the single array v8 needs, for all three cohorts.

Both of v8's new heads are functions of one quantity:

    action_root = sqrt(p)[..., action tokens, :]
    flow_speed[k] = || action_root[step k+1] - action_root[step k] || / sqrt(2),
                    averaged over the ten action tokens

  * front/back head   = ratio of sum_k flow_speed over front vs back layers
                        (the invariant search verified flow_path = 9 * mean
                        flow_speed to 2.4e-4, so the sum is the whole content)
  * 3-step curvature  = second difference of flow_speed over steps 1, 5, 9,
                        back layers only

So v8's entire data contract is one [episode, chunk, layer, 9] array.  Building
it directly is far cheaper than rebuilding `layer_graphs` (11 quantities) and
`step_profiles` (8 quantities x 10 steps), and it is the only route available
for legacy_main16x32, which has neither cache.
"""

from __future__ import annotations

import sys
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
BATCH = 512


def flow_speed(raw: np.ndarray) -> np.ndarray:
    """[q, 8, 10, 11, 32] -> [q, 8, 9], float32, matching the published formula."""
    p = np.clip(raw.astype(np.float32), 0.0, None)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    root = np.sqrt(p[:, :, :, 1:, :])              # action tokens only
    delta = root[:, :, 1:] - root[:, :, :-1]       # along the denoising axis
    return (np.linalg.norm(delta, axis=-1) / np.sqrt(2.0)).mean(axis=-1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cohort, cfg in COHORTS.items():
        with np.load(cfg["cache"], allow_pickle=False) as archive:
            cache = {k: np.asarray(archive[k]) for k in archive.files}
        task_names = cache["task_names"].astype(str)
        task_index = cache["task_index"].astype(int)
        episodes = cache["episode"].astype(int)
        max_query = cache["valid"].shape[1]
        run_id = cfg["run_id"] or str(cache["run_id"])
        n = len(episodes)

        out = np.full((n, max_query, 8, 9), np.nan, dtype=np.float32)
        for position, task in enumerate(task_names):
            rows = np.flatnonzero(task_index == position)
            path = cfg["root"] / task / run_id / "server/routes.zarr"
            if not path.is_dir():
                raise FileNotFoundError(path)
            group = zarr.open_group(str(path), mode="r")
            source = group["hb_router_probs"]
            if tuple(source.shape[1:]) != (8, 10, 11, 32):
                raise ValueError(f"unexpected shape at {path}: {source.shape}")
            raw_episode = np.asarray(group["episode_id"][:], dtype=int)
            starts = np.flatnonzero(np.r_[True, np.diff(raw_episode) != 0])
            ends = np.r_[starts[1:], len(raw_episode)]
            block = {int(raw_episode[s]): (int(s), int(e))
                     for s, e in zip(starts, ends)}
            for row in rows:
                ep = int(episodes[row])
                if ep not in block:
                    continue
                lo, hi = block[ep]
                for start in range(lo, hi, BATCH):
                    stop = min(start + BATCH, hi)
                    speed = flow_speed(np.asarray(source[start:stop]))
                    a = start - lo
                    b = min(a + speed.shape[0], max_query)
                    if b > a:
                        out[row, a:b] = speed[: b - a]
            print(f"  {cohort} {task[:46]:46s} {len(rows):5d}", flush=True)

        np.savez_compressed(OUT / f"{cohort}_flow_speed.npz", flow_speed=out,
                            layer_names=np.array(["L2", "L3", "L4", "L5",
                                                  "L12", "L13", "L14", "L15"]))
        finite = float(np.isfinite(out).mean())
        print(f"wrote {cohort}: {out.shape}, finite {finite:.3f}")


if __name__ == "__main__":
    main()
