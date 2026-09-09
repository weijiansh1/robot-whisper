"""Per-denoising-step adjacent-query mobility, all three cohorts.

v7 reads mobility at the final denoising step only (`FINAL_FLOW`), aggregates
the median over L12..L15, self-baselines against q1..q4 and smooths with a
causal width-6 mean.  Every one of those steps is a *smoother*, and the signal
in the two suites v8 handles worst is sharply chunk-localised: on development,
libero_object at chunk 9 scores 0.846-0.892 within-task AUC across all eight
layers, and libero_spatial peaks at chunk 7.  A width-6 trailing mean spreads a
one-chunk peak over six chunks and buries it.

So this extracts mobility per (layer, denoising step) without any smoothing,
for a chunk-gated head to use directly.

    mobility[q, l, s] = Hellinger( p[q, l, s, action tokens, :],
                                   p[q-1, l, s, action tokens, :] )
                        averaged over the ten action tokens

This is v7's own primitive with the step axis left free instead of fixed at 9.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "results"
V4 = ROOT / "moe-v4-0904/results/layerwise_mobility"

COHORTS = {
    "development_main": {"cache": V4 / "main_reference.npz",
                         "root": ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA",
                         "run_id": None},
    "external_8b": {"cache": V4 / "external_8b.npz",
                    "root": ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA",
                    "run_id": None},
    "legacy_main16x32": {
        "cache": ROOT / "moe-v4-0904/results/cache16x32_v4/layerwise_mobility.npz",
        "root": ROOT / "VLA_MUI_HUB/cache/HiMoE-VLA", "run_id": "right-16x32"},
}


def hellinger(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.clip(a.astype(np.float32), 0.0, None)
    a /= np.maximum(a.sum(-1, keepdims=True), 1e-12)
    b = np.clip(b.astype(np.float32), 0.0, None)
    b /= np.maximum(b.sum(-1, keepdims=True), 1e-12)
    return np.sqrt(np.clip(1.0 - np.sqrt(a * b).sum(-1), 0.0, 1.0))


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

        out = np.full((n, max_query, 8, 10), np.nan, dtype=np.float32)
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
                if hi - lo < 2:
                    continue
                arr = np.asarray(source[lo:hi])[:, :, :, 1:, :]   # action tokens
                d = hellinger(arr[1:], arr[:-1]).mean(axis=-1)    # [q-1, 8, 10]
                q = min(d.shape[0], max_query - 1)
                out[row, 1:q + 1] = d[:q]
            print(f"  {cohort} {task[:46]:46s} {len(rows):5d}", flush=True)

        np.savez_compressed(OUT / f"{cohort}_step_mobility.npz", mobility=out)
        print(f"wrote {cohort}: {out.shape}, finite {np.isfinite(out).mean():.3f}")


if __name__ == "__main__":
    main()
