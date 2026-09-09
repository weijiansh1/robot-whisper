"""Which distribution distance should adjacent-query routing change be measured in?

v7 uses **Hellinger** for adjacent-query mobility and a **weighted Jaccard**
(Ruzicka) for its long-lag recurrence head.  Bhattacharyya appears in
`action_consensus`, and a generalised Jensen-Shannon appears implicitly as
`token_differentiation = H(mean_t p_t) - mean_t H(p_t)` - but only *across the
ten action tokens at one chunk*, never *between adjacent queries*.  So the
temporal distance has never been ablated; Hellinger was simply the first choice.

This computes six temporal distances on the same tensor, at the same place v7
measures mobility (final denoising step, action tokens, per layer), so the only
thing that differs between arms is the distance.
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
ROUTE_ROOT = ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"

COHORTS = {
    "development_main": V4 / "main_reference.npz",
    "external_8b": V4 / "external_8b.npz",
}
DISTANCES = ("hellinger", "jensen_shannon", "total_variation",
             "weighted_jaccard", "bhattacharyya", "chi_square")
EPS = 1e-12


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 0.0, None)
    return p / np.maximum(p.sum(axis=-1, keepdims=True), EPS)


def pairwise(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    """Six distances between adjacent-query routing distributions.

    All are normalised so 0 = identical and 1 = maximally different, and all
    are averaged over the ten action tokens afterwards, exactly as v7 averages
    Hellinger.  `bhattacharyya` and `weighted_jaccard` are similarities, so
    they are reported as 1 - similarity to keep the orientation uniform.
    """
    a, b = normalize(a), normalize(b)
    root_a, root_b = np.sqrt(a), np.sqrt(b)
    bc = (root_a * root_b).sum(axis=-1).clip(0.0, 1.0)
    m = 0.5 * (a + b)

    def ent(p):
        return -(p * np.log(np.maximum(p, EPS))).sum(axis=-1)

    jsd = ent(m) - 0.5 * (ent(a) + ent(b))
    return {
        # v7's choice: sqrt(1 - Bhattacharyya coefficient)
        "hellinger": np.sqrt(np.maximum(1.0 - bc, 0.0)),
        # never used in the temporal direction before
        "jensen_shannon": np.sqrt(np.maximum(jsd, 0.0) / np.log(2.0)),
        "total_variation": 0.5 * np.abs(a - b).sum(axis=-1),
        # v7 uses this one, but only for long-lag recurrence
        "weighted_jaccard": 1.0 - np.minimum(a, b).sum(axis=-1)
        / np.maximum(np.maximum(a, b).sum(axis=-1), EPS),
        "bhattacharyya": 1.0 - bc,
        # symmetric chi-square, heavier tail than the others
        "chi_square": ((a - b) ** 2 / np.maximum(a + b, EPS)).sum(axis=-1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cohort, cache_path in COHORTS.items():
        with np.load(cache_path, allow_pickle=False) as archive:
            cache = {k: np.asarray(archive[k]) for k in archive.files}
        task_names = cache["task_names"].astype(str)
        task_index = cache["task_index"].astype(int)
        episodes = cache["episode"].astype(int)
        max_query = cache["valid"].shape[1]
        run_id = str(cache["run_id"])
        n = len(episodes)

        out = {d: np.full((n, max_query, 8), np.nan, np.float32) for d in DISTANCES}
        for position, task in enumerate(task_names):
            rows = np.flatnonzero(task_index == position)
            path = ROUTE_ROOT / task / run_id / "server/routes.zarr"
            if not path.is_dir():
                raise FileNotFoundError(path)
            group = zarr.open_group(str(path), mode="r")
            source = group["hb_router_probs"]
            if tuple(source.shape[1:]) != (8, 10, 11, 32):
                raise ValueError(f"unexpected shape at {path}: {source.shape}")
            raw_episode = np.asarray(group["episode_id"][:], dtype=int)
            lookup = {int(e): i for i, e in enumerate(raw_episode)}
            # Queries are stored consecutively per episode; recover the blocks.
            starts = np.flatnonzero(np.r_[True, np.diff(raw_episode) != 0])
            ends = np.r_[starts[1:], len(raw_episode)]
            block = {int(raw_episode[s]): (int(s), int(e))
                     for s, e in zip(starts, ends)}
            for row in rows:
                ep = int(episodes[row])
                if ep not in block:
                    continue
                lo, hi = block[ep]
                # final denoising step, action tokens only: [q, 8, 10, 32]
                chunk = np.asarray(source[lo:hi, :, -1, 1:, :])
                if chunk.shape[0] < 2:
                    continue
                d = pairwise(chunk[1:], chunk[:-1])
                q = min(chunk.shape[0] - 1, max_query - 1)
                for name, value in d.items():
                    # average over the ten action tokens, as v7 does
                    out[name][row, 1:q + 1] = value.mean(axis=-1)[:q]
            print(f"  {cohort} {task[:44]:44s} {len(rows):5d} rows", flush=True)

        np.savez_compressed(
            OUT / f"{cohort}_temporal_distances.npz",
            layer_names=np.array(["L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15"]),
            distance_names=np.array(DISTANCES),
            **out,
        )
        print(f"wrote {cohort}: {n} episodes x {max_query} queries x 8 layers "
              f"x {len(DISTANCES)} distances")


if __name__ == "__main__":
    main()
