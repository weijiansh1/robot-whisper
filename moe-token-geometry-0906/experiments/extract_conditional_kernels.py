#!/usr/bin/env python3
"""Cache the per-query conditional token kernel for every HB layer.

The conditional kernel is the Schur complement that ``compute_batch`` in
``moe-hb-front-back-0905/experiments/extract_layer_graphs_gpu.py`` forms at the
final denoising step:

    root       = sqrt(normalised hb_router_probs[:, :, -1])      (11 x 32)
    gram       = root root^T                                     (11 x 11)
    conditional = gram[1:, 1:] - gram[0, 1:] gram[0, 1:]^T       (10 x 10)

Because ``gram`` is PSD and ``gram[0, 0] == 1`` (the state token's root vector
is a unit vector), the conditional block is exactly the Schur complement of the
state token and is therefore PSD.  A PSD kernel over the ten action tokens is a
Gram matrix of ten points in Euclidean space, which is what the geometry
analysis in this bundle consumes.

Only the 55 upper-triangular entries (diagonal included) are stored, in float32,
packed over valid queries.  Nothing about outcomes, wall-clock or episode length
enters this file.  CPU only: the cluster GPUs are saturated by other jobs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import zarr  # noqa: E402


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
V4 = PROJECT / "moe-v4-0904/results/layerwise_mobility"
DEFAULT_OUTPUT = BUNDLE / "cache/kernels"

COHORTS: dict[str, Path] = {
    "development_main": V4 / "main_reference.npz",
    "development_extra": V4 / "extra_reference.npz",
    "external_8b": V4 / "external_8b.npz",
}
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
UPPER = np.triu_indices(10, 0)  # 55 entries, diagonal included
SCHEMA = "himoe.token_geometry.conditional_kernel.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--workers", type=int, default=20)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def conditional_upper(raw: np.ndarray) -> np.ndarray:
    """[n, 8, 10, 11, 32] float16 -> [n, 8, 55] float32 conditional kernel."""
    probability = np.asarray(raw[:, :, -1], dtype=np.float32)
    np.clip(probability, 0.0, None, out=probability)
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
    root = np.sqrt(probability)
    gram = np.einsum("nlie,nlje->nlij", root, root, optimize=True)
    state = gram[:, :, 0, 1:]
    conditional = gram[:, :, 1:, 1:] - state[:, :, :, None] * state[:, :, None, :]
    return np.ascontiguousarray(conditional[:, :, UPPER[0], UPPER[1]])


def run_task(job: tuple[str, int, str, str, int]) -> tuple[int, np.ndarray, np.ndarray]:
    _, task_position, task, run_id, batch = job
    path = ROUTE_ROOT / task / run_id / "server/routes.zarr"
    if not path.is_dir():
        raise FileNotFoundError(path)
    group = zarr.open_group(str(path), mode="r")
    source = group["hb_router_probs"]
    if tuple(source.shape[1:]) != (8, 10, 11, 32):
        raise ValueError(f"unexpected HB shape at {path}: {source.shape}")
    count = int(source.shape[0])
    out = np.empty((count, 8, 55), dtype=np.float32)
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        out[start:stop] = conditional_upper(np.asarray(source[start:stop]))
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    return task_position, out, episode_id


def build_cohort(
    cohort: str, output: Path, batch: int, workers: int
) -> dict[str, Any]:
    cache = load_npz(COHORTS[cohort])
    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    lengths = cache["length"].astype(int)
    valid = cache["valid"].astype(bool)
    run_id = str(cache["run_id"])
    n_episode, max_query = valid.shape

    dense = np.full((n_episode, max_query, 8, 55), np.nan, dtype=np.float32)
    jobs = [
        (cohort, position, task, run_id, batch)
        for position, task in enumerate(task_names)
    ]
    context = mp.get_context("spawn")
    with context.Pool(processes=min(workers, len(jobs))) as pool:
        for done, (position, block, episode_id) in enumerate(
            pool.imap_unordered(run_task, jobs), start=1
        ):
            rows = np.flatnonzero(task_index == position)
            lookup = {int(e): int(r) for r, e in zip(rows, episodes[rows])}
            for episode in np.unique(episode_id):
                where = np.flatnonzero(episode_id == episode)
                if len(where) > 1 and not np.all(np.diff(where) == 1):
                    raise ValueError(f"non-contiguous episode {episode} in {position}")
                row = lookup.get(int(episode))
                if row is None:
                    raise ValueError(f"episode {episode} missing for {task_names[position]}")
                if len(where) != int(lengths[row]):
                    raise ValueError(
                        f"length mismatch {task_names[position]} ep {episode}: "
                        f"{len(where)} != {int(lengths[row])}"
                    )
                dense[row, : len(where)] = block[where]
            print(f"[{cohort} {done}/{len(jobs)}] {task_names[position]}", flush=True)

    if np.isnan(dense[valid]).any():
        raise ValueError(f"{cohort}: NaN on a valid query")
    if np.isfinite(dense[~valid]).any():
        raise ValueError(f"{cohort}: value written on an invalid query")

    packed = np.ascontiguousarray(dense[valid])
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f"{cohort}_conditional55.npy", packed)
    np.savez_compressed(
        output / f"{cohort}_index.npz",
        schema=np.asarray(SCHEMA),
        cohort=np.asarray(cohort),
        run_id=np.asarray(run_id),
        source_cache=np.asarray(str(COHORTS[cohort])),
        task_names=task_names,
        task_index=cache["task_index"],
        episode=cache["episode"],
        init_state_id=cache["init_state_id"],
        flow_noise_seed=cache["flow_noise_seed"],
        length=cache["length"],
        valid=valid,
        layer_names=np.asarray(LAYER_NAMES),
        upper_row=UPPER[0].astype(np.int16),
        upper_col=UPPER[1].astype(np.int16),
    )
    return {
        "cohort": cohort,
        "run_id": run_id,
        "episodes": int(n_episode),
        "tasks": int(len(task_names)),
        "valid_queries": int(valid.sum()),
        "packed_bytes": int((output / f"{cohort}_conditional55.npy").stat().st_size),
    }


def main() -> None:
    args = parse_args()
    records = [
        build_cohort(cohort, args.output, args.batch, args.workers)
        for cohort in COHORTS
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "build_summary.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "device": "cpu",
                "denoising_step": "final (index -1)",
                "stored": "upper triangle incl. diagonal of the 10x10 conditional kernel",
                "outcomes_loaded": False,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
