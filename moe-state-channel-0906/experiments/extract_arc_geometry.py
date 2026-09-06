#!/usr/bin/env python3
"""Extract the per-flow-step geometry of the action-token configuration.

The published step profiles (`moe-flow-semantics-0906/results/step_profiles`)
keep only rotation-invariant scalars: mean pairwise similarity, mean state
alignment, the Schur-complement trace.  Nothing there records the *ordering* of
the ten action tokens, so the "one-dimensional ordered arc" cannot be scored
from them.  This script re-reads `hb_router_probs` once and emits, per
(episode, query, layer, denoising step), the twelve quantities in
``arc_lib.ARC_NAMES``.

Row alignment, episode ordering, task list and validity mask are taken from the
frozen v4 layerwise-mobility caches, exactly as
`moe-flow-semantics-0906/experiments/extract_flow_steps.py` does, so the output
is row-for-row joinable with the published step profiles and with the label
CSVs.  No outcome, length, wall clock or task identity enters any quantity.

Output: `results/arc/{cohort}_arc.npy`  [episode, query, 8 layers, 10 steps, 12]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import zarr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arc_lib import ARC_NAMES, LAYER_NAMES, arc_features, normalize  # noqa: E402


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
V4 = PROJECT / "moe-v4-0904/results/layerwise_mobility"
DEFAULT_OUTPUT = BUNDLE / "results/arc"

COHORTS: dict[str, Path] = {
    "development_main": V4 / "main_reference.npz",
    "development_extra": V4 / "extra_reference.npz",
    "external_8b": V4 / "external_8b.npz",
}
N_LAYERS, N_STEPS, N_TOKENS, N_EXPERTS = 8, 10, 11, 32
BATCH = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--cohorts", nargs="*", default=list(COHORTS), choices=list(COHORTS))
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def process_task(job: dict[str, Any]) -> dict[str, Any]:
    group = zarr.open_group(job["route"], mode="r")
    source = group["hb_router_probs"]
    if tuple(source.shape[1:]) != (N_LAYERS, N_STEPS, N_TOKENS, N_EXPERTS):
        raise ValueError(f"unexpected router shape at {job['route']}: {source.shape}")
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    count = int(source.shape[0])
    if np.any(np.diff(episode_id) < 0):
        raise ValueError(f"episode_id is not non-decreasing in {job['route']}")

    episodes = np.asarray(job["episodes"], dtype=np.int64)
    lengths = np.asarray(job["lengths"], dtype=np.int64)
    starts = np.searchsorted(episode_id, episodes, side="left")
    stops = np.searchsorted(episode_id, episodes, side="right")
    if not np.array_equal(stops - starts, lengths) or int(lengths.sum()) != count:
        raise ValueError(f"episode lengths in {job['route']} disagree with the v4 cache")

    features = np.empty((count, N_LAYERS, N_STEPS, len(ARC_NAMES)), dtype=np.float32)
    batch = int(job["batch"])
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        root = np.sqrt(normalize(np.asarray(source[start:stop])))
        features[start:stop] = arc_features(root)

    dense = np.lib.format.open_memmap(
        Path(job["output"]) / f"{job['cohort']}_arc.npy", mode="r+"
    )
    for row, episode_start, length in zip(
        np.asarray(job["rows"], dtype=np.int64), starts, lengths, strict=True
    ):
        block = slice(int(episode_start), int(episode_start) + int(length))
        dense[row, :length] = features[block]
    dense.flush()
    return {"task": job["task"], "task_position": job["task_position"], "queries": count}


def build_jobs(cohort: str, output: Path, batch: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    cache = load_npz(COHORTS[cohort])
    task_names = cache["task_names"].astype(str)
    episodes = cache["episode"].astype(int)
    task_index = cache["task_index"].astype(int)
    run_id = str(cache["run_id"])
    max_query = cache["valid"].shape[1]

    array = np.lib.format.open_memmap(
        output / f"{cohort}_arc.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(episodes), max_query, N_LAYERS, N_STEPS, len(ARC_NAMES)),
    )
    array[...] = np.nan
    array.flush()
    del array

    jobs = []
    for task_position, task in enumerate(task_names):
        rows = np.flatnonzero(task_index == task_position)
        rows = rows[np.argsort(episodes[rows], kind="stable")]
        jobs.append(
            {
                "cohort": cohort,
                "task": task,
                "task_position": task_position,
                "route": str(ROUTE_ROOT / task / run_id / "server/routes.zarr"),
                "rows": rows,
                "episodes": episodes[rows],
                "lengths": cache["length"].astype(int)[rows],
                "output": str(output),
                "batch": batch,
            }
        )
    return jobs, cache


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ".gitignore").write_text("*.npy\n", encoding="utf-8")

    records = []
    context = mp.get_context("fork")
    for cohort in args.cohorts:
        jobs, cache = build_jobs(cohort, args.output, args.batch)
        with context.Pool(processes=min(args.workers, len(jobs))) as pool:
            done = 0
            for result in pool.imap_unordered(process_task, jobs):
                done += 1
                print(f"[{cohort} {done}/{len(jobs)}] {result['task']}", flush=True)

        valid = cache["valid"].astype(bool)
        block = np.load(args.output / f"{cohort}_arc.npy", mmap_mode="r")
        observed = np.isfinite(np.asarray(block[:, :, 0, 9, 0]))
        if not np.array_equal(observed, valid):
            raise ValueError(f"{cohort}: arc validity mask does not match the v4 cache")
        records.append(
            {
                "cohort": cohort,
                "run_id": str(cache["run_id"]),
                "episodes": int(len(cache["episode"])),
                "valid_queries": int(valid.sum()),
                "tasks": int(len(jobs)),
            }
        )
        np.savez_compressed(
            args.output / f"{cohort}_arc_index.npz",
            schema=np.asarray("himoe.state_channel.arc_index.v1"),
            cohort=np.asarray(cohort),
            run_id=cache["run_id"],
            task_names=cache["task_names"],
            task_index=cache["task_index"],
            episode=cache["episode"],
            init_state_id=cache["init_state_id"],
            flow_noise_seed=cache["flow_noise_seed"],
            length=cache["length"],
            valid=cache["valid"],
            layer_names=np.asarray(LAYER_NAMES),
            arc_names=np.asarray(ARC_NAMES),
        )
        print(json.dumps(records[-1], indent=2), flush=True)

    (args.output / "extraction_summary.json").write_text(
        json.dumps(
            {
                "schema": "himoe.state_channel.arc_extraction.v1",
                "source": "VLA_MUI_HUB/cache_new/HiMoE-VLA/<task>/<run_id>/server/routes.zarr :: hb_router_probs",
                "device": "cpu",
                "outcomes_loaded": False,
                "task_identity_used_inside_any_quantity": False,
                "arc_names": list(ARC_NAMES),
                "layer_names": list(LAYER_NAMES),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
