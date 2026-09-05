#!/usr/bin/env python3
"""Build coupled routing-transfer profiles from original HiMoE router tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
SOURCE_FEATURES = PROJECT / "analysis_moe_phenotype/features"
DEFAULT_OUTPUT = BUNDLE / "results/transfer_profiles"
sys.path.insert(0, str(BUNDLE / "transfer"))

from field import (  # noqa: E402
    CHANNEL_NAMES,
    QUERY_FEATURE_NAMES,
    RELATION_VIEWS,
    SCHEMA,
    WITHIN_FEATURE_NAMES,
    compute_query_transfer,
    compute_within_transfer,
    episode_local_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(limit: int | None) -> list[dict[str, Any]]:
    jobs = []
    for meta_path in sorted(SOURCE_FEATURES.glob("*/*/*/meta.json")):
        corpus, suite, task = meta_path.parts[-4:-1]
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        route_path = Path(metadata["task_dir"]) / "server/routes.zarr"
        rows_path = meta_path.with_name("rows.npz")
        if not route_path.is_dir() or not rows_path.is_file():
            raise FileNotFoundError(f"missing source for {corpus}/{suite}/{task}")
        jobs.append(
            {
                "corpus": corpus,
                "suite": suite,
                "task": task,
                "rows": str(rows_path),
                "meta": str(meta_path),
                "routes": str(route_path),
                "count": int(metadata["rows"]),
                "episodes": int(metadata["episodes"]),
            }
        )
    observed = {
        corpus: sum(job["corpus"] == corpus for job in jobs)
        for corpus in ("main16x32", "grid50x8")
    }
    if observed != {"main16x32": 5, "grid50x8": 40}:
        raise RuntimeError(f"source inventory mismatch: {observed}")
    return jobs[:limit] if limit is not None else jobs


def aligned_positions(
    group: zarr.Group, episode_id: np.ndarray, control_step: np.ndarray
) -> np.ndarray:
    raw_episode = np.asarray(group["episode_id"], dtype=np.int64)
    raw_step = np.asarray(group["control_step"], dtype=np.int64)
    raw_key = (raw_episode << 32) | raw_step
    target_key = (episode_id.astype(np.int64) << 32) | control_step.astype(np.int64)
    order = np.argsort(raw_key, kind="stable")
    ordered = raw_key[order]
    location = np.searchsorted(ordered, target_key)
    if np.any(location >= len(ordered)) or not np.array_equal(ordered[location], target_key):
        raise ValueError("profile rows do not align to raw router rows")
    positions = order[location]
    if len(np.unique(positions)) != len(positions):
        raise ValueError("raw row alignment is not one-to-one")
    return positions


def read_probability(array: zarr.Array, positions: np.ndarray) -> np.ndarray:
    if len(positions) and np.all(np.diff(positions) == 1):
        return np.asarray(array[int(positions[0]) : int(positions[-1]) + 1])
    return np.asarray(array.oindex[positions, :, :, :, :])


def existing_record(job: dict[str, Any], destination: Path) -> dict[str, Any] | None:
    if not destination.is_file():
        return None
    try:
        with np.load(destination, allow_pickle=False) as archive:
            if str(archive["schema"]) != SCHEMA:
                return None
            if archive["within_features"].shape != (job["count"], len(WITHIN_FEATURE_NAMES)):
                return None
            if archive["query_features"].shape != (job["count"], len(QUERY_FEATURE_NAMES)):
                return None
            episodes = len(np.unique(archive["episode_id"]))
    except Exception:
        return None
    return {
        "corpus": job["corpus"],
        "suite": job["suite"],
        "task": job["task"],
        "rows": job["count"],
        "episodes": episodes,
        "raw_routes": job["routes"],
        "source_meta_sha256": sha256(Path(job["meta"])),
        "profile": str(destination),
        "profile_bytes": destination.stat().st_size,
        "profile_sha256": sha256(destination),
        "resumed": True,
    }


def _allocate_terminal(
    count: int,
    values: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: (
            np.empty((count, front.shape[1]), dtype=np.float32),
            np.empty((count, back.shape[1]), dtype=np.float32),
        )
        for name, (front, back) in values.items()
    }


def _store_terminal(
    destination: dict[str, tuple[np.ndarray, np.ndarray]],
    source: dict[str, tuple[np.ndarray, np.ndarray]],
    start: int,
    stop: int,
) -> None:
    for name, (front, back) in source.items():
        destination[name][0][start:stop] = front
        destination[name][1][start:stop] = back


def build_one(
    job: dict[str, Any],
    output_root: Path,
    device: torch.device,
    batch: int,
    resume: bool,
) -> dict[str, Any]:
    destination = output_root / job["corpus"] / job["suite"] / f"{job['task']}.npz"
    if resume:
        record = existing_record(job, destination)
        if record is not None:
            return record

    with np.load(job["rows"], allow_pickle=False) as rows:
        episode_id = np.asarray(rows["episode_id"], dtype=np.int32)
        control_step = np.asarray(rows["control_step"], dtype=np.int32)
        scene = np.asarray(rows["scene"], dtype=np.int16)
        repeat = np.asarray(rows["repeat"], dtype=np.int16)
    group = zarr.open_group(job["routes"], mode="r")
    probability_array = group["hb_router_probs"]
    if tuple(probability_array.shape[1:]) != (8, 10, 11, 32):
        raise ValueError(f"unexpected raw router shape in {job['routes']}")
    positions = aligned_positions(group, episode_id, control_step)
    count = len(positions)
    within = np.empty((count, len(WITHIN_FEATURE_NAMES)), dtype=np.float32)
    terminal_pairs = None
    terminal_channels = None

    for start in range(0, count, batch):
        stop = min(start + batch, count)
        raw = read_probability(probability_array, positions[start:stop])
        tensor = torch.as_tensor(raw, dtype=torch.float32, device=device)
        computed = compute_within_transfer(tensor)
        within[start:stop] = computed["features"]
        if terminal_pairs is None:
            terminal_pairs = _allocate_terminal(count, computed["terminal_pairs"])
            terminal_channels = _allocate_terminal(count, computed["terminal_channels"])
        _store_terminal(terminal_pairs, computed["terminal_pairs"], start, stop)
        _store_terminal(terminal_channels, computed["terminal_channels"], start, stop)
        del raw, tensor, computed

    assert terminal_pairs is not None and terminal_channels is not None
    query = compute_query_transfer(
        terminal_pairs, terminal_channels, within, episode_id, control_step
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(SCHEMA),
            episode_id=episode_id,
            query=episode_local_query(episode_id, control_step),
            control_step=control_step,
            scene=scene,
            repeat=repeat,
            within_feature_names=np.asarray(WITHIN_FEATURE_NAMES),
            within_features=within,
            query_feature_names=np.asarray(QUERY_FEATURE_NAMES),
            query_features=query,
        )
    os.replace(temporary, destination)
    return {
        "corpus": job["corpus"],
        "suite": job["suite"],
        "task": job["task"],
        "rows": count,
        "episodes": len(np.unique(episode_id)),
        "raw_routes": job["routes"],
        "source_meta_sha256": sha256(Path(job["meta"])),
        "profile": str(destination),
        "profile_bytes": destination.stat().st_size,
        "profile_sha256": sha256(destination),
        "resumed": False,
    }


def worker(
    device_index: int,
    jobs: list[dict[str, Any]],
    output: str,
    batch: int,
    resume: bool,
    queue: mp.Queue,
) -> None:
    try:
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        records = []
        for position, job in enumerate(jobs, 1):
            record = build_one(job, Path(output), device, batch, resume)
            records.append(record)
            print(
                f"[gpu{device_index} {position}/{len(jobs)}] "
                f"{job['corpus']}/{job['suite']}/{job['task']} rows={record['rows']} "
                f"size={record['profile_bytes'] / 2**20:.1f}MiB resumed={record['resumed']}",
                flush=True,
            )
        queue.put(
            {
                "device": device_index,
                "device_name": torch.cuda.get_device_name(device),
                "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "records": records,
            }
        )
    except Exception as error:  # pragma: no cover
        queue.put({"device": device_index, "error": repr(error)})


def balanced_partitions(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    partitions: list[list[dict[str, Any]]] = [[], []]
    loads = [0, 0]
    for job in sorted(jobs, key=lambda item: item["count"], reverse=True):
        target = int(np.argmin(loads))
        partitions[target].append(job)
        loads[target] += job["count"]
    return partitions


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices are required")
    jobs = discover(args.limit_tasks)
    partitions = balanced_partitions(jobs)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(device, partitions[device], str(args.output), args.batch, args.resume, queue),
        )
        for device in range(2)
    ]
    for process in processes:
        process.start()
    payloads = [queue.get() for _ in processes]
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"profile worker failed with exit code {process.exitcode}")
    failures = [payload for payload in payloads if "error" in payload]
    if failures:
        raise RuntimeError(str(failures))
    records = sorted(
        [record for payload in payloads for record in payload["records"]],
        key=lambda row: (row["corpus"], row["suite"], row["task"]),
    )
    summary = {
        "schema": "himoe.routing_transfer_profile_build.v2",
        "raw_router_shape": [8, 10, 11, 32],
        "source": "original server/routes.zarr hb_router_probs",
        "relation_views": list(RELATION_VIEWS),
        "channels": list(CHANNEL_NAMES),
        "outcomes_loaded": False,
        "fitted_weights": False,
        "task_conditioned_features": False,
        "storage": "float32 deflate",
        "within_feature_count": len(WITHIN_FEATURE_NAMES),
        "query_feature_count": len(QUERY_FEATURE_NAMES),
        "total_dynamic_axes": len(WITHIN_FEATURE_NAMES) + len(QUERY_FEATURE_NAMES),
        "tasks": len(records),
        "episodes": sum(record["episodes"] for record in records),
        "rows": sum(record["rows"] for record in records),
        "output_bytes": sum(record["profile_bytes"] for record in records),
        "resumed_profiles": sum(bool(record["resumed"]) for record in records),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "workers": [
            {key: value for key, value in payload.items() if key != "records"}
            for payload in payloads
        ],
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
