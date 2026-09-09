#!/usr/bin/env python3
"""Build outcome-blind routing-transfer profiles for right-50x8b."""

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
WORKSPACE = BUNDLE.parent
RAW_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
FEATURE_CACHE = (
    WORKSPACE
    / "double-selete/trainfree/results/online_precision_cascade_external"
    / "unlabeled_query_features.npz"
)
FIELD_ROOT = WORKSPACE / "moe-routing-transfer-v11-0905/transfer"
sys.path.insert(0, str(FIELD_ROOT))

from field import (  # noqa: E402
    QUERY_FEATURE_NAMES,
    SCHEMA,
    WITHIN_FEATURE_NAMES,
    compute_query_transfer,
    compute_within_transfer,
    episode_local_query,
)


RUN_ID = "right-50x8b-20260903"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BUNDLE / "results/graph_profiles/external_8b"
    )
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def task_inventory() -> list[dict[str, Any]]:
    with np.load(FEATURE_CACHE, allow_pickle=False) as cache:
        task_names = cache["task_names"].astype(str)
    jobs = []
    for task in task_names:
        suite, name = task.split("/", 1)
        route = RAW_ROOT / task / RUN_ID / "server/routes.zarr"
        if not route.is_dir():
            raise FileNotFoundError(route)
        group = zarr.open_group(str(route), mode="r")
        jobs.append(
            {
                "task": task,
                "suite": suite,
                "name": name,
                "route": str(route),
                "rows": int(group["hb_router_probs"].shape[0]),
            }
        )
    if len(jobs) != 39:
        raise ValueError(f"expected 39 tasks, found {len(jobs)}")
    return jobs


def allocate_terminal(
    count: int, values: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: (
            np.empty((count, front.shape[1]), dtype=np.float32),
            np.empty((count, back.shape[1]), dtype=np.float32),
        )
        for name, (front, back) in values.items()
    }


def store_terminal(
    destination: dict[str, tuple[np.ndarray, np.ndarray]],
    source: dict[str, tuple[np.ndarray, np.ndarray]],
    start: int,
    stop: int,
) -> None:
    for name, (front, back) in source.items():
        destination[name][0][start:stop] = front
        destination[name][1][start:stop] = back


def existing(path: Path, rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                str(archive["schema"]) == SCHEMA
                and archive["within_features"].shape
                == (rows, len(WITHIN_FEATURE_NAMES))
                and archive["query_features"].shape
                == (rows, len(QUERY_FEATURE_NAMES))
            )
    except Exception:
        return False


def build_one(
    job: dict[str, Any], output: Path, device: torch.device, batch: int, resume: bool
) -> dict[str, Any]:
    destination = output / job["suite"] / f"{job['name']}.npz"
    if resume and existing(destination, job["rows"]):
        return {
            **job,
            "profile": str(destination),
            "profile_bytes": destination.stat().st_size,
            "profile_sha256": sha256(destination),
            "resumed": True,
        }

    group = zarr.open_group(job["route"], mode="r")
    probability = group["hb_router_probs"]
    if tuple(probability.shape[1:]) != (8, 10, 11, 32):
        raise ValueError(f"unexpected router shape in {job['route']}")
    episode_id = np.asarray(group["episode_id"], dtype=np.int32)
    control_step = np.asarray(group["control_step"], dtype=np.int32)
    if len(np.unique(episode_id)) != 400:
        raise ValueError(f"expected 400 episodes in {job['task']}")
    count = len(episode_id)
    within = np.empty((count, len(WITHIN_FEATURE_NAMES)), dtype=np.float32)
    terminal_pairs = None
    terminal_channels = None
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        raw = np.asarray(probability[start:stop])
        tensor = torch.as_tensor(raw, dtype=torch.float32, device=device)
        computed = compute_within_transfer(tensor)
        within[start:stop] = computed["features"]
        if terminal_pairs is None:
            terminal_pairs = allocate_terminal(count, computed["terminal_pairs"])
            terminal_channels = allocate_terminal(count, computed["terminal_channels"])
        store_terminal(terminal_pairs, computed["terminal_pairs"], start, stop)
        store_terminal(terminal_channels, computed["terminal_channels"], start, stop)
        del raw, tensor, computed

    assert terminal_pairs is not None and terminal_channels is not None
    query_features = compute_query_transfer(
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
            within_feature_names=np.asarray(WITHIN_FEATURE_NAMES),
            within_features=within,
            query_feature_names=np.asarray(QUERY_FEATURE_NAMES),
            query_features=query_features,
        )
    os.replace(temporary, destination)
    return {
        **job,
        "profile": str(destination),
        "profile_bytes": destination.stat().st_size,
        "profile_sha256": sha256(destination),
        "resumed": False,
    }


def partition(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    partitions: list[list[dict[str, Any]]] = [[], []]
    loads = [0, 0]
    for job in sorted(jobs, key=lambda item: item["rows"], reverse=True):
        target = int(np.argmin(loads))
        partitions[target].append(job)
        loads[target] += job["rows"]
    return partitions


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
                f"[gpu{device_index} {position}/{len(jobs)}] {job['task']} "
                f"rows={job['rows']} size={record['profile_bytes'] / 2**20:.1f}MiB "
                f"resumed={record['resumed']}",
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
    except Exception as error:
        queue.put({"device": device_index, "error": repr(error)})


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("run with exactly two visible CUDA devices")
    jobs = task_inventory()
    partitions = partition(jobs)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(index, partitions[index], str(args.output), args.batch, args.resume, queue),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    payloads = [queue.get() for _ in processes]
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"worker exited with {process.exitcode}")
    failures = [payload for payload in payloads if "error" in payload]
    if failures:
        raise RuntimeError(str(failures))
    records = sorted(
        [record for payload in payloads for record in payload["records"]],
        key=lambda item: item["task"],
    )
    summary = {
        "schema": "himoe.feedback_responsiveness.external_graph_profiles.v1",
        "run_id": RUN_ID,
        "source": "original hb_router_probs",
        "raw_router_shape": [8, 10, 11, 32],
        "outcomes_loaded": False,
        "fitted_weights": False,
        "tasks": len(records),
        "episodes": 400 * len(records),
        "rows": sum(record["rows"] for record in records),
        "within_feature_count": len(WITHIN_FEATURE_NAMES),
        "query_feature_count": len(QUERY_FEATURE_NAMES),
        "output_bytes": sum(record["profile_bytes"] for record in records),
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
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "records"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
