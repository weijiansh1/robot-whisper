#!/usr/bin/env python3
"""Build outcome-free graph-rich assurance profiles on two GPUs."""

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


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
SOURCE_ROOT = WORKSPACE / "analysis_moe_phenotype/features"
DEFAULT_OUTPUT = BUNDLE / "results/profiles"
ASSURANCE = BUNDLE / "assurance"
sys.path.insert(0, str(ASSURANCE))

from profile_schema import LAYER_NAMES, SCHEMA, episode_local_query  # noqa: E402


CORPORA = ("main16x32", "grid50x8")
RAW_FEATURES = (
    "late_flow_volatility",
    "route_acceleration",
    "gate_entropy",
    "top12_margin",
    "token_consensus",
    "token_dispersion",
    "layer_disagreement",
    "state_action_gap",
    "flow_com",
    "flow_total",
    "mob1_w8",
)
LAYER_FEATURES = (
    "commitment",
    "entropy_normalized",
    "top1_mass",
    "top4_mass",
    "top12_margin",
    "token_consensus",
    "token_dispersion",
    "effective_rank",
    "stable_rank",
    "token_expert_mi",
    "occupancy_concentration",
    "occupancy_entropy",
    "support_union_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch", type=int, default=1024)
    return parser.parse_args()


def discover() -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for corpus in CORPORA:
        root = SOURCE_ROOT / corpus
        for rows_path in sorted(root.glob("*/*/rows.npz")):
            jobs.append(
                {
                    "corpus": corpus,
                    "suite": rows_path.parent.parent.name,
                    "task": rows_path.parent.name,
                    "rows": str(rows_path),
                    "reps": str(rows_path.with_name("reps.npy")),
                }
            )
    expected = {"main16x32": 5, "grid50x8": 40}
    observed = {corpus: sum(job["corpus"] == corpus for job in jobs) for corpus in CORPORA}
    if observed != expected:
        raise RuntimeError(f"profile source inventory mismatch: {observed}")
    return jobs


def entropy(probability: torch.Tensor) -> torch.Tensor:
    return -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)


def graph_features(reps: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    output = np.empty((len(reps), len(LAYER_NAMES), len(LAYER_FEATURES)), dtype=np.float32)
    pair = torch.triu_indices(10, 10, offset=1, device=device)
    log_experts = float(np.log(32.0))
    for start in range(0, len(reps), batch):
        stop = min(start + batch, len(reps))
        root = torch.as_tensor(
            np.asarray(reps[start:stop], dtype=np.float32).reshape(-1, 4, 10, 32),
            dtype=torch.float32,
            device=device,
        )
        probability = root.square()
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        ordered, ids = torch.sort(probability, dim=-1, descending=True)
        row_entropy = entropy(probability)
        mean_distribution = probability.mean(dim=2)
        mean_entropy = row_entropy.mean(dim=2)

        left = probability[:, :, pair[0], :]
        right = probability[:, :, pair[1], :]
        wj = torch.minimum(left, right).sum(dim=-1) / torch.maximum(left, right).sum(
            dim=-1
        ).clamp_min(1e-12)
        affinity = torch.sqrt(left * right).sum(dim=-1)
        dispersion = torch.sqrt((1.0 - affinity).clamp(0.0, 1.0))

        singular = torch.linalg.svdvals(probability)
        singular_mass = singular / singular.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        effective_rank = torch.exp(entropy(singular_mass))
        stable_rank = singular.square().sum(dim=-1) / singular[..., 0].square().clamp_min(1e-12)

        top4_ids = ids[..., :4]
        occupancy = torch.nn.functional.one_hot(top4_ids, num_classes=32).sum(dim=(2, 3)).float()
        occupancy_fraction = occupancy / 40.0
        occupied = occupancy > 0
        occupancy_entropy = entropy(occupancy_fraction) / log_experts

        block = torch.stack(
            (
                1.0 - mean_entropy / log_experts,
                mean_entropy / log_experts,
                ordered[..., 0].mean(dim=2),
                ordered[..., :4].sum(dim=-1).mean(dim=2),
                (ordered[..., 0] - ordered[..., 1]).mean(dim=2),
                wj.mean(dim=2),
                dispersion.mean(dim=2),
                effective_rank,
                stable_rank,
                (entropy(mean_distribution) - mean_entropy) / log_experts,
                occupancy_fraction.max(dim=-1).values,
                occupancy_entropy,
                occupied.sum(dim=-1).float() / 32.0,
            ),
            dim=-1,
        )
        output[start:stop] = block.detach().cpu().numpy()
    torch.cuda.synchronize(device)
    return output


def build_one(job: dict[str, str], output_root: Path, device: torch.device, batch: int) -> dict[str, Any]:
    rows_path = Path(job["rows"])
    reps_path = Path(job["reps"])
    with np.load(rows_path, allow_pickle=False) as source:
        allowed = set(RAW_FEATURES) | {
            "episode_id",
            "control_step",
            "scene",
            "repeat",
            "flow_profile",
            "mob_k",
        }
        data = {name: np.asarray(source[name]) for name in source.files if name in allowed}
    missing = allowed - set(data)
    if missing:
        raise ValueError(f"{rows_path} is missing {sorted(missing)}")
    reps = np.load(reps_path, mmap_mode="r")
    if reps.shape != (len(data["episode_id"]), 1280):
        raise ValueError(f"representation shape mismatch in {reps_path}: {reps.shape}")
    layer = graph_features(reps, device, batch)

    aggregate_names = list(RAW_FEATURES)
    aggregate = [np.asarray(data[name], dtype=np.float32) for name in RAW_FEATURES]
    for index, name in enumerate(LAYER_FEATURES):
        aggregate_names.extend((f"graph_{name}_mean", f"graph_{name}_layer_std"))
        aggregate.extend((layer[:, :, index].mean(axis=1), layer[:, :, index].std(axis=1)))
    recurrence = np.asarray(data["mob_k"], dtype=np.float32)
    lag1 = recurrence[:, 0]
    best_long = np.nanmin(recurrence[:, 1:4], axis=1)
    aggregate_names.extend(("lag1_distance", "best_lag2_4_distance", "long_lag_return_advantage"))
    aggregate.extend((lag1, best_long, lag1 - best_long))
    features = np.column_stack(aggregate).astype(np.float32)

    destination = output_root / job["corpus"] / job["suite"] / f"{job['task']}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema=np.asarray(SCHEMA),
        episode_id=np.asarray(data["episode_id"], dtype=np.int32),
        query=episode_local_query(data["episode_id"], data["control_step"]),
        scene=np.asarray(data["scene"], dtype=np.int16),
        repeat=np.asarray(data["repeat"], dtype=np.int16),
        feature_names=np.asarray(aggregate_names),
        features=features,
        layer_names=np.asarray(LAYER_NAMES),
        layer_feature_names=np.asarray(LAYER_FEATURES),
        layer_features=layer.astype(np.float16),
        flow_profile=np.asarray(data["flow_profile"], dtype=np.float32),
        recurrence_distance=recurrence,
    )
    return {
        "corpus": job["corpus"],
        "suite": job["suite"],
        "task": job["task"],
        "rows": len(features),
        "episodes": len(np.unique(data["episode_id"])),
        "profile": str(destination),
        "profile_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def worker(device_index: int, jobs: list[dict[str, str]], output: str, batch: int, queue: mp.Queue) -> None:
    try:
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        records = [build_one(job, Path(output), device, batch) for job in jobs]
        queue.put(
            {
                "device": device_index,
                "device_name": torch.cuda.get_device_name(device),
                "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
                "records": records,
            }
        )
    except Exception as error:  # pragma: no cover
        queue.put({"device": device_index, "error": repr(error)})


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices are required")
    jobs = discover()
    context = mp.get_context("spawn")
    queue = context.Queue()
    partitions = [jobs[::2], jobs[1::2]]
    processes = [
        context.Process(target=worker, args=(device, partitions[device], str(args.output), args.batch, queue))
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
        "schema": "himoe.moe_assurance_profile_build.v1",
        "outcomes_loaded": False,
        "task_conditioned_features": False,
        "source_representation": "sqrt soft-router probabilities at L12-L15/final-flow/action tokens",
        "raw_flow_features_reused": list(RAW_FEATURES),
        "graph_layer_features": list(LAYER_FEATURES),
        "aggregate_feature_count": len(RAW_FEATURES) + 2 * len(LAYER_FEATURES) + 3,
        "tasks": len(records),
        "episodes": sum(record["episodes"] for record in records),
        "rows": sum(record["rows"] for record in records),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "workers": [
            {key: value for key, value in payload.items() if key != "records"}
            for payload in payloads
        ],
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "build_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "records"},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
