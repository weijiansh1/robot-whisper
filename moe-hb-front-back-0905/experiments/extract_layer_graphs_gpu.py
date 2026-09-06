#!/usr/bin/env python3
"""Extract layer-resolved HB routing geometry on two GPUs.

The extractor never reads outcomes.  It retains all eight HB layers, all ten
flow steps, all eleven tokens, and the full 32-expert soft routing vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
LEGACY_ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache/HiMoE-VLA"
V4 = PROJECT / "moe-v4-0904/results/layerwise_mobility"
DEFAULT_OUTPUT = BUNDLE / "results/layer_graphs"

COHORTS: dict[str, dict[str, Any]] = {
    "development_main": {"cache": V4 / "main_reference.npz", "root": ROUTE_ROOT},
    "development_extra": {"cache": V4 / "extra_reference.npz", "root": ROUTE_ROOT},
    "external_8b": {"cache": V4 / "external_8b.npz", "root": ROUTE_ROOT},
    "legacy_main16x32": {
        "cache": PROJECT / "moe-v4-0904/results/cache16x32_v4/layerwise_mobility.npz",
        "root": LEGACY_ROUTE_ROOT,
        "run_id": "right-16x32",
    },
}
METRIC_NAMES = (
    "flow_path",
    "flow_settling_log_ratio",
    "flow_endpoint",
    "action_consensus",
    "state_action_alignment",
    "conditional_energy",
    "conditional_effective_rank",
    "partial_edge_std",
    "expert_load_effective_rank",
    "conditional_query_d1",
    "partial_query_d1",
)
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def normalize(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp_min(0.0)
    return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def entropy_effective_rank(mass: torch.Tensor, maximum: int) -> torch.Tensor:
    mass = mass.clamp_min(0.0)
    probability = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy.exp() / float(maximum)


def compute_batch(
    raw: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability = normalize(torch.as_tensor(raw, dtype=torch.float32, device=device))
    root = probability.sqrt()

    action_root = root[:, :, :, 1:, :]
    flow_speed = (
        torch.linalg.vector_norm(
            action_root[:, :, 1:] - action_root[:, :, :-1], dim=-1
        )
        / math.sqrt(2.0)
    ).mean(dim=-1)
    flow_path = flow_speed.sum(dim=-1)
    flow_settling = torch.log(
        (flow_speed[:, :, -3:].mean(dim=-1) + EPSILON)
        / (flow_speed[:, :, :3].mean(dim=-1) + EPSILON)
    )
    flow_endpoint = (
        torch.linalg.vector_norm(
            action_root[:, :, -1] - action_root[:, :, 0], dim=-1
        )
        / math.sqrt(2.0)
    ).mean(dim=-1)

    terminal = root[:, :, -1]
    gram = torch.matmul(terminal, terminal.transpose(-1, -2))
    action_gram = gram[:, :, 1:, 1:]
    state_action = gram[:, :, 0, 1:]
    conditional = action_gram - state_action.unsqueeze(-1) * state_action.unsqueeze(-2)
    diagonal = torch.diagonal(conditional, dim1=-2, dim2=-1).clamp_min(0.0)
    denominator = torch.sqrt(
        diagonal.unsqueeze(-1) * diagonal.unsqueeze(-2)
    ).clamp_min(EPSILON)
    partial = (conditional / denominator).clamp(-1.0, 1.0)

    offdiag = torch.triu_indices(10, 10, offset=1, device=device)
    upper = torch.triu_indices(10, 10, offset=0, device=device)
    action_edges = action_gram[..., offdiag[0], offdiag[1]]
    conditional_edges = conditional[..., offdiag[0], offdiag[1]]
    partial_edges = partial[..., offdiag[0], offdiag[1]]
    eigenvalues = torch.linalg.eigvalsh(conditional).clamp_min(0.0)
    expert_load = probability[:, :, -1, 1:, :].mean(dim=-2)

    base = torch.stack(
        (
            flow_path,
            flow_settling,
            flow_endpoint,
            action_edges.mean(dim=-1),
            state_action.mean(dim=-1),
            diagonal.mean(dim=-1),
            entropy_effective_rank(eigenvalues, 10),
            partial_edges.std(dim=-1, unbiased=False),
            entropy_effective_rank(expert_load, 32),
        ),
        dim=-1,
    )
    conditional_vector = conditional[..., upper[0], upper[1]] / math.sqrt(55.0)
    partial_vector = partial_edges / math.sqrt(45.0)
    return tuple(
        value.detach().cpu().numpy().astype(np.float32, copy=False)
        for value in (
            base,
            conditional_vector,
            partial_vector,
            conditional_edges,
            partial_edges,
        )
    )


def validate_existing(path: Path, expected_episodes: int, max_query: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["schema"]) != "himoe.hb_layer_graphs.v2":
                return None
            if archive["metrics"].shape != (
                expected_episodes,
                max_query,
                8,
                len(METRIC_NAMES),
            ):
                return None
            tasks = len(archive["task_names"])
            valid_queries = int(archive["valid"].sum())
    except Exception:
        return None
    return {
        "profile": str(path),
        "profile_bytes": path.stat().st_size,
        "episodes": expected_episodes,
        "tasks": tasks,
        "queries": valid_queries,
        "resumed": True,
    }


def build_cohort(
    cohort: str,
    cache_path: Path,
    route_root: Path,
    fixed_run_id: str | None,
    output: Path,
    device: torch.device,
    batch: int,
    resume: bool,
) -> dict[str, Any]:
    cache = load_npz(cache_path)
    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    lengths = cache["length"].astype(int)
    valid = cache["valid"].astype(bool)
    max_query = valid.shape[1]
    destination = output / f"{cohort}.npz"
    if resume:
        existing = validate_existing(destination, len(episodes), max_query)
        if existing is not None:
            existing["cohort"] = cohort
            return existing

    dense = np.full(
        (len(episodes), max_query, 8, len(METRIC_NAMES)), np.nan, dtype=np.float32
    )
    conditional_edge_mean = np.full((len(task_names), 8, 45), np.nan, np.float32)
    partial_edge_mean = np.full_like(conditional_edge_mean, np.nan)
    run_id = fixed_run_id if fixed_run_id is not None else str(cache["run_id"])
    for task_position, task in enumerate(task_names):
        global_rows = np.flatnonzero(task_index == task_position)
        task_dir = route_root / task / run_id
        route_path = task_dir / "server/routes.zarr"
        if not route_path.is_dir():
            raise FileNotFoundError(route_path)
        group = zarr.open_group(str(route_path), mode="r")
        source = group["hb_router_probs"]
        if tuple(source.shape[1:]) != (8, 10, 11, 32):
            raise ValueError(f"unexpected HB shape at {route_path}: {source.shape}")
        raw_episode = np.asarray(group["episode_id"][:], dtype=int)
        count = len(raw_episode)
        base_rows = np.empty((count, 8, 9), dtype=np.float32)
        conditional_vectors = np.empty((count, 8, 55), dtype=np.float32)
        partial_vectors = np.empty((count, 8, 45), dtype=np.float32)
        conditional_sum = np.zeros((8, 45), dtype=np.float64)
        partial_sum = np.zeros((8, 45), dtype=np.float64)
        for start in range(0, count, batch):
            stop = min(start + batch, count)
            computed = compute_batch(np.asarray(source[start:stop]), device)
            base, conditional, partial, conditional_edges, partial_edges = computed
            base_rows[start:stop] = base
            conditional_vectors[start:stop] = conditional
            partial_vectors[start:stop] = partial
            conditional_sum += conditional_edges.sum(axis=0, dtype=np.float64)
            partial_sum += partial_edges.sum(axis=0, dtype=np.float64)

        conditional_edge_mean[task_position] = conditional_sum / count
        partial_edge_mean[task_position] = partial_sum / count
        task_lookup = {int(episode): int(row) for row, episode in zip(global_rows, episodes[global_rows])}
        for episode in np.unique(raw_episode):
            positions = np.flatnonzero(raw_episode == episode)
            if len(positions) > 1 and not np.all(np.diff(positions) == 1):
                raise ValueError(f"non-contiguous episode {episode} in {route_path}")
            row = task_lookup.get(int(episode))
            if row is None:
                raise ValueError(f"episode {episode} missing from cache for {task}")
            length = int(lengths[row])
            if len(positions) != length:
                raise ValueError(
                    f"length mismatch for {task} episode {episode}: {len(positions)} != {length}"
                )
            dense[row, :length, :, :9] = base_rows[positions]
            if length > 1:
                conditional_d1 = np.linalg.norm(
                    conditional_vectors[positions[1:]] - conditional_vectors[positions[:-1]], axis=-1
                )
                partial_d1 = np.linalg.norm(
                    partial_vectors[positions[1:]] - partial_vectors[positions[:-1]], axis=-1
                )
                dense[row, 1:length, :, 9] = conditional_d1
                dense[row, 1:length, :, 10] = partial_d1
        print(
            f"[{cohort} {task_position + 1}/{len(task_names)}] {task} rows={count}",
            flush=True,
        )

    if not np.isfinite(dense[..., :9][valid[:, :, None, None].repeat(8, axis=2).repeat(9, axis=3)]).all():
        raise ValueError(f"{cohort}: non-finite base metrics on valid queries")
    expected_d1 = valid.copy()
    expected_d1[:, 0] = False
    observed_d1 = np.isfinite(dense[..., 9:]).all(axis=(2, 3))
    if not np.array_equal(expected_d1, observed_d1):
        raise ValueError(f"{cohort}: query-difference mask mismatch")

    output.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray("himoe.hb_layer_graphs.v2"),
            cohort=np.asarray(cohort),
            source_cache=np.asarray(str(cache_path)),
            run_id=np.asarray(run_id),
            task_names=task_names,
            task_index=cache["task_index"],
            episode=cache["episode"],
            init_state_id=cache["init_state_id"],
            flow_noise_seed=cache["flow_noise_seed"],
            length=cache["length"],
            valid=valid,
            layer_names=np.asarray(LAYER_NAMES),
            metric_names=np.asarray(METRIC_NAMES),
            metrics=dense,
            conditional_edge_mean=conditional_edge_mean,
            partial_edge_mean=partial_edge_mean,
        )
    os.replace(temporary, destination)
    return {
        "cohort": cohort,
        "profile": str(destination),
        "profile_bytes": destination.stat().st_size,
        "episodes": len(episodes),
        "tasks": len(task_names),
        "queries": int(valid.sum()),
        "run_id": run_id,
        "resumed": False,
    }


def worker(
    device_index: int,
    cohorts: list[str],
    output: str,
    batch: int,
    resume: bool,
    queue: mp.Queue,
) -> None:
    try:
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        records = [
            build_cohort(
                cohort,
                COHORTS[cohort]["cache"],
                COHORTS[cohort]["root"],
                COHORTS[cohort].get("run_id"),
                Path(output),
                device,
                batch,
                resume,
            )
            for cohort in cohorts
        ]
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices are required")
    context = mp.get_context("spawn")
    queue = context.Queue()
    assignments = (
        ("development_main", "development_extra"),
        ("external_8b", "legacy_main16x32"),
    )
    processes = [
        context.Process(
            target=worker,
            args=(device, list(cohorts), str(args.output), args.batch, args.resume, queue),
        )
        for device, cohorts in enumerate(assignments)
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
    records = [record for payload in payloads for record in payload["records"]]
    summary = {
        "schema": "himoe.hb_layer_graph_build.v2",
        "raw_router_shape": [8, 10, 11, 32],
        "outcomes_loaded": False,
        "fitted_weights": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "metrics": list(METRIC_NAMES),
        "episodes": sum(record["episodes"] for record in records),
        "queries": sum(record["queries"] for record in records),
        "records": sorted(records, key=lambda record: record["cohort"]),
        "workers": [
            {key: value for key, value in payload.items() if key != "records"}
            for payload in payloads
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
