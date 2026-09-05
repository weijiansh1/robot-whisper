#!/usr/bin/env python3
"""Replay raw router prefixes on two GPUs and verify the sealed v7 alarms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
METHOD = BUNDLE / "method"
sys.path.insert(0, str(METHOD))

from intrinsic_guard_monitor import (  # noqa: E402
    ACTION,
    BACK,
    FINAL_FLOW,
    LAGS,
    GlobalIntrinsicProfile,
    IntrinsicGuardMonitor,
    intrinsic_score_arrays,
)


CACHE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
LAYER_CACHE = WORKSPACE / "moe-v4-0904/results/layerwise_mobility/external_8b.npz"
FEATURE_CACHE = (
    WORKSPACE
    / "double-selete/trainfree/results/online_precision_cascade_external/unlabeled_query_features.npz"
)
DEFAULT_RESULT_ROOT = BUNDLE / "results/intrinsic_guard_v7"
RUN_ID = "right-50x8b-20260903"
FEATURE_TOLERANCE = 2e-5
SCORE_TOLERANCE = 3e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--samples-per-class", type=int, default=8)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def feature(cache: dict[str, np.ndarray], name: str) -> np.ndarray:
    names = cache["feature_names"].astype(str).tolist()
    return np.asarray(cache["features"][:, :, names.index(name)], dtype=np.float32)


def finite_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return float("inf")
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def evenly_spaced(indices: np.ndarray, count: int) -> list[int]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < count:
        raise ValueError(f"need {count} samples, only {len(indices)} are available")
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    return indices[positions].astype(int).tolist()


def select_rows(alarms: dict[str, np.ndarray], count: int) -> list[tuple[int, str]]:
    freeze = alarms["external_freeze"] >= 0
    turbulence = alarms["external_turbulence"] >= 0
    groups = {
        "freeze_only": np.flatnonzero(freeze & ~turbulence),
        "turbulence_only": np.flatnonzero(turbulence & ~freeze),
        "no_alarm": np.flatnonzero(~freeze & ~turbulence),
    }
    selected: list[tuple[int, str]] = []
    for group, rows in groups.items():
        selected.extend((row, group) for row in evenly_spaced(rows, count))
    if len({row for row, _ in selected}) != len(selected):
        raise RuntimeError("sample classes unexpectedly overlap")
    return selected


def torch_features(raw: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = torch.as_tensor(raw, dtype=torch.float32, device=device).clamp_min_(0.0)
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min_(1e-12)
    final_action = probability[:, :, FINAL_FLOW, ACTION, :]

    mobility = torch.full(
        (len(raw), final_action.shape[1]), float("nan"), dtype=torch.float32, device=device
    )
    if len(raw) > 1:
        affinity = torch.sqrt(final_action[1:] * final_action[:-1]).sum(dim=-1)
        mobility[1:] = torch.sqrt((1.0 - affinity).clamp(0.0, 1.0)).mean(dim=2)

    action_flow = probability[:, BACK, :, ACTION, :]
    root = torch.sqrt(action_flow)
    second = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
    acceleration = torch.linalg.vector_norm(second, dim=-1).mean(dim=(1, 2, 3)) / math.sqrt(2.0)

    action_route = final_action[:, BACK].reshape(len(raw), 40, 32)
    similarities = torch.full(
        (len(raw), len(LAGS)), float("nan"), dtype=torch.float32, device=device
    )
    for lag_index, lag in enumerate(LAGS):
        if len(raw) <= lag:
            continue
        left = action_route[lag:]
        right = action_route[:-lag]
        numerator = torch.minimum(left, right).sum(dim=-1)
        denominator = torch.maximum(left, right).sum(dim=-1).clamp_min_(1e-12)
        similarities[lag:, lag_index] = (numerator / denominator).mean(dim=1)
    periodicity = torch.full((len(raw),), float("nan"), dtype=torch.float32, device=device)
    if len(raw) > 2:
        long_lag = torch.nan_to_num(similarities[:, 1:], nan=-torch.inf).amax(dim=1)
        good = torch.isfinite(similarities[:, 0]) & torch.isfinite(long_lag)
        periodicity[good] = long_lag[good] - similarities[good, 0]

    torch.cuda.synchronize(device)
    return tuple(
        value.detach().cpu().numpy().astype(np.float32, copy=False)
        for value in (mobility, acceleration, periodicity)
    )


def load_raw_episode(task: str, episode: int, expected_length: int) -> tuple[np.ndarray, str]:
    route_path = CACHE_ROOT / task / RUN_ID / "server/routes.zarr"
    group = zarr.open_group(str(route_path), mode="r")
    episode_ids = np.asarray(group["episode_id"][:], dtype=np.int32)
    positions = np.flatnonzero(episode_ids == episode)
    if len(positions) != expected_length or not np.all(np.diff(positions) == 1):
        raise ValueError(f"raw episode mismatch for {task} episode {episode}")
    raw = np.asarray(
        group["hb_router_probs"][positions[0] : positions[-1] + 1], dtype=np.float16
    )
    digest = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
    return raw, digest


def replay_monitor(raw: np.ndarray, profile_path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    monitor = IntrinsicGuardMonitor(GlobalIntrinsicProfile.load(profile_path))
    records = [monitor.update(query) for query in raw]
    streams = {
        "mobility": np.stack([row["layer_mobility"] for row in records]),
        "acceleration": np.asarray([row["route_acceleration"] for row in records], dtype=np.float32),
        "periodicity": np.asarray([row["lag_periodicity"] for row in records], dtype=np.float32),
        "freeze_score": np.asarray([row["freeze_score"] for row in records], dtype=np.float32),
        "acceleration_score": np.asarray(
            [row["acceleration_score"] for row in records], dtype=np.float32
        ),
        "periodicity_score": np.asarray(
            [row["periodicity_score"] for row in records], dtype=np.float32
        ),
    }
    first = {
        "freeze": monitor.first_freeze_query,
        "acceleration": monitor.first_acceleration_query,
        "periodicity": monitor.first_periodicity_query,
        "turbulence": monitor.first_turbulence_query,
        "guard": monitor.first_alarm_query,
    }
    return streams, first


def verify_prefix_invariance(raw: np.ndarray, profile_path: Path) -> bool:
    cut = min(max(12, len(raw) // 2), len(raw))
    original = IntrinsicGuardMonitor(GlobalIntrinsicProfile.load(profile_path))
    altered = IntrinsicGuardMonitor(GlobalIntrinsicProfile.load(profile_path))
    first_records = [original.update(query) for query in raw]
    changed = raw.copy()
    if cut < len(raw):
        changed[cut:] = np.roll(changed[cut:], shift=7, axis=-1)
    second_records = [altered.update(query) for query in changed]
    fields = (
        "alarm",
        "first_alarm_query",
        "mechanism",
        "freeze_score",
        "acceleration_score",
        "periodicity_score",
    )
    for left, right in zip(first_records[:cut], second_records[:cut]):
        for field in fields:
            if isinstance(left[field], float):
                if not np.allclose(left[field], right[field], equal_nan=True, atol=0.0, rtol=0.0):
                    return False
            elif left[field] != right[field]:
                return False
    return True


def verify_one(job: dict[str, Any], device_index: int, profile_path: Path) -> dict[str, Any]:
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    raw, raw_sha256 = load_raw_episode(job["task"], job["episode"], job["length"])
    gpu_mobility, gpu_acceleration, gpu_periodicity = torch_features(raw, device)
    monitor, first = replay_monitor(raw, profile_path)
    profile = GlobalIntrinsicProfile.load(profile_path)
    batch = intrinsic_score_arrays(
        job["mobility"][None],
        job["acceleration"][None],
        job["periodicity"][None],
        profile.periodicity_scale,
    )
    errors = {
        "gpu_mobility": finite_max_abs(gpu_mobility, job["mobility"]),
        "gpu_acceleration": finite_max_abs(gpu_acceleration, job["acceleration"]),
        "gpu_periodicity": finite_max_abs(gpu_periodicity, job["periodicity"]),
        "stream_mobility": finite_max_abs(monitor["mobility"], job["mobility"]),
        "stream_acceleration": finite_max_abs(monitor["acceleration"], job["acceleration"]),
        "stream_periodicity": finite_max_abs(monitor["periodicity"], job["periodicity"]),
        "freeze_score": finite_max_abs(monitor["freeze_score"], batch["freeze"][0]),
        "acceleration_score": finite_max_abs(
            monitor["acceleration_score"], batch["acceleration_persistent"][0]
        ),
        "periodicity_score": finite_max_abs(
            monitor["periodicity_score"], batch["periodicity_persistent"][0]
        ),
    }
    expected_first = job["first"]
    feature_pass = all(
        error <= (SCORE_TOLERANCE if "score" in name else FEATURE_TOLERANCE)
        for name, error in errors.items()
    )
    result = {
        "row": job["row"],
        "sample_class": job["sample_class"],
        "task": job["task"],
        "episode": job["episode"],
        "queries": job["length"],
        "device": device_index,
        "device_name": torch.cuda.get_device_name(device),
        "raw_episode_sha256": raw_sha256,
        **{f"max_abs_{name}": value for name, value in errors.items()},
        "expected_first_guard": expected_first["guard"],
        "replayed_first_guard": first["guard"],
        "first_alarm_exact": first == expected_first,
        "prefix_invariant_to_future_mutation": verify_prefix_invariance(raw, profile_path),
        "feature_tolerance_pass": feature_pass,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }
    result["passed"] = bool(
        result["first_alarm_exact"]
        and result["prefix_invariant_to_future_mutation"]
        and result["feature_tolerance_pass"]
    )
    return result


def worker(
    device_index: int,
    jobs: list[dict[str, Any]],
    profile_path: str,
    queue: mp.Queue,
) -> None:
    try:
        results = [verify_one(job, device_index, Path(profile_path)) for job in jobs]
        queue.put({"device": device_index, "results": results})
    except Exception as error:  # pragma: no cover - reported to the parent process
        queue.put({"device": device_index, "error": repr(error)})


def build_jobs(result_root: Path, samples_per_class: int) -> list[dict[str, Any]]:
    layer = load_npz(LAYER_CACHE)
    route = load_npz(FEATURE_CACHE)
    alarms = load_npz(result_root / "sealed_first_alarms.npz")
    if not np.array_equal(layer["task_index"], route["task_index"]):
        raise ValueError("external feature caches are not aligned")
    selected = select_rows(alarms, samples_per_class)
    acceleration = feature(route, "route_acceleration")
    periodicity = feature(route, "lag_periodicity")
    jobs: list[dict[str, Any]] = []
    for row, sample_class in selected:
        length = int(layer["length"][row])
        jobs.append(
            {
                "row": row,
                "sample_class": sample_class,
                "task": str(layer["task_names"][int(layer["task_index"][row])]),
                "episode": int(layer["episode"][row]),
                "length": length,
                "mobility": np.asarray(layer["mobility"][row, :length], dtype=np.float32),
                "acceleration": np.asarray(acceleration[row, :length], dtype=np.float32),
                "periodicity": np.asarray(periodicity[row, :length], dtype=np.float32),
                "first": {
                    name: int(alarms[f"external_{name}"][row])
                    for name in ("freeze", "acceleration", "periodicity", "turbulence", "guard")
                },
            }
        )
    return jobs


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices are required; set CUDA_VISIBLE_DEVICES=6,7")
    profile_path = args.result_root / "global_profile.npz"
    jobs = build_jobs(args.result_root, args.samples_per_class)
    partitions = [jobs[::2], jobs[1::2]]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=worker, args=(device, partitions[device], str(profile_path), queue))
        for device in range(2)
    ]
    for process in processes:
        process.start()
    payloads = [queue.get() for _ in processes]
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"GPU replay worker exited with code {process.exitcode}")
    errors = [payload for payload in payloads if "error" in payload]
    if errors:
        raise RuntimeError(f"GPU replay failed: {errors}")
    results = sorted(
        [row for payload in payloads for row in payload["results"]], key=lambda row: row["row"]
    )
    table = pd.DataFrame(results)
    csv_path = args.result_root / "raw_causal_gpu_samples.csv"
    table.to_csv(csv_path, index=False)
    error_columns = [column for column in table if column.startswith("max_abs_")]
    visible_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_devices: list[int] | str | None
    if visible_spec and all(part.strip().isdigit() for part in visible_spec.split(",")):
        physical_devices = [int(part.strip()) for part in visible_spec.split(",")]
    else:
        physical_devices = visible_spec
    summary = {
        "schema": "himoe.intrinsic_guard_v7.raw_causal_gpu.v1",
        "cuda_visible_devices": visible_spec,
        "visible_device_count": torch.cuda.device_count(),
        "physical_devices_requested": physical_devices,
        "sample_policy": (
            f"{args.samples_per_class} evenly spaced freeze-only, turbulence-only, "
            "and no-alarm episodes per class"
        ),
        "outcomes_loaded": False,
        "raw_router_tensor_used": True,
        "future_query_access_by_monitor": False,
        "counterfactual_future_mutation_test": True,
        "feature_tolerance": FEATURE_TOLERANCE,
        "score_tolerance": SCORE_TOLERANCE,
        "episodes": len(table),
        "queries": int(table["queries"].sum()),
        "devices": sorted(table["device"].unique().astype(int).tolist()),
        "device_names": sorted(table["device_name"].unique().tolist()),
        "max_peak_gpu_memory_mb": float(table["peak_gpu_memory_mb"].max()),
        "max_absolute_errors": {
            column.removeprefix("max_abs_"): float(table[column].max())
            for column in error_columns
        },
        "exact_first_alarm_episodes": int(table["first_alarm_exact"].sum()),
        "prefix_invariant_episodes": int(table["prefix_invariant_to_future_mutation"].sum()),
        "passed_episodes": int(table["passed"].sum()),
        "all_passed": bool(table["passed"].all()),
        "sample_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output_path = args.result_root / "raw_causal_gpu_verification.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not summary["all_passed"]:
        raise RuntimeError("one or more raw causal replay checks failed")


if __name__ == "__main__":
    main()
