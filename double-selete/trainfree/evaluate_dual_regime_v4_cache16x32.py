#!/usr/bin/env python3
"""Replay frozen v4 and legacy alarms on VLA_MUI_HUB/cache right-16x32."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

import evaluate_dual_regime_v4 as v4
import evaluate_layerwise_alarm_development as dev


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_CACHE = WORKSPACE / "VLA_MUI_HUB/cache/HiMoE-VLA"
DEFAULT_DUAL_PROFILE = HERE / "results/online_dual_regime_v4/deployment_profiles.npz"
DEFAULT_LEGACY_PROFILE = (
    HERE / "results/online_precision_cascade_external/deployment_profiles.npz"
)
DEFAULT_OUTPUT = HERE / "results/online_dual_regime_v4_cache16x32"
RUN_ID = "right-16x32"
MAX_QUERY = 52
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dual-profile", type=Path, default=DEFAULT_DUAL_PROFILE)
    parser.add_argument("--legacy-profile", type=Path, default=DEFAULT_LEGACY_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def normalize(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return output / np.maximum(output.sum(axis=-1, keepdims=True), 1e-12)


def discover(cache_root: Path) -> list[Path]:
    runs = sorted(cache_root.glob(f"libero_*/*/{RUN_ID}"))
    complete: list[Path] = []
    for run in runs:
        meta_path = run / "meta.json"
        summary_path = run / "client/summaries.json"
        route_path = run / "server/routes.zarr"
        if not (meta_path.exists() and summary_path.exists() and route_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sampling = meta.get("sampling", {})
        if (
            meta.get("status") == "complete"
            and sampling.get("complete") is True
            and int(sampling.get("actual_episodes", -1)) == 512
        ):
            complete.append(run)
    if len(complete) != 5:
        raise ValueError(
            f"expected five complete right-16x32 runs, found {len(complete)}"
        )
    return complete


def extract(
    runs: list[Path], cache_root: Path
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    n_episode = 512 * len(runs)
    mobility = np.full((n_episode, MAX_QUERY, 8), np.nan, np.float32)
    valid = np.zeros((n_episode, MAX_QUERY), dtype=bool)
    task_index = np.repeat(np.arange(len(runs), dtype=np.int16), 512)
    episode = np.tile(np.arange(512, dtype=np.int16), len(runs))
    init_state = np.empty(n_episode, np.int16)
    flow_seed = np.empty(n_episode, np.int32)
    length = np.empty(n_episode, np.int16)
    label_rows: list[dict[str, Any]] = []
    task_names: list[str] = []

    for task_position, run in enumerate(runs):
        task = str(run.relative_to(cache_root).parent)
        task_names.append(task)
        summaries = json.loads(
            (run / "client/summaries.json").read_text(encoding="utf-8")
        )
        summaries.sort(key=lambda row: int(row["episode_index"]))
        if [int(row["episode_index"]) for row in summaries] != list(range(512)):
            raise ValueError(f"non-contiguous episodes for {task}")
        group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        source = group["hb_router_probs"]
        if source.shape[1:] != (8, 10, 11, 32):
            raise ValueError(f"unexpected router shape for {task}: {source.shape}")
        route = normalize(np.asarray(source[:, :, 9, 1:11, :]))
        affinity = np.sqrt(route[1:] * route[:-1]).sum(axis=-1)
        adjacent = np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0)).mean(axis=-1)
        route_episode = np.asarray(group["episode_id"][:], dtype=int)

        for row in summaries:
            local_episode = int(row["episode_index"])
            global_row = task_position * 512 + local_episode
            indices = np.flatnonzero(route_episode == local_episode)
            observed_length = int(row["inference_calls"])
            if (
                len(indices) != observed_length
                or observed_length > MAX_QUERY
                or (len(indices) > 1 and not np.all(np.diff(indices) == 1))
            ):
                raise ValueError(
                    f"route/summary mismatch for {task} episode {local_episode}"
                )
            valid[global_row, :observed_length] = True
            if observed_length > 1:
                mobility[global_row, 1:observed_length] = adjacent[indices[:-1]]
            init_state[global_row] = int(row["init_state_id"])
            flow_seed[global_row] = int(row["flow_noise_seed"])
            length[global_row] = observed_length
            label_rows.append(
                {
                    "row": global_row,
                    "task": task,
                    "episode": local_episode,
                    "init_state_id": int(row["init_state_id"]),
                    "flow_noise_seed": int(row["flow_noise_seed"]),
                    "length": observed_length,
                    "success": bool(row["success"]),
                    "failure": not bool(row["success"]),
                }
            )
        print(f"[cache extraction {task_position + 1}/{len(runs)}] {task}", flush=True)

    cache = {
        "schema": np.asarray("himoe.cache16x32.layer_mobility.v1"),
        "task_names": np.asarray(task_names),
        "task_index": task_index,
        "episode": episode,
        "init_state_id": init_state,
        "flow_noise_seed": flow_seed,
        "length": length,
        "valid": valid,
        "layer_names": np.asarray(LAYER_NAMES),
        "mobility": mobility,
    }
    labels = pd.DataFrame(label_rows).sort_values("row").reset_index(drop=True)
    if not np.array_equal(labels["row"].to_numpy(), np.arange(n_episode)):
        raise ValueError("label row construction is incomplete")
    return cache, labels


def task_thresholds(
    cache: dict[str, np.ndarray], profile_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(profile_path, allow_pickle=False) as profile:
        if str(profile["schema"]) != "himoe.dual_regime_v4.profile.v1":
            raise ValueError("unknown v4 deployment profile")
        profile_tasks = profile["task_names"].astype(str).tolist()
        lock = np.empty(len(cache["episode"]), np.float32)
        instability = np.empty(len(cache["episode"]), np.float32)
        for task_position, task in enumerate(cache["task_names"].astype(str)):
            source_position = profile_tasks.index(task)
            take = cache["task_index"] == task_position
            lock[take] = profile["lock_thresholds"][source_position]
            instability[take] = profile["instability_thresholds"][source_position]
    return lock, instability


def legacy_thresholds(cache: dict[str, np.ndarray], profile_path: Path) -> np.ndarray:
    with np.load(profile_path, allow_pickle=False) as profile:
        if str(profile["schema"]) != "himoe.precision_cascade.profile.v1":
            raise ValueError("unknown legacy deployment profile")
        profile_tasks = profile["task_names"].astype(str).tolist()
        detector = (
            profile["detector_names"].astype(str).tolist().index("mobility_w4_k1")
        )
        quantile = int(
            np.flatnonzero(np.isclose(profile["quantiles"].astype(float), 0.95))[0]
        )
        output = np.empty(len(cache["episode"]), np.float32)
        for task_position, task in enumerate(cache["task_names"].astype(str)):
            source_position = profile_tasks.index(task)
            output[cache["task_index"] == task_position] = profile["thresholds"][
                source_position, detector, quantile
            ]
    return output


def alarm_arrays(
    cache: dict[str, np.ndarray], dual_profile: Path, legacy_profile: Path
) -> dict[str, np.ndarray]:
    valid = cache["valid"].astype(bool)
    lock_instant, lock_persistent = v4.head_scores(cache, "lock")
    instability_instant, instability_persistent = v4.head_scores(cache, "instability")
    del lock_instant, instability_instant
    lock_threshold, instability_threshold = task_thresholds(cache, dual_profile)
    lock = v4.first_from_threshold(lock_persistent, valid, lock_threshold)
    instability = v4.first_from_threshold(
        instability_persistent, valid, instability_threshold
    )

    back_mean = cache["mobility"][:, :, 4:8].mean(axis=2)
    back_mean[~valid] = np.nan
    legacy_instant = -dev.trailing_mean(back_mean, 4)
    legacy_score = dev.persistent_score(legacy_instant, 2)
    legacy = v4.first_from_threshold(
        legacy_score, valid, legacy_thresholds(cache, legacy_profile)
    )
    return {
        "legacy_back_mean_q95_k2": legacy,
        "lock_layer_median_q75_k4": lock,
        "instability_L5_q80_k8": instability,
        "dual_regime_or": v4.first_or(lock, instability),
    }


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def metric(
    detector: str,
    first: np.ndarray,
    labels: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    alarm = first >= 0
    failure = labels["failure"].to_numpy(dtype=bool)
    success = ~failure
    lead = labels["length"].to_numpy(dtype=int) - 1 - first
    tp = int((alarm & failure).sum())
    fp = int((alarm & success).sum())
    tasks = labels["task"].to_numpy(dtype=str)
    row: dict[str, Any] = {
        "detector": detector,
        "episodes": len(labels),
        "failure_n": int(failure.sum()),
        "success_n": int(success.sum()),
        "tp": tp,
        "fp": fp,
        "fn": int((~alarm & failure).sum()),
        "tn": int((~alarm & success).sum()),
        "failure_recall": ratio(tp, failure.sum()),
        "success_fpr": ratio(fp, success.sum()),
        "precision": ratio(tp, tp + fp),
        "detected_failure_lead_median": (
            float(np.median(lead[alarm & failure])) if tp else float("nan")
        ),
    }
    for name, numerator, denominator in (
        ("failure_recall", alarm & failure, failure),
        ("success_fpr", alarm & success, success),
        ("precision", alarm & failure, alarm),
    ):
        low, high = v3_cluster(tasks, numerator, denominator, draws, rng)
        row[f"{name}_ci_low"] = low
        row[f"{name}_ci_high"] = high
    for early in (2, 4, 8, 12):
        row[f"early{early}_failure_recall"] = ratio(
            (alarm & failure & (lead >= early)).sum(), failure.sum()
        )
    return row


def v3_cluster(
    tasks: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    names = np.unique(tasks)
    task_num = np.asarray([numerator[tasks == name].sum() for name in names])
    task_den = np.asarray([denominator[tasks == name].sum() for name in names])
    samples = rng.integers(0, len(names), size=(draws, len(names)))
    values = task_num[samples].sum(axis=1) / np.maximum(
        task_den[samples].sum(axis=1), 1
    )
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def main() -> None:
    args = parse_args()
    runs = discover(args.cache)
    cache, labels = extract(runs, args.cache)
    alarms = alarm_arrays(cache, args.dual_profile, args.legacy_profile)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "layerwise_mobility.npz", **cache)
    np.savez_compressed(
        args.output / "frozen_first_alarms.npz",
        schema=np.asarray("himoe.dual_regime_v4.cache16x32.first.v1"),
        **alarms,
    )
    episode = labels.copy()
    for detector, first in alarms.items():
        episode[f"first_{detector}_query"] = first
    episode.to_csv(args.output / "episode_alarms.csv", index=False)

    rng = np.random.default_rng(args.seed)
    metrics = pd.DataFrame(
        [
            metric(name, first, labels, args.bootstrap, rng)
            for name, first in alarms.items()
        ]
    )
    metrics.to_csv(args.output / "outcome_metrics.csv", index=False)
    task_rows: list[dict[str, Any]] = []
    for task in labels["task"].unique():
        take = labels["task"].to_numpy(dtype=str) == task
        block = labels.loc[take].reset_index(drop=True)
        for detector, first in alarms.items():
            row = metric(detector, first[take], block, args.bootstrap, rng)
            row["task"] = task
            task_rows.append(row)
    pd.DataFrame(task_rows).to_csv(
        args.output / "outcome_metrics_by_task.csv", index=False
    )

    summary = {
        "schema": "himoe.dual_regime_v4.cache16x32.evaluation.v1",
        "status": "frozen-rule transfer replay on previously existing right-16x32 data",
        "data_root": str(args.cache),
        "run_id": RUN_ID,
        "tasks": len(runs),
        "episodes": len(labels),
        "sampling": "16 initial states x 32 flow-noise seeds per task",
        "runtime_moe_only": True,
        "v4_parameters_retuned": False,
        "v4_thresholds_recalibrated_on_cache16x32": False,
        "label_target": "raw_original_horizon_failure",
        "plus10_timeout_cleanup_available": False,
        "metrics": metrics.to_dict(orient="records"),
        "artifacts": {
            "dual_profile_sha256": sha256(args.dual_profile),
            "legacy_profile_sha256": sha256(args.legacy_profile),
            "outcome_metrics_sha256": sha256(args.output / "outcome_metrics.csv"),
        },
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        metrics[
            [
                "detector",
                "tp",
                "fp",
                "failure_recall",
                "success_fpr",
                "precision",
                "early4_failure_recall",
                "detected_failure_lead_median",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
