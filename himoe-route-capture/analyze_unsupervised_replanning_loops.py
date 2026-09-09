#!/usr/bin/env python3
"""Discover physical-routing-action loops before reading rollout outcomes.

The label-blind stage selects one physical leave-and-return pair per eligible
query, measures whether aligned action-token routing and action chunks also
leave and return, scores short replay, and clusters high-score motifs. Rollout
outcomes are loaded only after every discovery decision has been made.

See analysis/unsupervised-replanning-loops/PREREG.md.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


HERE = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/unsupervised-replanning-loops"
BLOCK = 512
PRIMARY_MIN_LAG = 3
SENSITIVITY_MIN_LAG = 4
DEFAULT_PERMUTATIONS = 5000
DEFAULT_BOOTSTRAPS = 5000
QUANTILES = {
    "q995": 0.995,
    "q99": 0.99,
    "q98": 0.98,
    "q95": 0.95,
}
PRIMARY_QUANTILE = "q99"
DISCOVERY_FEATURES = {
    "physical_seed_strength": True,
    "route_return_gain": True,
    "action_return_gain": True,
    "replay_physical_distance": False,
    "replay_route_overlap": True,
    "replay_action_distance": False,
    "effect_distance": False,
}
PRIMARY_METRICS = (
    "max_loop_score",
    "loop_score_p90",
    "candidate_q99_present",
)
SENSITIVITY_METRICS = (
    "candidate_q995_present",
    "candidate_q98_present",
    "candidate_q95_present",
)
POSTHOC_RECOVERY_THRESHOLDS = {
    "001": 0.01,
    "0025": 0.025,
    "005": 0.05,
    "010": 0.10,
    "025": 0.25,
}
POSTHOC_RECOVERY_METRICS = tuple(
    f"candidate_q99_recovery_{name}_present" for name in POSTHOC_RECOVERY_THRESHOLDS
)
CLUSTER_FEATURES = (
    "lag_fraction",
    "phase",
    "physical_closure_fraction",
    "log_physical_excursion_scale",
    "path_inefficiency",
    "physical_seed_strength_pct",
    "route_return_gain_pct",
    "action_return_gain_pct",
    "replay_physical_distance_pct",
    "replay_route_overlap_pct",
    "replay_action_distance_pct",
    "effect_distance_pct",
    "route_return_overlap",
    "route_return_gain",
    "action_return_gain",
)


POPCOUNT16 = np.fromiter(
    (value.bit_count() for value in range(1 << 16)),
    dtype=np.uint8,
    count=1 << 16,
)


@dataclass
class TaskData:
    key: str
    run: pathlib.Path
    episode_indices: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    scenes: np.ndarray
    seeds: np.ndarray
    prefix: int
    qpos_z: list[np.ndarray]
    action_z: list[np.ndarray]
    route_masks: np.ndarray
    adjacent_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def episode_path(client: pathlib.Path, episode_index: int) -> pathlib.Path:
    return client / ("episode_%02d.npz" % episode_index)


def discover_runs(cache_root: pathlib.Path) -> list[pathlib.Path]:
    runs = []
    pattern = "libero_*/*/right-16x32/client/summaries.json"
    for summary_path in sorted(cache_root.glob(pattern)):
        run = summary_path.parents[1]
        if (run / "server/routes.zarr").exists() and (
            run / "client/sim_layout.json"
        ).exists():
            runs.append(run)
    if not runs:
        raise RuntimeError("no complete right-16x32 runs found")
    return runs


def canonical_qpos(sim: np.ndarray, layout: dict) -> np.ndarray:
    nq = int(layout["nq"])
    qpos = np.asarray(sim[:, 1 : 1 + nq], np.float32).copy()
    for joint in layout["joints"]:
        lo = int(joint["state_lo"]) - 1
        hi = int(joint["state_hi"]) - 1
        if hi - lo != 7:
            continue
        quaternion = qpos[:, lo + 3 : hi]
        quaternion[quaternion[:, 0] < 0] *= -1
    return qpos


def standardize_parts(parts: list[np.ndarray]) -> list[np.ndarray]:
    lengths = [len(part) for part in parts]
    values = np.concatenate(parts).astype(np.float32, copy=False)
    scale = values.std(axis=0)
    keep = scale > 1e-6
    if not np.any(keep):
        raise ValueError("all channels are constant")
    values = (values[:, keep] - values[:, keep].mean(axis=0)) / scale[keep]
    return [
        part.astype(np.float32, copy=False)
        for part in np.split(values, np.cumsum(lengths)[:-1])
    ]


def action_route_masks(ids: np.ndarray) -> np.ndarray:
    if ids.ndim != 5 or ids.shape[1:] != (8, 10, 11, 4):
        raise ValueError(f"unexpected hb_expert_ids shape {ids.shape}")
    action_ids = ids[:, :, :, 1:, :].reshape(len(ids), -1, 4)
    masks = np.zeros(action_ids.shape[:-1], np.uint32)
    for slot in range(4):
        masks |= np.left_shift(
            np.uint32(1), action_ids[..., slot].astype(np.uint32, copy=False)
        )
    return masks


def route_overlaps(
    masks: np.ndarray, reference_row: int, other_rows: np.ndarray
) -> np.ndarray:
    intersection = np.bitwise_and(masks[other_rows], masks[reference_row])
    low = POPCOUNT16[np.bitwise_and(intersection, np.uint32(0xFFFF))]
    high = POPCOUNT16[np.right_shift(intersection, np.uint32(16))]
    return (low.astype(np.float32) + high.astype(np.float32)).mean(axis=1) / 4.0


def rms_rows(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values - reference), axis=1))


def load_task_label_blind(run: pathlib.Path, cache_root: pathlib.Path) -> TaskData:
    key = str(run.relative_to(cache_root).parent)
    # The JSON object is parsed here, but the outcome field is deliberately not
    # accessed. load_outcomes() re-opens it only after discovery and clustering.
    raw_rows = json.loads((run / "client/summaries.json").read_text())
    metadata = sorted(
        (
            {
                "episode_index": int(row["episode_index"]),
                "inference_calls": int(row["inference_calls"]),
                "init_state_id": int(row["init_state_id"]),
                "flow_noise_seed": int(row["flow_noise_seed"]),
            }
            for row in raw_rows
        ),
        key=lambda row: row["episode_index"],
    )
    episode_indices = np.asarray([row["episode_index"] for row in metadata], np.int32)
    if not np.array_equal(episode_indices, np.arange(len(metadata))):
        raise ValueError(f"non-contiguous episode indices in {key}")
    lengths = np.asarray([row["inference_calls"] for row in metadata], np.int32)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    scenes = np.asarray([row["init_state_id"] for row in metadata], np.int32)
    seeds = np.asarray([row["flow_noise_seed"] for row in metadata], np.int32)
    layout = json.loads((run / "client/sim_layout.json").read_text())

    qpos_parts = []
    action_parts = []
    for row, length in zip(metadata, lengths):
        path = episode_path(run / "client", row["episode_index"])
        with np.load(path, allow_pickle=False) as episode:
            sim = np.asarray(episode["sim_state"], np.float32)
            action = np.asarray(episode["actions"], np.float32).reshape(int(length), -1)
        if len(sim) != length or len(action) != length:
            raise ValueError(f"client length mismatch in {path}")
        qpos_parts.append(canonical_qpos(sim, layout))
        action_parts.append(action)
    qpos_z = standardize_parts(qpos_parts)
    action_z = standardize_parts(action_parts)

    route = zarr.open(str(run / "server/routes.zarr"), mode="r")
    expected_ids = np.repeat(episode_indices, lengths)
    recorded_ids = np.asarray(route["episode_id"][:], np.int32)
    if not np.array_equal(expected_ids, recorded_ids):
        raise ValueError(f"server/client episode alignment failed in {key}")
    total = int(lengths.sum())
    masks = np.empty((total, 8 * 10 * 10), np.uint32)
    expert_ids = route["hb_expert_ids"]
    for first in range(0, total, BLOCK):
        stop = min(first + BLOCK, total)
        masks[first:stop] = action_route_masks(
            np.asarray(expert_ids[first:stop], np.uint8)
        )

    adjacent = np.concatenate(
        [
            np.sqrt(np.mean(np.square(np.diff(values, axis=0)), axis=1))
            for values in qpos_z
            if len(values) > 1
        ]
    )
    positive = adjacent[adjacent > 1e-8]
    adjacent_scale = float(np.median(positive)) if len(positive) else 1.0
    return TaskData(
        key=key,
        run=run,
        episode_indices=episode_indices,
        lengths=lengths,
        offsets=offsets,
        scenes=scenes,
        seeds=seeds,
        prefix=int(lengths.min()),
        qpos_z=qpos_z,
        action_z=action_z,
        route_masks=masks,
        adjacent_scale=adjacent_scale,
    )


def physical_seed_pair(
    qpos: np.ndarray, current: int, min_lag: int, adjacent_scale: float
) -> dict[str, float | int]:
    step = np.sqrt(np.mean(np.square(np.diff(qpos[: current + 1], axis=0)), axis=1))
    cumulative = np.r_[0.0, np.cumsum(step)]
    best: dict[str, float | int] | None = None
    for past in range(0, current - min_lag + 1):
        distances = rms_rows(qpos[past + 1 : current + 1], qpos[past])
        interior = distances[:-1]
        turn_offset = int(np.argmax(interior))
        excursion = float(interior[turn_offset])
        closure = float(distances[-1])
        path_length = float(cumulative[current] - cumulative[past])
        closure_fraction = max(1.0 - closure / max(excursion, 1e-12), 0.0)
        path_inefficiency = max(1.0 - closure / max(path_length, 1e-12), 0.0)
        excursion_strength = 1.0 - math.exp(-excursion / max(adjacent_scale, 1e-12))
        strength = closure_fraction * path_inefficiency * excursion_strength
        row = {
            "past_query": int(past),
            "turn_query": int(past + 1 + turn_offset),
            "physical_closure_distance": closure,
            "physical_excursion_distance": excursion,
            "physical_excursion_scale": excursion / max(adjacent_scale, 1e-12),
            "physical_closure_fraction": closure_fraction,
            "path_length": path_length,
            "path_inefficiency": path_inefficiency,
            "physical_seed_strength": strength,
        }
        if best is None or strength > float(best["physical_seed_strength"]):
            best = row
    if best is None:
        raise ValueError(f"query {current} has no history at minimum lag {min_lag}")
    return best


def extract_events(data: TaskData, min_lag: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode, length_value in enumerate(data.lengths):
        length = int(length_value)
        qpos = data.qpos_z[episode]
        action = data.action_z[episode]
        offset = int(data.offsets[episode])
        for current in range(min_lag, length - 1):
            physical = physical_seed_pair(qpos, current, min_lag, data.adjacent_scale)
            past = int(physical["past_query"])
            interior = np.arange(past + 1, current, dtype=np.int64)
            route_all = route_overlaps(
                data.route_masks,
                offset + past,
                offset + np.arange(past + 1, current + 1, dtype=np.int64),
            )
            route_return = float(route_all[-1])
            route_excursion = float(route_all[:-1].min())
            route_gain = route_return - route_excursion
            route_recovery = float(
                np.clip(route_gain / max(1.0 - route_excursion, 1e-12), 0.0, 1.0)
            )

            action_all = rms_rows(action[past + 1 : current + 1], action[past])
            action_return = float(action_all[-1])
            action_excursion = float(action_all[:-1].max())
            action_gain = action_excursion - action_return
            action_recovery = float(
                np.clip(action_gain / max(action_excursion, 1e-12), 0.0, 1.0)
            )
            minimum_recovery = min(
                float(physical["physical_closure_fraction"]),
                route_recovery,
                action_recovery,
            )

            replay_x = float(
                np.sqrt(
                    np.mean(
                        np.square(qpos[past : past + 2] - qpos[current : current + 2])
                    )
                )
            )
            replay_action = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            action[past : past + 2] - action[current : current + 2]
                        )
                    )
                )
            )
            replay_route = (
                float(
                    route_overlaps(
                        data.route_masks,
                        offset + past,
                        np.asarray([offset + current], np.int64),
                    )[0]
                    + route_overlaps(
                        data.route_masks,
                        offset + past + 1,
                        np.asarray([offset + current + 1], np.int64),
                    )[0]
                )
                / 2.0
            )
            effect_distance = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            (qpos[past + 1] - qpos[past])
                            - (qpos[current + 1] - qpos[current])
                        )
                    )
                )
            )
            rows.append(
                {
                    "task": data.key,
                    "episode": int(episode),
                    "scene": int(data.scenes[episode]),
                    "flow_noise_seed": int(data.seeds[episode]),
                    "episode_length": length,
                    "shared_prefix": int(data.prefix),
                    "min_lag": int(min_lag),
                    "current_query": int(current),
                    "lag": int(current - past),
                    "lag_fraction": float((current - past) / max(length - 1, 1)),
                    "phase": float(current / max(length - 1, 1)),
                    **physical,
                    "log_physical_excursion_scale": float(
                        math.log1p(float(physical["physical_excursion_scale"]))
                    ),
                    "route_return_overlap": route_return,
                    "route_excursion_overlap": route_excursion,
                    "route_return_gain": route_gain,
                    "route_recovery_fraction": route_recovery,
                    "route_excursion_query": int(interior[np.argmin(route_all[:-1])]),
                    "action_return_distance": action_return,
                    "action_excursion_distance": action_excursion,
                    "action_return_gain": action_gain,
                    "action_recovery_fraction": action_recovery,
                    "action_excursion_query": int(interior[np.argmax(action_all[:-1])]),
                    "replay_physical_distance": replay_x,
                    "replay_route_overlap": replay_route,
                    "replay_action_distance": replay_action,
                    "effect_distance": effect_distance,
                    "minimum_xra_recovery_fraction": minimum_recovery,
                }
            )
    return rows


def empirical_percentile(
    sorted_reference: np.ndarray, values: np.ndarray, higher_is_better: bool
) -> np.ndarray:
    left = np.searchsorted(sorted_reference, values, side="left")
    right = np.searchsorted(sorted_reference, values, side="right")
    average_rank = (left + right + 1.0) / 2.0
    percentile = average_rank / (len(sorted_reference) + 1.0)
    if not higher_is_better:
        percentile = 1.0 - percentile
    epsilon = 1.0 / (2.0 * (len(sorted_reference) + 1.0))
    return np.clip(percentile, epsilon, 1.0 - epsilon)


def calibrate_and_score(
    events: list[dict[str, Any]], prefix: int
) -> tuple[dict[str, Any], dict[str, float]]:
    training = [row for row in events if int(row["current_query"]) < prefix]
    if not training:
        raise ValueError("no equal-prefix events for label-free calibration")
    calibration: dict[str, Any] = {
        "prefix": int(prefix),
        "training_events": len(training),
        "features": {},
    }
    for name, higher_is_better in DISCOVERY_FEATURES.items():
        reference = np.sort(
            np.asarray([float(row[name]) for row in training], np.float64)
        )
        values = np.asarray([float(row[name]) for row in events], np.float64)
        percentiles = empirical_percentile(reference, values, higher_is_better)
        percentile_name = f"{name}_pct"
        for row, value in zip(events, percentiles):
            row[percentile_name] = float(value)
        calibration["features"][name] = {
            "higher_is_better": higher_is_better,
            "q05": float(np.quantile(reference, 0.05)),
            "median": float(np.quantile(reference, 0.50)),
            "q95": float(np.quantile(reference, 0.95)),
        }

    percentile_names = [f"{name}_pct" for name in DISCOVERY_FEATURES]
    for row in events:
        values = np.asarray([float(row[name]) for name in percentile_names])
        row["loop_score"] = float(np.exp(np.mean(np.log(values))))

    training_scores = np.asarray(
        [float(row["loop_score"]) for row in training], np.float64
    )
    thresholds = {
        name: float(np.quantile(training_scores, quantile))
        for name, quantile in QUANTILES.items()
    }
    calibration["score_thresholds"] = thresholds
    for row in events:
        positive_topology = bool(
            float(row["physical_seed_strength"]) > 0
            and float(row["route_return_gain"]) > 0
            and float(row["action_return_gain"]) > 0
        )
        row["positive_topology"] = positive_topology
        for name, threshold in thresholds.items():
            row[f"candidate_{name}"] = bool(
                positive_topology and float(row["loop_score"]) >= threshold
            )
        for name, recovery in POSTHOC_RECOVERY_THRESHOLDS.items():
            row[f"candidate_q99_recovery_{name}"] = bool(
                row["candidate_q99"]
                and float(row["minimum_xra_recovery_fraction"]) >= recovery
            )
    return calibration, thresholds


def nonmaximum_suppression(events: list[dict[str, Any]], flag: str) -> list[dict]:
    by_episode: dict[tuple[str, int], list[dict]] = {}
    for row in events:
        if bool(row[flag]):
            by_episode.setdefault((str(row["task"]), int(row["episode"])), []).append(
                row
            )
    kept = []
    for group in by_episode.values():
        accepted: list[dict] = []
        for row in sorted(
            group, key=lambda item: float(item["loop_score"]), reverse=True
        ):
            duplicate = any(
                abs(int(row["past_query"]) - int(other["past_query"])) <= 1
                and abs(int(row["current_query"]) - int(other["current_query"])) <= 1
                for other in accepted
            )
            if not duplicate:
                accepted.append(row)
        kept.extend(accepted)
    return kept


def mean_silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    values = np.zeros(len(labels), np.float64)
    unique = np.unique(labels)
    for index in range(len(labels)):
        own = np.flatnonzero(labels == labels[index])
        own = own[own != index]
        if not len(own):
            continue
        within = float(distance[index, own].mean())
        between = min(
            float(distance[index, np.flatnonzero(labels == label)].mean())
            for label in unique
            if label != labels[index]
        )
        values[index] = (between - within) / max(within, between, 1e-12)
    return float(values.mean())


def label_blind_cluster_profiles(candidates: list[dict]) -> dict[str, dict]:
    profiles = {}
    for cluster in sorted({int(row["cluster"]) for row in candidates}):
        subset = [row for row in candidates if int(row["cluster"]) == cluster]
        profiles[str(cluster)] = {
            "events": len(subset),
            "episodes": len(
                {(str(row["task"]), int(row["episode"])) for row in subset}
            ),
            "task_counts": {
                task: sum(str(row["task"]) == task for row in subset)
                for task in sorted({str(row["task"]) for row in subset})
            },
            "prefix_event_fraction": float(
                np.mean(
                    [
                        int(row["current_query"]) < int(row["shared_prefix"])
                        for row in subset
                    ]
                )
            ),
            "medians": {
                name: float(np.median([float(row[name]) for row in subset]))
                for name in (
                    "loop_score",
                    "lag",
                    "lag_fraction",
                    "phase",
                    "physical_closure_fraction",
                    "physical_excursion_scale",
                    "path_inefficiency",
                    "route_return_overlap",
                    "route_return_gain",
                    "action_return_distance",
                    "action_return_gain",
                    "replay_physical_distance",
                    "replay_route_overlap",
                    "replay_action_distance",
                    "effect_distance",
                )
            },
        }
    return profiles


def cluster_candidates(candidates: list[dict]) -> dict[str, Any]:
    if len(candidates) < 4:
        for row in candidates:
            row["cluster"] = 1
        return {
            "status": "too_few_candidates",
            "candidates": len(candidates),
            "selected_clusters": 1 if candidates else 0,
            "profiles": label_blind_cluster_profiles(candidates) if candidates else {},
        }

    matrix = np.asarray(
        [[float(row[name]) for name in CLUSTER_FEATURES] for row in candidates],
        np.float64,
    )
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    keep = scale > 1e-10
    standardized = (matrix[:, keep] - center[keep]) / scale[keep]
    tree = linkage(standardized, method="ward", optimal_ordering=True)
    distance = squareform(pdist(standardized))
    minimum_size = max(5, int(math.ceil(0.02 * len(candidates))))
    trials = []
    for count in range(2, min(6, len(candidates) - 1) + 1):
        labels = fcluster(tree, count, criterion="maxclust").astype(np.int32)
        sizes = [int(np.sum(labels == label)) for label in np.unique(labels)]
        trials.append(
            {
                "clusters": int(len(sizes)),
                "silhouette": mean_silhouette(distance, labels),
                "sizes": sizes,
                "size_guard": bool(min(sizes) >= minimum_size),
                "labels": labels,
            }
        )
    eligible = [trial for trial in trials if trial["size_guard"]]
    pool = eligible if eligible else trials[:1]
    selected = max(pool, key=lambda trial: float(trial["silhouette"]))
    raw_labels = np.asarray(selected["labels"], np.int32)
    ordered = sorted(
        np.unique(raw_labels),
        key=lambda label: (
            float(
                np.median(
                    [
                        float(row["lag_fraction"])
                        for row, value in zip(candidates, raw_labels)
                        if value == label
                    ]
                )
            ),
            -float(
                np.median(
                    [
                        float(row["loop_score"])
                        for row, value in zip(candidates, raw_labels)
                        if value == label
                    ]
                )
            ),
        ),
    )
    remap = {int(label): index + 1 for index, label in enumerate(ordered)}
    for row, label in zip(candidates, raw_labels):
        row["cluster"] = remap[int(label)]
    profiles = label_blind_cluster_profiles(candidates)
    task_purity = float(
        sum(max(profile["task_counts"].values()) for profile in profiles.values())
        / len(candidates)
    )
    return {
        "status": "ok" if eligible else "size_guard_failed",
        "candidates": len(candidates),
        "minimum_cluster_size": minimum_size,
        "selected_clusters": int(len(np.unique(raw_labels))),
        "selected_silhouette": float(selected["silhouette"]),
        "features": list(CLUSTER_FEATURES),
        "active_features": [
            name for name, active in zip(CLUSTER_FEATURES, keep) if active
        ],
        "trials": [
            {key: value for key, value in trial.items() if key != "labels"}
            for trial in trials
        ],
        "task_purity": task_purity,
        "profiles": profiles,
    }


def load_outcomes(run: pathlib.Path) -> dict[int, bool]:
    rows = json.loads((run / "client/summaries.json").read_text())
    return {int(row["episode_index"]): not bool(row["success"]) for row in rows}


def attach_outcomes(
    events: list[dict[str, Any]], outcomes: dict[str, dict[int, bool]]
) -> None:
    for row in events:
        row["failure"] = bool(outcomes[str(row["task"])][int(row["episode"])])


def build_episode_records(
    events: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[int, bool]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in events:
        grouped.setdefault((str(row["task"]), int(row["episode"])), []).append(row)
    records = []
    for task, info in catalog.items():
        for episode in range(int(info["episodes"])):
            prefix = int(info["prefix"])
            selected = [
                row
                for row in grouped[(task, episode)]
                if int(row["current_query"]) < prefix
            ]
            if not selected:
                raise ValueError(f"no prefix events for {task} episode {episode}")
            scores = np.asarray([float(row["loop_score"]) for row in selected])
            record: dict[str, Any] = {
                "task": task,
                "episode": episode,
                "scene": int(info["scenes"][episode]),
                "flow_noise_seed": int(info["seeds"][episode]),
                "episode_length": int(info["lengths"][episode]),
                "prefix": prefix,
                "event_opportunities": len(selected),
                "failure": bool(outcomes[task][episode]),
                "max_loop_score": float(scores.max()),
                "loop_score_p90": float(np.quantile(scores, 0.90)),
            }
            for name in QUANTILES:
                record[f"candidate_{name}_present"] = float(
                    any(bool(row[f"candidate_{name}"]) for row in selected)
                )
                record[f"candidate_{name}_count"] = int(
                    sum(bool(row[f"candidate_{name}"]) for row in selected)
                )
            for name in POSTHOC_RECOVERY_THRESHOLDS:
                field = f"candidate_q99_recovery_{name}"
                record[f"{field}_present"] = float(
                    any(bool(row[field]) for row in selected)
                )
            records.append(record)
    return records


def group_key(record: dict) -> str:
    return f"{record['task']}|{record['scene']}"


def stratified_test(
    records: list[dict],
    metric_names: tuple[str, ...],
    permutations: int,
    bootstraps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    labels = np.asarray([bool(record["failure"]) for record in records])
    values = np.column_stack(
        [[float(record[name]) for record in records] for name in metric_names]
    )
    groups = np.asarray([group_key(record) for record in records])
    valid_groups = []
    valid = np.zeros(len(records), bool)
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        if labels[index].any() and (~labels[index]).any():
            valid_groups.append(index)
            valid[index] = True
    use = np.flatnonzero(valid)
    if not len(valid_groups):
        return {"status": "no_mixed_strata", "records": len(records)}

    local = {original: index for index, original in enumerate(use)}
    group_local = [
        np.asarray([local[index] for index in group]) for group in valid_groups
    ]
    y = labels[use]
    x = values[use]
    ranks = np.empty_like(x, dtype=np.float64)
    base = 0.0
    denominator = 0.0
    components = []
    for index in group_local:
        n1 = int(y[index].sum())
        n0 = int((~y[index]).sum())
        group_ranks = np.column_stack(
            [rankdata(x[index, column]) for column in range(x.shape[1])]
        )
        ranks[index] = group_ranks
        group_base = n1 * (n1 + 1) / 2.0
        wins = y[index].astype(np.float64) @ group_ranks - group_base
        pairs = float(n1 * n0)
        components.append((wins, pairs))
        base += group_base
        denominator += pairs
    observed = (y.astype(np.float64) @ ranks - base) / denominator

    null_parts = []
    for first in range(0, permutations, 100):
        batch = min(100, permutations - first)
        permuted = np.empty((batch, len(use)), np.float64)
        for index in group_local:
            order = np.argsort(rng.random((batch, len(index))), axis=1)
            permuted[:, index] = y[index][order]
        null_parts.append((permuted @ ranks - base) / denominator)
    null = np.concatenate(null_parts)
    max_null = np.max(null - 0.5, axis=1)
    max_abs_null = np.max(np.abs(null - 0.5), axis=1)

    bootstrap_auc = np.empty((bootstraps, len(metric_names)), np.float64)
    for draw in range(bootstraps):
        selected = rng.integers(0, len(components), len(components))
        wins = sum(
            (components[index][0] for index in selected),
            np.zeros(len(metric_names)),
        )
        pairs = sum(components[index][1] for index in selected)
        bootstrap_auc[draw] = wins / pairs

    metrics = {}
    for column, name in enumerate(metric_names):
        effect = observed[column] - 0.5
        metrics[name] = {
            "auc_failure": float(observed[column]),
            "cluster_bootstrap_ci": [
                float(np.quantile(bootstrap_auc[:, column], 0.025)),
                float(np.quantile(bootstrap_auc[:, column], 0.975)),
            ],
            "permutation_p_one_sided": float(
                (1 + np.sum(null[:, column] - 0.5 >= effect - 1e-15))
                / (permutations + 1)
            ),
            "permutation_p_two_sided": float(
                (1 + np.sum(np.abs(null[:, column] - 0.5) >= abs(effect) - 1e-15))
                / (permutations + 1)
            ),
            "familywise_p_one_sided": float(
                (1 + np.sum(max_null >= effect - 1e-15)) / (permutations + 1)
            ),
            "familywise_p_two_sided": float(
                (1 + np.sum(max_abs_null >= abs(effect) - 1e-15)) / (permutations + 1)
            ),
        }
    return {
        "status": "ok",
        "records_in_mixed_strata": int(len(use)),
        "failures_in_mixed_strata": int(y.sum()),
        "successes_in_mixed_strata": int((~y).sum()),
        "mixed_strata": len(valid_groups),
        "permutations": permutations,
        "bootstraps": bootstraps,
        "metrics": metrics,
    }


def summarize_discovery(
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    calibrations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for task in sorted(calibrations):
        subset = [row for row in events if str(row["task"]) == task]
        selected = [row for row in candidates if str(row["task"]) == task]
        prefix = int(calibrations[task]["prefix"])
        result[task] = {
            "seed_pairs": len(subset),
            "prefix_seed_pairs": sum(
                int(row["current_query"]) < prefix for row in subset
            ),
            "positive_topology_pairs": sum(
                bool(row["positive_topology"]) for row in subset
            ),
            "raw_primary_candidates": sum(
                bool(row[f"candidate_{PRIMARY_QUANTILE}"]) for row in subset
            ),
            "nms_primary_candidates": len(selected),
            "candidate_episodes": len({int(row["episode"]) for row in selected}),
            "score_thresholds": calibrations[task]["score_thresholds"],
            "lag_counts": {
                str(lag): sum(int(row["lag"]) == lag for row in selected)
                for lag in sorted({int(row["lag"]) for row in selected})
            },
        }
    return result


def add_cluster_outcomes(
    profiles: dict[str, dict],
    candidates: list[dict],
    catalog: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[int, bool]],
) -> dict[str, dict]:
    group_failure_rate = {}
    for task, info in catalog.items():
        scenes = np.asarray(info["scenes"], np.int32)
        for scene in np.unique(scenes):
            episodes = np.flatnonzero(scenes == scene)
            group_failure_rate[(task, int(scene))] = float(
                np.mean([outcomes[task][int(episode)] for episode in episodes])
            )

    enriched = json.loads(json.dumps(profiles))
    for cluster, profile in enriched.items():
        subset = [row for row in candidates if int(row["cluster"]) == int(cluster)]
        episode_keys = sorted(
            {(str(row["task"]), int(row["episode"])) for row in subset}
        )
        failures = [outcomes[task][episode] for task, episode in episode_keys]
        expected = [
            group_failure_rate[(task, int(catalog[task]["scenes"][episode]))]
            for task, episode in episode_keys
        ]
        profile["unblinded_outcome"] = {
            "unique_episodes": len(episode_keys),
            "failure_episodes": int(sum(failures)),
            "success_episodes": int(len(failures) - sum(failures)),
            "failure_fraction": float(np.mean(failures)),
            "matched_task_scene_baseline": float(np.mean(expected)),
            "failure_excess_descriptive": float(np.mean(failures) - np.mean(expected)),
            "failure_event_fraction": float(
                np.mean([bool(row["failure"]) for row in subset])
            ),
        }
    return enriched


def candidate_episode_summary(
    candidates: list[dict],
    catalog: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[int, bool]],
) -> dict[str, Any]:
    result = {}
    for task, info in catalog.items():
        candidate_episodes = {
            int(row["episode"]) for row in candidates if str(row["task"]) == task
        }
        failure_episodes = {
            episode for episode, failure in outcomes[task].items() if failure
        }
        success_episodes = set(range(int(info["episodes"]))) - failure_episodes
        result[task] = {
            "candidate_episodes": len(candidate_episodes),
            "failure_candidate_episodes": len(candidate_episodes & failure_episodes),
            "success_candidate_episodes": len(candidate_episodes & success_episodes),
            "all_failure_episodes": len(failure_episodes),
            "all_success_episodes": len(success_episodes),
        }
    return result


def per_task_validation(
    records: list[dict],
    permutations: int,
    bootstraps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    result = {}
    for task in sorted({str(row["task"]) for row in records}):
        subset = [row for row in records if str(row["task"]) == task]
        result[task] = stratified_test(
            subset, PRIMARY_METRICS, permutations, bootstraps, rng
        )
    return result


def leave_one_task_out_validation(
    records: list[dict],
    permutations: int,
    bootstraps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    tasks_with_failures = sorted(
        {str(row["task"]) for row in records if bool(row["failure"])}
    )
    return {
        omitted: stratified_test(
            [row for row in records if str(row["task"]) != omitted],
            PRIMARY_METRICS,
            permutations,
            bootstraps,
            rng,
        )
        for omitted in tasks_with_failures
    }


def task_candidate_counts(records: list[dict]) -> dict[str, dict[str, int]]:
    result = {}
    for task in sorted({str(row["task"]) for row in records}):
        subset = [row for row in records if str(row["task"]) == task]
        failure = [row for row in subset if bool(row["failure"])]
        success = [row for row in subset if not bool(row["failure"])]
        result[task] = {
            "failure_episodes": len(failure),
            "failure_candidate_episodes": sum(
                bool(row["candidate_q99_present"]) for row in failure
            ),
            "success_episodes": len(success),
            "success_candidate_episodes": sum(
                bool(row["candidate_q99_present"]) for row in success
            ),
        }
    return result


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# 无监督 replanning loop 发现",
        "",
        "本实验先完成物理回返选择、无标签分位数标定、候选筛选和聚类，",
        "之后才读取 episode outcome。failure/success 没有参与任何发现决策。",
        "",
        "## 数据覆盖",
        "",
        "| task | episodes | failures | queries | shared prefix |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["coverage"]:
        lines.append(
            "| %s | %d | %d | %d | %d |"
            % (
                row["task"],
                row["episodes"],
                row["failures"],
                row["queries"],
                row["shared_prefix"],
            )
        )

    lines.extend(
        [
            "",
            "## 标签盲发现",
            "",
            "每个当前 query 先仅凭 qpos 选择一个“离开后返回”的历史起点。",
            "主候选要求无标签联合分数超过各任务等长前缀的 99% 分位，",
            "并且 routing 与 action 的返回增益都为正。",
            "",
            "| task | seed pairs | positive topology | q99 threshold | raw candidates | NMS candidates / episodes |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task, row in result["discovery_by_task"].items():
        lines.append(
            "| %s | %d | %d | %s | %d | %d / %d |"
            % (
                task,
                row["seed_pairs"],
                row["positive_topology_pairs"],
                fmt(row["score_thresholds"]["q99"]),
                row["raw_primary_candidates"],
                row["nms_primary_candidates"],
                row["candidate_episodes"],
            )
        )

    clustering = result["clustering"]
    lines.extend(
        [
            "",
            "## 无监督 motif 聚类",
            "",
            "Ward 聚类在看不到 outcome 的情况下选择了 %d 个簇，silhouette=%s，状态=%s。"
            % (
                clustering["selected_clusters"],
                fmt(clustering.get("selected_silhouette", float("nan"))),
                clustering["status"],
            ),
            "Cluster-task purity=%s；越接近 1 表示聚类越像在区分任务而非共享 motif。"
            % fmt(clustering.get("task_purity", float("nan"))),
            "",
            "| cluster | events / episodes | median score | median lag | median phase | X closure | route gain | action gain | failure episodes after unblinding | matched baseline |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cluster, profile in result["cluster_profiles_unblinded"].items():
        medians = profile["medians"]
        outcome = profile["unblinded_outcome"]
        lines.append(
            "| %s | %d / %d | %s | %s | %s | %s | %+.3f | %+.3f | %d/%d | %s |"
            % (
                cluster,
                profile["events"],
                profile["episodes"],
                fmt(medians["loop_score"]),
                fmt(medians["lag"]),
                fmt(medians["phase"]),
                fmt(medians["physical_closure_fraction"]),
                medians["route_return_gain"],
                medians["action_return_gain"],
                outcome["failure_episodes"],
                outcome["unique_episodes"],
                fmt(outcome["matched_task_scene_baseline"]),
            )
        )

    validation = result["outcome_validation"]
    lines.extend(
        [
            "",
            "## Outcome 解盲验证",
            "",
            "这里只使用各任务等长前缀，并在 task x initial-state 内比较。",
            "AUC > 0.5 表示标签盲发现分数在最终失败 episode 中更高。",
            "",
            "| endpoint | failure AUC | cluster 95% CI | p one-sided | family-wise p |",
            "|---|---:|---|---:|---:|",
        ]
    )
    labels = {
        "max_loop_score": "maximum loop score",
        "loop_score_p90": "90th-percentile loop score",
        "candidate_q99_present": "q99 candidate present",
    }
    if validation.get("status") == "ok":
        for name in PRIMARY_METRICS:
            row = validation["metrics"][name]
            lines.append(
                "| %s | %s | [%s, %s] | %.4f | %.4f |"
                % (
                    labels[name],
                    fmt(row["auc_failure"]),
                    fmt(row["cluster_bootstrap_ci"][0]),
                    fmt(row["cluster_bootstrap_ci"][1]),
                    row["permutation_p_one_sided"],
                    row["familywise_p_one_sided"],
                )
            )

    lines.extend(
        [
            "",
            "### 逐任务异质性",
            "",
            "| task | failure candidates | success candidates | q99 candidate AUC | p one-sided |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task, counts in result["task_candidate_counts"].items():
        test = result["per_task_validation"][task]
        if test.get("status") == "ok":
            metric = test["metrics"]["candidate_q99_present"]
            auc = fmt(metric["auc_failure"])
            p_value = f"{metric['permutation_p_one_sided']:.4f}"
        else:
            auc = "all success"
            p_value = "-"
        lines.append(
            "| %s | %d/%d | %d/%d | %s | %s |"
            % (
                task,
                counts["failure_candidate_episodes"],
                counts["failure_episodes"],
                counts["success_candidate_episodes"],
                counts["success_episodes"],
                auc,
                p_value,
            )
        )

    lines.extend(
        [
            "",
            "Leave-one-task-out 是解盲后的稳健性检查。",
            "",
            "| omitted task | remaining q99 candidate AUC | p one-sided |",
            "|---|---:|---:|",
        ]
    )
    for task, test in result["leave_one_task_out"].items():
        row = test["metrics"]["candidate_q99_present"]
        lines.append(
            "| %s | %s | %.4f |"
            % (task, fmt(row["auc_failure"]), row["permutation_p_one_sided"])
        )

    lines.extend(
        [
            "",
            "### 阈值和最小周期敏感性",
            "",
            "| endpoint | failure AUC | 95% CI | p one-sided |",
            "|---|---:|---|---:|",
        ]
    )
    threshold_test = result["threshold_sensitivity"]
    threshold_labels = {
        "candidate_q995_present": "top 0.5% candidate",
        "candidate_q98_present": "top 2% candidate",
        "candidate_q95_present": "top 5% candidate",
    }
    if threshold_test.get("status") == "ok":
        for name in SENSITIVITY_METRICS:
            row = threshold_test["metrics"][name]
            lines.append(
                "| %s | %s | [%s, %s] | %.4f |"
                % (
                    threshold_labels[name],
                    fmt(row["auc_failure"]),
                    fmt(row["cluster_bootstrap_ci"][0]),
                    fmt(row["cluster_bootstrap_ci"][1]),
                    row["permutation_p_one_sided"],
                )
            )
    minlag = result["min_lag_4_validation"]
    if minlag.get("status") == "ok":
        for name in PRIMARY_METRICS:
            row = minlag["metrics"][name]
            lines.append(
                "| min lag 4: %s | %s | [%s, %s] | %.4f |"
                % (
                    labels[name],
                    fmt(row["auc_failure"]),
                    fmt(row["cluster_bootstrap_ci"][0]),
                    fmt(row["cluster_bootstrap_ci"][1]),
                    row["permutation_p_one_sided"],
                )
            )

    recovery = result["posthoc_recovery_audit"]
    lines.extend(
        [
            "",
            "### Post-hoc 绝对回返幅度审计",
            "",
            "该检查在首次 outcome 解盲后新增，不属于预注册主检验。",
            "它要求 X、routing、action 各自至少恢复指定比例的离开幅度，",
            "用于区分微小回摆和有实际幅度的闭环。",
            "",
            "| minimum recovered departure | failure AUC | 95% CI | p one-sided | family-wise p |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    if recovery.get("status") == "ok":
        for name, threshold in POSTHOC_RECOVERY_THRESHOLDS.items():
            metric = recovery["metrics"][f"candidate_q99_recovery_{name}_present"]
            lines.append(
                "| %.1f%% | %s | [%s, %s] | %.4f | %.4f |"
                % (
                    100 * threshold,
                    fmt(metric["auc_failure"]),
                    fmt(metric["cluster_bootstrap_ci"][0]),
                    fmt(metric["cluster_bootstrap_ci"][1]),
                    metric["permutation_p_one_sided"],
                    metric["familywise_p_one_sided"],
                )
            )

    primary = validation.get("metrics", {})
    candidate_test = primary.get("candidate_q99_present", {})
    score_test = primary.get("max_loop_score", {})
    enriched = bool(
        candidate_test.get("auc_failure", 0.5) > 0.5
        and candidate_test.get("permutation_p_one_sided", 1.0) < 0.05
    )
    score_enriched = bool(
        score_test.get("auc_failure", 0.5) > 0.5
        and score_test.get("permutation_p_one_sided", 1.0) < 0.05
    )
    lines.extend(["", "## 结论", ""])
    if enriched:
        lines.append(
            "标签盲 q99 回环候选在最终失败中富集；这支持 recurrent geometry 与 outcome 的关联。"
        )
    elif score_enriched:
        lines.append("连续的标签盲 loop score 在失败中升高，但高阈值候选没有稳定富集。")
    else:
        lines.append("标签盲 loop score 和高阈值候选均未在最终失败中稳定富集。")
    supporting_tasks = []
    for task, test in result["per_task_validation"].items():
        if test.get("status") != "ok":
            continue
        metric = test["metrics"]["candidate_q99_present"]
        if metric["auc_failure"] > 0.5 and metric["permutation_p_one_sided"] < 0.05:
            supporting_tasks.append(task)
    if len(supporting_tasks) == 1:
        lines.append(
            "该方向只在一个任务中成立：%s；移除它后总体关联消失。" % supporting_tasks[0]
        )
    recovery_005 = recovery.get("metrics", {}).get(
        "candidate_q99_recovery_005_present", {}
    )
    if recovery_005.get("permutation_p_one_sided", 1.0) >= 0.05:
        lines.append(
            "要求三种表示都至少恢复 5% 离开幅度后不再显著，因此当前证据更像任务特定的微回返，而非强 routing basin 闭环。"
        )
    lines.extend(
        [
            "这项分析不需要知道失败原因，但也不能把任何几何簇命名为滑落、空抓或目标错误。",
            "cluster outcome 比例来自全轨迹，受 episode 长度和终止阶段影响，只作描述。",
            "",
            "## 限制",
            "",
            "- X 是 MuJoCo qpos，不是 RGB 或 vision embedding。",
            "- 检测使用当前及下一 query，因此是事件发现，不是提前预警。",
            "- MoE 信号是 action-token top-4 routing identity，不是 expert output。",
            "- Outcome 富集是关联证据，不是 routing 导致失败的因果证据。",
            "- 没有使用或生成语义 failure-cause 标签。",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    ids = np.empty((2, 8, 10, 11, 4), np.uint8)
    ids[0] = np.asarray([0, 1, 2, 3], np.uint8)
    ids[1] = np.asarray([0, 1, 4, 5], np.uint8)
    masks = action_route_masks(ids)
    overlap = route_overlaps(masks, 0, np.asarray([1], np.int64))[0]
    assert abs(float(overlap) - 0.5) < 1e-8

    qpos = np.asarray([[0.0], [1.0], [2.0], [1.0], [0.0], [0.5]])
    pair = physical_seed_pair(qpos, current=4, min_lag=3, adjacent_scale=1.0)
    assert int(pair["past_query"]) == 0
    assert int(pair["turn_query"]) == 2
    assert abs(float(pair["physical_closure_distance"])) < 1e-12
    assert float(pair["physical_seed_strength"]) > 0.8

    reference = np.asarray([0.0, 1.0, 2.0, 3.0])
    values = np.asarray([0.0, 3.0])
    high = empirical_percentile(reference, values, True)
    low = empirical_percentile(reference, values, False)
    assert high[1] > high[0]
    assert low[1] < low[0]

    events = [
        {
            "task": "task",
            "episode": 0,
            "past_query": 1,
            "current_query": 5,
            "loop_score": 0.9,
            "candidate_q99": True,
        },
        {
            "task": "task",
            "episode": 0,
            "past_query": 2,
            "current_query": 6,
            "loop_score": 0.8,
            "candidate_q99": True,
        },
    ]
    assert len(nonmaximum_suppression(events, "candidate_q99")) == 1
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.permutations < 1 or args.bootstraps < 1:
        raise ValueError("permutations and bootstraps must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.cache_root)
    primary_events: list[dict[str, Any]] = []
    min_lag_4_events: list[dict[str, Any]] = []
    calibrations: dict[str, dict[str, Any]] = {}
    min_lag_4_calibrations: dict[str, dict[str, Any]] = {}
    catalog: dict[str, dict[str, Any]] = {}
    task_runs: dict[str, pathlib.Path] = {}

    # Label-blind phase. No success field is accessed before clustering below.
    for run in runs:
        data = load_task_label_blind(run, args.cache_root)
        task_runs[data.key] = run
        catalog[data.key] = {
            "episodes": len(data.lengths),
            "queries": int(data.lengths.sum()),
            "prefix": int(data.prefix),
            "lengths": data.lengths.astype(int).tolist(),
            "scenes": data.scenes.astype(int).tolist(),
            "seeds": data.seeds.astype(int).tolist(),
            "adjacent_scale": float(data.adjacent_scale),
        }

        task_primary = extract_events(data, PRIMARY_MIN_LAG)
        calibration, _ = calibrate_and_score(task_primary, data.prefix)
        calibrations[data.key] = calibration
        primary_events.extend(task_primary)

        task_min_lag_4 = extract_events(data, SENSITIVITY_MIN_LAG)
        calibration_4, _ = calibrate_and_score(task_min_lag_4, data.prefix)
        min_lag_4_calibrations[data.key] = calibration_4
        min_lag_4_events.extend(task_min_lag_4)
        print(
            "discovered %s: %d episodes, %d min-lag-3 seeds, %d min-lag-4 seeds"
            % (
                data.key,
                len(data.lengths),
                len(task_primary),
                len(task_min_lag_4),
            )
        )
        del data, task_primary, task_min_lag_4
        gc.collect()

    candidates = nonmaximum_suppression(primary_events, f"candidate_{PRIMARY_QUANTILE}")
    discovery_by_task = summarize_discovery(primary_events, candidates, calibrations)
    min_lag_4_candidates = nonmaximum_suppression(
        min_lag_4_events, f"candidate_{PRIMARY_QUANTILE}"
    )
    min_lag_4_discovery = summarize_discovery(
        min_lag_4_events, min_lag_4_candidates, min_lag_4_calibrations
    )
    clustering = cluster_candidates(candidates)
    print(
        "label-blind discovery complete: %d primary candidates, %d clusters"
        % (len(candidates), int(clustering["selected_clusters"]))
    )

    # Outcome unblinding starts here, after candidates and cluster assignments.
    outcomes = {task: load_outcomes(run) for task, run in task_runs.items()}
    attach_outcomes(primary_events, outcomes)
    attach_outcomes(min_lag_4_events, outcomes)
    episode_records = build_episode_records(primary_events, catalog, outcomes)
    min_lag_4_episode_records = build_episode_records(
        min_lag_4_events, catalog, outcomes
    )

    rng = np.random.default_rng(args.seed)
    outcome_validation = stratified_test(
        episode_records,
        PRIMARY_METRICS,
        args.permutations,
        args.bootstraps,
        rng,
    )
    threshold_sensitivity = stratified_test(
        episode_records,
        SENSITIVITY_METRICS,
        args.permutations,
        args.bootstraps,
        rng,
    )
    min_lag_4_validation = stratified_test(
        min_lag_4_episode_records,
        PRIMARY_METRICS,
        args.permutations,
        args.bootstraps,
        rng,
    )
    per_task = per_task_validation(
        episode_records,
        args.permutations,
        args.bootstraps,
        rng,
    )
    posthoc_recovery_audit = stratified_test(
        episode_records,
        POSTHOC_RECOVERY_METRICS,
        args.permutations,
        args.bootstraps,
        rng,
    )
    leave_one_task_out = leave_one_task_out_validation(
        episode_records,
        args.permutations,
        args.bootstraps,
        rng,
    )

    cluster_profiles_unblinded = add_cluster_outcomes(
        clustering.get("profiles", {}), candidates, catalog, outcomes
    )
    coverage = [
        {
            "task": task,
            "episodes": int(info["episodes"]),
            "failures": int(sum(outcomes[task].values())),
            "queries": int(info["queries"]),
            "shared_prefix": int(info["prefix"]),
        }
        for task, info in catalog.items()
    ]
    top_candidates = []
    for row in sorted(
        candidates, key=lambda item: float(item["loop_score"]), reverse=True
    )[:25]:
        top_candidates.append(
            {
                name: row[name]
                for name in (
                    "task",
                    "episode",
                    "scene",
                    "flow_noise_seed",
                    "failure",
                    "cluster",
                    "past_query",
                    "turn_query",
                    "current_query",
                    "lag",
                    "phase",
                    "loop_score",
                    "physical_closure_fraction",
                    "physical_excursion_scale",
                    "route_return_overlap",
                    "route_return_gain",
                    "route_recovery_fraction",
                    "action_return_distance",
                    "action_return_gain",
                    "action_recovery_fraction",
                    "minimum_xra_recovery_fraction",
                    "replay_physical_distance",
                    "replay_route_overlap",
                    "replay_action_distance",
                    "effect_distance",
                )
            }
        )

    result = {
        "schema": "unsupervised-replanning-loops/1",
        "source": str(args.cache_root),
        "run_id": "right-16x32",
        "preregistered_spec": str(args.out_dir / "PREREG.md"),
        "discovery_completed_before_outcome_access": True,
        "definitions": {
            "physical": "canonicalized within-task standardized MuJoCo qpos",
            "routing": "actual action-token top-4 overlap over 8 HB layers x 10 denoise x 10 action tokens",
            "action": "within-task standardized recorded 10x7 action chunk",
            "primary_min_lag": PRIMARY_MIN_LAG,
            "sensitivity_min_lag": SENSITIVITY_MIN_LAG,
            "primary_candidate": "task equal-prefix q99 loop score and positive X/R/A return topology",
            "outcome_role": "external validation only after discovery and clustering",
        },
        "coverage": coverage,
        "calibrations": calibrations,
        "discovery_by_task": discovery_by_task,
        "clustering": clustering,
        "outcome_validation": outcome_validation,
        "threshold_sensitivity": threshold_sensitivity,
        "min_lag_4_discovery": min_lag_4_discovery,
        "min_lag_4_validation": min_lag_4_validation,
        "per_task_validation": per_task,
        "task_candidate_counts": task_candidate_counts(episode_records),
        "leave_one_task_out": leave_one_task_out,
        "posthoc_recovery_audit": posthoc_recovery_audit,
        "candidate_episode_summary": candidate_episode_summary(
            candidates, catalog, outcomes
        ),
        "cluster_profiles_unblinded": cluster_profiles_unblinded,
        "top_candidates": top_candidates,
        "limitations": [
            "qpos is not a stored RGB or vision embedding",
            "event discovery uses the next query and is not an early warning test",
            "clusters are geometric and have no semantic failure-cause labels",
            "full-trajectory cluster outcomes are length and terminal-phase confounded",
            "outcome association is not a causal routing intervention",
        ],
    }

    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(result))
    write_csv(args.out_dir / "seed_pairs.csv", primary_events)
    write_csv(args.out_dir / "candidate_events.csv", candidates)
    write_csv(args.out_dir / "episode_scores.csv", episode_records)
    write_csv(
        args.out_dir / "episode_scores_min_lag_4.csv",
        min_lag_4_episode_records,
    )
    print(f"wrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
