#!/usr/bin/env python3
"""Build and replay a label-free causal multi-head MoE alarm on VLA_MUI_HUB."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import zarr


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_CACHE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
DEFAULT_OUTPUT = HERE / "results/online_multihead_hub"
PROTOCOL = HERE / "ONLINE_MULTIHEAD_PROTOCOL.md"
RUN_ID = "right-50x8-20260903"

FRONT = slice(0, 4)
BACK = slice(4, 8)
FINAL_FLOW = 9
LATE_FLOW_START = 6
ACTION = slice(1, 11)
N_EXPERTS = 32
LAGS = (1, 2, 3, 4)
RAW_SCORE_START_QUERY = 2
WARMUP_QUERY = 4
SMOOTHING_WIDTH = 3
MIN_REFERENCE = 32
OPERATING_QUANTILES = (0.90, 0.95, 0.975)

FEATURES = (
    "late_flow_volatility",
    "route_acceleration",
    "gate_entropy",
    "top12_margin",
    "top4_union",
    "token_disagreement",
    "front_state_action_gap",
    "layer5_state_action_gap",
    "route_mobility",
    "lag_recurrence",
    "lag_periodicity",
    "front_feedback_split",
)

HEADS: dict[str, tuple[tuple[str, int], ...]] = {
    "instability": (
        ("late_flow_volatility", 1),
        ("route_acceleration", 1),
        ("route_mobility", 1),
        ("lag_periodicity", 1),
    ),
    "lock_in": (
        ("route_mobility", -1),
        ("lag_recurrence", 1),
        ("top4_union", -1),
        ("token_disagreement", -1),
    ),
    "flat_narrow_support": (
        ("gate_entropy", 1),
        ("top12_margin", -1),
        ("top4_union", -1),
    ),
    "feedback_decoupling": (
        ("front_state_action_gap", 1),
        ("layer5_state_action_gap", 1),
        ("front_feedback_split", 1),
    ),
}

DETECTORS = (
    *HEADS,
    "dual_mean",
    "dual_max",
    "multi_max",
    "instant_multi_max",
    "clock",
)

TASK_QUERY_LIMITS = {
    "libero_spatial": 22,
    "libero_object": 28,
    "libero_goal": 30,
    "libero_long": 52,
}

# These tasks completed after the frozen 14,800-trajectory retrospective cohort.
# They are excluded from the main replay and can be scored separately as a holdout.
EXTERNAL_TASKS = {
    "libero_long/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "libero_long/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "libero_long/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
}

EXPECTED = {
    "main": (37, 14_800),
    "external": (3, 1_200),
    "all": (40, 16_000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort", choices=tuple(EXPECTED), default="main")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_probability(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return output / np.maximum(output.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probability(left)
    right = normalize_probability(right)
    affinity = np.sqrt(left * right).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))


def weighted_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probability(left)
    right = normalize_probability(right)
    numerator = np.minimum(left, right).sum(axis=-1)
    denominator = np.maximum(left, right).sum(axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def task_key(run: Path, cache_root: Path) -> str:
    return str(run.relative_to(cache_root).parent)


def discover_runs(cache_root: Path, run_id: str, cohort: str) -> list[Path]:
    runs: list[Path] = []
    for summary_path in sorted(cache_root.glob(f"libero_*/*/{run_id}/client/summaries.json")):
        run = summary_path.parents[1]
        meta_path = run / "meta.json"
        route_path = run / "server/routes.zarr"
        if not meta_path.exists() or not route_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sampling = meta.get("sampling", {})
        complete = (
            meta.get("status") == "complete"
            and sampling.get("complete") is True
            and int(sampling.get("actual_episodes", -1))
            == int(sampling.get("designed_episodes", -2))
        )
        if not complete:
            continue
        task = task_key(run, cache_root)
        if cohort == "main" and task in EXTERNAL_TASKS:
            continue
        if cohort == "external" and task not in EXTERNAL_TASKS:
            continue
        runs.append(run)

    task_count, episode_count = EXPECTED[cohort]
    observed_episodes = 0
    for run in runs:
        rows = json.loads((run / "client/summaries.json").read_text(encoding="utf-8"))
        observed_episodes += len(rows)
    if len(runs) != task_count or observed_episodes != episode_count:
        raise RuntimeError(
            f"frozen {cohort} cohort mismatch: tasks={len(runs)}, "
            f"episodes={observed_episodes}; expected {task_count}, {episode_count}"
        )
    return runs


def load_unlabeled_index(run: Path) -> list[dict[str, int]]:
    """Whitelist metadata fields; outcome fields in summaries are never returned."""
    raw = json.loads((run / "client/summaries.json").read_text(encoding="utf-8"))
    rows = [
        {
            "episode": int(row["episode_index"]),
            "init_state_id": int(row["init_state_id"]),
            "flow_noise_seed": int(row["flow_noise_seed"]),
            "length": int(row["inference_calls"]),
        }
        for row in raw
    ]
    rows.sort(key=lambda row: row["episode"])
    if [row["episode"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"non-contiguous episode IDs in {run}")
    if len(rows) != 400:
        raise ValueError(f"expected 400 episodes in {run}, found {len(rows)}")
    init, counts = np.unique([row["init_state_id"] for row in rows], return_counts=True)
    if len(init) != 50 or not np.all(counts == 8):
        raise ValueError(f"expected 50 init states x 8 draws in {run}")
    return rows


def top4_union_fraction(ids: np.ndarray) -> np.ndarray:
    values = np.asarray(ids, dtype=np.int16)
    present = np.any(
        values[..., None] == np.arange(N_EXPERTS, dtype=np.int16),
        axis=(2, 3),
    )
    return present.sum(axis=-1).mean(axis=1, dtype=np.float32) / N_EXPERTS


def extract_task_features(
    run: Path, rows: list[dict[str, int]]
) -> tuple[np.ndarray, np.ndarray]:
    group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    router = group["hb_router_probs"]
    expert_ids = group["hb_expert_ids"]
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    total = int(router.shape[0])
    if episode_id.shape != (total,):
        raise ValueError(f"route episode-id mismatch in {run}")

    scalar = {
        name: np.full(total, np.nan, dtype=np.float32)
        for name in FEATURES
        if name not in {
            "route_mobility",
            "lag_recurrence",
            "lag_periodicity",
            "front_feedback_split",
        }
    }
    back_action = np.empty((total, 4, 10, N_EXPERTS), dtype=np.float16)
    front_state = np.empty((total, 4, N_EXPERTS), dtype=np.float16)
    front_action = np.empty((total, 4, N_EXPERTS), dtype=np.float16)

    for start in range(0, total, 64):
        stop = min(start + 64, total)
        probability = normalize_probability(router[start:stop])
        final_action = probability[:, :, FINAL_FLOW, ACTION, :]
        final_state = probability[:, :, FINAL_FLOW, 0, :]
        front_action_mean = normalize_probability(final_action[:, FRONT].mean(axis=2))
        back_action_mean = normalize_probability(final_action[:, BACK].mean(axis=2))

        entropy = -(
            final_action[:, BACK]
            * np.log(np.maximum(final_action[:, BACK], 1e-12))
        ).sum(axis=-1)
        scalar["gate_entropy"][start:stop] = entropy.mean(axis=(1, 2)) / np.log(N_EXPERTS)

        ordered = np.partition(final_action[:, BACK], -2, axis=-1)
        scalar["top12_margin"][start:stop] = (
            ordered[..., -1] - ordered[..., -2]
        ).mean(axis=(1, 2))
        scalar["token_disagreement"][start:stop] = hellinger(
            final_action[:, BACK], back_action_mean[:, :, None, :]
        ).mean(axis=(1, 2))
        scalar["front_state_action_gap"][start:stop] = hellinger(
            final_state[:, FRONT], front_action_mean
        ).mean(axis=1)
        scalar["layer5_state_action_gap"][start:stop] = hellinger(
            final_state[:, 3], front_action_mean[:, 3]
        )

        ids = np.asarray(expert_ids[start:stop, BACK, FINAL_FLOW, ACTION, :])
        scalar["top4_union"][start:stop] = top4_union_fraction(ids)

        action_flow = probability[:, BACK, :, ACTION, :]
        late = action_flow[:, :, LATE_FLOW_START:]
        scalar["late_flow_volatility"][start:stop] = (
            1.0 - weighted_jaccard(late[:, :, 1:], late[:, :, :-1])
        ).mean(axis=(1, 2, 3))
        root = np.sqrt(action_flow)
        acceleration = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
        scalar["route_acceleration"][start:stop] = (
            np.linalg.norm(acceleration, axis=-1).mean(axis=(1, 2, 3)) / np.sqrt(2.0)
        )

        back_action[start:stop] = final_action[:, BACK]
        front_state[start:stop] = final_state[:, FRONT]
        front_action[start:stop] = front_action_mean

    max_query = max(row["length"] for row in rows)
    output = np.full((len(rows), max_query, len(FEATURES)), np.nan, dtype=np.float32)
    valid = np.zeros((len(rows), max_query), dtype=bool)
    feature_index = {name: index for index, name in enumerate(FEATURES)}

    for row in rows:
        episode = row["episode"]
        indices = np.flatnonzero(episode_id == episode)
        if len(indices) != row["length"]:
            raise ValueError(
                f"route length mismatch in {run.name}/{episode}: "
                f"{len(indices)} != {row['length']}"
            )
        if len(indices) > 1 and not np.all(np.diff(indices) == 1):
            raise ValueError(f"non-contiguous route rows in {run.name}/{episode}")
        length = len(indices)
        valid[episode, :length] = True
        for name, values in scalar.items():
            output[episode, :length, feature_index[name]] = values[indices]

        action_route = normalize_probability(back_action[indices]).reshape(
            length, 40, N_EXPERTS
        )
        state_route = normalize_probability(front_state[indices])
        action_front = normalize_probability(front_action[indices])

        if length > 1:
            output[episode, 1:length, feature_index["route_mobility"]] = hellinger(
                action_route[1:], action_route[:-1]
            ).mean(axis=1)
            state_jump = hellinger(state_route[1:], state_route[:-1]).mean(axis=1)
            action_jump = hellinger(action_front[1:], action_front[:-1]).mean(axis=1)
            output[episode, 1:length, feature_index["front_feedback_split"]] = (
                state_jump - action_jump
            )

        similarities = np.full((length, len(LAGS)), np.nan, dtype=np.float32)
        for lag_index, lag in enumerate(LAGS):
            if length > lag:
                similarities[lag:, lag_index] = weighted_jaccard(
                    action_route[lag:], action_route[:-lag]
                ).mean(axis=1)
        recurrence_valid = np.isfinite(similarities).any(axis=1)
        recurrence_query = np.flatnonzero(recurrence_valid)
        output[
            episode, recurrence_query, feature_index["lag_recurrence"]
        ] = np.nanmax(similarities[recurrence_valid], axis=1)
        periodic_valid = np.isfinite(similarities[:, 0]) & np.isfinite(
            similarities[:, 1:]
        ).any(axis=1)
        periodic_query = np.flatnonzero(periodic_valid)
        output[episode, periodic_query, feature_index["lag_periodicity"]] = (
            np.nanmax(similarities[periodic_valid, 1:], axis=1)
            - similarities[periodic_valid, 0]
        )

    finite = np.isfinite(output[:, RAW_SCORE_START_QUERY:]).all(axis=-1)
    expected_valid = valid[:, RAW_SCORE_START_QUERY:]
    if not np.array_equal(finite, expected_valid):
        bad = np.argwhere(finite != expected_valid)[0]
        raise ValueError(f"non-finite eligible feature at episode/query {bad.tolist()}")
    return output, valid


def build_feature_cache(
    runs: Iterable[Path], cache_root: Path, output_path: Path
) -> dict[str, np.ndarray]:
    runs = list(runs)
    all_features: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    task_names: list[str] = []
    task_index: list[int] = []
    episode: list[int] = []
    init_state: list[int] = []
    flow_seed: list[int] = []
    length: list[int] = []

    max_query = max(TASK_QUERY_LIMITS.values())
    for index, run in enumerate(runs):
        task = task_key(run, cache_root)
        rows = load_unlabeled_index(run)
        task_features, task_valid = extract_task_features(run, rows)
        padded = np.full(
            (len(rows), max_query, len(FEATURES)), np.nan, dtype=np.float32
        )
        valid_padded = np.zeros((len(rows), max_query), dtype=bool)
        padded[:, : task_features.shape[1]] = task_features
        valid_padded[:, : task_valid.shape[1]] = task_valid
        all_features.append(padded)
        all_valid.append(valid_padded)
        task_names.append(task)
        task_index.extend([index] * len(rows))
        episode.extend(row["episode"] for row in rows)
        init_state.extend(row["init_state_id"] for row in rows)
        flow_seed.extend(row["flow_noise_seed"] for row in rows)
        length.extend(row["length"] for row in rows)
        print(
            f"[features {index + 1}/{len(runs)}] {task}: "
            f"queries={int(task_valid.sum())}",
            flush=True,
        )

    cache = {
        "schema": np.asarray("himoe.online_multihead.features.v1"),
        "feature_names": np.asarray(FEATURES),
        "task_names": np.asarray(task_names),
        "task_index": np.asarray(task_index, dtype=np.int16),
        "episode": np.asarray(episode, dtype=np.int16),
        "init_state_id": np.asarray(init_state, dtype=np.int16),
        "flow_noise_seed": np.asarray(flow_seed, dtype=np.int16),
        "length": np.asarray(length, dtype=np.int16),
        "valid": np.concatenate(all_valid, axis=0),
        "features": np.concatenate(all_features, axis=0),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **cache)
    return cache


def load_feature_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: np.asarray(archive[name]) for name in archive.files}
    if str(cache["schema"]) != "himoe.online_multihead.features.v1":
        raise ValueError(f"unexpected feature-cache schema in {path}")
    if tuple(cache["feature_names"].tolist()) != FEATURES:
        raise ValueError("feature-cache feature order mismatch")
    return cache


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    if len(reference) < MIN_REFERENCE:
        return np.full(values.shape, np.nan, dtype=np.float32)
    return (
        np.searchsorted(reference, values, side="right") / len(reference)
    ).astype(np.float32)


def trailing_mean(values: np.ndarray, width: int = SMOOTHING_WIDTH) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.full_like(values, np.nan)
    for query in range(width - 1, values.shape[1]):
        window = values[:, query - width + 1 : query + 1]
        good = np.isfinite(window).all(axis=1)
        output[good, query] = window[good].mean(axis=1)
    return output


def nan_row_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.full(values.shape[0], np.nan, dtype=np.float32)
    good = np.isfinite(values).any(axis=1)
    output[good] = np.nanmax(values[good], axis=1)
    return output


def quantile_higher(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < MIN_REFERENCE:
        return float("nan")
    return float(np.quantile(finite, quantile, method="higher"))


def score_fold(
    features: np.ndarray,
    valid: np.ndarray,
    init_state: np.ndarray,
    held_out: int,
    query_limit: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    train = init_state != held_out
    if int(train.sum()) != 392:
        raise ValueError(f"held-out fold must have 392 references, got {train.sum()}")
    feature_index = {name: index for index, name in enumerate(FEATURES)}
    ranks: dict[tuple[str, int], np.ndarray] = {}

    directed_components = {
        component for members in HEADS.values() for component in members
    }
    for name, direction in sorted(directed_components):
        values = direction * features[:, :, feature_index[name]]
        percentile = np.full(values.shape, np.nan, dtype=np.float32)
        for query in range(RAW_SCORE_START_QUERY, values.shape[1]):
            reference = values[train & valid[:, query], query]
            reference = reference[np.isfinite(reference)]
            take = valid[:, query] & np.isfinite(values[:, query])
            percentile[take, query] = empirical_percentile(
                reference, values[take, query]
            )
        ranks[(name, direction)] = percentile

    instant_heads: dict[str, np.ndarray] = {}
    smooth_heads: dict[str, np.ndarray] = {}
    for head, members in HEADS.items():
        stack = np.stack([ranks[member] for member in members], axis=-1)
        complete = np.isfinite(stack).all(axis=-1)
        score = np.full(stack.shape[:2], np.nan, dtype=np.float32)
        score[complete] = stack[complete].mean(axis=-1)
        instant_heads[head] = score
        smooth_heads[head] = trailing_mean(score)

    scores: dict[str, np.ndarray] = dict(smooth_heads)
    scores["dual_mean"] = (
        smooth_heads["instability"] + smooth_heads["lock_in"]
    ) / 2.0
    scores["dual_max"] = np.maximum(
        smooth_heads["instability"], smooth_heads["lock_in"]
    )
    scores["multi_max"] = np.maximum.reduce(
        [smooth_heads[name] for name in HEADS]
    )
    scores["instant_multi_max"] = np.maximum.reduce(
        [instant_heads[name] for name in HEADS]
    )

    clock = np.full(valid.shape, np.nan, dtype=np.float32)
    query_values = np.arange(valid.shape[1], dtype=np.float32) / max(query_limit - 1, 1)
    clock[valid] = np.broadcast_to(query_values, valid.shape)[valid]
    scores["clock"] = clock

    eligible = valid.copy()
    eligible[:, :WARMUP_QUERY] = False
    for name in scores:
        scores[name][~eligible] = np.nan

    typed = np.stack([smooth_heads[name] for name in HEADS], axis=-1)
    winner = np.full(valid.shape, -1, dtype=np.int8)
    good = np.isfinite(typed).all(axis=-1) & eligible
    winner[good] = np.argmax(typed[good], axis=-1).astype(np.int8)
    return scores, winner, train


def calibrate_and_replay(cache: dict[str, np.ndarray], output_dir: Path) -> None:
    features = np.asarray(cache["features"], dtype=np.float32)
    valid = np.asarray(cache["valid"], dtype=bool)
    task_index = np.asarray(cache["task_index"], dtype=np.int16)
    task_names = np.asarray(cache["task_names"]).astype(str)
    init_state = np.asarray(cache["init_state_id"], dtype=np.int16)
    episode = np.asarray(cache["episode"], dtype=np.int16)
    flow_seed = np.asarray(cache["flow_noise_seed"], dtype=np.int16)
    length = np.asarray(cache["length"], dtype=np.int16)

    n_episode, n_query = valid.shape
    n_detector = len(DETECTORS)
    n_quantile = len(OPERATING_QUANTILES)
    detector_scores = np.full(
        (n_episode, n_query, n_detector), np.nan, dtype=np.float32
    )
    alarms = np.zeros(
        (n_episode, n_query, n_detector, n_quantile), dtype=bool
    )
    thresholds_by_episode = np.full(
        (n_episode, n_detector, n_quantile), np.nan, dtype=np.float32
    )
    winning_head = np.full((n_episode, n_query), -1, dtype=np.int8)
    threshold_rows: list[dict[str, Any]] = []

    for task_position, task in enumerate(task_names):
        take = np.flatnonzero(task_index == task_position)
        if len(take) != 400:
            raise ValueError(f"expected 400 episodes for {task}, found {len(take)}")
        suite = task.split("/", 1)[0]
        query_limit = TASK_QUERY_LIMITS[suite]
        task_features = features[take]
        task_valid = valid[take]
        task_init = init_state[take]

        for held_out in np.unique(task_init):
            scores, winner, train = score_fold(
                task_features, task_valid, task_init, int(held_out), query_limit
            )
            test = ~train
            global_test = take[test]
            winning_head[global_test] = winner[test]
            for detector_position, detector in enumerate(DETECTORS):
                values = scores[detector]
                detector_scores[global_test, :, detector_position] = values[test]
                reference_max = nan_row_max(values[train])
                for quantile_position, quantile in enumerate(OPERATING_QUANTILES):
                    threshold = quantile_higher(reference_max, quantile)
                    thresholds_by_episode[
                        global_test, detector_position, quantile_position
                    ] = threshold
                    trigger = values[test] > threshold
                    latched = np.maximum.accumulate(trigger, axis=1) & task_valid[test]
                    alarms[
                        global_test, :, detector_position, quantile_position
                    ] = latched
                    threshold_rows.append(
                        {
                            "task": task,
                            "held_out_init_state_id": int(held_out),
                            "detector": detector,
                            "quantile": quantile,
                            "threshold": threshold,
                            "unlabeled_reference_episodes": int(train.sum()),
                            "finite_reference_maxima": int(
                                np.isfinite(reference_max).sum()
                            ),
                        }
                    )
        print(
            f"[calibration {task_position + 1}/{len(task_names)}] {task}",
            flush=True,
        )

    if not np.isfinite(detector_scores[:, WARMUP_QUERY:]).any(axis=(1, 2)).all():
        raise ValueError("at least one episode has no eligible detector score")
    if not np.isfinite(thresholds_by_episode).all():
        raise ValueError("non-finite alarm threshold")

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "sealed_online_scores.npz"
    np.savez_compressed(
        score_path,
        schema=np.asarray("himoe.online_multihead.sealed.v1"),
        task_names=task_names,
        task_index=task_index,
        episode=episode,
        init_state_id=init_state,
        flow_noise_seed=flow_seed,
        length=length,
        valid=valid,
        feature_names=np.asarray(FEATURES),
        detector_names=np.asarray(DETECTORS),
        head_names=np.asarray(tuple(HEADS)),
        quantiles=np.asarray(OPERATING_QUANTILES, dtype=np.float32),
        scores=detector_scores,
        thresholds=thresholds_by_episode,
        alarms=alarms,
        winning_head=winning_head,
    )
    pd.DataFrame(threshold_rows).to_csv(output_dir / "unlabeled_thresholds.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for row in range(n_episode):
        item: dict[str, Any] = {
            "task": task_names[task_index[row]],
            "episode": int(episode[row]),
            "init_state_id": int(init_state[row]),
            "flow_noise_seed": int(flow_seed[row]),
            "length": int(length[row]),
        }
        for detector_position, detector in enumerate(DETECTORS):
            for quantile_position, quantile in enumerate(OPERATING_QUANTILES):
                indices = np.flatnonzero(
                    alarms[row, :, detector_position, quantile_position]
                )
                suffix = str(quantile).replace("0.", "q")
                item[f"first_{detector}_{suffix}"] = (
                    int(indices[0]) if len(indices) else -1
                )
        primary_indices = np.flatnonzero(
            alarms[row, :, DETECTORS.index("multi_max"), 1]
        )
        item["primary_winning_head"] = (
            tuple(HEADS)[int(winning_head[row, primary_indices[0]])]
            if len(primary_indices)
            else "none"
        )
        summary_rows.append(item)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "sealed_episode_alarms.csv", index=False
    )


def self_test() -> None:
    reference = np.arange(40, dtype=np.float64)
    values = np.asarray([-1.0, 0.0, 19.5, 39.0, 40.0])
    observed = empirical_percentile(reference, values)
    expected = np.asarray([0.0, 1 / 40, 0.5, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(observed, expected)

    curve = np.asarray([[1.0, 2.0, 3.0, 6.0]], dtype=np.float32)
    smooth = trailing_mean(curve)
    assert np.isnan(smooth[0, :2]).all()
    np.testing.assert_allclose(smooth[0, 2:], [2.0, 11.0 / 3.0])

    ids = np.zeros((1, 4, 10, 4), dtype=np.int16)
    ids[:, :, :, 1] = 1
    ids[:, :, :, 2] = 2
    ids[:, :, :, 3] = 3
    np.testing.assert_allclose(top4_union_fraction(ids), [4 / 32])
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    runs = discover_runs(args.cache_root, args.run_id, args.cohort)
    args.output.mkdir(parents=True, exist_ok=True)
    feature_path = args.output / "unlabeled_query_features.npz"
    if args.reuse_features and feature_path.exists():
        cache = load_feature_cache(feature_path)
        print(f"reused {feature_path}", flush=True)
    else:
        cache = build_feature_cache(runs, args.cache_root, feature_path)
    expected_tasks, expected_episodes = EXPECTED[args.cohort]
    if len(cache["task_names"]) != expected_tasks or len(cache["episode"]) != expected_episodes:
        raise ValueError("feature cache does not match requested frozen cohort")

    calibrate_and_replay(cache, args.output)
    manifest = {
        "schema": "himoe.online_multihead.manifest.v1",
        "cohort": args.cohort,
        "run_id": args.run_id,
        "tasks": expected_tasks,
        "episodes": expected_episodes,
        "features": list(FEATURES),
        "heads": plain(HEADS),
        "detectors": list(DETECTORS),
        "warmup_query": WARMUP_QUERY,
        "raw_score_start_query": RAW_SCORE_START_QUERY,
        "smoothing_width": SMOOTHING_WIDTH,
        "minimum_reference_episodes": MIN_REFERENCE,
        "operating_quantiles": list(OPERATING_QUANTILES),
        "calibration": "leave-one-init-state-out within task; all outcomes retained",
        "query_causal": True,
        "future_queries_used_by_online_update": False,
        "labels_used": [],
        "summary_fields_whitelisted": [
            "episode_index",
            "init_state_id",
            "flow_noise_seed",
            "inference_calls",
        ],
        "artifacts": {
            "protocol_sha256": sha256(PROTOCOL),
            "scorer_sha256": sha256(Path(__file__)),
            "feature_cache_sha256": sha256(feature_path),
            "sealed_scores_sha256": sha256(args.output / "sealed_online_scores.npz"),
            "thresholds_sha256": sha256(args.output / "unlabeled_thresholds.csv"),
            "episode_alarms_sha256": sha256(args.output / "sealed_episode_alarms.csv"),
        },
    }
    manifest_path = args.output / "sealed_manifest.json"
    manifest_path.write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"sealed {expected_episodes} episodes across {expected_tasks} tasks at {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
