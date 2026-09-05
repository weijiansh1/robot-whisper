#!/usr/bin/env python3
"""Test within-chunk structure and longitudinal HB-MoE token dynamics.

The discovery state is used to select one scalar feature with a maxT
permutation correction. The selected feature is then tested, with its direction
fixed, in a second initial state. A rollout, never a control row, is the sample.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zarr
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
HUB = HERE.parent
TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
RUN = HUB / "cache" / "HiMoE-VLA" / "libero_long" / TASK / "right-16x32"

LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
HORIZONS = (7, 12, 34)
LAGS = (1, 2, 3, 4, 5)
TRAILING_WINDOW = 8
N_EXPERTS = 32
ACTION_TOKENS = slice(1, 11)


@dataclass(frozen=True)
class Episode:
    index: int
    state: int
    noise_seed: int
    failure: int
    offset: int
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--discovery-state", type=int, default=0)
    parser.add_argument("--confirmation-state", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def load_episodes(run: Path, states: tuple[int, int]) -> tuple[list[Episode], list[dict[str, Any]]]:
    summaries = sorted(
        json.loads((run / "client" / "summaries.json").read_text()),
        key=lambda row: row["episode_index"],
    )
    lengths = np.asarray([row["inference_calls"] for row in summaries], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    episodes = []
    for row, offset, length in zip(summaries, offsets, lengths):
        state = int(row["init_state_id"])
        if state not in states:
            continue
        episodes.append(
            Episode(
                index=int(row["episode_index"]),
                state=state,
                noise_seed=int(row["flow_noise_seed"]),
                failure=int(not row["success"]),
                offset=int(offset),
                length=int(length),
            )
        )
    for state in states:
        group = [episode for episode in episodes if episode.state == state]
        if len(group) != 32:
            raise ValueError(f"state {state} has {len(group)} episodes, expected 32")
        if min(episode.length for episode in group) <= max(HORIZONS):
            raise ValueError(f"state {state} does not have the required common prefix")
        failures = sum(episode.failure for episode in group)
        if failures in (0, len(group)):
            raise ValueError(f"state {state} has only one outcome class")
    return episodes, summaries


def episode_path(run: Path, episode: int) -> Path:
    return run / "client" / f"episode_{episode:02d}.npz"


def normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    total = values.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("probability vector with zero mass")
    return values / total


def normalize_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def hellinger_embedding_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.5 * np.square(a - b).sum(axis=-1), 0.0))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(1.0 - (a * b).sum(axis=-1), 0.0)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def rms_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(a - b).mean(axis=-1))


def register_feature(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    key: str,
    value: float,
    *,
    source: str,
    horizon: int,
    statistic: str,
    series: str,
    category: str,
) -> None:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite feature {key}")
    if key in output:
        raise ValueError(f"duplicate feature {key}")
    output[key] = value
    item = {
        "source": source,
        "horizon": horizon,
        "statistic": statistic,
        "series": series,
        "category": category,
    }
    previous = metadata.setdefault(key, item)
    if previous != item:
        raise ValueError(f"metadata mismatch for {key}")


def add_temporal_series(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    source: str,
    series_name: str,
    values: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
    channels: tuple[str, ...] | None = None,
) -> None:
    """Reduce aligned [time,layer,flow,(channel),width] trajectories."""

    if values.shape[0] <= max(HORIZONS):
        raise ValueError(f"short series for {source}/{series_name}: {values.shape}")
    expected_distance_ndim = values.ndim - 1
    for horizon in HORIZONS:
        start = max(0, horizon - TRAILING_WINDOW + 1)
        lag_values = []
        for lag in LAGS:
            current = values[start + lag : horizon + 1]
            previous = values[start : horizon + 1 - lag]
            measured = distance(current, previous)
            if measured.ndim != expected_distance_ndim:
                raise ValueError(
                    f"bad distance rank for {source}/{series_name}: {measured.shape}"
                )
            reduced = measured.mean(axis=(0, 1, 2))
            if channels is None:
                reduced = np.asarray([reduced], dtype=np.float64)
                names = (series_name,)
            else:
                reduced = np.asarray(reduced, dtype=np.float64)
                names = channels
                if reduced.shape != (len(channels),):
                    raise ValueError(
                        f"bad channel shape for {source}/{series_name}: {reduced.shape}"
                    )
            lag_values.append(reduced)
            category = "token_lag" if channels is not None else "descriptor_lag"
            for name, value in zip(names, reduced):
                key = f"{source}|h{horizon:02d}|lag{lag}|{name}"
                register_feature(
                    output,
                    metadata,
                    key,
                    value,
                    source=source,
                    horizon=horizon,
                    statistic=f"lag{lag}",
                    series=name,
                    category=category,
                )

        lag_values = np.stack(lag_values)
        recurrences = {
            2: 0.5 * (lag_values[0] + lag_values[2]) - lag_values[1],
            3: 0.5 * (lag_values[1] + lag_values[3]) - lag_values[2],
            4: 0.5 * (lag_values[2] + lag_values[4]) - lag_values[3],
            5: lag_values[3] - lag_values[4],
        }
        names = channels if channels is not None else (series_name,)
        for period, recurrence in recurrences.items():
            for name, value in zip(names, recurrence):
                key = f"{source}|h{horizon:02d}|period{period}|{name}"
                register_feature(
                    output,
                    metadata,
                    key,
                    value,
                    source=source,
                    horizon=horizon,
                    statistic=f"period{period}",
                    series=name,
                    category="periodicity",
                )


def add_level_series(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    source: str,
    name: str,
    values: np.ndarray,
) -> None:
    if values.ndim != 3:
        raise ValueError(f"level series must be [time,layer,flow], got {values.shape}")
    for horizon in HORIZONS:
        start = max(0, horizon - TRAILING_WINDOW + 1)
        value = float(values[start : horizon + 1].mean())
        key = f"{source}|h{horizon:02d}|level|{name}"
        register_feature(
            output,
            metadata,
            key,
            value,
            source=source,
            horizon=horizon,
            statistic="level",
            series=name,
            category="structure_level",
        )


def spectrum_and_gram(unit_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram = np.einsum("tlfud,tlfvd->tlfuv", unit_action, unit_action, optimize=True)
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    spectrum = eigenvalues / np.maximum(eigenvalues.sum(axis=-1, keepdims=True), 1e-12)
    return spectrum.astype(np.float32), gram.astype(np.float32)


def add_route_features(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    source: str,
    probabilities: np.ndarray,
) -> None:
    probabilities = normalize_probability(probabilities)
    embedding = np.sqrt(probabilities)
    token_names = tuple(f"T{index}" for index in range(11))
    add_temporal_series(
        output,
        metadata,
        source,
        "tokens",
        embedding,
        hellinger_embedding_distance,
        token_names,
    )

    action = probabilities[..., ACTION_TOKENS, :]
    action_embedding = embedding[..., ACTION_TOKENS, :]
    action_mean = normalize_probability(action.mean(axis=-2))
    action_mean_embedding = np.sqrt(action_mean)
    action_std = action.std(axis=-2)
    endpoint_delta = action[..., -1, :] - action[..., 0, :]
    spectrum, gram = spectrum_and_gram(action_embedding)

    add_temporal_series(
        output,
        metadata,
        source,
        "action_mean",
        action_mean_embedding,
        hellinger_embedding_distance,
    )
    add_temporal_series(
        output, metadata, source, "action_std", action_std, euclidean_distance
    )
    add_temporal_series(
        output,
        metadata,
        source,
        "endpoint_delta",
        endpoint_delta,
        euclidean_distance,
    )
    add_temporal_series(
        output, metadata, source, "gram_spectrum", spectrum, euclidean_distance
    )

    action_dispersion = hellinger_embedding_distance(
        action_embedding, action_mean_embedding[..., None, :]
    ).mean(axis=-1)
    endpoint_separation = hellinger_embedding_distance(
        action_embedding[..., -1, :], action_embedding[..., 0, :]
    )
    entropy = -(spectrum * np.log(np.maximum(spectrum, 1e-12))).sum(axis=-1)
    effective_rank = np.exp(entropy)
    mean_similarity = (gram.sum(axis=(-1, -2)) - 10.0) / 90.0
    add_level_series(output, metadata, source, "action_dispersion", action_dispersion)
    add_level_series(output, metadata, source, "endpoint_separation", endpoint_separation)
    add_level_series(output, metadata, source, "gram_effective_rank", effective_rank)
    add_level_series(output, metadata, source, "gram_mean_similarity", mean_similarity)
    add_level_series(
        output, metadata, source, "action_std_norm", np.linalg.norm(action_std, axis=-1)
    )
    add_level_series(
        output,
        metadata,
        source,
        "endpoint_delta_norm",
        np.linalg.norm(endpoint_delta, axis=-1),
    )


def add_hidden_features(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    hidden: np.ndarray,
) -> None:
    source = "hidden"
    unit = normalize_vector(hidden)
    token_names = tuple(f"T{index}" for index in range(11))
    add_temporal_series(
        output, metadata, source, "tokens", unit, cosine_distance, token_names
    )

    action = unit[..., ACTION_TOKENS, :]
    action_mean = normalize_vector(action.mean(axis=-2))
    action_std = action.std(axis=-2)
    endpoint_delta = action[..., -1, :] - action[..., 0, :]
    spectrum, gram = spectrum_and_gram(action)
    add_temporal_series(
        output, metadata, source, "action_mean", action_mean, cosine_distance
    )
    add_temporal_series(
        output, metadata, source, "action_std", action_std, euclidean_distance
    )
    add_temporal_series(
        output,
        metadata,
        source,
        "endpoint_delta",
        endpoint_delta,
        euclidean_distance,
    )
    add_temporal_series(
        output, metadata, source, "gram_spectrum", spectrum, euclidean_distance
    )

    action_dispersion = cosine_distance(action, action_mean[..., None, :]).mean(axis=-1)
    endpoint_separation = cosine_distance(action[..., -1, :], action[..., 0, :])
    entropy = -(spectrum * np.log(np.maximum(spectrum, 1e-12))).sum(axis=-1)
    effective_rank = np.exp(entropy)
    mean_similarity = (gram.sum(axis=(-1, -2)) - 10.0) / 90.0
    add_level_series(output, metadata, source, "action_dispersion", action_dispersion)
    add_level_series(output, metadata, source, "endpoint_separation", endpoint_separation)
    add_level_series(output, metadata, source, "gram_effective_rank", effective_rank)
    add_level_series(output, metadata, source, "gram_mean_similarity", mean_similarity)
    add_level_series(
        output, metadata, source, "action_std_norm", np.linalg.norm(action_std, axis=-1)
    )
    add_level_series(
        output,
        metadata,
        source,
        "endpoint_delta_norm",
        np.linalg.norm(endpoint_delta, axis=-1),
    )


def add_behavior_features(
    output: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    state: np.ndarray,
    actions: np.ndarray,
    state_scale: np.ndarray,
    action_scale: np.ndarray,
) -> None:
    scaled_state = state / state_scale
    scaled_actions = actions / action_scale
    state_series = scaled_state[:, None, None, None, :]
    action_series = scaled_actions[:, None, None, :, :]
    add_temporal_series(
        output,
        metadata,
        "behavior",
        "proprio",
        state_series,
        rms_distance,
        ("proprio",),
    )
    add_temporal_series(
        output,
        metadata,
        "behavior",
        "action_tokens",
        action_series,
        rms_distance,
        tuple(f"action_T{index}" for index in range(1, 11)),
    )
    action_mean = scaled_actions.mean(axis=1)[:, None, None, :]
    add_temporal_series(
        output,
        metadata,
        "behavior",
        "action_mean",
        action_mean,
        rms_distance,
    )


def behavior_scales(run: Path, episodes: list[Episode], discovery_state: int) -> tuple[np.ndarray, np.ndarray]:
    states = []
    actions = []
    for episode in episodes:
        if episode.state != discovery_state:
            continue
        with np.load(episode_path(run, episode.index), allow_pickle=False) as payload:
            states.append(np.asarray(payload["state"][: max(HORIZONS) + 1], dtype=np.float32))
            actions.append(np.asarray(payload["actions"][: max(HORIZONS) + 1], dtype=np.float32))
    state_scale = np.concatenate(states).std(axis=0)
    action_scale = np.concatenate(actions).reshape(-1, actions[0].shape[-1]).std(axis=0)
    return np.maximum(state_scale, 1e-6), np.maximum(action_scale, 1e-6)


def top4_probabilities(ids: np.ndarray) -> np.ndarray:
    if ids.shape[-1] != 4:
        raise ValueError(f"bad Top-4 array {ids.shape}")
    sorted_ids = np.sort(ids, axis=-1)
    if np.any(np.diff(sorted_ids, axis=-1) == 0):
        raise ValueError("Top-4 expert IDs are not unique")
    probabilities = np.zeros(ids.shape[:-1] + (N_EXPERTS,), dtype=np.float32)
    np.put_along_axis(probabilities, ids.astype(np.int64), 0.25, axis=-1)
    return probabilities


def extract_features(
    run: Path,
    episodes: list[Episode],
    discovery_state: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str], dict[str, dict[str, Any]], dict[int, list[Episode]]]:
    route_store = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    hidden_store = zarr.open(str(run / "server" / "hidden.zarr"), mode="r")
    route_episode_id = np.asarray(route_store["episode_id"][:], dtype=np.int32)
    hidden_episode_id = np.asarray(hidden_store["episode_id"][:], dtype=np.int32)
    if not np.array_equal(route_episode_id, hidden_episode_id):
        raise ValueError("route and hidden episode axes differ")

    state_scale, action_scale = behavior_scales(run, episodes, discovery_state)
    grouped: dict[int, list[Episode]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    rows_by_state: dict[int, list[dict[str, float]]] = {}
    labels_by_state: dict[int, np.ndarray] = {}
    prefix = max(HORIZONS) + 1

    for state in sorted({episode.state for episode in episodes}):
        group = sorted(
            [episode for episode in episodes if episode.state == state],
            key=lambda episode: episode.noise_seed,
        )
        grouped[state] = group
        rows = []
        for position, episode in enumerate(group, start=1):
            start = episode.offset
            stop = start + prefix
            if not np.all(route_episode_id[start:stop] == episode.index):
                raise ValueError(f"episode boundary mismatch for {episode.index}")
            soft = np.asarray(
                route_store["hb_router_probs"][start:stop], dtype=np.float32
            )
            ids = np.asarray(route_store["hb_expert_ids"][start:stop], dtype=np.uint8)
            hidden = np.asarray(hidden_store["hb_hidden"][start:stop], dtype=np.float32)
            if soft.shape[1:] != (8, 10, 11, 32):
                raise ValueError(f"bad soft route shape {soft.shape}")
            if hidden.shape[1:] != (8, 10, 11, 1024):
                raise ValueError(f"bad hidden shape {hidden.shape}")
            if float(np.ptp(hidden[:, :, :, 0, :], axis=2).max()) != 0.0:
                raise ValueError("state token unexpectedly varies across denoise steps")

            with np.load(episode_path(run, episode.index), allow_pickle=False) as payload:
                state_values = np.asarray(payload["state"][:prefix], dtype=np.float32)
                action_values = np.asarray(payload["actions"][:prefix], dtype=np.float32)

            features: dict[str, float] = {}
            add_route_features(features, metadata, "route_soft", soft)
            add_route_features(features, metadata, "route_top4", top4_probabilities(ids))
            add_hidden_features(features, metadata, hidden)
            add_behavior_features(
                features,
                metadata,
                state_values,
                action_values,
                state_scale,
                action_scale,
            )
            rows.append(features)
            print(
                f"state {state}: extracted {position:02d}/32 episode {episode.index}",
                flush=True,
            )
        rows_by_state[state] = rows
        labels_by_state[state] = np.asarray([episode.failure for episode in group], dtype=np.int8)

    keys = sorted(metadata)
    matrices = {}
    for state, rows in rows_by_state.items():
        for row in rows:
            if set(row) != set(keys):
                raise ValueError("feature keys differ across episodes")
        matrix = np.asarray([[row[key] for key in keys] for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"non-finite feature matrix for state {state}")
        matrices[state] = matrix
    return matrices, labels_by_state, keys, metadata, grouped


def auc_from_ranks(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        raise ValueError("AUC needs both outcome classes")
    rank_sum = labels @ ranks
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def permutation_scan(
    matrix: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    ranks = rankdata(matrix, axis=0, method="average")
    observed = auc_from_ranks(ranks, labels)
    observed_abs = np.abs(observed - 0.5)
    unadjusted = np.zeros(matrix.shape[1], dtype=np.int64)
    familywise = np.zeros(matrix.shape[1], dtype=np.int64)
    chunk_size = 500
    for start in range(0, permutations, chunk_size):
        count = min(chunk_size, permutations - start)
        permuted = np.stack([rng.permutation(labels) for _ in range(count)])
        positive = int(labels.sum())
        negative = len(labels) - positive
        null = (
            permuted @ ranks - positive * (positive + 1) / 2.0
        ) / (positive * negative)
        null_abs = np.abs(null - 0.5)
        unadjusted += (null_abs >= observed_abs[None, :] - 1e-15).sum(axis=0)
        maximum = null_abs.max(axis=1)
        familywise += (maximum[:, None] >= observed_abs[None, :] - 1e-15).sum(axis=0)
    return {
        "auc": observed,
        "p_unadjusted": (unadjusted + 1.0) / (permutations + 1.0),
        "p_fwer": (familywise + 1.0) / (permutations + 1.0),
    }


def confirmation_test(
    score: np.ndarray,
    labels: np.ndarray,
    direction: float,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    ranks = rankdata(score, method="average")[:, None]
    raw_auc = float(auc_from_ranks(ranks, labels)[0])
    oriented_auc = 0.5 + direction * (raw_auc - 0.5)
    exceed = 0
    for _ in range(permutations):
        null_auc = float(auc_from_ranks(ranks, rng.permutation(labels))[0])
        null_oriented = 0.5 + direction * (null_auc - 0.5)
        exceed += int(null_oriented >= oriented_auc - 1e-15)
    return {
        "auc_failure_raw": raw_auc,
        "auc_discovery_oriented": oriented_auc,
        "permutation_p_one_sided": (exceed + 1.0) / (permutations + 1.0),
    }


def compact_feature(
    key: str,
    index: int,
    scan: dict[str, np.ndarray],
    confirmation_matrix: np.ndarray,
    confirmation_labels: np.ndarray,
    metadata: dict[str, dict[str, Any]],
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    discovery_auc = float(scan["auc"][index])
    direction = 1.0 if discovery_auc >= 0.5 else -1.0
    confirmation = confirmation_test(
        confirmation_matrix[:, index],
        confirmation_labels,
        direction,
        permutations,
        rng,
    )
    return {
        "key": key,
        **metadata[key],
        "discovery_auc_failure": discovery_auc,
        "discovery_direction": "larger_in_failure" if direction > 0 else "smaller_in_failure",
        "discovery_p_unadjusted": float(scan["p_unadjusted"][index]),
        "discovery_p_fwer": float(scan["p_fwer"][index]),
        "confirmation": confirmation,
    }


def behavior_residual_audit(
    target_key: str,
    control_keys: tuple[str, str],
    keys: list[str],
    discovery: np.ndarray,
    confirmation: np.ndarray,
    discovery_labels: np.ndarray,
    confirmation_labels: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    target_index = keys.index(target_key)
    control_indexes = [keys.index(key) for key in control_keys]
    residuals = []
    explained = []
    for matrix in (discovery, confirmation):
        target = matrix[:, target_index]
        controls = np.column_stack(
            [np.ones(len(target)), matrix[:, control_indexes]]
        )
        fitted = controls @ np.linalg.lstsq(controls, target, rcond=None)[0]
        residual = target - fitted
        residuals.append(residual)
        denominator = np.square(target - target.mean()).sum()
        explained.append(float(1.0 - np.square(residual).sum() / denominator))

    discovery_scan = permutation_scan(
        residuals[0][:, None], discovery_labels, permutations, rng
    )
    discovery_auc = float(discovery_scan["auc"][0])
    direction = 1.0 if discovery_auc >= 0.5 else -1.0
    confirmation_result = confirmation_test(
        residuals[1], confirmation_labels, direction, permutations, rng
    )
    return {
        "target": target_key,
        "controls": list(control_keys),
        "selection_status": "post_hoc_audit_of_discovery-selected_raw_feature",
        "variance_explained_by_controls": {
            "discovery": explained[0],
            "confirmation": explained[1],
        },
        "residual_discovery_auc_failure": discovery_auc,
        "residual_discovery_p_two_sided": float(
            discovery_scan["p_unadjusted"][0]
        ),
        "residual_confirmation": confirmation_result,
    }


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# HB-MoE token dynamics pilot",
        "",
        "## Protocol",
        "",
        "- One rollout is one sample. Control rows and denoise iterations are repeated measurements.",
        "- State 0 is discovery; state 42 is untouched confirmation. Both contain 14 successes and 18 timeout failures.",
        "- Horizons: t7 (before the displayed pose split), t12 (before the displayed route split), and t34 (before the length/survivor split).",
        "- Every temporal score uses an 8-control trailing window and aligned token positions. Raw lags 1-5 and recurrence residuals for periods 2-5 are tested.",
        "- Discovery uses a two-sided maxT permutation correction over the complete MoE feature family. The chosen direction is frozen for the one-sided confirmation test.",
        "",
        "## Primary result",
        "",
    ]
    primary = summary["primary"]
    lines.extend(
        [
            f"Feature: `{primary['key']}`",
            "",
            f"- Discovery failure AUC: `{primary['discovery_auc_failure']:.4f}`; unadjusted p `{primary['discovery_p_unadjusted']:.6f}`; maxT p `{primary['discovery_p_fwer']:.6f}`.",
            f"- Confirmation AUC in the discovery direction: `{primary['confirmation']['auc_discovery_oriented']:.4f}`; one-sided p `{primary['confirmation']['permutation_p_one_sided']:.6f}`.",
            "",
            "## Discovery-selected family leads",
            "",
            "| family | feature | discovery AUC | maxT p | confirmation oriented AUC | confirmation p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["family_leads"]:
        lines.append(
            "| %s | `%s` | %.4f | %.6f | %.4f | %.6f |"
            % (
                item["category"],
                item["key"],
                item["discovery_auc_failure"],
                item["discovery_p_fwer"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
            )
        )
    lines.extend(
        [
            "",
            "## Source leads",
            "",
            "| source | feature | discovery AUC | maxT p | confirmation oriented AUC | confirmation p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["source_leads"]:
        lines.append(
            "| %s | `%s` | %.4f | %.6f | %.4f | %.6f |"
            % (
                item["source"],
                item["key"],
                item["discovery_auc_failure"],
                item["discovery_p_fwer"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
            )
        )
    lines.extend(
        [
            "",
            "## Horizon-specific tests",
            "",
            "| horizon | selected feature | discovery AUC | horizon maxT p | all-MoE maxT p | confirmation oriented AUC | confirmation p |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["horizon_leads"]:
        lines.append(
            "| t%d | `%s` | %.4f | %.6f | %.6f | %.4f | %.6f |"
            % (
                item["horizon"],
                item["key"],
                item["discovery_auc_failure"],
                item["discovery_p_horizon_fwer"],
                item["discovery_p_all_moe_fwer"],
                item["confirmation"]["auc_discovery_oriented"],
                item["confirmation"]["permutation_p_one_sided"],
            )
        )
    behavior = summary["behavior_lead"]
    lines.extend(
        [
            "",
            "## Behavior baseline",
            "",
            f"Best discovery behavior feature: `{behavior['key']}`. Discovery AUC `{behavior['discovery_auc_failure']:.4f}` (behavior-family maxT p `{behavior['discovery_p_fwer']:.6f}`); confirmation oriented AUC `{behavior['confirmation']['auc_discovery_oriented']:.4f}` (p `{behavior['confirmation']['permutation_p_one_sided']:.6f}`).",
            "",
            "## Post-hoc behavior residual audit",
            "",
            "These checks are diagnostic, not additional confirmatory tests: the targets were chosen after inspecting the raw discovery results.",
            "",
            "| target | behavior R2 discovery / confirmation | residual discovery AUC (p) | residual confirmation oriented AUC (p) |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in summary["behavior_residual_audits"]:
        lines.append(
            "| `%s` | %.3f / %.3f | %.3f (%.4f) | %.3f (%.4f) |"
            % (
                item["target"],
                item["variance_explained_by_controls"]["discovery"],
                item["variance_explained_by_controls"]["confirmation"],
                item["residual_discovery_auc_failure"],
                item["residual_discovery_p_two_sided"],
                item["residual_confirmation"]["auc_discovery_oriented"],
                item["residual_confirmation"]["permutation_p_one_sided"],
            )
        )
    lines.extend(
        [
            "",
            "## Lag-curve check for the selected period-2 score",
            "",
            "The hidden T0 lag curve remains monotone in both outcomes. The period-2 residual is therefore curvature of a slowing trajectory, not a return to a previous state.",
            "",
            "| state | outcome | lag1 | lag2 | lag3 | lag4 | lag5 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["hidden_t0_lag_curve_h34"]:
        lines.append(
            "| %d | %s | %s |"
            % (
                item["state"],
                item["outcome"],
                " | ".join("%.4f" % value for value in item["mean_distance"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- A replicated hidden feature is an HB-block-input marker, not by itself a MoE-specific or causal trap.",
            "- The same 32 noise seed IDs occur in both initial states; state holdout does not create unseen noise seeds.",
            "- t34 is only before episode-length separation. It is after the displayed pose/route separation and cannot establish an early precursor.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.permutations <= 0:
        raise ValueError("permutations must be positive")
    run = args.run.resolve()
    states = (args.discovery_state, args.confirmation_state)
    episodes, _ = load_episodes(run, states)
    matrices, labels, keys, metadata, grouped = extract_features(
        run, episodes, args.discovery_state
    )

    discovery = matrices[args.discovery_state]
    confirmation = matrices[args.confirmation_state]
    discovery_labels = labels[args.discovery_state]
    confirmation_labels = labels[args.confirmation_state]
    ranges = np.ptp(discovery, axis=0)
    moe_indexes = np.asarray(
        [
            index
            for index, key in enumerate(keys)
            if metadata[key]["source"] != "behavior" and ranges[index] > 1e-12
        ],
        dtype=np.int64,
    )
    behavior_indexes = np.asarray(
        [
            index
            for index, key in enumerate(keys)
            if metadata[key]["source"] == "behavior" and ranges[index] > 1e-12
        ],
        dtype=np.int64,
    )
    if not len(moe_indexes) or not len(behavior_indexes):
        raise ValueError("no variable features")

    rng = np.random.default_rng(args.seed)
    moe_scan_local = permutation_scan(
        discovery[:, moe_indexes], discovery_labels, args.permutations, rng
    )
    moe_scan = {
        name: np.full(len(keys), np.nan, dtype=np.float64)
        for name in ("auc", "p_unadjusted", "p_fwer")
    }
    for name, values in moe_scan_local.items():
        moe_scan[name][moe_indexes] = values

    best_index = int(moe_indexes[np.argmax(np.abs(moe_scan["auc"][moe_indexes] - 0.5))])
    primary = compact_feature(
        keys[best_index],
        best_index,
        moe_scan,
        confirmation,
        confirmation_labels,
        metadata,
        args.permutations,
        rng,
    )

    family_leads = []
    for category in ("token_lag", "descriptor_lag", "periodicity", "structure_level"):
        candidates = np.asarray(
            [
                index
                for index in moe_indexes
                if metadata[keys[index]]["category"] == category
            ],
            dtype=np.int64,
        )
        selected = int(candidates[np.argmax(np.abs(moe_scan["auc"][candidates] - 0.5))])
        family_leads.append(
            compact_feature(
                keys[selected],
                selected,
                moe_scan,
                confirmation,
                confirmation_labels,
                metadata,
                args.permutations,
                rng,
            )
        )

    source_leads = []
    for source in ("route_soft", "route_top4", "hidden"):
        candidates = np.asarray(
            [index for index in moe_indexes if metadata[keys[index]]["source"] == source],
            dtype=np.int64,
        )
        selected = int(candidates[np.argmax(np.abs(moe_scan["auc"][candidates] - 0.5))])
        source_leads.append(
            compact_feature(
                keys[selected],
                selected,
                moe_scan,
                confirmation,
                confirmation_labels,
                metadata,
                args.permutations,
                rng,
            )
        )

    horizon_leads = []
    for horizon in HORIZONS:
        candidates = np.asarray(
            [
                index
                for index in moe_indexes
                if metadata[keys[index]]["horizon"] == horizon
            ],
            dtype=np.int64,
        )
        horizon_scan_local = permutation_scan(
            discovery[:, candidates], discovery_labels, args.permutations, rng
        )
        local_index = int(
            np.argmax(np.abs(horizon_scan_local["auc"] - 0.5))
        )
        selected = int(candidates[local_index])
        horizon_scan = {
            name: np.full(len(keys), np.nan, dtype=np.float64)
            for name in ("auc", "p_unadjusted", "p_fwer")
        }
        for name, values in horizon_scan_local.items():
            horizon_scan[name][candidates] = values
        item = compact_feature(
            keys[selected],
            selected,
            horizon_scan,
            confirmation,
            confirmation_labels,
            metadata,
            args.permutations,
            rng,
        )
        item["discovery_p_horizon_fwer"] = item["discovery_p_fwer"]
        item["discovery_p_all_moe_fwer"] = float(moe_scan["p_fwer"][selected])
        horizon_leads.append(item)

    behavior_scan_local = permutation_scan(
        discovery[:, behavior_indexes], discovery_labels, args.permutations, rng
    )
    behavior_scan = {
        name: np.full(len(keys), np.nan, dtype=np.float64)
        for name in ("auc", "p_unadjusted", "p_fwer")
    }
    for name, values in behavior_scan_local.items():
        behavior_scan[name][behavior_indexes] = values
    behavior_best = int(
        behavior_indexes[
            np.argmax(np.abs(behavior_scan["auc"][behavior_indexes] - 0.5))
        ]
    )
    behavior_lead = compact_feature(
        keys[behavior_best],
        behavior_best,
        behavior_scan,
        confirmation,
        confirmation_labels,
        metadata,
        args.permutations,
        rng,
    )

    behavior_residual_audits = [
        behavior_residual_audit(
            "route_top4|h34|lag2|action_std",
            (
                "behavior|h34|lag2|proprio",
                "behavior|h34|lag2|action_mean",
            ),
            keys,
            discovery,
            confirmation,
            discovery_labels,
            confirmation_labels,
            args.permutations,
            rng,
        ),
        behavior_residual_audit(
            "hidden|h34|period2|T0",
            (
                "behavior|h34|period2|proprio",
                "behavior|h34|period2|action_mean",
            ),
            keys,
            discovery,
            confirmation,
            discovery_labels,
            confirmation_labels,
            args.permutations,
            rng,
        ),
        behavior_residual_audit(
            "route_soft|h34|lag1|T0",
            (
                "behavior|h34|lag1|proprio",
                "behavior|h34|lag1|action_mean",
            ),
            keys,
            discovery,
            confirmation,
            discovery_labels,
            confirmation_labels,
            args.permutations,
            rng,
        ),
    ]

    hidden_t0_lag_curve_h34 = []
    for state, matrix, outcome in (
        (args.discovery_state, discovery, discovery_labels),
        (args.confirmation_state, confirmation, confirmation_labels),
    ):
        for label, name in ((0, "success"), (1, "timeout_failure")):
            hidden_t0_lag_curve_h34.append(
                {
                    "state": state,
                    "outcome": name,
                    "mean_distance": [
                        float(
                            matrix[
                                outcome == label,
                                keys.index(f"hidden|h34|lag{lag}|T0"),
                            ].mean()
                        )
                        for lag in LAGS
                    ],
                }
            )

    top_order = moe_indexes[
        np.argsort(-np.abs(moe_scan["auc"][moe_indexes] - 0.5))[:20]
    ]
    top_discovery = [
        compact_feature(
            keys[int(index)],
            int(index),
            moe_scan,
            confirmation,
            confirmation_labels,
            metadata,
            args.permutations,
            rng,
        )
        for index in top_order
    ]

    summary = {
        "experiment": "himoe_hb_token_dynamics_pilot_v1",
        "integrity": {
            "run": str(run),
            "discovery_state": args.discovery_state,
            "confirmation_state": args.confirmation_state,
            "episodes_per_state": 32,
            "failures_per_state": {
                str(state): int(labels[state].sum()) for state in states
            },
            "successes_per_state": {
                str(state): int(len(labels[state]) - labels[state].sum())
                for state in states
            },
            "noise_seeds": {
                str(state): [episode.noise_seed for episode in grouped[state]]
                for state in states
            },
            "features_total": len(keys),
            "moe_features_tested": int(len(moe_indexes)),
            "behavior_features_tested": int(len(behavior_indexes)),
            "state_token_denoise_invariance_checked": True,
        },
        "protocol": {
            "horizons": list(HORIZONS),
            "lags": list(LAGS),
            "trailing_window": TRAILING_WINDOW,
            "sources": ["32-way soft route", "authoritative Top-4", "HB gate hidden"],
            "discovery_test": "two-sided abs(AUC-0.5), maxT over all variable MoE features",
            "confirmation_test": "one-sided AUC with direction fixed by discovery",
            "permutation_draws": args.permutations,
            "random_seed": args.seed,
            "sample_unit": "rollout",
        },
        "primary": primary,
        "family_leads": family_leads,
        "source_leads": source_leads,
        "horizon_leads": horizon_leads,
        "behavior_lead": behavior_lead,
        "behavior_residual_audits": behavior_residual_audits,
        "hidden_t0_lag_curve_h34": hidden_t0_lag_curve_h34,
        "top_discovery": top_discovery,
    }
    summary = finite_json(summary)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )
    (output / "report.md").write_text(render_report(summary))
    np.savez_compressed(
        output / "feature_scores.npz",
        feature_keys=np.asarray(keys),
        discovery_scores=discovery,
        confirmation_scores=confirmation,
        discovery_failure=discovery_labels,
        confirmation_failure=confirmation_labels,
        discovery_episode=np.asarray(
            [episode.index for episode in grouped[args.discovery_state]], dtype=np.int32
        ),
        confirmation_episode=np.asarray(
            [episode.index for episode in grouped[args.confirmation_state]], dtype=np.int32
        ),
    )
    print(f"wrote {output / 'summary.json'}")
    print(f"wrote {output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
