#!/usr/bin/env python3
"""Cross-fitted offline simulation of a causal HB-MoE failure alarm.

The detector uses only routing values available at the current query and its
past.  Success episodes in disjoint flow-noise-seed folds define the reference
CDF and alarm threshold; no failure label is used to fit a threshold or weight.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr


HERE = Path(__file__).resolve().parent
HUB = HERE.parent
EPISODE_LABELS = HUB / "physical-failure-labels/results/episodes.csv"
DEFAULT_OUTPUT = HERE / "results"

HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
TOKENS = ("state",) + tuple(f"T{i}" for i in range(1, 11))
FEATURE_NAMES = (
    "back_top4_churn",
    "back_tail_T6_T10_top4_churn",
    "front_top4_churn",
    "back_action_entropy",
    "front_action_entropy",
)
FEATURE_DIRECTION = "lower_is_riskier"
N_EXPERTS = 32
TOP_K = 4
N_FOLDS = 5
ROLLING_WINDOW = 3
PERSISTENCE = 2
MIN_INIT_REFERENCE = 3
MIN_RUN_REFERENCE = 20
TARGET_FPRS = (0.01, 0.05, 0.10)
PRIMARY_FPR = 0.05
HORIZONS = (7, 12, 20, 27, 34)
REASON_ZH = {
    "object_released_or_dropped_before_goal": "目标前释放或掉落",
    "stable_grasp_not_observed": "未观测到稳定抓取",
    "object_moved_but_goal_unmet": "物体移动但目标未满足",
    "goal_predicate_regressed": "目标谓词回退",
    "timeout_while_holding_target": "持物超时",
    "object_released_outside_goal": "目标区外释放",
    "approached_target_without_observed_contact": "接近但未观测到接触",
    "mechanism_threshold_not_reached": "机构阈值未达到",
    "no_meaningful_target_progress": "无显著目标进展",
}
REASON_SHORT = {
    "object_released_or_dropped_before_goal": "drop / early release",
    "stable_grasp_not_observed": "no stable grasp",
    "object_moved_but_goal_unmet": "moved, goal unmet",
    "goal_predicate_regressed": "goal regressed",
    "timeout_while_holding_target": "holding timeout",
    "object_released_outside_goal": "released outside goal",
    "approached_target_without_observed_contact": "approach, no contact",
    "mechanism_threshold_not_reached": "mechanism shortfall",
    "no_meaningful_target_progress": "no progress",
}


@dataclass(frozen=True)
class EpisodeSpec:
    episode_row: int
    data_root: str
    source_run: str
    suite: str
    task_name: str
    run_id: str
    checkpoint_sha256: str
    episode_index: int
    init_state_id: int
    flow_noise_seed: int
    fold: int
    success: bool
    primary_failure_reason: str
    run_regime: str
    query_start: int
    inference_calls: int


@dataclass(frozen=True)
class RunSpec:
    source_run: str
    query_start: int
    query_rows: int
    episode_indices: tuple[int, ...]
    episode_lengths: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=EPISODE_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def run_regime(run_id: str) -> str:
    if run_id == "pin-base":
        return "duplicate_pin_base"
    if run_id in ("pin-on", "pin-off"):
        return "front_layer_intervention"
    if run_id == "pin-smoke":
        return "smoke"
    if run_id.startswith("right-"):
        return "observational"
    raise ValueError(f"unknown run regime {run_id!r}")


def build_index(path: Path) -> tuple[list[EpisodeSpec], list[RunSpec], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    keyed = {
        (row["source_run"], int(row["episode_index"])): row for row in raw
    }
    if len(keyed) != len(raw):
        raise ValueError("episode label index contains duplicate rows")

    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        by_run[row["source_run"]].append(row)

    episodes: list[EpisodeSpec] = []
    runs: list[RunSpec] = []
    query_cursor = 0
    checkpoints: set[str] = set()
    for source_run in sorted(by_run):
        run_path = HUB / source_run
        summaries = sorted(
            read_json(run_path / "client/summaries.json"),
            key=lambda item: int(item["episode_index"]),
        )
        metadata = read_json(run_path / "meta.json")
        server = read_json(run_path / "client/server_metadata.json")
        checkpoint = str(server["checkpoint_sha256"])
        checkpoints.add(checkpoint)
        csv_rows = by_run[source_run]
        if len(csv_rows) != len(summaries):
            raise ValueError(f"{source_run}: labels/summaries row count differs")

        run_start = query_cursor
        indices: list[int] = []
        lengths: list[int] = []
        for summary in summaries:
            episode_index = int(summary["episode_index"])
            label = keyed[(source_run, episode_index)]
            success = parse_bool(label["recorded_success"])
            if success != bool(summary["success"]):
                raise ValueError(f"{source_run}/{episode_index}: outcome mismatch")
            if int(label["init_state_id"]) != int(summary["init_state_id"]):
                raise ValueError(f"{source_run}/{episode_index}: init-state mismatch")
            if int(label["flow_noise_seed"]) != int(summary["flow_noise_seed"]):
                raise ValueError(f"{source_run}/{episode_index}: seed mismatch")
            length = int(summary["inference_calls"])
            if length <= 0:
                raise ValueError(f"{source_run}/{episode_index}: empty episode")
            seed = int(summary["flow_noise_seed"])
            current_run_id = str(metadata["run_id"])
            reason = label["primary_failure_reason"].strip() or "success"
            episodes.append(
                EpisodeSpec(
                    episode_row=len(episodes),
                    data_root=str(label["data_root"]),
                    source_run=source_run,
                    suite=str(label["suite"]),
                    task_name=str(label["task_name"]),
                    run_id=current_run_id,
                    checkpoint_sha256=checkpoint,
                    episode_index=episode_index,
                    init_state_id=int(summary["init_state_id"]),
                    flow_noise_seed=seed,
                    fold=seed % N_FOLDS,
                    success=success,
                    primary_failure_reason=reason,
                    run_regime=run_regime(current_run_id),
                    query_start=query_cursor,
                    inference_calls=length,
                )
            )
            indices.append(episode_index)
            lengths.append(length)
            query_cursor += length
        runs.append(
            RunSpec(
                source_run=source_run,
                query_start=run_start,
                query_rows=query_cursor - run_start,
                episode_indices=tuple(indices),
                episode_lengths=tuple(lengths),
            )
        )

    integrity = {
        "episodes": len(episodes),
        "successes": sum(item.success for item in episodes),
        "failures": sum(not item.success for item in episodes),
        "source_runs": len(runs),
        "query_rows": query_cursor,
        "checkpoints": len(checkpoints),
        "run_regime_episodes": dict(Counter(item.run_regime for item in episodes)),
        "run_regime_failures": dict(
            Counter(item.run_regime for item in episodes if not item.success)
        ),
    }
    return episodes, runs, integrity


def top4_mask(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint8)
    if ids.shape[-1] != TOP_K:
        raise ValueError(f"expected Top-{TOP_K} IDs, found {ids.shape}")
    if np.any(ids >= N_EXPERTS):
        raise ValueError("expert ID outside [0, 31]")
    mask = np.zeros(ids.shape[:-1], dtype=np.uint32)
    for position in range(TOP_K):
        mask |= np.left_shift(np.uint32(1), ids[..., position].astype(np.uint32))
    if np.any(np.bitwise_count(mask) != TOP_K):
        raise ValueError("duplicate expert ID in a Top-4 set")
    return mask


def mask_churn(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("Top-4 masks must have equal shape")
    intersection = np.bitwise_count(
        np.asarray(left, np.uint32) & np.asarray(right, np.uint32)
    ).astype(np.float32)
    return 1.0 - intersection / np.maximum(2.0 * TOP_K - intersection, 1.0)


def extract_run_features(run: RunSpec) -> dict[str, Any]:
    path = HUB / run.source_run / "server/routes.zarr"
    store = zarr.open_group(str(path), mode="r")
    episode_axis = np.asarray(store["episode_id"][:], dtype=np.int32)
    expected_axis = np.repeat(
        np.asarray(run.episode_indices, dtype=np.int32),
        np.asarray(run.episode_lengths, dtype=np.int32),
    )
    if not np.array_equal(episode_axis, expected_axis):
        raise ValueError(f"{run.source_run}: route episode alignment failed")
    if len(episode_axis) != run.query_rows:
        raise ValueError(f"{run.source_run}: route row count differs")

    ids = np.asarray(store["hb_expert_ids"][:], dtype=np.uint8)
    entropy = np.asarray(store["hb_entropy"][:], dtype=np.float16)
    expected_ids = (run.query_rows, len(HB_LAYERS), 10, len(TOKENS), TOP_K)
    expected_entropy = expected_ids[:-1]
    if ids.shape != expected_ids or entropy.shape != expected_entropy:
        raise ValueError(
            f"{run.source_run}: unexpected tensors {ids.shape}, {entropy.shape}"
        )

    mask = top4_mask(ids)
    del ids
    output = np.full((run.query_rows, len(FEATURE_NAMES)), np.nan, np.float32)
    if run.query_rows > 1:
        churn = mask_churn(mask[1:], mask[:-1])
        output[1:, 0] = churn[:, 4:, :, 1:].mean(axis=(1, 2, 3), dtype=np.float32)
        output[1:, 1] = churn[:, 4:, :, 6:].mean(axis=(1, 2, 3), dtype=np.float32)
        output[1:, 2] = churn[:, :4, :, 1:].mean(axis=(1, 2, 3), dtype=np.float32)
        boundary_rows = np.flatnonzero(episode_axis[1:] != episode_axis[:-1]) + 1
        output[boundary_rows, :3] = np.nan
        del churn
    output[:, 3] = entropy[:, 4:, :, 1:].mean(
        axis=(1, 2, 3), dtype=np.float32
    ) / math.log(N_EXPERTS)
    output[:, 4] = entropy[:, :4, :, 1:].mean(
        axis=(1, 2, 3), dtype=np.float32
    ) / math.log(N_EXPERTS)
    if np.any((output[:, 3:] < 0.0) | (output[:, 3:] > 1.0005)):
        raise ValueError(f"{run.source_run}: entropy outside normalized range")
    return {
        "source_run": run.source_run,
        "query_start": run.query_start,
        "features": output,
        "finite_rows": int(np.isfinite(output).all(axis=1).sum()),
    }


def extract_all_features(
    runs: list[RunSpec], total_rows: int, workers: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.full((total_rows, len(FEATURE_NAMES)), np.nan, np.float32)
    audits: list[dict[str, Any]] = []
    if workers <= 1:
        iterator: Iterable[dict[str, Any]] = (
            extract_run_features(run) for run in runs
        )
        for position, result in enumerate(iterator, start=1):
            start = int(result["query_start"])
            values = result.pop("features")
            output[start : start + len(values)] = values
            audits.append(result)
            print(f"  extracted run {position}/{len(runs)}: {result['source_run']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_run_features, run): run for run in runs}
            for position, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                start = int(result["query_start"])
                values = result.pop("features")
                output[start : start + len(values)] = values
                audits.append(result)
                print(
                    f"  extracted run {position}/{len(runs)}: {result['source_run']}",
                    flush=True,
                )
    return output, sorted(audits, key=lambda item: item["source_run"])


def save_feature_cache(
    path: Path,
    features: np.ndarray,
    episodes: list[EpisodeSpec],
    audits: list[dict[str, Any]],
) -> None:
    np.savez_compressed(
        path,
        features=features,
        feature_names=np.asarray(FEATURE_NAMES),
        episode_starts=np.asarray([item.query_start for item in episodes], np.int64),
        episode_lengths=np.asarray([item.inference_calls for item in episodes], np.int32),
        source_runs=np.asarray([item.source_run for item in episodes]),
        episode_indices=np.asarray([item.episode_index for item in episodes], np.int32),
        extraction_audit_json=np.asarray(json.dumps(plain(audits), sort_keys=True)),
    )


def load_feature_cache(path: Path, episodes: list[EpisodeSpec]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"feature cache does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if tuple(payload["feature_names"].tolist()) != FEATURE_NAMES:
            raise ValueError("cached feature schema differs")
        checks = (
            np.array_equal(
                payload["episode_starts"],
                np.asarray([item.query_start for item in episodes], np.int64),
            ),
            np.array_equal(
                payload["episode_lengths"],
                np.asarray([item.inference_calls for item in episodes], np.int32),
            ),
            np.array_equal(
                payload["source_runs"],
                np.asarray([item.source_run for item in episodes]),
            ),
            np.array_equal(
                payload["episode_indices"],
                np.asarray([item.episode_index for item in episodes], np.int32),
            ),
        )
        if not all(checks):
            raise ValueError("cached feature episode index differs")
        features = np.asarray(payload["features"], np.float32)
        audits = json.loads(str(payload["extraction_audit_json"].item()))
    return features, audits


def rolling_features(
    features: np.ndarray, starts: np.ndarray, lengths: np.ndarray, window: int
) -> np.ndarray:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    output = np.full_like(features, np.nan, dtype=np.float32)
    for start, length in zip(starts, lengths):
        start = int(start)
        length = int(length)
        if length <= window:
            continue
        transitions = np.asarray(features[start + 1 : start + length], np.float64)
        if np.any(~np.isfinite(transitions)):
            raise ValueError("non-finite feature inside an episode transition sequence")
        cumulative = np.vstack(
            (np.zeros((1, features.shape[1]), np.float64), np.cumsum(transitions, axis=0))
        )
        output[start + window : start + length] = (
            cumulative[window:] - cumulative[:-window]
        ) / float(window)
    return output


def empirical_lower_tail_risk(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    reference = np.asarray(reference, np.float32)
    if values.ndim != 2 or reference.ndim != 2 or values.shape[1] != reference.shape[1]:
        raise ValueError("risk transform expects aligned two-dimensional matrices")
    if not len(reference) or np.any(~np.isfinite(reference)):
        raise ValueError("risk reference must be non-empty and finite")
    result = np.empty_like(values, dtype=np.float32)
    for feature in range(values.shape[1]):
        ordered = np.sort(reference[:, feature])
        lower = np.searchsorted(ordered, values[:, feature], side="left")
        upper = np.searchsorted(ordered, values[:, feature], side="right")
        result[:, feature] = 1.0 - (lower + upper) / (2.0 * len(ordered))
    return result


def sustained_scores(
    score: np.ndarray, starts: np.ndarray, lengths: np.ndarray, persistence: int
) -> tuple[np.ndarray, np.ndarray]:
    if persistence <= 0:
        raise ValueError("persistence must be positive")
    sustained = np.full_like(score, np.nan, dtype=np.float32)
    maximum = np.full(len(starts), np.nan, dtype=np.float32)
    for episode, (start, length) in enumerate(zip(starts, lengths)):
        start = int(start)
        length = int(length)
        sequence = score[start : start + length]
        for query in range(persistence - 1, length):
            window = sequence[query - persistence + 1 : query + 1]
            if np.all(np.isfinite(window)):
                sustained[start + query] = float(np.min(window))
        finite = sustained[start : start + length]
        if np.any(np.isfinite(finite)):
            maximum[episode] = float(np.nanmax(finite))
    return sustained, maximum


def conservative_threshold(values: np.ndarray, target_fpr: float) -> tuple[float, float]:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("cannot calibrate threshold without finite success scores")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target FPR must lie in (0, 1)")
    for candidate in np.unique(values):
        achieved = float(np.mean(values >= candidate))
        if achieved <= target_fpr:
            return float(candidate), achieved
    threshold = float(np.nextafter(values.max(), np.inf))
    return threshold, 0.0


def length_threshold(lengths: np.ndarray, target_fpr: float) -> tuple[int, float]:
    lengths = np.asarray(lengths, np.int32)
    if not len(lengths):
        raise ValueError("cannot calibrate length threshold without successes")
    max_query = lengths - 1
    for query in range(int(max_query.max()) + 2):
        achieved = float(np.mean(max_query >= query))
        if achieved <= target_fpr:
            return query, achieved
    raise AssertionError("unreachable length threshold")


def first_alarm(sequence: np.ndarray, threshold: float) -> int:
    positions = np.flatnonzero(np.asarray(sequence) >= threshold)
    return int(positions[0]) if len(positions) else -1


def query_axes(episodes: list[EpisodeSpec]) -> tuple[np.ndarray, np.ndarray]:
    episode_axis = np.repeat(
        np.arange(len(episodes), dtype=np.int32),
        np.asarray([item.inference_calls for item in episodes], np.int32),
    )
    query_axis = np.concatenate(
        [np.arange(item.inference_calls, dtype=np.int16) for item in episodes]
    )
    return episode_axis, query_axis


def cross_fitted_alarm(
    features: np.ndarray,
    episodes: list[EpisodeSpec],
    runs: list[RunSpec],
) -> dict[str, Any]:
    starts = np.asarray([item.query_start for item in episodes], np.int64)
    lengths = np.asarray([item.inference_calls for item in episodes], np.int32)
    folds = np.asarray([item.fold for item in episodes], np.int8)
    success = np.asarray([item.success for item in episodes], np.bool_)
    init_states = np.asarray([item.init_state_id for item in episodes], np.int16)
    observational = np.asarray(
        [item.run_regime == "observational" for item in episodes], np.bool_
    )
    rolled = rolling_features(features, starts, lengths, ROLLING_WINDOW)

    run_episode_rows: list[np.ndarray] = []
    cursor = 0
    for run in runs:
        count = len(run.episode_indices)
        run_episode_rows.append(np.arange(cursor, cursor + count, dtype=np.int32))
        cursor += count
    if cursor != len(episodes):
        raise ValueError("run/episode indexing differs")

    oof_score = np.full(len(features), np.nan, np.float32)
    oof_sustained = np.full(len(features), np.nan, np.float32)
    oof_maximum = np.full(len(episodes), np.nan, np.float32)
    supported_queries = np.zeros(len(episodes), np.int16)
    alarm_queries = {
        fpr: np.full(len(episodes), -1, np.int16) for fpr in TARGET_FPRS
    }
    length_alarm_queries = {
        fpr: np.full(len(episodes), -1, np.int16) for fpr in TARGET_FPRS
    }
    thresholds: list[dict[str, Any]] = []

    for test_fold in range(N_FOLDS):
        calibration_fold = (test_fold + 1) % N_FOLDS
        reference_fold_mask = (folds != test_fold) & (folds != calibration_fold)
        target_fold_mask = (folds == test_fold) | (folds == calibration_fold)
        component_risk = np.full_like(features, np.nan, dtype=np.float32)

        for run_index, episode_rows in enumerate(run_episode_rows):
            run_lengths = lengths[episode_rows]
            for query in range(ROLLING_WINDOW, int(run_lengths.max())):
                alive = episode_rows[run_lengths > query]
                rows = starts[alive] + query
                finite = np.isfinite(rolled[rows]).all(axis=1)
                alive = alive[finite]
                rows = rows[finite]
                if not len(alive):
                    continue
                pooled_reference_mask = success[alive] & reference_fold_mask[alive]
                pooled_reference_rows = rows[pooled_reference_mask]
                if len(pooled_reference_rows) < MIN_RUN_REFERENCE:
                    continue
                for init_state in np.unique(init_states[alive]):
                    target_mask = (
                        (init_states[alive] == init_state) & target_fold_mask[alive]
                    )
                    if not np.any(target_mask):
                        continue
                    init_reference_mask = (
                        (init_states[alive] == init_state)
                        & success[alive]
                        & reference_fold_mask[alive]
                    )
                    reference_rows = rows[init_reference_mask]
                    if len(reference_rows) < MIN_INIT_REFERENCE:
                        reference_rows = pooled_reference_rows
                    target_rows = rows[target_mask]
                    component_risk[target_rows] = empirical_lower_tail_risk(
                        rolled[target_rows], rolled[reference_rows]
                    )

        score = np.where(
            np.isfinite(component_risk).all(axis=1),
            component_risk.mean(axis=1),
            np.nan,
        ).astype(np.float32)
        sustained, maximum = sustained_scores(score, starts, lengths, PERSISTENCE)
        test_episode_mask = folds == test_fold
        test_query_mask = np.repeat(test_episode_mask, lengths)
        oof_score[test_query_mask] = score[test_query_mask]
        oof_sustained[test_query_mask] = sustained[test_query_mask]
        oof_maximum[test_episode_mask] = maximum[test_episode_mask]
        for episode in np.flatnonzero(test_episode_mask):
            start = int(starts[episode])
            length = int(lengths[episode])
            supported_queries[episode] = int(
                np.isfinite(sustained[start : start + length]).sum()
            )

        global_calibration_success = (
            (folds == calibration_fold) & success & observational & np.isfinite(maximum)
        )
        if global_calibration_success.sum() < 100:
            raise ValueError(f"fold {test_fold}: too few calibration successes")
        for target_fpr in TARGET_FPRS:
            for run, run_episodes in zip(runs, run_episode_rows):
                calibration_success = run_episodes[
                    (folds[run_episodes] == calibration_fold)
                    & success[run_episodes]
                    & np.isfinite(maximum[run_episodes])
                ]
                threshold_scope = "source_run"
                if len(calibration_success) < MIN_RUN_REFERENCE:
                    calibration_success = np.flatnonzero(global_calibration_success)
                    threshold_scope = "global_fallback"
                threshold, calibration_fpr = conservative_threshold(
                    maximum[calibration_success], target_fpr
                )
                duration_threshold, duration_calibration_fpr = length_threshold(
                    lengths[calibration_success], target_fpr
                )
                thresholds.append(
                    {
                        "source_run": run.source_run,
                        "threshold_scope": threshold_scope,
                        "test_fold": test_fold,
                        "calibration_fold": calibration_fold,
                        "reference_folds": ",".join(
                            str(fold)
                            for fold in range(N_FOLDS)
                            if fold not in (test_fold, calibration_fold)
                        ),
                        "target_fpr": target_fpr,
                        "moe_threshold": threshold,
                        "moe_calibration_fpr": calibration_fpr,
                        "length_threshold_query": duration_threshold,
                        "length_calibration_fpr": duration_calibration_fpr,
                        "calibration_successes": int(len(calibration_success)),
                    }
                )
                test_episodes = run_episodes[folds[run_episodes] == test_fold]
                for episode in test_episodes:
                    start = int(starts[episode])
                    length = int(lengths[episode])
                    alarm_queries[target_fpr][episode] = first_alarm(
                        sustained[start : start + length], threshold
                    )
                    if length - 1 >= duration_threshold:
                        length_alarm_queries[target_fpr][episode] = duration_threshold

        print(
            f"  replayed fold {test_fold + 1}/{N_FOLDS} "
            f"(reference={','.join(str(x) for x in range(N_FOLDS) if x not in (test_fold, calibration_fold))}, "
            f"calibration={calibration_fold})",
            flush=True,
        )

    if np.any(np.isnan(oof_maximum) & (supported_queries > 0)):
        raise RuntimeError("supported episode lacks an OOF maximum score")
    return {
        "raw_features": features,
        "rolling_features": rolled,
        "risk_score": oof_score,
        "sustained_score": oof_sustained,
        "episode_maximum": oof_maximum,
        "supported_queries": supported_queries,
        "alarm_queries": alarm_queries,
        "length_alarm_queries": length_alarm_queries,
        "thresholds": thresholds,
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def auc_higher_is_failure(score: np.ndarray, failure: np.ndarray) -> float:
    score = np.asarray(score, np.float64)
    failure = np.asarray(failure, np.bool_)
    finite = np.isfinite(score)
    score = score[finite]
    failure = failure[finite]
    positives = int(failure.sum())
    negatives = int((~failure).sum())
    if not positives or not negatives:
        return float("nan")
    ranks = average_ranks(score)
    return float(
        (ranks[failure].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def classifier_metrics(
    success: np.ndarray, alarm: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    success = np.asarray(success, np.bool_)
    alarm = np.asarray(alarm, np.int32) >= 0
    mask = np.asarray(mask, np.bool_)
    failure_mask = mask & ~success
    success_mask = mask & success
    true_positive = int((alarm & failure_mask).sum())
    false_positive = int((alarm & success_mask).sum())
    failures = int(failure_mask.sum())
    successes = int(success_mask.sum())
    recall = true_positive / failures if failures else float("nan")
    fpr = false_positive / successes if successes else float("nan")
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else float("nan")
    )
    recall_ci = wilson_interval(true_positive, failures)
    fpr_ci = wilson_interval(false_positive, successes)
    return {
        "episodes": int(mask.sum()),
        "successes": successes,
        "failures": failures,
        "true_alarms": true_positive,
        "false_alarms": false_positive,
        "recall": recall,
        "recall_ci_low": recall_ci[0],
        "recall_ci_high": recall_ci[1],
        "false_positive_rate": fpr,
        "fpr_ci_low": fpr_ci[0],
        "fpr_ci_high": fpr_ci[1],
        "precision": precision,
    }


def task_macro_rates(
    episodes: list[EpisodeSpec], alarm: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    alarm_bool = np.asarray(alarm) >= 0
    success = np.asarray([item.success for item in episodes], np.bool_)
    tasks = np.asarray([f"{item.suite}/{item.task_name}" for item in episodes])
    recalls = []
    fprs = []
    for task in np.unique(tasks[mask]):
        current = mask & (tasks == task)
        failed = current & ~success
        passed = current & success
        if failed.any():
            recalls.append(float(alarm_bool[failed].mean()))
        if passed.any():
            fprs.append(float(alarm_bool[passed].mean()))
    return float(np.mean(recalls)), float(np.mean(fprs))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{key: plain(value) for key, value in row.items()} for row in rows])


def summarize(
    episodes: list[EpisodeSpec],
    replay: dict[str, Any],
    integrity: dict[str, Any],
    audits: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    success = np.asarray([item.success for item in episodes], np.bool_)
    failure = ~success
    observational = np.asarray(
        [item.run_regime == "observational" for item in episodes], np.bool_
    )
    suites = np.asarray([item.suite for item in episodes])
    roots = np.asarray([item.data_root for item in episodes])
    reasons = np.asarray([item.primary_failure_reason for item in episodes])
    tasks = np.asarray([f"{item.suite}/{item.task_name}" for item in episodes])
    lengths = np.asarray([item.inference_calls for item in episodes], np.int32)
    maximum = replay["episode_maximum"]
    supported = replay["supported_queries"]
    primary_alarm = replay["alarm_queries"][PRIMARY_FPR]
    primary_length_alarm = replay["length_alarm_queries"][PRIMARY_FPR]

    operating_rows: list[dict[str, Any]] = []
    for target_fpr in TARGET_FPRS:
        for detector, alarm in (
            ("HB_MoE", replay["alarm_queries"][target_fpr]),
            ("length_only_same_run", replay["length_alarm_queries"][target_fpr]),
        ):
            metric = classifier_metrics(success, alarm, observational)
            macro_recall, macro_fpr = task_macro_rates(episodes, alarm, observational)
            operating_rows.append(
                {
                    "detector": detector,
                    "target_calibration_fpr": target_fpr,
                    **metric,
                    "task_macro_recall": macro_recall,
                    "task_macro_false_positive_rate": macro_fpr,
                }
            )

    group_rows: list[dict[str, Any]] = []
    group_specs = [("all_observational", "all", observational)]
    group_specs.extend(
        ("suite", suite, observational & (suites == suite))
        for suite in np.unique(suites[observational])
    )
    group_specs.extend(
        ("data_root", root, observational & (roots == root))
        for root in np.unique(roots[observational])
    )
    regimes = np.asarray([item.run_regime for item in episodes])
    group_specs.extend(
        ("run_regime", regime, regimes == regime) for regime in np.unique(regimes)
    )
    for group_type, group, mask in group_specs:
        for detector, alarm in (
            ("HB_MoE", primary_alarm),
            ("length_only_same_run", primary_length_alarm),
        ):
            group_rows.append(
                {
                    "group_type": group_type,
                    "group": group,
                    "detector": detector,
                    **classifier_metrics(success, alarm, mask),
                }
            )

    reason_rows: list[dict[str, Any]] = []
    for reason in sorted(
        np.unique(reasons[observational & failure]),
        key=lambda item: int((observational & failure & (reasons == item)).sum()),
        reverse=True,
    ):
        mask = observational & failure & (reasons == reason)
        alarmed = mask & (primary_alarm >= 0)
        alarm_phases = primary_alarm[alarmed] / np.maximum(lengths[alarmed] - 1, 1)
        remaining = lengths[alarmed] - 1 - primary_alarm[alarmed]
        reason_rows.append(
            {
                "failure_reason": reason,
                "failure_reason_zh": REASON_ZH.get(reason, reason),
                "failures": int(mask.sum()),
                "supported_failures": int((mask & (supported > 0)).sum()),
                "alarms": int(alarmed.sum()),
                "recall": float(alarmed.sum() / mask.sum()),
                "median_alarm_query": float(np.median(primary_alarm[alarmed])) if alarmed.any() else None,
                "median_alarm_phase": float(np.median(alarm_phases)) if alarmed.any() else None,
                "median_remaining_queries": float(np.median(remaining)) if alarmed.any() else None,
            }
        )

    horizon_rows: list[dict[str, Any]] = []
    starts = np.asarray([item.query_start for item in episodes], np.int64)
    sustained = replay["sustained_score"]
    for horizon in HORIZONS:
        at_risk = observational & (lengths > horizon)
        rows = starts[at_risk] + horizon
        scores = sustained[rows]
        labels = failure[at_risk]
        finite = np.isfinite(scores)
        cumulative_alarm = (primary_alarm[at_risk] >= 0) & (
            primary_alarm[at_risk] <= horizon
        )
        horizon_rows.append(
            {
                "query_index": horizon,
                "actions_generated_through_query": (horizon + 1) * 10,
                "at_risk_episodes": int(at_risk.sum()),
                "at_risk_successes": int((~labels).sum()),
                "at_risk_failures": int(labels.sum()),
                "score_supported": int(finite.sum()),
                "point_score_auc": auc_higher_is_failure(scores, labels),
                "cumulative_failure_recall": float(cumulative_alarm[labels].mean()) if labels.any() else None,
                "cumulative_success_fpr": float(cumulative_alarm[~labels].mean()) if (~labels).any() else None,
            }
        )

    primary_metric = classifier_metrics(success, primary_alarm, observational)
    length_metric = classifier_metrics(success, primary_length_alarm, observational)
    alarmed_failure = observational & failure & (primary_alarm >= 0)
    alarm_phase = primary_alarm[alarmed_failure] / np.maximum(
        lengths[alarmed_failure] - 1, 1
    )
    remaining_queries = lengths[alarmed_failure] - 1 - primary_alarm[alarmed_failure]
    timing = {
        "alarmed_failures": int(alarmed_failure.sum()),
        "median_first_alarm_query": float(np.median(primary_alarm[alarmed_failure])) if alarmed_failure.any() else None,
        "median_first_alarm_action": float(np.median(primary_alarm[alarmed_failure] * 10)) if alarmed_failure.any() else None,
        "median_alarm_episode_phase": float(np.median(alarm_phase)) if alarmed_failure.any() else None,
        "median_remaining_queries": float(np.median(remaining_queries)) if alarmed_failure.any() else None,
        "alarm_by_half_episode_recall": float(
            np.mean((primary_alarm[observational & failure] >= 0) & (
                primary_alarm[observational & failure]
                <= 0.5 * np.maximum(lengths[observational & failure] - 1, 1)
            ))
        ),
        "alarm_by_three_quarter_episode_recall": float(
            np.mean((primary_alarm[observational & failure] >= 0) & (
                primary_alarm[observational & failure]
                <= 0.75 * np.maximum(lengths[observational & failure] - 1, 1)
            ))
        ),
        "alarm_at_least_three_queries_before_end_recall": float(
            np.mean((primary_alarm[observational & failure] >= 0) & (
                primary_alarm[observational & failure]
                <= lengths[observational & failure] - 4
            ))
        ),
    }
    length_alarmed_failure = observational & failure & (primary_length_alarm >= 0)
    length_alarm_phase = primary_length_alarm[length_alarmed_failure] / np.maximum(
        lengths[length_alarmed_failure] - 1, 1
    )
    length_remaining = (
        lengths[length_alarmed_failure]
        - 1
        - primary_length_alarm[length_alarmed_failure]
    )
    length_timing = {
        "alarmed_failures": int(length_alarmed_failure.sum()),
        "median_first_alarm_query": float(
            np.median(primary_length_alarm[length_alarmed_failure])
        ),
        "median_alarm_episode_phase": float(np.median(length_alarm_phase)),
        "median_remaining_queries": float(np.median(length_remaining)),
        "alarm_by_half_episode_recall": float(
            np.mean(
                (primary_length_alarm[observational & failure] >= 0)
                & (
                    primary_length_alarm[observational & failure]
                    <= 0.5 * np.maximum(lengths[observational & failure] - 1, 1)
                )
            )
        ),
    }
    moe_before_length = (
        observational
        & failure
        & (primary_alarm >= 0)
        & (primary_alarm < primary_length_alarm)
    )
    run_maximum_length = np.zeros(len(episodes), np.int32)
    for source_run in np.unique(np.asarray([item.source_run for item in episodes])):
        current = np.asarray(
            [item.source_run == source_run for item in episodes], np.bool_
        )
        run_maximum_length[current] = int(lengths[current].max())
    failure_at_run_horizon = observational & failure & (
        lengths == run_maximum_length
    )
    success_at_run_horizon = observational & success & (
        lengths == run_maximum_length
    )
    duration_audit = {
        "failures_at_source_run_maximum_length": int(failure_at_run_horizon.sum()),
        "observational_failures": int((observational & failure).sum()),
        "successes_at_source_run_maximum_length": int(success_at_run_horizon.sum()),
        "observational_successes": int((observational & success).sum()),
        "moe_alarms_before_length_alarm": int(moe_before_length.sum()),
        "median_query_lead_when_moe_is_earlier": float(
            np.median(
                primary_length_alarm[moe_before_length]
                - primary_alarm[moe_before_length]
            )
        ),
    }

    task_rows = []
    for task in np.unique(tasks[observational]):
        mask = observational & (tasks == task)
        row = {"task": task, **classifier_metrics(success, primary_alarm, mask)}
        task_rows.append(row)
    task_rows.sort(key=lambda row: (-(row["false_positive_rate"] or 0.0), row["task"]))

    episode_rows: list[dict[str, Any]] = []
    for item in episodes:
        row = asdict(item)
        row.update(
            {
                "supported_alarm_queries": int(supported[item.episode_row]),
                "max_oof_risk_score": (
                    float(maximum[item.episode_row])
                    if np.isfinite(maximum[item.episode_row])
                    else None
                ),
            }
        )
        for target_fpr in TARGET_FPRS:
            suffix = str(int(target_fpr * 100))
            alarm = int(replay["alarm_queries"][target_fpr][item.episode_row])
            duration = int(
                replay["length_alarm_queries"][target_fpr][item.episode_row]
            )
            row[f"moe_alarm_query_fpr{suffix}"] = alarm
            row[f"length_alarm_query_fpr{suffix}"] = duration
        episode_rows.append(row)

    summary = {
        "schema": "himoe.hb_moe_failure_alarm.v1",
        "integrity": integrity,
        "detector": {
            "causal": True,
            "uses_future_queries": False,
            "uses_failure_labels_for_threshold_or_weights": False,
            "feature_family_selected_from_prior_labeled_analysis": True,
            "input": "actual HB Top-4 expert IDs and saved HB entropy",
            "feature_names": FEATURE_NAMES,
            "feature_direction": FEATURE_DIRECTION,
            "rolling_transition_window": ROLLING_WINDOW,
            "persistence_queries": PERSISTENCE,
            "fold_assignment": "flow_noise_seed modulo 5",
            "split_per_test_fold": "three reference folds, one disjoint success-calibration fold, one test fold",
            "reference": "same source run + same init state + same absolute query; fallback same run/query",
            "episode_alarm_threshold": "success-only calibration within each source run",
            "minimum_init_reference_successes": MIN_INIT_REFERENCE,
            "minimum_run_reference_successes": MIN_RUN_REFERENCE,
            "risk_combination": "equal mean of five lower-tail empirical percentiles",
            "primary_target_calibration_fpr": PRIMARY_FPR,
        },
        "feature_extraction": {
            "runs": audits,
            "finite_query_rows": int(np.isfinite(replay["raw_features"]).all(axis=1).sum()),
            "raw_feature_min": float(np.nanmin(replay["raw_features"])),
            "raw_feature_max": float(np.nanmax(replay["raw_features"])),
        },
        "primary_observational": {
            "HB_MoE": primary_metric,
            "length_only": length_metric,
            "score_auc": auc_higher_is_failure(maximum[observational], failure[observational]),
            "length_auc": auc_higher_is_failure(lengths[observational], failure[observational]),
            "score_supported_episodes": int((observational & np.isfinite(maximum)).sum()),
            "score_unsupported_episodes": int((observational & ~np.isfinite(maximum)).sum()),
            "timing": timing,
            "length_timing": length_timing,
            "duration_audit": duration_audit,
        },
        "thresholds": replay["thresholds"],
        "operating_points": operating_rows,
        "reasons": reason_rows,
        "horizons": horizon_rows,
    }
    tables = {
        "operating_points": operating_rows,
        "groups": group_rows,
        "reasons": reason_rows,
        "horizons": horizon_rows,
        "tasks": task_rows,
        "episodes": episode_rows,
        "thresholds": replay["thresholds"],
    }
    return summary, tables


def make_figure(
    output: Path,
    episodes: list[EpisodeSpec],
    replay: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    success = np.asarray([item.success for item in episodes], np.bool_)
    failure = ~success
    observational = np.asarray(
        [item.run_regime == "observational" for item in episodes], np.bool_
    )
    lengths = np.asarray([item.inference_calls for item in episodes], np.int32)
    starts = np.asarray([item.query_start for item in episodes], np.int64)
    primary_alarm = replay["alarm_queries"][PRIMARY_FPR]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {"HB_MoE": "#256d5a", "length_only_same_run": "#c44e52"}
    for detector in ("HB_MoE", "length_only_same_run"):
        rows = [
            row
            for row in summary["operating_points"]
            if row["detector"] == detector
        ]
        axes[0, 0].plot(
            [row["false_positive_rate"] for row in rows],
            [row["recall"] for row in rows],
            marker="o",
            label=detector,
            color=colors[detector],
        )
        for row in rows:
            axes[0, 0].annotate(
                f"{int(row['target_calibration_fpr'] * 100)}%",
                (row["false_positive_rate"], row["recall"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0, 0].set_xlabel("observed success episode false-alarm rate")
    axes[0, 0].set_ylabel("failure episode recall")
    axes[0, 0].set_title("Cross-fitted operating points")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    for label, mask, color in (
        ("failure alarm", observational & failure & (primary_alarm >= 0), "#c44e52"),
        ("success false alarm", observational & success & (primary_alarm >= 0), "#4c72b0"),
    ):
        phase = primary_alarm[mask] / np.maximum(lengths[mask] - 1, 1)
        if len(phase):
            ordered = np.sort(phase)
            axes[0, 1].plot(
                ordered,
                np.arange(1, len(ordered) + 1) / len(ordered),
                label=f"{label} (n={len(phase)})",
                color=color,
            )
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("retrospective episode phase at first alarm")
    axes[0, 1].set_ylabel("cumulative fraction")
    axes[0, 1].set_title("First-alarm timing at 5% calibration FPR")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    max_query = min(40, int(lengths.max() - 1))
    for outcome, mask, color in (
        ("failure", observational & failure, "#c44e52"),
        ("success", observational & success, "#4c72b0"),
    ):
        medians = []
        lows = []
        highs = []
        query_values = []
        for query in range(ROLLING_WINDOW + PERSISTENCE - 1, max_query + 1):
            alive = np.flatnonzero(mask & (lengths > query))
            values = replay["sustained_score"][starts[alive] + query]
            values = values[np.isfinite(values)]
            if len(values) < 10:
                continue
            query_values.append(query)
            medians.append(float(np.median(values)))
            lows.append(float(np.quantile(values, 0.25)))
            highs.append(float(np.quantile(values, 0.75)))
        axes[1, 0].plot(query_values, medians, label=outcome, color=color)
        axes[1, 0].fill_between(query_values, lows, highs, color=color, alpha=0.15)
    axes[1, 0].set_xlabel("absolute query index (0-based)")
    axes[1, 0].set_ylabel("sustained HB-MoE risk score")
    axes[1, 0].set_title("At-risk score by absolute query")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    reason_rows = summary["reasons"]
    y = np.arange(len(reason_rows))
    axes[1, 1].barh(
        y,
        [row["recall"] for row in reason_rows],
        color="#256d5a",
    )
    axes[1, 1].set_yticks(
        y, [REASON_SHORT[row["failure_reason"]] for row in reason_rows]
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_xlabel("failure recall")
    axes[1, 1].set_title("Recall by physical failure reason")
    axes[1, 1].grid(axis="x", alpha=0.2)
    fig.suptitle("Causal HB-MoE offline alarm simulation", fontsize=16)
    fig.savefig(output / "alarm_overview.png", dpi=180)
    plt.close(fig)


def format_rate(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100.0 * value:.1f}%"


def build_report(output: Path, summary: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    integrity = summary["integrity"]
    primary = summary["primary_observational"]
    moe = primary["HB_MoE"]
    duration = primary["length_only"]
    timing = primary["timing"]
    length_timing = primary["length_timing"]
    duration_audit = primary["duration_audit"]
    op_rows = tables["operating_points"]
    reason_rows = tables["reasons"]
    horizon_rows = tables["horizons"]
    threshold_rows = tables["thresholds"]
    suite_rows = [
        row
        for row in tables["groups"]
        if row["group_type"] == "suite" and row["detector"] == "HB_MoE"
    ]
    primary_thresholds = np.asarray([
        row["moe_threshold"]
        for row in threshold_rows
        if row["target_fpr"] == PRIMARY_FPR
        and row["threshold_scope"] == "source_run"
    ], np.float64)

    lines = [
        "# HB-MoE 失败报警器：全轨迹离线回放",
        "",
        "## 结论",
        "",
        f"报警器已在全部 `{integrity['episodes']}` 条轨迹、`{integrity['query_rows']}` 个 query 上完成逐步回放；主观察性评估含 `{integrity['run_regime_episodes']['observational']}` 条轨迹和 `{integrity['run_regime_failures']['observational']}` 条物理失败。每个 query 只使用当前及历史 HB 路由。",
        "",
        f"在成功校准目标 5% 下，测试成功轨迹实际误报率为 `{format_rate(moe['false_positive_rate'])}`（95% Wilson CI `{format_rate(moe['fpr_ci_low'])}`–`{format_rate(moe['fpr_ci_high'])}`），失败召回为 `{format_rate(moe['recall'])}`（CI `{format_rate(moe['recall_ci_low'])}`–`{format_rate(moe['recall_ci_high'])}`），precision 为 `{format_rate(moe['precision'])}`。episode 最大风险分数的 AUC 为 `{primary['score_auc']:.3f}`。",
        "",
        f"只看轨迹是否超过同 run 成功时长阈值的同误报预算基线，误报率 `{format_rate(duration['false_positive_rate'])}`、失败召回 `{format_rate(duration['recall'])}`、全局长度 AUC `{primary['length_auc']:.3f}`。它之所以近乎完美，是因为 `{duration_audit['failures_at_source_run_maximum_length']}/{duration_audit['observational_failures']}` 个自然失败都运行到各 source run 的最大 horizon，而成功只有 `{duration_audit['successes_at_source_run_maximum_length']}/{duration_audit['observational_successes']}` 到达该 horizon。",
        "",
        "## 报警规则",
        "",
        "固定使用五个低值风险信号：后层 L12–15 全动作 token 的相邻 query Top-4 churn、后层 T6–T10 churn、前层 L2–5 churn、后层动作 entropy、前层动作 entropy。前三个 query 做因果滚动均值，再在同 run、同初态、同绝对 query 的参考成功轨迹中转换为 lower-tail percentile，五项等权平均。风险连续两个 query 越线才报警。",
        "",
        "每个测试 seed-fold 使用三个完全不同的 seed-fold 建参考分布、另一个 fold 的成功轨迹定阈值；失败标签不参与权重、参考分布或阈值拟合。若同初态参考成功少于 3 条，回退到同 run/同 query；少于 20 条则该 query 不报警。",
        "",
        f"各 run、各折的 5% MoE 阈值中位数为 `{np.median(primary_thresholds):.3f}`，10%–90% 范围 `{np.quantile(primary_thresholds, 0.1):.3f}`–`{np.quantile(primary_thresholds, 0.9):.3f}`。",
        "",
        "## 工作点",
        "",
        "| 检测器 | 校准误报目标 | 实测误报 | 失败召回 | precision | task 宏召回 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in op_rows:
        lines.append(
            f"| `{row['detector']}` | {format_rate(row['target_calibration_fpr'])} | "
            f"{format_rate(row['false_positive_rate'])} | {format_rate(row['recall'])} | "
            f"{format_rate(row['precision'])} | {format_rate(row['task_macro_recall'])} |"
        )
    lines.extend(
        [
            "",
            "1% 档只有约 50–100 条同 run 校准成功可用于估计极端尾部，虽然校准折内不超预算，测试误报仍漂到 2.7%；它不适合作为稳定生产工作点。5% 档的测试误报 4.7%，是本次建议工作点。",
            "",
            "## Suite 异质性",
            "",
            "| suite | 成功 | 失败 | 误报 | 召回 | precision |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in suite_rows:
        lines.append(
            f"| `{row['group']}` | {row['successes']} | {row['failures']} | "
            f"{format_rate(row['false_positive_rate'])} | {format_rate(row['recall'])} | "
            f"{format_rate(row['precision'])} |"
        )
    lines.extend(
        [
            "",
            "## 报警时间",
            "",
            f"在被 MoE 捕获的失败中，首次报警 query 中位数为 `{timing['median_first_alarm_query']:.1f}`（当时约已执行 `{timing['median_first_alarm_action']:.0f}` 个动作），回看 episode 相位中位数 `{timing['median_alarm_episode_phase']:.3f}`，到记录终点尚余 query 中位数 `{timing['median_remaining_queries']:.1f}`。全部失败中，在半程前报警 `{format_rate(timing['alarm_by_half_episode_recall'])}`，四分之三程前报警 `{format_rate(timing['alarm_by_three_quarter_episode_recall'])}`，至少提前 3 个 query 报警 `{format_rate(timing['alarm_at_least_three_queries_before_end_recall'])}`。",
            "",
            f"同 run 时长警报捕获全部失败，但首报相位中位数为 `{length_timing['median_alarm_episode_phase']:.3f}`、尚余 query 中位数 `{length_timing['median_remaining_queries']:.1f}`。MoE 在 `{duration_audit['moe_alarms_before_length_alarm']}` 条失败上更早触发，提前量中位数 `{duration_audit['median_query_lead_when_moe_is_earlier']:.1f}` 个 query。因此合理部署是把 MoE 当作可恢复的早期黄色预警，把时长阈值当作较晚的红色警报。所有相位和剩余量只用于离线评价，不进入在线分数。",
            "",
            "## 按物理失败原因",
            "",
            "| 原因 | 失败 | 有支持窗口 | 报警 | 召回 | 首报相位中位数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in reason_rows:
        lines.append(
            f"| {row['failure_reason_zh']} | {row['failures']} | {row['supported_failures']} | "
            f"{row['alarms']} | {format_rate(row['recall'])} | "
            f"{row['median_alarm_phase']:.3f} |"
            if row["median_alarm_phase"] is not None
            else f"| {row['failure_reason_zh']} | {row['failures']} | {row['supported_failures']} | {row['alarms']} | {format_rate(row['recall'])} | NA |"
        )
    lines.extend(
        [
            "",
            "## 固定 query 审计",
            "",
            "固定绝对 query 只比较当时仍在运行的轨迹，避免把未来长度直接放入分数。",
            "",
            "| query | 在险轨迹 | 失败 | point-score AUC | 累计失败召回 | 累计成功误报 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in horizon_rows:
        lines.append(
            f"| {row['query_index']} | {row['at_risk_episodes']} | {row['at_risk_failures']} | "
            f"{row['point_score_auc']:.3f} | {format_rate(row['cumulative_failure_recall'])} | "
            f"{format_rate(row['cumulative_success_fpr'])} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这是对同一批发现数据的交叉回放，不是新采集任务上的独立外部验证；特征方向来自先前的 1442 条失败分析，因此性能仍可能乐观。",
            "- 报警表示路由进入与历史失败相关的低变化/高集中状态，不等于已证明 MoE 导致物理失败，也不能区分停滞、摆动或周期。",
            "- 全局失败率较低，因此即使误报约 5%，precision 也可能有限；线上使用应触发降速、额外观察或重规划，而不是直接中止任务。",
            "- 该原型需要同任务/source run 的成功校准轨迹；新任务冷启动时不能直接套用这里的阈值。",
            "- 不在成功参考轨迹支持范围内的 query 不会报警；这避免利用“成功都已结束”这一未来长度泄漏，但会降低末期召回。",
            "- `pin-base`、`pin-on/off` 和 `pin-smoke` 全部做了离线回放，但不进入主自然轨迹指标。",
            "",
            "## 产物",
            "",
            "- `episode_alarm_results.csv`: 全部轨迹的 OOF 风险、三档报警时间及长度基线。",
            "- `query_scores.npz`: 每个 query 的五维原始特征、滚动特征、OOF 风险和持续风险。",
            "- `operating_points.csv`: 1%/5%/10% 误报预算的 MoE 与长度基线比较。",
            "- `reason_metrics.csv`, `group_metrics.csv`, `task_metrics.csv`: 失败原因、suite/root、任务细分。",
            "- `horizon_metrics.csv`: 固定绝对 query 的在险评估。",
            "- `thresholds.csv`: 每折完全可复核的参考/校准/测试切分和阈值。",
            "- `alarm_overview.png`: 工作点、报警时机、在险风险曲线和原因召回图。",
        ]
    )
    (output / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_episode_index(path: Path, episodes: list[EpisodeSpec]) -> None:
    write_csv(path, [asdict(item) for item in episodes])


def run_self_test() -> None:
    left = top4_mask(np.asarray([[0, 1, 2, 3]], np.uint8))
    same = top4_mask(np.asarray([[3, 2, 1, 0]], np.uint8))
    half = top4_mask(np.asarray([[0, 1, 4, 5]], np.uint8))
    assert np.allclose(mask_churn(left, same), 0.0)
    assert np.allclose(mask_churn(left, half), 1.0 - 2.0 / 6.0)
    risk = empirical_lower_tail_risk(
        np.asarray([[0.0], [1.0], [2.0]], np.float32),
        np.asarray([[0.0], [1.0], [2.0]], np.float32),
    )
    assert np.allclose(risk[:, 0], [5 / 6, 0.5, 1 / 6])
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    print("building full trajectory index", flush=True)
    episodes, runs, integrity = build_index(args.episodes)
    write_episode_index(args.output / "episode_index.csv", episodes)
    cache_path = args.output / "query_features.npz"
    if args.reuse_features:
        print("loading validated query feature cache", flush=True)
        features, audits = load_feature_cache(cache_path, episodes)
    else:
        print("extracting causal Top-4/entropy query features", flush=True)
        features, audits = extract_all_features(
            runs, integrity["query_rows"], args.workers
        )
        save_feature_cache(cache_path, features, episodes, audits)
    expected_finite = integrity["query_rows"] - integrity["episodes"]
    finite = int(np.isfinite(features).all(axis=1).sum())
    if finite != expected_finite:
        raise RuntimeError(
            f"expected one unsupported first query per episode: {finite} != {expected_finite}"
        )
    print("running disjoint reference/calibration/test replay", flush=True)
    replay = cross_fitted_alarm(features, episodes, runs)
    episode_axis, query_axis = query_axes(episodes)
    np.savez_compressed(
        args.output / "query_scores.npz",
        episode_row=episode_axis,
        query_index=query_axis,
        feature_names=np.asarray(FEATURE_NAMES),
        raw_features=replay["raw_features"],
        rolling_features=replay["rolling_features"],
        risk_score=replay["risk_score"],
        sustained_score=replay["sustained_score"],
    )
    summary, tables = summarize(episodes, replay, integrity, audits)
    write_csv(args.output / "operating_points.csv", tables["operating_points"])
    write_csv(args.output / "group_metrics.csv", tables["groups"])
    write_csv(args.output / "reason_metrics.csv", tables["reasons"])
    write_csv(args.output / "horizon_metrics.csv", tables["horizons"])
    write_csv(args.output / "task_metrics.csv", tables["tasks"])
    write_csv(args.output / "episode_alarm_results.csv", tables["episodes"])
    write_csv(args.output / "thresholds.csv", tables["thresholds"])
    (args.output / "summary.json").write_text(
        json.dumps(plain(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_figure(args.output, episodes, replay, summary)
    build_report(args.output, summary, tables)
    print(f"wrote {args.output / 'report.zh.md'}", flush=True)


if __name__ == "__main__":
    main()
