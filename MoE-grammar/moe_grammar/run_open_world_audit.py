from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import HDBSCAN

from moe_grammar.evaluate_global_prefix import (
    fast_state_blocked_auc_difference,
    fast_state_blocked_interval,
)
from moe_grammar.open_world import (
    AXIS_NAMES,
    AnchorTemplate,
    HealthyDynamicsModel,
    HealthyPrefixGrammar,
    HealthyReturnTable,
    PhaseConditionalCDF,
    RoutingPhaseModel,
    TaskRobustScaler,
    build_dynamic_features,
    causal_window_signatures,
    open_set_scores,
    phenotype_axes,
)
from moe_grammar.statistics import auc_pairwise, stratified_pair_auc

FULL_HORIZONS = (3, 7, 12)
ANCHOR_HORIZONS = (7, 12, 20, 27, 34)
AGGREGATORS = ("instant", "learned_persistent", "dwell_persistent", "ewma_persistent")
GENERIC_METHODS = (
    *AGGREGATORS,
    *(f"grammar_{name}" for name in AGGREGATORS),
    *(f"joint_{name}" for name in AGGREGATORS),
)
ANCHOR_METHODS = (*GENERIC_METHODS, "known_stasis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/open-world-phenotypes.npz"))
    parser.add_argument(
        "--fold-dirs",
        nargs="+",
        type=Path,
        default=[
            Path("results-full40"),
            Path("results-full40-fold1"),
            Path("results-full40-fold2"),
            Path("results-full40-fold3"),
            Path("results-full40-fold4"),
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results-open-world"))
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--healthy-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conditioning", choices=("global", "task"), default="global")
    parser.add_argument("--phase-states", type=int, default=8)
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            name: np.asarray(payload[name]) for name in payload.files if name != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    return arrays, metadata


def episode_rows(
    starts: np.ndarray, lengths: np.ndarray, episode_indexes: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(starts[index], starts[index] + lengths[index], dtype=np.int64)
            for index in np.asarray(episode_indexes, dtype=np.int64)
        ]
    )


def split_train_states(
    states: np.ndarray, seed: int, density_count: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=np.int16).copy()
    np.random.default_rng(seed).shuffle(values)
    return np.sort(values[:density_count]), np.sort(values[density_count:])


def select_episodes(
    state: np.ndarray,
    states: np.ndarray,
    success: np.ndarray | None = None,
    success_value: bool = True,
) -> np.ndarray:
    selected = np.isin(state, states)
    if success is not None:
        selected &= success == success_value
    return np.flatnonzero(selected)


def running_max_flat(values: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for start, length in zip(starts, lengths):
        start = int(start)
        stop = start + int(length)
        output[start:stop] = np.maximum.accumulate(values[start:stop])
    return output


def episode_value(
    values: np.ndarray, starts: np.ndarray, episode_indexes: np.ndarray, position: int
) -> np.ndarray:
    return np.asarray(
        [values[int(starts[index]) + position] for index in episode_indexes],
        dtype=np.float64,
    )


def episode_maxima(
    values: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    episode_indexes: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            np.max(values[int(starts[index]) : int(starts[index] + lengths[index])])
            for index in episode_indexes
        ],
        dtype=np.float64,
    )


def horizon_task_thresholds(
    values: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    task: np.ndarray,
    calibration_success: np.ndarray,
    horizon: int,
    task_count: int,
    healthy_fpr: float,
    condition_on_task: bool,
) -> dict[int, float]:
    eligible = calibration_success[lengths[calibration_success] > horizon]
    all_values = episode_value(values, starts, eligible, horizon)
    global_threshold = float(np.quantile(all_values, 1.0 - healthy_fpr, method="higher"))
    output = {}
    for task_index in range(task_count):
        if not condition_on_task:
            output[task_index] = global_threshold
            continue
        selected = eligible[task[eligible] == task_index]
        output[task_index] = (
            float(
                np.quantile(
                    episode_value(values, starts, selected, horizon),
                    1.0 - healthy_fpr,
                    method="higher",
                )
            )
            if len(selected) >= 20
            else global_threshold
        )
    return output


def anytime_task_thresholds(
    values: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    task: np.ndarray,
    calibration_success: np.ndarray,
    task_count: int,
    healthy_fpr: float,
    condition_on_task: bool,
) -> dict[int, float]:
    all_values = episode_maxima(values, starts, lengths, calibration_success)
    global_threshold = float(np.quantile(all_values, 1.0 - healthy_fpr, method="higher"))
    output = {}
    for task_index in range(task_count):
        if not condition_on_task:
            output[task_index] = global_threshold
            continue
        selected = calibration_success[task[calibration_success] == task_index]
        output[task_index] = (
            float(
                np.quantile(
                    episode_maxima(values, starts, lengths, selected),
                    1.0 - healthy_fpr,
                    method="higher",
                )
            )
            if len(selected) >= 20
            else global_threshold
        )
    return output


def first_crossing(
    values: np.ndarray,
    start: int,
    length: int,
    threshold: float,
    stop: int | None = None,
) -> int | None:
    count = length if stop is None else min(length, stop + 1)
    crossing = np.flatnonzero(values[start : start + count] > threshold)
    return int(crossing[0]) if len(crossing) else None


def phase_velocity(dynamic: np.ndarray, base_dimensions: int) -> np.ndarray:
    return dynamic[:, 2 * base_dimensions + 7]


def calibrated_score_family(
    raw: np.ndarray,
    phase: np.ndarray,
    query_task: np.ndarray,
    calibration_rows: np.ndarray,
    velocity: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    return_episodes: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    calibrator = PhaseConditionalCDF().fit(raw, query_task, phase, calibration_rows)
    percentile = calibrator.transform(raw, query_task, phase)
    return_model = HealthyReturnTable().fit(
        percentile,
        velocity,
        starts,
        lengths,
        return_episodes,
    )
    return_probability, dwell = return_model.predict(percentile, velocity, starts, lengths)
    scores = open_set_scores(percentile, return_probability, dwell, starts, lengths)
    derived = {
        "calibrator": calibrator,
        "percentile": percentile,
        "return_model": return_model,
        "return_probability": return_probability,
        "dwell": dwell,
    }
    return scores, derived


def fit_fold_models(
    arrays: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    seed: int,
    device: str,
    conditioning: str,
    phase_states: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    base = np.asarray(arrays["full_base"], dtype=np.float32)
    starts = arrays["full_starts"]
    lengths = arrays["full_lengths"]
    episode_state = arrays["full_state"]
    success = arrays["full_success"]
    query_task = arrays["full_query_task"]
    model_task = query_task if conditioning == "task" else np.zeros(len(query_task), dtype=np.int16)
    progress = np.asarray(arrays["full_query_progress"], dtype=np.float32)
    task_count = int(model_task.max()) + 1
    density_states, return_states = split_train_states(split["train"], seed)
    density_episodes = select_episodes(episode_state, density_states, success, True)
    return_episodes = select_episodes(episode_state, return_states, success, True)
    density_rows = episode_rows(starts, lengths, density_episodes)
    return_rows = episode_rows(starts, lengths, return_episodes)

    scaler = TaskRobustScaler().fit(base, model_task, density_rows, task_count)
    scaled = scaler.transform(base, model_task)
    phase_model = RoutingPhaseModel().fit(scaled, model_task, progress, density_rows, task_count)
    phase = phase_model.predict(scaled, model_task)
    dynamic = build_dynamic_features(scaled, phase, starts, lengths)
    healthy = HealthyDynamicsModel().fit(dynamic, model_task, phase, density_rows, task_count)
    raw_distance = healthy.score(dynamic, model_task, phase)
    velocity = phase_velocity(dynamic, base.shape[1])
    current_scores, current_derived = calibrated_score_family(
        raw_distance,
        phase,
        model_task,
        return_rows,
        velocity,
        starts,
        lengths,
        return_episodes,
    )

    prefix_grammar = HealthyPrefixGrammar(phase_states=phase_states).fit(
        scaled,
        progress,
        starts,
        lengths,
        density_episodes,
    )
    grammar_raw, grammar_phase, grammar_belief_entropy = prefix_grammar.score(
        scaled, starts, lengths, device=device
    )
    grammar_clock_raw, _, _ = prefix_grammar.score(
        scaled,
        starts,
        lengths,
        update_with_observations=False,
        device=device,
    )
    grammar_scores, grammar_derived = calibrated_score_family(
        grammar_raw,
        grammar_phase,
        model_task,
        return_rows,
        velocity,
        starts,
        lengths,
        return_episodes,
    )
    joint_raw = np.maximum(current_derived["percentile"], grammar_derived["percentile"])
    joint_scores, joint_derived = calibrated_score_family(
        joint_raw,
        phase,
        model_task,
        return_rows,
        velocity,
        starts,
        lengths,
        return_episodes,
    )
    scores = dict(current_scores)
    scores.update({f"grammar_{name}": value for name, value in grammar_scores.items()})
    scores.update({f"joint_{name}": value for name, value in joint_scores.items()})
    axes = phenotype_axes(scaled, dynamic, current_derived["percentile"])
    model = {
        "scaler": scaler,
        "phase_model": phase_model,
        "healthy_dynamics": healthy,
        "calibrator": current_derived["calibrator"],
        "return_model": current_derived["return_model"],
        "prefix_grammar": prefix_grammar,
        "grammar_calibrator": grammar_derived["calibrator"],
        "grammar_return_model": grammar_derived["return_model"],
        "joint_calibrator": joint_derived["calibrator"],
        "joint_return_model": joint_derived["return_model"],
        "conditioning": conditioning,
        "phase_states": phase_states,
        "density_states": density_states,
        "return_states": return_states,
    }
    derived = {
        "scaled": scaled,
        "phase": phase,
        "dynamic": dynamic,
        "raw_distance": raw_distance,
        "percentile": current_derived["percentile"],
        "return_probability": current_derived["return_probability"],
        "dwell": current_derived["dwell"],
        "grammar_raw": grammar_raw,
        "grammar_clock_raw": grammar_clock_raw,
        "grammar_phase": grammar_phase,
        "grammar_belief_entropy": grammar_belief_entropy,
        "model_task": model_task,
        "grammar_percentile": grammar_derived["percentile"],
        "joint_percentile": joint_derived["percentile"],
        "axes": axes,
    }
    return model, scores, derived


def score_anchor(
    arrays: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    model: dict[str, Any],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], AnchorTemplate]:
    base = np.asarray(arrays["anchor_base"], dtype=np.float32)
    starts = arrays["anchor_starts"]
    lengths = arrays["anchor_lengths"]
    query_task = arrays["anchor_query_task"]
    model_task = (
        query_task if model["conditioning"] == "task" else np.zeros(len(query_task), dtype=np.int16)
    )
    episode_state = arrays["anchor_state"]
    success = arrays["anchor_success"]
    onsets = arrays["anchor_stasis_onset"]
    train_success = select_episodes(episode_state, split["train"], success, True)
    train_success_rows = episode_rows(starts, lengths, train_success)

    scaled = model["scaler"].transform(base, model_task)
    phase = model["phase_model"].predict(scaled, model_task)
    dynamic = build_dynamic_features(scaled, phase, starts, lengths)
    raw_distance = model["healthy_dynamics"].score(dynamic, model_task, phase)
    velocity = phase_velocity(dynamic, base.shape[1])
    current_scores, current_derived = calibrated_score_family(
        raw_distance,
        phase,
        model_task,
        train_success_rows,
        velocity,
        starts,
        lengths,
        train_success,
    )
    grammar_raw, grammar_phase, grammar_belief_entropy = model["prefix_grammar"].score(
        scaled, starts, lengths, device=device
    )
    grammar_clock_raw, _, _ = model["prefix_grammar"].score(
        scaled,
        starts,
        lengths,
        update_with_observations=False,
        device=device,
    )
    grammar_scores, grammar_derived = calibrated_score_family(
        grammar_raw,
        grammar_phase,
        model_task,
        train_success_rows,
        velocity,
        starts,
        lengths,
        train_success,
    )
    joint_raw = np.maximum(current_derived["percentile"], grammar_derived["percentile"])
    joint_scores, joint_derived = calibrated_score_family(
        joint_raw,
        phase,
        model_task,
        train_success_rows,
        velocity,
        starts,
        lengths,
        train_success,
    )
    scores = dict(current_scores)
    scores.update({f"grammar_{name}": value for name, value in grammar_scores.items()})
    scores.update({f"joint_{name}": value for name, value in joint_scores.items()})
    axes = phenotype_axes(scaled, dynamic, current_derived["percentile"])
    signatures = causal_window_signatures(axes, starts, lengths)
    train_stasis = np.flatnonzero(np.isin(episode_state, split["train"]) & (onsets >= 0))
    anchor_template = AnchorTemplate().fit(signatures, starts, lengths, onsets, train_stasis)
    similarity = anchor_template.score(signatures)
    similarity_calibrator = PhaseConditionalCDF().fit(
        similarity, model_task, phase, train_success_rows
    )
    similarity_percentile = similarity_calibrator.transform(similarity, model_task, phase)
    scores["known_stasis"] = np.sqrt(
        np.clip(scores["learned_persistent"], 0.0, 1.0) * np.clip(similarity_percentile, 0.0, 1.0)
    ).astype(np.float32)
    derived = {
        "scaled": scaled,
        "phase": phase,
        "dynamic": dynamic,
        "raw_distance": raw_distance,
        "percentile": current_derived["percentile"],
        "return_probability": current_derived["return_probability"],
        "dwell": current_derived["dwell"],
        "grammar_raw": grammar_raw,
        "grammar_clock_raw": grammar_clock_raw,
        "grammar_phase": grammar_phase,
        "grammar_belief_entropy": grammar_belief_entropy,
        "grammar_percentile": grammar_derived["percentile"],
        "joint_percentile": joint_derived["percentile"],
        "axes": axes,
        "signatures": signatures,
        "stasis_similarity": similarity,
        "stasis_similarity_percentile": similarity_percentile,
    }
    return scores, derived, anchor_template


def full_detection_record(
    arrays: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    metadata: dict[str, Any],
    healthy_fpr: float,
    condition_on_task: bool,
) -> tuple[dict[tuple[int, str], dict[str, np.ndarray]], dict[str, dict[int, float]]]:
    starts = arrays["full_starts"]
    lengths = arrays["full_lengths"]
    task = arrays["full_task"]
    state = arrays["full_state"]
    success = arrays["full_success"]
    calibration_success = select_episodes(state, split["calibration"], success, True)
    test = select_episodes(state, split["test"])
    task_count = len(metadata["tasks"])
    records = {}
    anytime = {}
    for method in GENERIC_METHODS:
        running = running_max_flat(scores[method], starts, lengths)
        anytime[method] = anytime_task_thresholds(
            scores[method],
            starts,
            lengths,
            task,
            calibration_success,
            task_count,
            healthy_fpr,
            condition_on_task,
        )
        for horizon in FULL_HORIZONS:
            eligible = test[lengths[test] > horizon]
            values = episode_value(running, starts, eligible, horizon)
            labels = ~success[eligible]
            thresholds = horizon_task_thresholds(
                running,
                starts,
                lengths,
                task,
                calibration_success,
                horizon,
                task_count,
                healthy_fpr,
                condition_on_task,
            )
            alarms = np.asarray(
                [value > thresholds[int(task[index])] for value, index in zip(values, eligible)],
                dtype=np.bool_,
            )
            records[(horizon, method)] = {
                "labels": labels,
                "scores": values,
                "alarms": alarms,
                "task_names": np.asarray(
                    [metadata["tasks"][int(task[index])] for index in eligible],
                    dtype=object,
                ),
                "states": state[eligible],
            }
    return records, anytime


def anchor_detection_record(
    arrays: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    healthy_fpr: float,
) -> tuple[dict[tuple[int, str], dict[str, np.ndarray]], dict[str, float], dict[str, Any]]:
    starts = arrays["anchor_starts"]
    lengths = arrays["anchor_lengths"]
    state = arrays["anchor_state"]
    success = arrays["anchor_success"]
    onsets = arrays["anchor_stasis_onset"]
    calibration_success = select_episodes(state, split["calibration"], success, True)
    test = select_episodes(state, split["test"])
    labels_all = onsets >= 0
    records = {}
    anytime_threshold = {}
    event_results: dict[str, Any] = {}
    for method in ANCHOR_METHODS:
        running = running_max_flat(scores[method], starts, lengths)
        calibration_max = episode_maxima(scores[method], starts, lengths, calibration_success)
        threshold = float(np.quantile(calibration_max, 1.0 - healthy_fpr, method="higher"))
        anytime_threshold[method] = threshold
        for horizon in ANCHOR_HORIZONS:
            eligible = test[lengths[test] > horizon]
            records[(horizon, method)] = {
                "labels": labels_all[eligible],
                "scores": episode_value(running, starts, eligible, horizon),
                "states": state[eligible],
                "success": success[eligible],
            }

        stasis_test = test[labels_all[test]]
        detected_leads = []
        by_onset = 0
        by_onset_plus3 = 0
        for episode_index in stasis_test:
            alarm = first_crossing(
                scores[method],
                int(starts[episode_index]),
                int(lengths[episode_index]),
                threshold,
            )
            if alarm is not None:
                detected_leads.append(alarm - int(onsets[episode_index]))
                by_onset += int(alarm <= onsets[episode_index])
                by_onset_plus3 += int(alarm <= onsets[episode_index] + 3)
        success_test = test[success[test]]
        success_alarm = np.asarray(
            [
                episode_maxima(scores[method], starts, lengths, np.asarray([index]))[0] > threshold
                for index in success_test
            ]
        )
        event_results[method] = {
            "test_stasis": int(len(stasis_test)),
            "test_success": int(len(success_test)),
            "success_episode_fpr": float(success_alarm.mean()),
            "recall_by_physical_onset": float(by_onset / len(stasis_test)),
            "recall_by_onset_plus3": float(by_onset_plus3 / len(stasis_test)),
            "ever_recall": float(len(detected_leads) / len(stasis_test)),
            "median_alarm_minus_onset_detected": (
                float(np.median(detected_leads)) if detected_leads else None
            ),
        }

    instant_threshold = anytime_threshold["instant"]
    persistent_threshold = anytime_threshold["learned_persistent"]
    transient_episodes = 0
    persistent_alarm_on_transient = 0
    success_test = test[success[test]]
    for episode_index in success_test:
        start = int(starts[episode_index])
        length = int(lengths[episode_index])
        instant = scores["instant"][start : start + length]
        high = np.flatnonzero(instant > 0.95)
        observed_transient = any(
            np.any(instant[position + 1 : position + 4] < 0.8)
            for position in high
            if position + 1 < length
        )
        if observed_transient:
            transient_episodes += 1
            persistent_alarm_on_transient += int(
                np.max(scores["learned_persistent"][start : start + length]) > persistent_threshold
            )
    event_results["transient_correction_control"] = {
        "definition": "success episode has u>0.95 followed by u<0.8 within three queries",
        "episodes": transient_episodes,
        "instant_anytime_threshold": instant_threshold,
        "persistent_false_alarm_rate": (
            persistent_alarm_on_transient / transient_episodes if transient_episodes else None
        ),
    }
    return records, anytime_threshold, event_results


def offline_event_embedding(
    axes: np.ndarray, start: int, length: int, event_query: int
) -> np.ndarray:
    left = max(0, event_query - 2)
    right = min(length, event_query + 5)
    sequence = axes[start + left : start + right]
    time = np.arange(len(sequence), dtype=np.float32)
    time -= time.mean()
    slope = (
        (sequence * time[:, None]).sum(axis=0) / np.square(time).sum()
        if len(sequence) > 1
        else np.zeros(sequence.shape[1], dtype=np.float32)
    )
    return np.concatenate(
        [sequence.mean(axis=0), sequence.std(axis=0), slope, sequence.max(axis=0)]
    ).astype(np.float32)


def collect_full_events(
    arrays: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    axes: np.ndarray,
    thresholds: dict[int, float],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    events = []
    test = select_episodes(arrays["full_state"], split["test"])
    for episode_index in test:
        start = int(arrays["full_starts"][episode_index])
        length = int(arrays["full_lengths"][episode_index])
        task_index = int(arrays["full_task"][episode_index])
        event_query = first_crossing(
            scores["learned_persistent"],
            start,
            length,
            thresholds[task_index],
        )
        if event_query is None:
            continue
        events.append(
            {
                "embedding": offline_event_embedding(axes, start, length, event_query),
                "axis_mean": axes[
                    start + max(0, event_query - 2) : start + min(length, event_query + 5)
                ].mean(axis=0),
                "source": "full40_event",
                "failure": bool(not arrays["full_success"][episode_index]),
                "task": metadata["tasks"][task_index],
                "state": int(arrays["full_state"][episode_index]),
                "seed": int(arrays["full_seed"][episode_index]),
                "query": event_query,
            }
        )
    return events


def collect_anchor_events(
    arrays: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    axes: np.ndarray,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    events = []
    test_stasis = np.flatnonzero(
        np.isin(arrays["anchor_state"], split["test"]) & (arrays["anchor_stasis_onset"] >= 0)
    )
    for episode_index in test_stasis:
        start = int(arrays["anchor_starts"][episode_index])
        length = int(arrays["anchor_lengths"][episode_index])
        onset = int(arrays["anchor_stasis_onset"][episode_index])
        events.append(
            {
                "embedding": offline_event_embedding(axes, start, length, onset),
                "axis_mean": axes[start + max(0, onset - 2) : start + min(length, onset + 5)].mean(
                    axis=0
                ),
                "source": "physical_stasis_anchor",
                "failure": True,
                "task": metadata["anchor_task"],
                "state": int(arrays["anchor_state"][episode_index]),
                "seed": int(arrays["anchor_seed"][episode_index]),
                "query": onset,
            }
        )
    return events


def cluster_event_atlas(events: list[dict[str, Any]]) -> dict[str, Any]:
    if len(events) < 30:
        return {"events": len(events), "clusters": [], "noise_events": len(events)}
    embeddings = np.asarray([event["embedding"] for event in events], dtype=np.float64)
    center = np.median(embeddings, axis=0)
    low, high = np.quantile(embeddings, [0.25, 0.75], axis=0)
    scale = np.maximum(high - low, 1e-4)
    standardized = np.clip((embeddings - center) / scale, -10.0, 10.0)
    minimum = max(25, int(round(len(events) * 0.015)))
    labels = HDBSCAN(
        min_cluster_size=minimum,
        min_samples=max(10, minimum // 3),
        cluster_selection_method="eom",
        copy=False,
    ).fit_predict(standardized)
    clusters = []
    for label in sorted(set(labels) - {-1}):
        selected = np.flatnonzero(labels == label)
        selected_events = [events[index] for index in selected]
        axis_mean = np.mean([event["axis_mean"] for event in selected_events], axis=0)
        dominant = np.argsort(axis_mean)[::-1][:3]
        clusters.append(
            {
                "cluster": int(label),
                "events": int(len(selected)),
                "full40_events": int(
                    sum(event["source"] == "full40_event" for event in selected_events)
                ),
                "physical_stasis_anchors": int(
                    sum(event["source"] == "physical_stasis_anchor" for event in selected_events)
                ),
                "posthoc_failure_fraction_full40": (
                    float(
                        np.mean(
                            [
                                event["failure"]
                                for event in selected_events
                                if event["source"] == "full40_event"
                            ]
                        )
                    )
                    if any(event["source"] == "full40_event" for event in selected_events)
                    else None
                ),
                "tasks": int(len({event["task"] for event in selected_events})),
                "states": int(len({event["state"] for event in selected_events})),
                "axis_mean": {name: float(value) for name, value in zip(AXIS_NAMES, axis_mean)},
                "dominant_positive_axes": [AXIS_NAMES[index] for index in dominant],
            }
        )
    return {
        "events": len(events),
        "minimum_cluster_size": minimum,
        "clusters": clusters,
        "clustered_events": int(np.sum(labels >= 0)),
        "noise_events": int(np.sum(labels < 0)),
        "interpretation": (
            "unsupervised routing-phenotype candidates; posthoc failure fractions and "
            "stasis overlap do not turn clusters into validated Trap classes"
        ),
    }


def aggregate_full_detection(
    fold_records: list[dict[tuple[int, str], dict[str, np.ndarray]]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    output = {}
    for horizon in FULL_HORIZONS:
        output[str(horizon)] = {}
        for method_index, method in enumerate(GENERIC_METHODS):
            records = [fold[(horizon, method)] for fold in fold_records]
            labels = np.concatenate([record["labels"] for record in records])
            scores = np.concatenate([record["scores"] for record in records])
            alarms = np.concatenate([record["alarms"] for record in records])
            tasks = np.concatenate([record["task_names"] for record in records])
            states = np.concatenate([record["states"] for record in records])
            strata = np.asarray(
                [f"{task}|{state}" for task, state in zip(tasks, states)], dtype=object
            )
            within_auc, by_stratum, pairs = stratified_pair_auc(labels, scores, strata)
            output[str(horizon)][method] = {
                "episodes": int(len(labels)),
                "successes": int((~labels).sum()),
                "failures": int(labels.sum()),
                "within_task_state_auc": within_auc,
                "state_blocked_ci95": list(
                    fast_state_blocked_interval(
                        labels,
                        scores,
                        tasks,
                        states,
                        draws,
                        seed + horizon * 1009 + method_index,
                    )
                ),
                "healthy_fpr": float(alarms[~labels].mean()),
                "failure_recall": float(alarms[labels].mean()),
                "pooled_auc": auc_pairwise(labels, scores),
                "informative_task_state_cells": len(by_stratum),
                "success_failure_pairs": pairs,
            }
        instant = [fold[(horizon, "instant")] for fold in fold_records]
        learned = [fold[(horizon, "learned_persistent")] for fold in fold_records]
        labels = np.concatenate([record["labels"] for record in instant])
        tasks = np.concatenate([record["task_names"] for record in instant])
        states = np.concatenate([record["states"] for record in instant])
        output[str(horizon)]["learned_persistent_minus_instant"] = (
            fast_state_blocked_auc_difference(
                labels,
                np.concatenate([record["scores"] for record in learned]),
                np.concatenate([record["scores"] for record in instant]),
                tasks,
                states,
                draws,
                seed + horizon * 2017,
            )
        )
    return output


def aggregate_anchor_detection(
    fold_records: list[dict[tuple[int, str], dict[str, np.ndarray]]],
    fold_events: list[dict[str, Any]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {"fixed_horizon": {}}
    for horizon in ANCHOR_HORIZONS:
        output["fixed_horizon"][str(horizon)] = {}
        for method_index, method in enumerate(ANCHOR_METHODS):
            records = [fold[(horizon, method)] for fold in fold_records]
            labels = np.concatenate([record["labels"] for record in records])
            scores = np.concatenate([record["scores"] for record in records])
            states = np.concatenate([record["states"] for record in records])
            strata = states.astype(str)
            auc, by_state, pairs = stratified_pair_auc(labels, scores, strata)
            output["fixed_horizon"][str(horizon)][method] = {
                "episodes": int(len(labels)),
                "stasis": int(labels.sum()),
                "within_state_auc": auc,
                "state_blocked_ci95": list(
                    fast_state_blocked_interval(
                        labels,
                        scores,
                        np.full(len(labels), "scene8", dtype=object),
                        states,
                        draws,
                        seed + horizon * 1013 + method_index,
                    )
                ),
                "informative_states": len(by_state),
                "pairs": pairs,
            }
    output["physical_onset"] = {}
    for method in ANCHOR_METHODS:
        items = [fold[method] for fold in fold_events]
        stasis_count = sum(item["test_stasis"] for item in items)
        success_count = sum(item["test_success"] for item in items)
        output["physical_onset"][method] = {
            "test_stasis": stasis_count,
            "test_success": success_count,
            "success_episode_fpr": float(
                np.average(
                    [item["success_episode_fpr"] for item in items],
                    weights=[item["test_success"] for item in items],
                )
            ),
            "recall_by_physical_onset": float(
                np.average(
                    [item["recall_by_physical_onset"] for item in items],
                    weights=[item["test_stasis"] for item in items],
                )
            ),
            "recall_by_onset_plus3": float(
                np.average(
                    [item["recall_by_onset_plus3"] for item in items],
                    weights=[item["test_stasis"] for item in items],
                )
            ),
            "ever_recall": float(
                np.average(
                    [item["ever_recall"] for item in items],
                    weights=[item["test_stasis"] for item in items],
                )
            ),
        }
    transient = [fold["transient_correction_control"] for fold in fold_events]
    transient_count = sum(item["episodes"] for item in transient)
    output["transient_correction_control"] = {
        "episodes": transient_count,
        "persistent_false_alarm_rate": (
            float(
                np.average(
                    [item["persistent_false_alarm_rate"] for item in transient if item["episodes"]],
                    weights=[item["episodes"] for item in transient if item["episodes"]],
                )
            )
            if transient_count
            else None
        ),
    }
    return output


def write_report(path: Path, summary: dict[str, Any]) -> None:
    full = summary["full40_eventual_failure"]
    anchor = summary["scene8_physical_stasis_anchor"]
    phase = summary["routing_phase"]
    grammar = summary["healthy_prefix_grammar"]
    conditioning = summary["protocol"]["conditioning"]
    conditioning_description = (
        "全局 robust scaling、全局 routing-only phase 和 global×phase 健康密度"
        if conditioning == "global"
        else "任务内 robust scaling、task-specific routing-only phase 和 task×phase 健康密度"
    )
    lines = [
        "# MoE-only 开放集状态观察器审计",
        "",
        "## 协议",
        "",
        "通用模型只读取成功轨迹，不训练 Loop/Static/Failure 分类器。每折 30 个 train states 内部再拆为 "
        "20 个 density states 和 10 个 return states；原 calibration states 只定健康误报阈值，test states 只评价。",
        "",
        f"每个 query 使用 22 个可解释 MoE phenotype；{conditioning_description}给出到健康动力学的距离。"
        "另一个共享 phase-HMM 用完整"
        "已观测前缀递推 belief，直接计算当前 routing chord 对健康下一词分布的预测密度。"
        "归一化进度只用作离线 phase "
        "回归目标，在线 phase 估计的输入仍只有 MoE。健康 return table 估计异常后三个 query 内返回"
        "健康区的概率。Transformer 和失败标签都不参与通用分数。",
        "",
        "Scene8 的物理 stasis onset 只是 known-static anchor。其模板只用每折 train states 的 stasis，"
        "其他失败仍保持未命名；当前没有可靠 Loop anchor。",
        "",
        "## Routing Phase",
        "",
        f"held-out success 上 routing phase 与离线归一化进度的 Spearman 相关为 "
        f"{phase['spearman_success_progress']:.3f}；相邻 query 向前推进比例为 "
        f"{phase['positive_velocity_fraction']:.3f}。",
        "",
        "## 健康全前缀语法",
        "",
        "下表只比较 held-out success 上的下一 chord 密度。clock 对照仅根据健康转移矩阵推进；"
        "full-prefix 每步用已观测 routing 更新 belief。负差值表示历史改善预测。",
        "",
        "| 范围 | clock bits/phenotype | full-prefix bits/phenotype | full-clock |",
        "|---|---:|---:|---:|",
    ]
    for scope in ("q1+", "q4+"):
        item = grammar[scope]
        lines.append(
            f"| {scope} | {item['clock_bits_per_phenotype']:.4f} | "
            f"{item['prefix_bits_per_phenotype']:.4f} | "
            f"{item['prefix_minus_clock_bits_per_phenotype']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Full-40 最终失败读数",
            "",
            "这里的阳性仍只是 eventual failure，不等同于真实 Trap。",
            "",
            "| q | 方法 | task/state AUC | 95% CI | healthy FPR | failure recall |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for horizon in FULL_HORIZONS:
        for method in GENERIC_METHODS:
            item = full[str(horizon)][method]
            lines.append(
                f"| {horizon} | {method} | {item['within_task_state_auc']:.3f} | "
                f"[{item['state_blocked_ci95'][0]:.3f}, {item['state_blocked_ci95'][1]:.3f}] | "
                f"{item['healthy_fpr']:.3f} | {item['failure_recall']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Scene8 物理 Stasis 锚点",
            "",
            "阈值只由 calibration-success 的整条 episode 最大值给出，因此下表是 anytime 5% FPR 目标。",
            "",
            "| 方法 | success FPR | onset 前 recall | onset+3 recall | ever recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in ANCHOR_METHODS:
        item = anchor["physical_onset"][method]
        lines.append(
            f"| {method} | {item['success_episode_fpr']:.3f} | "
            f"{item['recall_by_physical_onset']:.3f} | "
            f"{item['recall_by_onset_plus3']:.3f} | {item['ever_recall']:.3f} |"
        )
    learned = anchor["physical_onset"]["learned_persistent"]
    known = anchor["physical_onset"]["known_stasis"]
    lines.extend(
        [
            "",
            f"在 {learned['test_stasis']} 条 cross-fit 物理 stasis 中，通用 persistent 分数到 onset+3 的召回为 "
            f"{learned['recall_by_onset_plus3']:.3f}；known-static 模板为 "
            f"{known['recall_by_onset_plus3']:.3f}。漏报只能称为 observer-unseen stasis；"
            "它们是 MoE-silent 候选，但尚不能排除特征或模型能力不足。",
            "",
            "## 未知表型 Atlas",
            "",
            f"跨折共收集 {summary['event_atlas']['events']} 个持久偏离事件，HDBSCAN 得到 "
            f"{len(summary['event_atlas']['clusters'])} 个 cluster；"
            f"{summary['event_atlas']['noise_events']} 个事件保留为 cluster noise。",
            "",
            "这些 cluster 是待审计 routing phenotypes，不是自动获得物理语义的 Trap 类别。"
            "只有同时偏离健康、具有持续性、且经过物理审计后，才能升级为新的 known anchor。",
            "",
            "## 结论边界",
            "",
            "- 通用 open-set detector 的训练和方法固定不读取 failure/stasis 标签；known-static 模板是单独的半监督锚点。",
            "- success 中的短暂高分只可称 transient-like correction；没有环境干预标签时不能断言它完成了物理纠错。",
            "- full-40 未报警的最终失败只能称 unobserved failure；即使 Scene8 有物理 onset，仍需穷尽合理 MoE 读出后才能称 MoE-silent。",
            "- cluster 的 post-hoc failure fraction 只用于安排人工审计，不能用于选择 detector 或报告无偏分类性能。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    started = time.time()
    arrays, metadata = load_dataset(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    full_fold_records = []
    anchor_fold_records = []
    anchor_fold_events = []
    atlas_events: list[dict[str, Any]] = []
    phase_prediction = []
    phase_target = []
    phase_velocity_values = []
    grammar_phase_prediction = []
    grammar_nll: dict[str, dict[str, list[float]]] = {
        scope: {"prefix": [], "clock": []} for scope in ("q1+", "q4+")
    }
    folds = []

    for fold, directory in enumerate(args.fold_dirs):
        print(f"Fold {fold}: fitting healthy open-set dynamics...", flush=True)
        fold_summary = json.loads((directory / "summary.json").read_text())
        split = {
            name: np.asarray(values, dtype=np.int16)
            for name, values in fold_summary["split"].items()
        }
        model, full_scores, full_derived = fit_fold_models(
            arrays,
            split,
            args.seed + fold * 101,
            args.device,
            args.conditioning,
            args.phase_states,
        )
        print(f"Fold {fold}: scoring physical stasis anchor...", flush=True)
        anchor_scores, anchor_derived, anchor_template = score_anchor(
            arrays, split, model, args.device
        )
        model["anchor_template"] = anchor_template
        joblib.dump(
            model,
            args.output_dir / "models" / f"fold_{fold}.joblib",
            compress=3,
        )

        full_record, full_anytime = full_detection_record(
            arrays,
            split,
            full_scores,
            metadata,
            args.healthy_fpr,
            args.conditioning == "task",
        )
        anchor_record, anchor_thresholds, anchor_events = anchor_detection_record(
            arrays, split, anchor_scores, args.healthy_fpr
        )
        full_fold_records.append(full_record)
        anchor_fold_records.append(anchor_record)
        anchor_fold_events.append(anchor_events)
        atlas_events.extend(
            collect_full_events(
                arrays,
                split,
                full_scores,
                full_derived["axes"],
                full_anytime["learned_persistent"],
                metadata,
            )
        )
        atlas_events.extend(collect_anchor_events(arrays, split, anchor_derived["axes"], metadata))

        test_success = select_episodes(
            arrays["full_state"], split["test"], arrays["full_success"], True
        )
        rows = episode_rows(arrays["full_starts"], arrays["full_lengths"], test_success)
        noninitial_rows = np.setdiff1d(
            rows,
            arrays["full_starts"][test_success],
            assume_unique=True,
        )
        phase_prediction.append(full_derived["phase"][rows])
        grammar_phase_prediction.append(full_derived["grammar_phase"][rows])
        phase_target.append(np.asarray(arrays["full_query_progress"][rows], dtype=np.float32))
        phase_velocity_values.append(
            phase_velocity(full_derived["dynamic"], arrays["full_base"].shape[1])[noninitial_rows]
        )
        for episode_index in test_success:
            start = int(arrays["full_starts"][episode_index])
            length = int(arrays["full_lengths"][episode_index])
            for scope, first_query in (("q1+", 1), ("q4+", 4)):
                if length <= first_query:
                    continue
                selected = slice(start + first_query, start + length)
                grammar_nll[scope]["prefix"].append(
                    float(np.mean(full_derived["grammar_raw"][selected]))
                )
                grammar_nll[scope]["clock"].append(
                    float(np.mean(full_derived["grammar_clock_raw"][selected]))
                )
        folds.append(
            {
                "fold": fold,
                "density_states": model["density_states"].tolist(),
                "return_states": model["return_states"].tolist(),
                "calibration_states": split["calibration"].tolist(),
                "test_states": split["test"].tolist(),
                "anchor_anytime_thresholds": anchor_thresholds,
            }
        )
        print(f"Fold {fold}: complete", flush=True)

    joined_test_states = sorted(state for fold in folds for state in fold["test_states"])
    if joined_test_states != list(range(50)):
        raise ValueError("fold test states must partition 0..49")
    predicted_phase = np.concatenate(phase_prediction)
    target_phase = np.concatenate(phase_target)
    predicted_grammar_phase = np.concatenate(grammar_phase_prediction)
    velocity = np.concatenate(phase_velocity_values)
    correlation = spearmanr(predicted_phase, target_phase).statistic
    grammar_correlation = spearmanr(predicted_grammar_phase, target_phase).statistic
    grammar_summary = {}
    for scope in ("q1+", "q4+"):
        prefix = np.asarray(grammar_nll[scope]["prefix"])
        clock = np.asarray(grammar_nll[scope]["clock"])
        grammar_summary[scope] = {
            "episodes": int(len(prefix)),
            "prefix_bits_per_phenotype": float(prefix.mean() / np.log(2.0)),
            "clock_bits_per_phenotype": float(clock.mean() / np.log(2.0)),
            "prefix_minus_clock_bits_per_phenotype": float(np.mean(prefix - clock) / np.log(2.0)),
        }

    summary = {
        "schema_version": 2,
        "protocol": {
            "development_status": "exploratory iteration after the v1 local-state audit",
            "generic_training": "full40 successful episodes only",
            "generic_labels": "none",
            "known_anchor": "Scene8 physical stasis onset; per-fold train states only",
            "density_return_split": "20/10 states inside each 30-state train split",
            "threshold": "calibration-success only; nominal 5% episode FPR",
            "test": "state cross-fit; every full40 state and every anchor state tested once",
            "conditioning": args.conditioning,
            "phase_states": args.phase_states,
            "online_inputs": (
                "MoE routing phenotypes only"
                if args.conditioning == "global"
                else "MoE routing phenotypes plus known task ID"
            ),
            "neural_sequence_model": False,
            "compute_device": args.device,
        },
        "counts": {
            "full_queries": int(len(arrays["full_base"])),
            "full_episodes": int(len(arrays["full_lengths"])),
            "full_successes": int(arrays["full_success"].sum()),
            "full_failures": int((~arrays["full_success"]).sum()),
            "anchor_episodes": int(len(arrays["anchor_lengths"])),
            "anchor_stasis": int((arrays["anchor_stasis_onset"] >= 0).sum()),
        },
        "folds": folds,
        "routing_phase": {
            "spearman_success_progress": float(correlation),
            "prefix_filter_spearman_success_progress": float(grammar_correlation),
            "positive_velocity_fraction": float(np.mean(velocity > 0.0)),
            "stall_or_regression_fraction": float(np.mean(velocity <= 0.0)),
            "note": (
                "normalized progress is an offline training/evaluation target for the "
                "routing-only phase regressor; it is never an online detector input"
            ),
        },
        "healthy_prefix_grammar": grammar_summary,
        "full40_eventual_failure": aggregate_full_detection(
            full_fold_records, args.bootstrap_draws, args.seed
        ),
        "scene8_physical_stasis_anchor": aggregate_anchor_detection(
            anchor_fold_records,
            anchor_fold_events,
            args.bootstrap_draws,
            args.seed + 1,
        ),
        "event_atlas": cluster_event_atlas(atlas_events),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_report(args.output_dir / "REPORT.zh.md", summary)
    print(
        f"Wrote open-world audit to {args.output_dir} in "
        f"{(time.time() - started) / 60.0:.1f} minutes",
        flush=True,
    )


if __name__ == "__main__":
    main()
