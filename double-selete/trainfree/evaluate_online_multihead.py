#!/usr/bin/env python3
"""Reveal outcomes only after sealed causal alarms exist, then evaluate events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_RESULT_DIR = HERE / "results/online_multihead_hub"
DEFAULT_CACHE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
LABEL_PATH = HERE / "results/hub_binary_audit/episode_physical_labels.csv"
TARGET_PATH = HERE / "results/hub_binary_audit/target_audit.json"
PROTOCOL = HERE / "ONLINE_MULTIHEAD_PROTOCOL.md"
RUN_ID = "right-50x8-20260903"
SEED = 20260904

STATIC_EEF_M = 0.005
STATIC_OBJECT_M = 0.003
STATIC_GRIPPER_M = 0.001
ACTIVE_EEF_M = 0.005
REVERSAL_COSINE = -0.25
APPROACH_M = 0.10
APPROACH_EXIT_M = 0.15
UNDO_EXIT_M = 0.075
LIFT_M = 0.01
GOAL_M = 0.05

ONSET_BEHAVIORS = (
    "stagnation",
    "gripper_cycling",
    "active_retry",
    "eef_oscillation",
    "goal_regression",
    "goal_approach_leave",
    "subtask_undo",
    "regrasp_or_drop",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--labels", type=Path, default=LABEL_PATH)
    parser.add_argument("--targets", type=Path, default=TARGET_PATH)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-onsets", action="store_true")
    parser.add_argument("--outcomes-from-summaries", action="store_true")
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


def verify_seal(result_dir: Path) -> dict[str, Any]:
    manifest_path = result_dir / "sealed_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("labels_used") != []:
        raise ValueError("sealed scorer manifest already contains labels")
    if not manifest.get("query_causal") or manifest.get(
        "future_queries_used_by_online_update"
    ):
        raise ValueError("sealed scorer does not assert causal query updates")
    checks = {
        "sealed_scores_sha256": result_dir / "sealed_online_scores.npz",
        "thresholds_sha256": result_dir / "unlabeled_thresholds.csv",
        "episode_alarms_sha256": result_dir / "sealed_episode_alarms.csv",
        "feature_cache_sha256": result_dir / "unlabeled_query_features.npz",
    }
    for key, path in checks.items():
        observed = sha256(path)
        expected = manifest["artifacts"][key]
        if observed != expected:
            raise ValueError(f"sealed artifact changed: {path}")
    if sha256(PROTOCOL) != manifest["artifacts"]["protocol_sha256"]:
        raise ValueError("protocol changed after alarms were sealed")
    return manifest


def load_sealed(result_dir: Path) -> dict[str, np.ndarray]:
    with np.load(result_dir / "sealed_online_scores.npz", allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    if str(values["schema"]) != "himoe.online_multihead.sealed.v1":
        raise ValueError("unknown sealed score schema")
    return values


def join_labels(sealed: dict[str, np.ndarray], label_path: Path) -> pd.DataFrame:
    task_names = sealed["task_names"].astype(str)
    index = pd.DataFrame(
        {
            "sealed_row": np.arange(len(sealed["episode"]), dtype=np.int64),
            "task": task_names[sealed["task_index"].astype(int)],
            "episode": sealed["episode"].astype(int),
            "init_state_id": sealed["init_state_id"].astype(int),
            "flow_noise_seed": sealed["flow_noise_seed"].astype(int),
            "sealed_length": sealed["length"].astype(int),
        }
    )
    labels = pd.read_csv(label_path)
    if labels.duplicated(["task", "episode"]).any():
        raise ValueError("duplicate physical-label episode key")
    merged = index.merge(
        labels,
        on=["task", "episode"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_label"),
    ).sort_values("sealed_row")
    if len(merged) != len(index) or merged["failure"].isna().any():
        raise ValueError("sealed episodes do not match frozen label table")
    if not np.array_equal(
        merged["sealed_length"].to_numpy(), merged["episode_length"].to_numpy()
    ):
        raise ValueError("episode lengths changed between seal and evaluation")
    if not np.array_equal(
        merged["init_state_id"].to_numpy(),
        merged["init_state_id_label"].to_numpy(),
    ):
        raise ValueError("initial-state IDs changed between seal and evaluation")
    if not np.array_equal(
        merged["flow_noise_seed"].to_numpy(),
        merged["flow_noise_seed_label"].to_numpy(),
    ):
        raise ValueError("flow-noise seeds changed between seal and evaluation")
    return merged.reset_index(drop=True)


def load_outcomes_from_summaries(
    sealed: dict[str, np.ndarray], cache_root: Path
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Outcome-only loader for a sealed cohort without a physical-label table."""
    task_names = sealed["task_names"].astype(str)
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for task_index, task in enumerate(task_names):
        path = cache_root / task / RUN_ID / "client/summaries.json"
        source_hashes[str(path)] = sha256(path)
        summaries = json.loads(path.read_text(encoding="utf-8"))
        by_episode = {int(row["episode_index"]): row for row in summaries}
        take = np.flatnonzero(sealed["task_index"].astype(int) == task_index)
        for sealed_row in take:
            episode = int(sealed["episode"][sealed_row])
            summary = by_episode[episode]
            expected = {
                "init_state_id": int(sealed["init_state_id"][sealed_row]),
                "flow_noise_seed": int(sealed["flow_noise_seed"][sealed_row]),
                "inference_calls": int(sealed["length"][sealed_row]),
            }
            for key, value in expected.items():
                if int(summary[key]) != value:
                    raise ValueError(f"external metadata mismatch: {task}/{episode}/{key}")
            rows.append(
                {
                    "sealed_row": int(sealed_row),
                    "task": task,
                    "episode": episode,
                    "init_state_id": expected["init_state_id"],
                    "flow_noise_seed": expected["flow_noise_seed"],
                    "sealed_length": expected["inference_calls"],
                    "episode_length": expected["inference_calls"],
                    "failure": not bool(summary["success"]),
                    "primary_behavior": "not_labeled",
                }
            )
    frame = pd.DataFrame(rows).sort_values("sealed_row").reset_index(drop=True)
    if not np.array_equal(frame["sealed_row"].to_numpy(), np.arange(len(frame))):
        raise ValueError("external outcome rows do not align to sealed score rows")
    return frame, source_hashes


def first_alarm_query(alarm: np.ndarray) -> np.ndarray:
    alarm = np.asarray(alarm, dtype=bool)
    any_alarm = alarm.any(axis=1)
    first = np.argmax(alarm, axis=1).astype(np.int16)
    first[~any_alarm] = -1
    return first


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def clustered_interval(
    numerators: np.ndarray,
    denominators: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    if draws <= 0:
        return float("nan"), float("nan")
    sample = rng.integers(0, len(numerators), size=(draws, len(numerators)))
    numerator = numerators[sample].sum(axis=1)
    denominator = denominators[sample].sum(axis=1)
    values = numerator / np.maximum(denominator, 1.0)
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


def count_by_task(
    tasks: np.ndarray,
    mask: np.ndarray,
    condition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = np.unique(tasks)
    numerator = np.asarray(
        [np.sum(mask & condition & (tasks == task)) for task in names],
        dtype=np.float64,
    )
    denominator = np.asarray(
        [np.sum(mask & (tasks == task)) for task in names], dtype=np.float64
    )
    return names, numerator, denominator


def outcome_tables(
    sealed: dict[str, np.ndarray],
    labels: pd.DataFrame,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, float], np.ndarray]]:
    detector_names = sealed["detector_names"].astype(str)
    quantiles = sealed["quantiles"].astype(float)
    alarms = sealed["alarms"].astype(bool)
    failure = labels["failure"].to_numpy(dtype=bool)
    success = ~failure
    lengths = sealed["length"].astype(int)
    tasks = labels["task"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    first_lookup: dict[tuple[str, float], np.ndarray] = {}

    for detector_index, detector in enumerate(detector_names):
        for quantile_index, quantile in enumerate(quantiles):
            first = first_alarm_query(alarms[:, :, detector_index, quantile_index])
            first_lookup[(detector, float(quantile))] = first
            alerted = first >= 0
            lead = lengths - 1 - first
            early4 = alerted & (lead >= 4)
            early2 = alerted & (lead >= 2)

            _, tp_task, fail_task = count_by_task(tasks, failure, alerted)
            _, fp_task, success_task = count_by_task(tasks, success, alerted)
            _, early4_task, _ = count_by_task(tasks, failure, early4)
            _, early2_task, _ = count_by_task(tasks, failure, early2)
            _, alarm_failure_task, alarm_task = count_by_task(
                tasks, alerted, failure
            )
            recall_ci = clustered_interval(tp_task, fail_task, draws, rng)
            fpr_ci = clustered_interval(fp_task, success_task, draws, rng)
            precision_ci = clustered_interval(
                alarm_failure_task, alarm_task, draws, rng
            )
            early4_ci = clustered_interval(early4_task, fail_task, draws, rng)
            early2_ci = clustered_interval(early2_task, fail_task, draws, rng)

            detected_lead = lead[failure & alerted]
            rows.append(
                {
                    "detector": detector,
                    "quantile": quantile,
                    "failure_n": int(failure.sum()),
                    "success_n": int(success.sum()),
                    "tp": int(np.sum(failure & alerted)),
                    "fp": int(np.sum(success & alerted)),
                    "fn": int(np.sum(failure & ~alerted)),
                    "tn": int(np.sum(success & ~alerted)),
                    "failure_recall": float(alerted[failure].mean()),
                    "failure_recall_ci_low": recall_ci[0],
                    "failure_recall_ci_high": recall_ci[1],
                    "success_fpr": float(alerted[success].mean()),
                    "success_fpr_ci_low": fpr_ci[0],
                    "success_fpr_ci_high": fpr_ci[1],
                    "precision": ratio(np.sum(failure & alerted), np.sum(alerted)),
                    "precision_ci_low": precision_ci[0],
                    "precision_ci_high": precision_ci[1],
                    "early4_recall": float(early4[failure].mean()),
                    "early4_recall_ci_low": early4_ci[0],
                    "early4_recall_ci_high": early4_ci[1],
                    "early2_recall": float(early2[failure].mean()),
                    "early2_recall_ci_low": early2_ci[0],
                    "early2_recall_ci_high": early2_ci[1],
                    "detected_failure_lead_median": (
                        float(np.median(detected_lead)) if len(detected_lead) else np.nan
                    ),
                    "detected_failure_lead_q25": (
                        float(np.quantile(detected_lead, 0.25))
                        if len(detected_lead)
                        else np.nan
                    ),
                    "detected_failure_lead_q75": (
                        float(np.quantile(detected_lead, 0.75))
                        if len(detected_lead)
                        else np.nan
                    ),
                }
            )

            for task in np.unique(tasks):
                take = tasks == task
                fail_take = take & failure
                success_take = take & success
                task_rows.append(
                    {
                        "task": task,
                        "detector": detector,
                        "quantile": quantile,
                        "failure_n": int(fail_take.sum()),
                        "success_n": int(success_take.sum()),
                        "tp": int(np.sum(fail_take & alerted)),
                        "fp": int(np.sum(success_take & alerted)),
                        "failure_recall": (
                            float(alerted[fail_take].mean())
                            if fail_take.any()
                            else np.nan
                        ),
                        "success_fpr": (
                            float(alerted[success_take].mean())
                            if success_take.any()
                            else np.nan
                        ),
                    }
                )

    outcome = pd.DataFrame(rows)
    by_task = pd.DataFrame(task_rows)
    comparisons = paired_comparisons(
        first_lookup, detector_names, quantiles, labels, lengths, draws, seed + 1
    )
    return outcome, by_task, comparisons, first_lookup


def paired_comparisons(
    first_lookup: dict[tuple[str, float], np.ndarray],
    detector_names: np.ndarray,
    quantiles: np.ndarray,
    labels: pd.DataFrame,
    lengths: np.ndarray,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    failure = labels["failure"].to_numpy(dtype=bool)
    success = ~failure
    tasks = labels["task"].astype(str).to_numpy()
    task_values = np.unique(tasks)
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    candidates = ("multi_max",)
    baselines = (
        "instability",
        "lock_in",
        "flat_narrow_support",
        "feedback_decoupling",
        "dual_mean",
        "dual_max",
        "instant_multi_max",
        "clock",
    )
    for quantile in quantiles:
        for candidate in candidates:
            candidate_first = first_lookup[(candidate, float(quantile))]
            candidate_alarm = candidate_first >= 0
            candidate_early4 = candidate_alarm & (
                lengths - 1 - candidate_first >= 4
            )
            for baseline in baselines:
                if baseline not in detector_names:
                    continue
                baseline_first = first_lookup[(baseline, float(quantile))]
                baseline_alarm = baseline_first >= 0
                baseline_early4 = baseline_alarm & (
                    lengths - 1 - baseline_first >= 4
                )

                count = {
                    "fail": [],
                    "success": [],
                    "candidate_tp": [],
                    "baseline_tp": [],
                    "candidate_fp": [],
                    "baseline_fp": [],
                    "candidate_early4": [],
                    "baseline_early4": [],
                }
                for task in task_values:
                    take = tasks == task
                    count["fail"].append(np.sum(take & failure))
                    count["success"].append(np.sum(take & success))
                    count["candidate_tp"].append(
                        np.sum(take & failure & candidate_alarm)
                    )
                    count["baseline_tp"].append(
                        np.sum(take & failure & baseline_alarm)
                    )
                    count["candidate_fp"].append(
                        np.sum(take & success & candidate_alarm)
                    )
                    count["baseline_fp"].append(
                        np.sum(take & success & baseline_alarm)
                    )
                    count["candidate_early4"].append(
                        np.sum(take & failure & candidate_early4)
                    )
                    count["baseline_early4"].append(
                        np.sum(take & failure & baseline_early4)
                    )
                count = {key: np.asarray(value, np.float64) for key, value in count.items()}
                sample = rng.integers(
                    0, len(task_values), size=(draws, len(task_values))
                )
                fail_denominator = np.maximum(count["fail"][sample].sum(axis=1), 1)
                success_denominator = np.maximum(
                    count["success"][sample].sum(axis=1), 1
                )
                recall_delta = (
                    count["candidate_tp"][sample].sum(axis=1)
                    - count["baseline_tp"][sample].sum(axis=1)
                ) / fail_denominator
                fpr_delta = (
                    count["candidate_fp"][sample].sum(axis=1)
                    - count["baseline_fp"][sample].sum(axis=1)
                ) / success_denominator
                early4_delta = (
                    count["candidate_early4"][sample].sum(axis=1)
                    - count["baseline_early4"][sample].sum(axis=1)
                ) / fail_denominator

                def summarize(values: np.ndarray) -> tuple[float, float, float]:
                    low, high = np.quantile(values, (0.025, 0.975))
                    p = min(
                        1.0,
                        2.0
                        * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))),
                    )
                    return float(low), float(high), p

                recall_ci = summarize(recall_delta)
                fpr_ci = summarize(fpr_delta)
                early4_ci = summarize(early4_delta)
                output.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "quantile": float(quantile),
                        "recall_delta": float(
                            candidate_alarm[failure].mean()
                            - baseline_alarm[failure].mean()
                        ),
                        "recall_delta_ci_low": recall_ci[0],
                        "recall_delta_ci_high": recall_ci[1],
                        "recall_delta_p": recall_ci[2],
                        "success_fpr_delta": float(
                            candidate_alarm[success].mean()
                            - baseline_alarm[success].mean()
                        ),
                        "success_fpr_delta_ci_low": fpr_ci[0],
                        "success_fpr_delta_ci_high": fpr_ci[1],
                        "success_fpr_delta_p": fpr_ci[2],
                        "early4_recall_delta": float(
                            candidate_early4[failure].mean()
                            - baseline_early4[failure].mean()
                        ),
                        "early4_delta_ci_low": early4_ci[0],
                        "early4_delta_ci_high": early4_ci[1],
                        "early4_delta_p": early4_ci[2],
                        "failure_candidate_only": int(
                            np.sum(failure & candidate_alarm & ~baseline_alarm)
                        ),
                        "failure_baseline_only": int(
                            np.sum(failure & baseline_alarm & ~candidate_alarm)
                        ),
                        "success_candidate_only": int(
                            np.sum(success & candidate_alarm & ~baseline_alarm)
                        ),
                        "success_baseline_only": int(
                            np.sum(success & baseline_alarm & ~candidate_alarm)
                        ),
                        "bootstrap_unit": "task",
                        "bootstrap_draws": draws,
                    }
                )
    return pd.DataFrame(output)


def longest_true_run(mask: np.ndarray) -> tuple[int, int, int]:
    values = np.asarray(mask, dtype=bool)
    best_length = best_start = best_end = 0
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        end = start + 1
        while end < len(values) and values[end]:
            end += 1
        if end - start > best_length:
            best_length, best_start, best_end = end - start, start, end
        start = end
    return best_length, best_start, best_end


def nonzero_sign_flip_indices(values: np.ndarray) -> np.ndarray:
    signs = np.sign(np.asarray(values, dtype=np.float64))
    nonzero = np.flatnonzero(signs != 0)
    if len(nonzero) < 2:
        return np.empty(0, dtype=np.int64)
    return nonzero[1:][signs[nonzero[1:]] != signs[nonzero[:-1]]]


def reversal_indices(delta: np.ndarray) -> np.ndarray:
    vector = np.asarray(delta, dtype=np.float64)
    norm = np.linalg.norm(vector, axis=1)
    output = []
    for index in range(1, len(vector)):
        if norm[index - 1] <= ACTIVE_EEF_M or norm[index] <= ACTIVE_EEF_M:
            continue
        cosine = float(
            np.dot(vector[index - 1], vector[index])
            / (norm[index - 1] * norm[index])
        )
        if cosine < REVERSAL_COSINE:
            output.append(index + 1)
    return np.asarray(output, dtype=np.int64)


def hysteresis_exits(
    distance: np.ndarray, enter_radius: float, exit_radius: float
) -> np.ndarray:
    inside = False
    exits = []
    for index, value in enumerate(np.asarray(distance, dtype=np.float64)):
        if not inside and value <= enter_radius:
            inside = True
        elif inside and value >= exit_radius:
            exits.append(index)
            inside = False
    return np.asarray(exits, dtype=np.int64)


def load_task_targets(
    task: str, cache_root: Path, audit: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    run = cache_root / task / RUN_ID
    layout = json.loads((run / "client/sim_layout.json").read_text(encoding="utf-8"))
    joints = {str(row["joint"]): row for row in layout["joints"]}
    targets = []
    task_audit = audit[task]
    for name, goal in zip(
        task_audit["targets"], task_audit["goal_positions"]
    ):
        joint = joints[name]
        lo = int(joint["state_lo"])
        targets.append(
            {
                "name": name,
                "lo": lo,
                "hi": lo + 3,
                "goal": np.asarray(goal, dtype=np.float64),
            }
        )
    return targets


def physical_onset(
    behavior: str,
    state: np.ndarray,
    actions: np.ndarray,
    sim: np.ndarray,
    targets: list[dict[str, Any]],
) -> tuple[int, str]:
    length = len(state)
    eef_delta = np.diff(np.asarray(state[:, :3], dtype=np.float64), axis=0)
    eef_step = np.linalg.norm(eef_delta, axis=1)
    gripper = np.asarray(state[:, 6:8], dtype=np.float64).mean(axis=1)
    gripper_step = np.abs(np.diff(gripper))

    positions = [
        np.asarray(sim[:, target["lo"] : target["hi"]], dtype=np.float64)
        for target in targets
    ]
    distances = [
        np.linalg.norm(position - target["goal"][None, :], axis=1)
        for position, target in zip(positions, targets)
    ]

    if behavior == "stagnation":
        target_step = (
            np.stack(
                [np.linalg.norm(np.diff(position, axis=0), axis=1) for position in positions],
                axis=1,
            ).max(axis=1)
            if positions
            else np.zeros(length - 1, dtype=np.float64)
        )
        static = (
            (eef_step <= STATIC_EEF_M)
            & (target_step <= STATIC_OBJECT_M)
            & (gripper_step <= STATIC_GRIPPER_M)
        )
        phase = (np.arange(length - 1, dtype=np.float64) + 0.5) / max(
            length - 1, 1
        )
        eligible = static & (phase <= 0.9 + 1e-12)
        run = longest_true_run(eligible)
        return (int(run[1]), "longest_pre90_static_run") if run[0] else (-1, "none")

    if behavior == "gripper_cycling":
        command = np.asarray(actions[:, :, 6], dtype=np.float64).mean(axis=1)
        flips = nonzero_sign_flip_indices(command)
        return (int(flips[2]), "third_gripper_flip") if len(flips) >= 3 else (-1, "none")

    if behavior in {"active_retry", "eef_oscillation"}:
        reversals = reversal_indices(eef_delta)
        return (int(reversals[2]), "third_eef_reversal") if len(reversals) >= 3 else (-1, "none")

    if behavior == "goal_regression" and distances:
        regrets = np.asarray([distance[-1] - distance.min() for distance in distances])
        target = int(np.argmax(regrets))
        return int(np.argmin(distances[target])), "minimum_goal_distance"

    if behavior == "goal_approach_leave" and distances:
        exits = [hysteresis_exits(distance, APPROACH_M, APPROACH_EXIT_M) for distance in distances]
        candidates = np.concatenate([value for value in exits if len(value)]) if any(
            len(value) for value in exits
        ) else np.empty(0, dtype=np.int64)
        return (int(candidates.min()), "first_goal_region_exit") if len(candidates) else (-1, "none")

    if behavior == "subtask_undo" and distances:
        candidates = []
        for distance in distances:
            at_goal = np.flatnonzero(distance <= GOAL_M)
            if not len(at_goal):
                continue
            after = np.flatnonzero(
                (np.arange(length) > at_goal[0]) & (distance >= UNDO_EXIT_M)
            )
            if len(after):
                candidates.append(int(after[0]))
        return (min(candidates), "first_subtask_undo") if candidates else (-1, "none")

    if behavior == "regrasp_or_drop" and distances:
        candidates = []
        for position, distance in zip(positions, distances):
            lifted = position[:, 2] - position[0, 2] > LIFT_M
            onset = np.flatnonzero(lifted & ~np.r_[False, lifted[:-1]])
            loss = np.flatnonzero(~lifted & np.r_[False, lifted[:-1]])
            offgoal_loss = loss[distance[loss] > GOAL_M] if len(loss) else loss
            if len(offgoal_loss):
                candidates.append((int(offgoal_loss[0]), "first_offgoal_lift_loss"))
            if len(onset) >= 2:
                candidates.append((int(onset[1]), "second_lift_onset"))
        return min(candidates, key=lambda value: value[0]) if candidates else (-1, "none")

    return -1, "none"


def build_onset_table(
    labels: pd.DataFrame, cache_root: Path, target_path: Path
) -> pd.DataFrame:
    raw_audit = json.loads(target_path.read_text(encoding="utf-8"))
    audit = {str(row["task"]): row for row in raw_audit}
    output: list[dict[str, Any]] = []
    failure_rows = labels[labels["failure"].astype(bool)]
    for task_index, (task, task_rows) in enumerate(
        failure_rows.groupby("task", sort=True), start=1
    ):
        run = cache_root / task / RUN_ID
        targets = load_task_targets(task, cache_root, audit)
        for row in task_rows.itertuples(index=False):
            behavior = str(row.primary_behavior)
            onset = -1
            source = "unsupported_behavior"
            if behavior in ONSET_BEHAVIORS:
                episode_path = run / "client" / f"episode_{int(row.episode):02d}.npz"
                with np.load(episode_path, allow_pickle=False) as data:
                    state = np.asarray(data["state"], dtype=np.float32)
                    actions = np.asarray(data["actions"], dtype=np.float32)
                    sim = np.asarray(data["sim_state"], dtype=np.float32)
                onset, source = physical_onset(
                    behavior, state, actions, sim, targets
                )
            output.append(
                {
                    "sealed_row": int(row.sealed_row),
                    "task": task,
                    "episode": int(row.episode),
                    "primary_behavior": behavior,
                    "onset_query": onset,
                    "onset_source": source,
                    "episode_length": int(row.episode_length),
                }
            )
        print(
            f"[onsets {task_index}/{failure_rows['task'].nunique()}] {task}",
            flush=True,
        )
    return pd.DataFrame(output)


def onset_tables(
    sealed: dict[str, np.ndarray],
    onsets: pd.DataFrame,
    first_lookup: dict[tuple[str, float], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detector_names = sealed["detector_names"].astype(str)
    head_names = sealed["head_names"].astype(str)
    quantiles = sealed["quantiles"].astype(float)
    primary_quantile = float(quantiles[np.argmin(np.abs(quantiles - 0.95))])
    usable = onsets[onsets["onset_query"] >= 0].copy()
    rows: list[dict[str, Any]] = []
    for behavior in ("all", *sorted(usable["primary_behavior"].unique())):
        subset = usable if behavior == "all" else usable[usable["primary_behavior"] == behavior]
        index = subset["sealed_row"].to_numpy(dtype=int)
        onset = subset["onset_query"].to_numpy(dtype=int)
        for detector in detector_names:
            first = first_lookup[(detector, primary_quantile)][index]
            relative = first - onset
            alerted = first >= 0
            rows.append(
                {
                    "primary_behavior": behavior,
                    "detector": detector,
                    "quantile": primary_quantile,
                    "onset_n": len(subset),
                    "alarm_any_rate": float(alerted.mean()) if len(subset) else np.nan,
                    "alarm_before_onset_rate": (
                        float(np.mean(alerted & (relative <= -1)))
                        if len(subset)
                        else np.nan
                    ),
                    "strict_precursor_m4_m1_rate": (
                        float(np.mean(alerted & (relative >= -4) & (relative <= -1)))
                        if len(subset)
                        else np.nan
                    ),
                    "timely_m2_p2_rate": (
                        float(np.mean(alerted & (relative >= -2) & (relative <= 2)))
                        if len(subset)
                        else np.nan
                    ),
                    "reaction_p0_p4_rate": (
                        float(np.mean(alerted & (relative >= 0) & (relative <= 4)))
                        if len(subset)
                        else np.nan
                    ),
                    "first_alarm_relative_median": (
                        float(np.median(relative[alerted])) if alerted.any() else np.nan
                    ),
                }
            )

    multi_first = first_lookup[("multi_max", primary_quantile)]
    winning = sealed["winning_head"].astype(int)
    composition: list[dict[str, Any]] = []
    for behavior, subset in usable.groupby("primary_behavior", sort=True):
        index = subset["sealed_row"].to_numpy(dtype=int)
        onset = subset["onset_query"].to_numpy(dtype=int)
        first = multi_first[index]
        relative = first - onset
        timely = (first >= 0) & (relative >= -2) & (relative <= 2)
        winners = winning[index[timely], first[timely]] if timely.any() else np.empty(0, int)
        for head_index, head in enumerate(head_names):
            count = int(np.sum(winners == head_index))
            composition.append(
                {
                    "primary_behavior": behavior,
                    "head": head,
                    "onset_n": len(subset),
                    "timely_multi_alarm_n": int(timely.sum()),
                    "winning_head_n": count,
                    "winning_head_rate_among_timely": ratio(count, timely.sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(composition)


def subtype_table(
    labels: pd.DataFrame,
    first_lookup: dict[tuple[str, float], np.ndarray],
) -> pd.DataFrame:
    output = []
    quantiles = sorted({quantile for _detector, quantile in first_lookup})
    primary_quantile = min(quantiles, key=lambda value: abs(value - 0.95))
    behaviors = ["all_failures", *sorted(labels.loc[labels.failure, "primary_behavior"].unique())]
    for behavior in behaviors:
        take = labels["failure"].to_numpy(dtype=bool).copy()
        if behavior != "all_failures":
            take &= labels["primary_behavior"].eq(behavior).to_numpy()
        for detector in (
            "instability",
            "lock_in",
            "flat_narrow_support",
            "feedback_decoupling",
            "dual_mean",
            "multi_max",
            "clock",
        ):
            first = first_lookup[(detector, primary_quantile)]
            output.append(
                {
                    "primary_behavior": behavior,
                    "detector": detector,
                    "n": int(take.sum()),
                    "alarm_n": int(np.sum(take & (first >= 0))),
                    "alarm_rate": (
                        float(np.mean(first[take] >= 0)) if take.any() else np.nan
                    ),
                }
            )
    return pd.DataFrame(output)


def self_test() -> None:
    alarm = np.asarray(
        [[False, False, True, True], [False, False, False, False]], dtype=bool
    )
    np.testing.assert_array_equal(first_alarm_query(alarm), [2, -1])
    assert longest_true_run([False, True, True, False, True]) == (2, 1, 3)
    np.testing.assert_array_equal(nonzero_sign_flip_indices([1, 1, -1, -1, 1]), [2, 4])
    delta = np.asarray([[1.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]])
    np.testing.assert_array_equal(reversal_indices(delta), [2, 3, 4])
    np.testing.assert_array_equal(
        hysteresis_exits([0.2, 0.09, 0.11, 0.16, 0.08, 0.2], 0.1, 0.15),
        [3, 5],
    )
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    manifest = verify_seal(args.result_dir)
    sealed = load_sealed(args.result_dir)
    if args.outcomes_from_summaries:
        labels, outcome_source_hashes = load_outcomes_from_summaries(
            sealed, args.cache_root
        )
    else:
        labels = join_labels(sealed, args.labels)
        outcome_source_hashes = {str(args.labels): sha256(args.labels)}
    print(
        f"labels revealed after seal: episodes={len(labels)}, "
        f"failures={int(labels.failure.sum())}, successes={int((~labels.failure).sum())}",
        flush=True,
    )

    outcome, by_task, comparisons, first_lookup = outcome_tables(
        sealed, labels, args.bootstrap, args.seed
    )
    subtype = subtype_table(labels, first_lookup)
    outcome.to_csv(args.result_dir / "outcome_metrics.csv", index=False)
    by_task.to_csv(args.result_dir / "outcome_metrics_by_task.csv", index=False)
    comparisons.to_csv(args.result_dir / "paired_detector_comparisons.csv", index=False)
    subtype.to_csv(args.result_dir / "failure_subtype_alarm_rates.csv", index=False)

    onset_metrics = pd.DataFrame()
    onset_composition = pd.DataFrame()
    if not args.skip_onsets and not args.outcomes_from_summaries:
        onsets = build_onset_table(labels, args.cache_root, args.targets)
        onset_metrics, onset_composition = onset_tables(sealed, onsets, first_lookup)
        onsets.to_csv(args.result_dir / "physical_proxy_onsets.csv", index=False)
        onset_metrics.to_csv(args.result_dir / "onset_event_metrics.csv", index=False)
        onset_composition.to_csv(
            args.result_dir / "onset_winning_head_composition.csv", index=False
        )

    primary = outcome[
        (outcome["detector"] == "multi_max")
        & np.isclose(outcome["quantile"], 0.95)
    ].iloc[0]
    summary = {
        "schema": "himoe.online_multihead.evaluation.v1",
        "seal_verified_before_labels": True,
        "sealed_manifest_sha256": sha256(args.result_dir / "sealed_manifest.json"),
        "label_files": outcome_source_hashes,
        "episodes": len(labels),
        "failures": int(labels.failure.sum()),
        "successes": int((~labels.failure).sum()),
        "tasks": int(labels.task.nunique()),
        "primary_detector": "multi_max",
        "primary_quantile": 0.95,
        "primary": plain(primary.to_dict()),
        "physical_onsets_available": int(
            (pd.read_csv(args.result_dir / "physical_proxy_onsets.csv")["onset_query"] >= 0).sum()
        )
        if not args.skip_onsets and not args.outcomes_from_summaries
        else 0,
        "bootstrap_unit": "task",
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
        "labels_used_by_evaluator": (
            ["success/failure"]
            if args.outcomes_from_summaries
            else [
                "success/failure",
                "physical proxy behavior labels",
                "physical proxy onset signals",
            ]
        ),
        "labels_used_by_scorer": manifest["labels_used"],
    }
    (args.result_dir / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plain(summary["primary"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
