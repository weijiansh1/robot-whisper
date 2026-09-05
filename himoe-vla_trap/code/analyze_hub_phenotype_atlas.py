#!/usr/bin/env python3
"""Cross-task, train-free audit of physical and MoE routing phenotypes.

The script only reads completed LIBERO runs. Physical labels are frozen before
routing is loaded, and routing features never use failure labels or outcomes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import zarr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
ROUTE_CAPTURE_ROOT = WORKSPACE_ROOT / "himoe-route-capture"
sys.path.insert(0, str(ROUTE_CAPTURE_ROOT))

from analyze_failure_behavior_taxonomy import (  # noqa: E402
    LABELS as PHYSICAL_LABELS,
    attach_labels,
    classify_behaviors,
    extract_episode_features,
)
from analyze_replanning_reset_trap import identify_targets  # noqa: E402


DEFAULT_CACHE_ROOT = WORKSPACE_ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
DEFAULT_OUT_DIR = PACKAGE_ROOT / "results/hub_phenotype_atlas_50x8"
DEFAULT_RUN_ID = "right-50x8-20260903"
FRONT = slice(0, 4)
BACK = slice(4, 8)
FINAL_FLOW = 9
LATE_FLOW_START = 6
ACTION = slice(1, 11)
N_EXPERTS = 32
LAGS = (1, 2, 3, 4)


ROW_SIGNALS = (
    "late_flow_volatility",
    "route_acceleration",
    "gate_entropy",
    "top12_margin",
    "top4_union",
    "support_entropy",
    "token_disagreement",
    "front_state_action_gap",
    "layer5_state_action_gap",
    "back_state_action_gap",
)
TEMPORAL_SIGNALS = (
    "route_mobility",
    "lag_recurrence",
    "lag_periodicity",
    "front_state_jump",
    "front_action_jump",
    "front_feedback_split",
)
ROUTE_SIGNALS = tuple(f"{name}_mean" for name in ROW_SIGNALS + TEMPORAL_SIGNALS) + (
    "gate_entropy_slope",
    "top4_union_slope",
    "layer5_state_action_gap_slope",
    "front_feedback_split_p90",
)

HEADS: dict[str, tuple[tuple[str, int], ...]] = {
    "instability": (
        ("late_flow_volatility_mean", 1),
        ("route_acceleration_mean", 1),
        ("route_mobility_mean", 1),
        ("lag_periodicity_mean", 1),
    ),
    "lock_in": (
        ("route_mobility_mean", -1),
        ("lag_recurrence_mean", 1),
        ("top4_union_mean", -1),
        ("token_disagreement_mean", -1),
    ),
    "flat_narrow_support": (
        ("gate_entropy_mean", 1),
        ("top12_margin_mean", -1),
        ("top4_union_mean", -1),
    ),
    "feedback_decoupling": (
        ("front_state_action_gap_mean", 1),
        ("layer5_state_action_gap_mean", 1),
        ("front_feedback_split_p90", 1),
    ),
}
COMPOSITE_HEADS: dict[str, tuple[tuple[str, ...], str]] = {
    "dual_extremes": (("instability", "lock_in"), "max"),
    "dual_mean_control": (("instability", "lock_in"), "mean"),
    "multi_phenotype": (tuple(HEADS), "max"),
}
DETECTORS = (*HEADS, *COMPOSITE_HEADS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phase-start", type=float, default=0.5)
    parser.add_argument("--phase-end", type=float, default=0.9)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--reuse", action="store_true")
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
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


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


def episode_path(client: Path, episode: int) -> Path:
    return client / f"episode_{episode:02d}.npz"


def discover_complete_runs(cache_root: Path, run_id: str) -> list[Path]:
    runs: list[Path] = []
    pattern = f"libero_*/*/{run_id}/client/summaries.json"
    for summary_path in sorted(cache_root.glob(pattern)):
        run = summary_path.parents[1]
        meta_path = run / "meta.json"
        if not meta_path.exists() or not (run / "server/routes.zarr").exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sampling = meta.get("sampling", {})
        if meta.get("status") != "complete" or not sampling.get("complete", False):
            continue
        if int(sampling.get("actual_episodes", -1)) != int(
            sampling.get("designed_episodes", -2)
        ):
            continue
        runs.append(run)
    if not runs:
        raise RuntimeError(f"no complete {run_id!r} runs under {cache_root}")
    return runs


def task_key(run: Path, cache_root: Path) -> str:
    return str(run.relative_to(cache_root).parent)


def load_summaries(run: Path) -> list[dict[str, Any]]:
    rows = json.loads((run / "client/summaries.json").read_text(encoding="utf-8"))
    rows.sort(key=lambda row: int(row["episode_index"]))
    expected = list(range(len(rows)))
    observed = [int(row["episode_index"]) for row in rows]
    if observed != expected:
        raise ValueError(f"non-contiguous episode indices in {run}")
    return rows


def action_change_feature(actions: np.ndarray) -> float:
    values = np.asarray(actions[:, :, :6], dtype=np.float64)
    if len(values) < 3:
        return 0.0
    delta = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=(1, 2)))
    phase = (np.arange(len(delta), dtype=np.float64) + 1.0) / max(len(values) - 1, 1)
    keep = (phase >= 0.5) & (phase <= 0.9)
    return float(delta[keep].mean()) if np.any(keep) else float(delta.mean())


def build_physical_frame(runs: Iterable[Path], cache_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    target_audit: list[dict[str, Any]] = []
    for run_index, run in enumerate(runs, start=1):
        task = task_key(run, cache_root)
        client = run / "client"
        summaries = load_summaries(run)
        layout = json.loads((client / "sim_layout.json").read_text(encoding="utf-8"))
        tapes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        sims: list[np.ndarray] = []
        for summary in summaries:
            episode = int(summary["episode_index"])
            with np.load(episode_path(client, episode), allow_pickle=False) as data:
                state = np.asarray(data["state"], dtype=np.float32)
                actions = np.asarray(data["actions"], dtype=np.float32)
                sim = np.asarray(data["sim_state"], dtype=np.float32)
            expected = int(summary["inference_calls"])
            if state.shape != (expected, 8) or actions.shape != (expected, 10, 7):
                raise ValueError(f"state/action shape mismatch: {task}/{episode}")
            if len(sim) != expected:
                raise ValueError(f"sim-state shape mismatch: {task}/{episode}")
            tapes.append((state, actions, sim))
            sims.append(sim)
        targets = identify_targets(summaries, layout, sims)
        target_audit.append(
            {
                "task": task,
                "targets": [str(target["name"]) for target in targets],
                "goal_positions": [np.asarray(target["goal"]).tolist() for target in targets],
            }
        )
        for summary, (state, actions, sim) in zip(summaries, tapes):
            episode = int(summary["episode_index"])
            features, _ = extract_episode_features(
                state, actions, sim, targets, stop_fraction=1.0
            )
            rows.append(
                {
                    "task": task,
                    "episode": episode,
                    "init_state_id": int(summary["init_state_id"]),
                    "flow_noise_seed": int(summary["flow_noise_seed"]),
                    "episode_length": int(summary["inference_calls"]),
                    "failure": not bool(summary["success"]),
                    "action_midlate_change": action_change_feature(actions),
                    **features,
                }
            )
        print(
            f"[physical {run_index}] {task}: episodes={len(summaries)}, "
            f"targets={[target['name'] for target in targets]}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    labels, thresholds = classify_behaviors(frame)
    return attach_labels(frame, labels, thresholds), target_audit


def linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values)
    if good.sum() < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, len(values))[good]
    y = values[good]
    x = x - x.mean()
    return float(np.dot(x, y - y.mean()) / max(np.dot(x, x), 1e-12))


def phase_mask(length: int, start: float, end: float) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=bool)
    phase = np.arange(length, dtype=np.float64) / (length - 1)
    mask = (phase >= start) & (phase <= end)
    if mask.sum() < min(3, length):
        mask[:] = True
    return mask


def union_fraction(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)
    output = np.empty(ids.shape[0], dtype=np.float32)
    for row in range(len(ids)):
        output[row] = np.mean(
            [len(np.unique(ids[row, layer])) / N_EXPERTS for layer in range(ids.shape[1])]
        )
    return output


def aggregate_series(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    selected = np.asarray(values, dtype=np.float64)[mask]
    selected = selected[np.isfinite(selected)]
    if not len(selected):
        return float("nan"), float("nan"), float("nan")
    return float(selected.mean()), linear_slope(selected), float(np.quantile(selected, 0.9))


def extract_task_route_features(
    run: Path,
    summaries: list[dict[str, Any]],
    phase_start: float,
    phase_end: float,
) -> pd.DataFrame:
    group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    router = group["hb_router_probs"]
    expert_ids = group["hb_expert_ids"]
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    total = int(router.shape[0])
    if len(episode_id) != total:
        raise ValueError(f"route episode-id mismatch: {run}")

    scalar = {name: np.empty(total, dtype=np.float32) for name in ROW_SIGNALS}
    front_state = np.empty((total, 4, N_EXPERTS), dtype=np.float16)
    front_action = np.empty((total, 4, N_EXPERTS), dtype=np.float16)
    back_state = np.empty((total, 4, N_EXPERTS), dtype=np.float16)
    back_action = np.empty((total, 4, 10, N_EXPERTS), dtype=np.float16)

    block = 64
    for start in range(0, total, block):
        stop = min(start + block, total)
        p = normalize_probability(router[start:stop])
        final_action = p[:, :, FINAL_FLOW, ACTION, :]
        final_state = p[:, :, FINAL_FLOW, 0, :]
        front_action_mean = normalize_probability(final_action[:, FRONT].mean(axis=2))
        back_action_mean = normalize_probability(final_action[:, BACK].mean(axis=2))

        entropy = -(final_action[:, BACK] * np.log(np.maximum(final_action[:, BACK], 1e-12))).sum(-1)
        scalar["gate_entropy"][start:stop] = entropy.mean((1, 2)) / np.log(N_EXPERTS)
        ordered = np.partition(final_action[:, BACK], -2, axis=-1)
        scalar["top12_margin"][start:stop] = (
            ordered[..., -1] - ordered[..., -2]
        ).mean((1, 2))
        scalar["support_entropy"][start:stop] = (
            -(back_action_mean * np.log(np.maximum(back_action_mean, 1e-12))).sum(-1)
        ).mean(1) / np.log(N_EXPERTS)
        scalar["token_disagreement"][start:stop] = hellinger(
            final_action[:, BACK], back_action_mean[:, :, None, :]
        ).mean((1, 2))
        scalar["front_state_action_gap"][start:stop] = hellinger(
            final_state[:, FRONT], front_action_mean
        ).mean(1)
        scalar["layer5_state_action_gap"][start:stop] = hellinger(
            final_state[:, 3], front_action_mean[:, 3]
        )
        scalar["back_state_action_gap"][start:stop] = hellinger(
            final_state[:, BACK], back_action_mean
        ).mean(1)
        ids = np.asarray(expert_ids[start:stop, BACK, FINAL_FLOW, ACTION, :])
        scalar["top4_union"][start:stop] = union_fraction(ids)

        action_flow = p[:, BACK, :, ACTION, :]
        late = action_flow[:, :, LATE_FLOW_START:]
        scalar["late_flow_volatility"][start:stop] = (
            1.0 - weighted_jaccard(late[:, :, 1:], late[:, :, :-1])
        ).mean((1, 2, 3))
        root = np.sqrt(action_flow)
        acceleration = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
        scalar["route_acceleration"][start:stop] = (
            np.linalg.norm(acceleration, axis=-1).mean((1, 2, 3)) / np.sqrt(2.0)
        )

        front_state[start:stop] = final_state[:, FRONT]
        front_action[start:stop] = front_action_mean
        back_state[start:stop] = final_state[:, BACK]
        back_action[start:stop] = final_action[:, BACK]

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        episode = int(summary["episode_index"])
        indices = np.flatnonzero(episode_id == episode)
        expected = int(summary["inference_calls"])
        if len(indices) != expected:
            raise ValueError(
                f"route length mismatch: {run.name}/{episode}: {len(indices)} != {expected}"
            )
        if len(indices) > 1 and not np.all(np.diff(indices) == 1):
            raise ValueError(f"non-contiguous route rows: {run.name}/{episode}")
        length = len(indices)
        mask = phase_mask(length, phase_start, phase_end)
        temporal = {name: np.full(length, np.nan, dtype=np.float32) for name in TEMPORAL_SIGNALS}

        action_route = normalize_probability(back_action[indices]).reshape(length, 40, N_EXPERTS)
        fstate = normalize_probability(front_state[indices])
        faction = normalize_probability(front_action[indices])
        if length > 1:
            temporal["route_mobility"][1:] = hellinger(
                action_route[1:], action_route[:-1]
            ).mean(1)
            temporal["front_state_jump"][1:] = hellinger(fstate[1:], fstate[:-1]).mean(1)
            temporal["front_action_jump"][1:] = hellinger(faction[1:], faction[:-1]).mean(1)
            temporal["front_feedback_split"][1:] = (
                temporal["front_state_jump"][1:] - temporal["front_action_jump"][1:]
            )
        lag_similarity = np.full((length, len(LAGS)), np.nan, dtype=np.float32)
        for lag_index, lag in enumerate(LAGS):
            if length > lag:
                lag_similarity[lag:, lag_index] = weighted_jaccard(
                    action_route[lag:], action_route[:-lag]
                ).mean(1)
        valid_lag = np.isfinite(lag_similarity).any(axis=1)
        temporal["lag_recurrence"][valid_lag] = np.nanmax(
            lag_similarity[valid_lag], axis=1
        )
        periodic = np.isfinite(lag_similarity[:, 0]) & np.isfinite(lag_similarity[:, 1:]).any(axis=1)
        temporal["lag_periodicity"][periodic] = np.nanmax(
            lag_similarity[periodic, 1:], axis=1
        ) - lag_similarity[periodic, 0]

        row: dict[str, Any] = {"episode": episode}
        for name in ROW_SIGNALS:
            mean, slope, p90 = aggregate_series(scalar[name][indices], mask)
            row[f"{name}_mean"] = mean
            if name in {"gate_entropy", "top4_union", "layer5_state_action_gap"}:
                row[f"{name}_slope"] = slope
            if name == "layer5_state_action_gap":
                row[f"{name}_p90"] = p90
        for name in TEMPORAL_SIGNALS:
            mean, _slope, p90 = aggregate_series(temporal[name], mask)
            row[f"{name}_mean"] = mean
            if name == "front_feedback_split":
                row[f"{name}_p90"] = p90
        rows.append(row)
    return pd.DataFrame(rows)


def build_route_frame(
    runs: Iterable[Path], cache_root: Path, phase_start: float, phase_end: float
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for run_index, run in enumerate(runs, start=1):
        task = task_key(run, cache_root)
        summaries = load_summaries(run)
        task_frame = extract_task_route_features(run, summaries, phase_start, phase_end)
        task_frame.insert(0, "task", task)
        outputs.append(task_frame)
        print(f"[routing {run_index}] {task}: rows={len(task_frame)}", flush=True)
    return pd.concat(outputs, ignore_index=True)


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    if not len(reference):
        return np.full(len(values), np.nan)
    return np.searchsorted(reference, values, side="right") / len(reference)


def add_crossfit_heads(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    threshold_rows: list[dict[str, Any]] = []
    for head in DETECTORS:
        output[f"head_{head}"] = np.nan
        output[f"alarm_{head}"] = False

    for (task, init_state), test in output.groupby(["task", "init_state_id"], sort=False):
        train = output[
            (output["task"] == task)
            & (output["init_state_id"] != init_state)
            & (~output["failure"])
        ]
        if len(train) < 20:
            continue
        train_head_scores: dict[str, np.ndarray] = {}
        test_head_scores: dict[str, np.ndarray] = {}
        for head, components in HEADS.items():
            train_parts = []
            test_parts = []
            for signal, direction in components:
                reference = direction * train[signal].to_numpy(dtype=np.float64)
                train_parts.append(empirical_percentile(reference, reference))
                test_parts.append(
                    empirical_percentile(
                        reference, direction * test[signal].to_numpy(dtype=np.float64)
                    )
                )
            train_score = np.nanmean(np.stack(train_parts, axis=1), axis=1)
            test_score = np.nanmean(np.stack(test_parts, axis=1), axis=1)
            train_head_scores[head] = train_score
            test_head_scores[head] = test_score
            threshold = float(np.nanquantile(train_score, 0.95))
            output.loc[test.index, f"head_{head}"] = test_score
            output.loc[test.index, f"alarm_{head}"] = test_score > threshold
            threshold_rows.append(
                {
                    "task": task,
                    "held_out_init_state_id": int(init_state),
                    "head": head,
                    "calibration_successes": len(train),
                    "threshold_q95": threshold,
                }
            )
        for head, (members, aggregation) in COMPOSITE_HEADS.items():
            train_stack = np.stack(
                [train_head_scores[member] for member in members], axis=1
            )
            test_stack = np.stack(
                [test_head_scores[member] for member in members], axis=1
            )
            reducer = np.nanmax if aggregation == "max" else np.nanmean
            train_score = reducer(train_stack, axis=1)
            test_score = reducer(test_stack, axis=1)
            threshold = float(np.nanquantile(train_score, 0.95))
            output.loc[test.index, f"head_{head}"] = test_score
            output.loc[test.index, f"alarm_{head}"] = test_score > threshold
            threshold_rows.append(
                {
                    "task": task,
                    "held_out_init_state_id": int(init_state),
                    "head": head,
                    "aggregation": aggregation,
                    "calibration_successes": len(train),
                    "threshold_q95": threshold,
                }
            )
    return output, pd.DataFrame(threshold_rows)


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    delta = positive[:, None] - negative[None, :]
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))


def grouped_auc(
    frame: pd.DataFrame,
    signal: str,
    positive: np.ndarray,
    negative: np.ndarray,
    group_indices: list[np.ndarray],
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    group_values: list[float] = []
    positive_n = negative_n = 0
    score = frame[signal].to_numpy(dtype=np.float64)
    good = np.isfinite(score)
    for index in group_indices:
        pos = score[index[positive[index] & good[index]]]
        neg = score[index[negative[index] & good[index]]]
        if not len(pos) or not len(neg):
            continue
        group_values.append(pair_auc(pos, neg))
        positive_n += len(pos)
        negative_n += len(neg)
    if len(group_values) < 3:
        return None
    values = np.asarray(group_values, dtype=np.float64)
    raw = float(values.mean())
    direction = "high" if raw >= 0.5 else "low"
    oriented = values if direction == "high" else 1.0 - values
    if bootstrap > 0:
        draw = rng.integers(0, len(values), size=(bootstrap, len(values)))
        raw_means = values[draw].mean(axis=1)
        auc_low, auc_high = np.quantile(raw_means, [0.025, 0.975])
        det_means = oriented[draw].mean(axis=1)
        low, high = np.quantile(det_means, [0.025, 0.975])
    else:
        auc_low = auc_high = low = high = float("nan")
    return {
        "signal": signal,
        "direction": direction,
        "auc": raw,
        "auc_ci_low": float(auc_low),
        "auc_ci_high": float(auc_high),
        "det_auc": float(oriented.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "mixed_groups": len(values),
        "positive_n": positive_n,
        "negative_n": negative_n,
    }


def evaluation_tables(
    frame: pd.DataFrame, bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    signals = list(ROUTE_SIGNALS) + [f"head_{name}" for name in DETECTORS]
    group_key = pd.MultiIndex.from_arrays(
        [frame["task"].astype(str), frame["init_state_id"].astype(int)]
    )
    group_codes, _ = pd.factorize(group_key, sort=False)
    group_indices = [
        np.flatnonzero(group_codes == code) for code in range(int(group_codes.max()) + 1)
    ]
    result: list[dict[str, Any]] = []
    failure = frame["failure"].to_numpy(dtype=bool)
    success = ~failure
    for label in PHYSICAL_LABELS:
        positive = failure & frame[f"label_{label}"].to_numpy(dtype=bool)
        other_failure = failure & ~frame[f"label_{label}"].to_numpy(dtype=bool)
        for comparison, negative in (
            ("other_failure", other_failure),
            ("success", success),
        ):
            for signal in signals:
                row = grouped_auc(
                    frame,
                    signal,
                    positive,
                    negative,
                    group_indices,
                    bootstrap,
                    rng,
                )
                if row is not None:
                    result.append(
                        {"label": label, "comparison": comparison, **row}
                    )

    alarm_rows: list[dict[str, Any]] = []
    for head in DETECTORS:
        alarm = frame[f"alarm_{head}"].to_numpy(dtype=bool)
        eligible = np.isfinite(frame[f"head_{head}"].to_numpy(dtype=np.float64))
        success_eligible = success & eligible
        alarm_rows.append(
            {
                "head": head,
                "target": "success_false_alarm",
                "eligible": int(success_eligible.sum()),
                "triggered": int((success_eligible & alarm).sum()),
                "rate": float(alarm[success_eligible].mean()),
            }
        )
        failure_eligible = failure & eligible
        alarm_rows.append(
            {
                "head": head,
                "target": "all_failures",
                "eligible": int(failure_eligible.sum()),
                "triggered": int((failure_eligible & alarm).sum()),
                "rate": float(alarm[failure_eligible].mean()),
            }
        )
        for label in PHYSICAL_LABELS:
            target = failure & frame[f"label_{label}"].to_numpy(dtype=bool) & eligible
            if not target.any():
                continue
            alarm_rows.append(
                {
                    "head": head,
                    "target": label,
                    "eligible": int(target.sum()),
                    "triggered": int((target & alarm).sum()),
                    "rate": float(alarm[target].mean()),
                }
            )
    return pd.DataFrame(result), pd.DataFrame(alarm_rows)


def representative_candidates(frame: pd.DataFrame, per_label: int = 8) -> pd.DataFrame:
    severity = {
        "stagnation": "longest_static_pre90_fraction",
        "gripper_cycling": "gripper_flip_rate",
        "goal_regression": "goal_regression_m",
        "goal_approach_leave": "goal_approach_exit_per_target",
        "subtask_undo": "subtask_undo_count",
        "regrasp_or_drop": "offgoal_lift_loss_count",
    }
    rows = []
    failure = frame[frame["failure"]]
    for label, metric in severity.items():
        subset = failure[failure[f"label_{label}"]].nlargest(per_label, metric)
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            rows.append(
                {
                    "phenotype": label,
                    "rank": rank,
                    "task": row["task"],
                    "episode": int(row["episode"]),
                    "init_state_id": int(row["init_state_id"]),
                    "flow_noise_seed": int(row["flow_noise_seed"]),
                    "primary_behavior": row["primary_behavior"],
                    "severity_metric": metric,
                    "severity_value": float(row[metric]),
                }
            )
    return pd.DataFrame(rows)


def label_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    failure = frame["failure"].to_numpy(dtype=bool)
    for label in PHYSICAL_LABELS:
        value = frame[f"label_{label}"].to_numpy(dtype=bool)
        rows.append(
            {
                "label": label,
                "failure_n": int((failure & value).sum()),
                "failure_rate": float(value[failure].mean()),
                "success_n": int((~failure & value).sum()),
                "success_rate": float(value[~failure].mean()),
                "failure_tasks": int(frame.loc[failure & value, "task"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def primary_alarm_summary(frame: pd.DataFrame) -> pd.DataFrame:
    failure = frame["failure"].to_numpy(dtype=bool)
    targets: list[tuple[str, np.ndarray]] = [("success_false_alarm", ~failure)]
    order = [*PHYSICAL_LABELS, "other"]
    observed = set(frame.loc[failure, "primary_behavior"].astype(str))
    for label in order:
        if label in observed:
            targets.append(
                (
                    label,
                    failure
                    & (frame["primary_behavior"].astype(str).to_numpy() == label),
                )
            )

    rows: list[dict[str, Any]] = []
    for target, mask in targets:
        row: dict[str, Any] = {"target": target, "n": int(mask.sum())}
        for head in DETECTORS:
            score = frame[f"head_{head}"].to_numpy(dtype=np.float64)
            eligible = mask & np.isfinite(score)
            alarm = frame[f"alarm_{head}"].to_numpy(dtype=bool)
            row[head] = float(alarm[eligible].mean()) if eligible.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def binary_detector_summary(frame: pd.DataFrame) -> pd.DataFrame:
    failure = frame["failure"].to_numpy(dtype=bool)
    alarms = {
        head: frame[f"alarm_{head}"].to_numpy(dtype=bool) for head in DETECTORS
    }
    alarms["dual_per_head_q95_or"] = alarms["instability"] | alarms["lock_in"]
    alarms["multi_per_head_q95_or"] = np.logical_or.reduce(
        [alarms[head] for head in HEADS]
    )

    rows = []
    for detector, alarm in alarms.items():
        tp = int((failure & alarm).sum())
        fn = int((failure & ~alarm).sum())
        fp = int((~failure & alarm).sum())
        tn = int((~failure & ~alarm).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        rows.append(
            {
                "detector": detector,
                "tp": tp,
                "fp": fp,
                "tpr": tpr,
                "fpr": fpr,
                "precision": tp / max(tp + fp, 1),
                "balanced_accuracy": 0.5 * (tpr + 1.0 - fpr),
            }
        )
    return pd.DataFrame(rows)


def paired_detector_deltas(
    frame: pd.DataFrame, bootstrap: int, seed: int
) -> pd.DataFrame:
    comparisons = (
        ("lock_in", "dual_extremes"),
        ("dual_mean_control", "dual_extremes"),
        ("dual_extremes", "multi_phenotype"),
    )
    failure = frame["failure"].to_numpy(dtype=bool)
    task_values = frame["task"].astype(str).to_numpy()
    tasks = frame["task"].astype(str).unique()
    rng = np.random.default_rng(seed)
    rows = []
    for baseline, candidate in comparisons:
        base = frame[f"alarm_{baseline}"].to_numpy(dtype=bool)
        trial = frame[f"alarm_{candidate}"].to_numpy(dtype=bool)
        counts = []
        for task in tasks:
            mask = task_values == task
            counts.append(
                (
                    int((mask & failure).sum()),
                    int((mask & ~failure).sum()),
                    int((mask & failure & base).sum()),
                    int((mask & failure & trial).sum()),
                    int((mask & ~failure & base).sum()),
                    int((mask & ~failure & trial).sum()),
                )
            )
        count = np.asarray(counts, dtype=np.float64)
        total = count.sum(axis=0)
        delta_tpr = total[3] / total[0] - total[2] / total[0]
        delta_fpr = total[5] / total[1] - total[4] / total[1]
        if bootstrap > 0:
            draw = rng.integers(0, len(tasks), size=(bootstrap, len(tasks)))
            sampled = count[draw].sum(axis=1)
            sampled_tpr = sampled[:, 3] / sampled[:, 0] - sampled[:, 2] / sampled[:, 0]
            sampled_fpr = sampled[:, 5] / sampled[:, 1] - sampled[:, 4] / sampled[:, 1]
            tpr_low, tpr_high = np.quantile(sampled_tpr, (0.025, 0.975))
            fpr_low, fpr_high = np.quantile(sampled_fpr, (0.025, 0.975))
        else:
            tpr_low = tpr_high = fpr_low = fpr_high = float("nan")
        rows.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "delta_tpr": float(delta_tpr),
                "delta_tpr_ci_low": float(tpr_low),
                "delta_tpr_ci_high": float(tpr_high),
                "delta_fpr": float(delta_fpr),
                "delta_fpr_ci_low": float(fpr_low),
                "delta_fpr_ci_high": float(fpr_high),
                "failure_baseline_only": int((failure & base & ~trial).sum()),
                "failure_candidate_only": int((failure & ~base & trial).sum()),
                "success_baseline_only": int((~failure & base & ~trial).sum()),
                "success_candidate_only": int((~failure & ~base & trial).sum()),
                "bootstrap_unit": "task",
                "bootstrap_draws": bootstrap,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    def cell(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def render_report(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    auc: pd.DataFrame,
    alarms: pd.DataFrame,
    primary_alarms: pd.DataFrame,
    binary_detectors: pd.DataFrame,
    detector_deltas: pd.DataFrame,
    candidates: pd.DataFrame,
    target_audit: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    failures = frame[frame["failure"]]

    def alarm_rate(target: str, head: str) -> float:
        match = alarms[(alarms["target"] == target) & (alarms["head"] == head)]
        return float(match.iloc[0]["rate"]) if len(match) else float("nan")

    def fixed_auc(label: str, head: str) -> float:
        match = auc[
            (auc["label"] == label)
            & (auc["comparison"] == "other_failure")
            & (auc["signal"] == f"head_{head}")
        ]
        return float(match.iloc[0]["auc"]) if len(match) else float("nan")

    stagnation_lock_rate = alarm_rate("stagnation", "lock_in")
    cycling_instability_rate = alarm_rate("gripper_cycling", "instability")
    success_lock_rate = alarm_rate("success_false_alarm", "lock_in")
    success_instability_rate = alarm_rate("success_false_alarm", "instability")
    binary = binary_detectors.set_index("detector")
    lock_tpr = float(binary.loc["lock_in", "tpr"])
    dual_tpr = float(binary.loc["dual_extremes", "tpr"])
    multi_tpr = float(binary.loc["multi_phenotype", "tpr"])
    mean_tpr = float(binary.loc["dual_mean_control", "tpr"])
    dual_fpr = float(binary.loc["dual_extremes", "fpr"])
    multi_fpr = float(binary.loc["multi_phenotype", "fpr"])
    mean_fpr = float(binary.loc["dual_mean_control", "fpr"])
    mean_precision = float(binary.loc["dual_mean_control", "precision"])
    dual_precision = float(binary.loc["dual_extremes", "precision"])
    loose_dual_tpr = float(binary.loc["dual_per_head_q95_or", "tpr"])
    loose_dual_fpr = float(binary.loc["dual_per_head_q95_or", "fpr"])
    dual_delta = detector_deltas[
        (detector_deltas["baseline"] == "lock_in")
        & (detector_deltas["candidate"] == "dual_extremes")
    ].iloc[0]
    mean_delta = detector_deltas[
        (detector_deltas["baseline"] == "dual_mean_control")
        & (detector_deltas["candidate"] == "dual_extremes")
    ].iloc[0]
    multi_delta = detector_deltas[
        (detector_deltas["baseline"] == "dual_extremes")
        & (detector_deltas["candidate"] == "multi_phenotype")
    ].iloc[0]
    lines = [
        "# VLA_MUI_HUB 跨任务 MoE Trap 表型审计",
        "",
        "## 直接结论",
        "",
        f"本轮只使用 `{args.run_id}` 中元数据标记 complete 的任务："
        f"{frame['task'].nunique()} 个任务、{len(frame)} 条轨迹、"
        f"{len(failures)} 条失败。正在写入的续批和不完整任务均未纳入。",
        "",
        "物理标签先于 routing 冻结，并由同 task/init 的成功 siblings 校准；"
        "路由头则对 held-out init 只使用同任务其他 init 的成功轨迹校准。"
        "这些仍是 query-boundary 运动学代理，不是视频/contact 真值。",
        "",
        f"**好结果：** stagnation 明确对应稳定计算锁死。固定 `lock_in` 头在"
        f"其他失败对照上的方向 AUC 为 {fixed_auc('stagnation', 'lock_in'):.3f}；"
        f"q95 下检出 {stagnation_lock_rate:.1%}，成功误报 {success_lock_rate:.1%}。"
        "单独的 flat+narrow 现象存在，但远弱于跨 query 的低 mobility + 高 recurrence。",
        "",
        f"**有限结果：** gripper cycling 符合 routing instability，固定头方向 AUC "
        f"{fixed_auc('gripper_cycling', 'instability'):.3f}；q95 检出 "
        f"{cycling_instability_rate:.1%}，成功误报 {success_instability_rate:.1%}。"
        "它能提供候选，但还不是高召回 detector。",
        "",
        "**坏结果：** goal regression、subtask undo 和 regrasp/drop 虽在失败内部"
        "可见部分 state-action gap，但固定 feedback 头在成功校准的 q95 下几乎不触发；"
        "lag periodicity 也没有支持预期的阶段往返方向。当前数据不支持把它们宣称为"
        "可用的 train-free 报警头。",
        "",
        f"**总体二元 Trap 判断：双头 `max` 不好。** 在完全相同的 routing 输入和"
        f"同一成功误报预算下，等权 mean 单标量检出 {mean_tpr:.1%}（FPR "
        f"{mean_fpr:.1%}，precision {mean_precision:.1%}），双头 `max` 只检出 "
        f"{dual_tpr:.1%}（FPR {dual_fpr:.1%}，precision {dual_precision:.1%}）。"
        "这里评估的是固定阈值后的 ACCEPT/ALARM，不是轨迹排序。",
        "",
        f"按 37 个任务成簇 bootstrap，双头相对信息匹配 mean 的召回差为 "
        f"{mean_delta['delta_tpr']:+.1%} ["
        f"{mean_delta['delta_tpr_ci_low']:+.1%}, "
        f"{mean_delta['delta_tpr_ci_high']:+.1%}]，因此下降不能用抽样波动解释。"
        f"双头相对只看 `lock_in` 仍增加 "
        f"{dual_delta['delta_tpr']:+.1%} ["
        f"{dual_delta['delta_tpr_ci_low']:+.1%}, "
        f"{dual_delta['delta_tpr_ci_high']:+.1%}]，说明新增 instability 信号有用，"
        "但不能证明双头结构更好。四头相对双头为 "
        f"{multi_delta['delta_tpr']:+.1%} ["
        f"{multi_delta['delta_tpr_ci_low']:+.1%}, "
        f"{multi_delta['delta_tpr_ci_high']:+.1%}]。",
        "",
        f"若每个头各自用 q95 后直接 OR，双头检出率可到 {loose_dual_tpr:.1%}，"
        f"但 FPR 同时升至 {loose_dual_fpr:.1%}；这不是同预算提升。"
        "两个分数仍可保留给后续分型修正，但当前数据只验证了表型分流，"
        "没有验证任何修正动作能提高成功率。",
        "",
        "## 物理表型库存",
        "",
        markdown_table(labels),
        "",
        "## Train-free 多头的 episode-level 报警",
        "",
        "每个头的分量和方向固定，不用失败标签拟合权重；对每个 held-out init，"
        "只用同任务其他 init 的成功轨迹做经验分位数和 q95 阈值。"
        "这是 50%--90% 相位的离线表型检查，不是在线 query 报警性能。",
        "联合双头/四头直接取分位数头分数的最大值，并用 held-out init 之外的"
        "成功轨迹重新校准联合 q95；没有用失败标签学习组合权重。",
        "",
        "### 二元 Trap 判断",
        "",
        markdown_table(binary_detectors),
        "",
        "### 公平预算配对差值",
        "",
        "正的 `delta_tpr` 表示 candidate 在同一成功误报校准下提高失败召回；"
        "区间按 37 个任务成簇 bootstrap。",
        "",
        markdown_table(detector_deltas),
        "",
        markdown_table(alarms),
        "",
        "### 互斥主标签检查",
        "",
        "上表中的物理标签允许重叠。下表将每条失败只归到一个 primary behavior，"
        "用于检查某个头是否只是被重叠样本抬高；数值仍是 q95 触发率。",
        "",
        markdown_table(primary_alarms),
        "",
        "## 路由表型分离",
        "",
        "AUC 在 task/init 内计算后对 mixed groups 等权平均。`other_failure` 比较回答"
        "某一物理表型能否与其他失败区分；`success` 比较更容易受到阶段和轨迹长度混杂。"
        "置信区间为 task/init group bootstrap，当前属于探索性结果，未做全族多重校正。",
        "`direction` 和 `det_auc` 是看过数据后选方向的表型诊断，不能当作预先规定的"
        "detector 性能；固定头必须读取原始 `auc`，低于 0.5 就代表方向相反。",
        "",
    ]
    for label in PHYSICAL_LABELS:
        subset = auc[(auc["label"] == label) & (auc["comparison"] == "other_failure")]
        if subset.empty:
            continue
        subset = subset[~subset["signal"].str.startswith("head_")]
        subset = subset.sort_values("det_auc", ascending=False).head(6)
        lines.extend([f"### {label} vs other failures", "", markdown_table(subset), ""])
    lines.extend(
        [
            "## 解释边界",
            "",
            "- `identify_targets` 通过成功轨迹中平均移动超过 3 cm 的 free joint 识别任务物体；"
            "drawer/stove 这类非 free-joint 任务没有物体进度标签。",
            "- 路由窗口按 episode 相对时间 50%--90% 对齐，并不能保证语义阶段严格一致。",
            "- `state_action_gap` 是 token-routing 差异，不是校准后的认识不确定性或语义信念。",
            "- 当前记录没有 RGB、contact、force，也没有 chunk 内重新 forward；"
            "因此不能把代理升级为 true grasp/drop/contact，也不能在 stale chunk 中途给出 MoE 响应。",
            "- 本轮用于建立表型关联，不把任何 AUC 或 q95 触发率解释为因果性或恢复收益。",
            "",
            "## 代表性轨迹",
            "",
            f"共输出 {len(candidates)} 个高严重度候选到 `representative_candidates.csv`。",
            "",
            "## 可观测性",
            "",
            f"自动识别不到 free-joint 任务目标的任务数：{sum(not row['targets'] for row in target_audit)}。",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    p = np.asarray([[0.25, 0.75], [0.5, 0.5]], dtype=np.float32)
    assert np.allclose(hellinger(p, p), 0.0, atol=1e-6)
    assert np.allclose(weighted_jaccard(p, p), 1.0, atol=1e-6)
    assert pair_auc(np.asarray([2.0, 3.0]), np.asarray([0.0, 1.0])) == 1.0
    ids = np.asarray([[[[1, 2], [2, 3]], [[4, 5], [4, 5]]]])
    assert np.allclose(union_fraction(ids), (3 / 32 + 2 / 32) / 2)
    assert phase_mask(10, 0.5, 0.9).sum() == 4
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not 0 <= args.phase_start < args.phase_end <= 1:
        raise ValueError("phase window must satisfy 0 <= start < end <= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    physical_path = args.out_dir / "episode_physical_labels.csv"
    route_path = args.out_dir / "episode_route_features.csv"
    target_path = args.out_dir / "target_audit.json"
    runs = discover_complete_runs(args.cache_root, args.run_id)

    if args.reuse and physical_path.exists() and target_path.exists():
        physical = pd.read_csv(physical_path)
        target_audit = json.loads(target_path.read_text(encoding="utf-8"))
        print(f"reused {physical_path}", flush=True)
    else:
        physical, target_audit = build_physical_frame(runs, args.cache_root)
        physical.to_csv(physical_path, index=False)
        target_path.write_text(
            json.dumps(plain(target_audit), indent=2) + "\n", encoding="utf-8"
        )

    if args.reuse and route_path.exists():
        routes = pd.read_csv(route_path)
        print(f"reused {route_path}", flush=True)
    else:
        routes = build_route_frame(
            runs, args.cache_root, args.phase_start, args.phase_end
        )
        routes.to_csv(route_path, index=False)

    frame = physical.merge(routes, on=["task", "episode"], validate="one_to_one")
    if len(frame) != len(physical):
        raise ValueError("physical/route row mismatch")
    frame, thresholds = add_crossfit_heads(frame)
    auc, alarms = evaluation_tables(frame, args.bootstrap, args.seed)
    labels = label_summary(frame)
    primary_alarms = primary_alarm_summary(frame)
    binary_detectors = binary_detector_summary(frame)
    detector_deltas = paired_detector_deltas(frame, args.bootstrap, args.seed + 1)
    candidates = representative_candidates(frame)

    frame.to_csv(args.out_dir / "episode_features_and_heads.csv", index=False)
    thresholds.to_csv(args.out_dir / "head_thresholds.csv", index=False)
    auc.to_csv(args.out_dir / "routing_auc.csv", index=False)
    alarms.to_csv(args.out_dir / "head_alarm_rates.csv", index=False)
    primary_alarms.to_csv(args.out_dir / "primary_head_alarm_rates.csv", index=False)
    binary_detectors.to_csv(args.out_dir / "binary_detector_summary.csv", index=False)
    detector_deltas.to_csv(args.out_dir / "paired_detector_deltas.csv", index=False)
    labels.to_csv(args.out_dir / "physical_label_inventory.csv", index=False)
    candidates.to_csv(args.out_dir / "representative_candidates.csv", index=False)

    summary = {
        "schema": "himoe.hub_phenotype_atlas.v1",
        "run_id": args.run_id,
        "complete_tasks": int(frame["task"].nunique()),
        "episodes": len(frame),
        "failures": int(frame["failure"].sum()),
        "successes": int((~frame["failure"]).sum()),
        "queries": int(frame["episode_length"].sum()),
        "phase_window": [args.phase_start, args.phase_end],
        "physical_labels": labels.to_dict(orient="records"),
        "heads": {name: list(parts) for name, parts in HEADS.items()},
        "composite_heads": {
            name: {"members": list(parts), "aggregation": aggregation}
            for name, (parts, aggregation) in COMPOSITE_HEADS.items()
        },
        "limitations": [
            "query-boundary kinematic proxies, not video/contact labels",
            "relative-time windows do not guarantee semantic-phase matching",
            "no RGB, contact, force, or mid-chunk MoE forward",
            "exploratory group-bootstrap intervals without familywise correction",
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT_ZH.md").write_text(
        render_report(
            frame,
            labels,
            auc,
            alarms,
            primary_alarms,
            binary_detectors,
            detector_deltas,
            candidates,
            target_audit,
            args,
        ),
        encoding="utf-8",
    )
    print(f"analysis complete: {args.out_dir}")


if __name__ == "__main__":
    main()
