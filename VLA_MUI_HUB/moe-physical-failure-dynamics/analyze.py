#!/usr/bin/env python3
"""Analyze HB-MoE dynamics for the physically annotated LIBERO failures.

The rollout is the statistical unit.  Raw expert identities are never pooled
across checkpoints.  Outcome comparisons use successes from the same source
run and initial state; the primary observational cohort excludes duplicated
and intervened pin runs while retaining them in the descriptive inventory.
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
FAILURES = HUB / "physical-failure-labels/results/failures.jsonl"
DEFAULT_OUTPUT = HERE / "results"

HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
LAYER_GROUPS = ("front_L2_5", "back_L12_15")
TOKENS = ("state",) + tuple(f"T{index}" for index in range(1, 11))
DENOISE = tuple(range(10))
PHASE_POINTS = np.linspace(0.0, 1.0, 11, dtype=np.float64)
CROSS_PHASE_POINTS = np.linspace(0.1, 1.0, 10, dtype=np.float64)
PHASE_BANDS = ("early_0_33", "middle_33_67", "late_67_100")
N_EXPERTS = 32
TOP_K = 4
MATCH_RTOL = 2e-6
MATCH_ATOL = 1e-8

TRAJECTORY_METRICS = (
    "action_entropy",
    "action_top1_mass",
    "action_top4_mass",
    "action_token_dispersion",
    "state_action_gap",
    "denoise_soft_change",
    "denoise_hard_churn",
    "query_soft_change",
    "query_hard_churn",
    "state_query_soft_change",
)
WITHIN_METRICS = ("entropy", "top1_mass", "top4_mass")
CROSS_METRICS = ("lag1_soft_change", "lag1_hard_churn", "lag2_soft_change")
BOUNDARY_METRICS = ("T10_to_next_T1_soft_change", "T10_to_next_T1_hard_churn")
AS_METRICS = ("entropy", "top1_mass", "query_soft_change", "query_hard_churn")

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


@dataclass(frozen=True)
class EpisodeSpec:
    feature_row: int
    source_run: str
    episode_index: int
    offset: int
    inference_calls: int
    init_state_id: int
    flow_noise_seed: int
    suite: str
    task_name: str
    run_id: str
    checkpoint_sha256: str
    success: bool
    primary_failure_reason: str
    completion_state: str
    confidence: str
    run_regime: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", type=Path, default=FAILURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="reuse the validated episode_features.npz and only redo matching/aggregation",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def normalize_probability(value: np.ndarray) -> np.ndarray:
    probability = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = probability.sum(axis=-1, keepdims=True)
    if np.any(total <= 0.0) or np.any(~np.isfinite(total)):
        raise ValueError("router probability contains invalid mass")
    return probability / total


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32)
    return -np.sum(
        probability * np.log(np.maximum(probability, 1e-12)), axis=-1
    ) / math.log(probability.shape[-1])


def top_mass(probability: np.ndarray, count: int) -> np.ndarray:
    return np.partition(probability, -count, axis=-1)[..., -count:].sum(axis=-1)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.maximum(np.asarray(left, dtype=np.float32), 0.0)
    right = np.maximum(np.asarray(right, dtype=np.float32), 0.0)
    return np.sqrt(
        0.5
        * np.sum(
            np.square(np.sqrt(left) - np.sqrt(right)),
            axis=-1,
        )
    )


def top4_churn(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.shape[-1] != TOP_K:
        raise ValueError("top-4 arrays must have equal [..., 4] shapes")
    intersection = (left[..., :, None] == right[..., None, :]).any(axis=-1).sum(axis=-1)
    jaccard = intersection / np.maximum(2 * TOP_K - intersection, 1)
    return 1.0 - jaccard


def resample_at(
    values: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if values.ndim < 1 or len(values) != len(source) or not len(values):
        raise ValueError("resampling requires a non-empty aligned first axis")
    if len(values) == 1:
        return np.repeat(values, len(target), axis=0)
    if np.any(np.diff(source) <= 0):
        raise ValueError("source coordinates must increase strictly")
    high = np.searchsorted(source, target, side="left")
    high = np.clip(high, 1, len(source) - 1)
    low = high - 1
    weight = (target - source[low]) / (source[high] - source[low])
    weight = np.clip(weight, 0.0, 1.0).astype(np.float32)
    shape = (len(weight),) + (1,) * (values.ndim - 1)
    return values[low] * (1.0 - weight.reshape(shape)) + values[high] * weight.reshape(shape)


def regular_resample(values: np.ndarray, target: np.ndarray = PHASE_POINTS) -> np.ndarray:
    return resample_at(
        values,
        np.linspace(0.0, 1.0, len(values), dtype=np.float64),
        target,
    )


def phase_band_indices(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, 1.0, length, dtype=np.float64)
    bands = (
        np.flatnonzero(phase <= 1.0 / 3.0),
        np.flatnonzero((phase > 1.0 / 3.0) & (phase <= 2.0 / 3.0)),
        np.flatnonzero(phase > 2.0 / 3.0),
    )
    output = []
    centers = (1.0 / 6.0, 0.5, 5.0 / 6.0)
    for indices, center in zip(bands, centers):
        if not len(indices):
            indices = np.asarray([int(np.argmin(np.abs(phase - center)))])
        output.append(indices)
    return tuple(output)  # type: ignore[return-value]


def grouped_layer_mean(values: np.ndarray) -> np.ndarray:
    """Convert [query, layer, ...] to [query, front/back, ...]."""
    if values.shape[1] != len(HB_LAYERS):
        raise ValueError(f"expected eight HB layers, found {values.shape}")
    return np.stack((values[:, :4].mean(axis=1), values[:, 4:].mean(axis=1)), axis=1)


def prepend_first_transition(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros((length, *values.shape[1:]), dtype=np.float32)
    if len(values) != length - 1:
        raise ValueError("transition sequence has the wrong length")
    return np.concatenate((values[:1], values), axis=0).astype(np.float32, copy=False)


def expert_load(ids: np.ndarray) -> np.ndarray:
    """Per-episode hard load: [phase band, front/back, token, expert]."""
    result = np.zeros((3, 2, len(TOKENS), N_EXPERTS), dtype=np.float32)
    for band, queries in enumerate(phase_band_indices(len(ids))):
        for group, layers in enumerate((slice(0, 4), slice(4, 8))):
            current = ids[queries, layers]
            for token in range(len(TOKENS)):
                counts = np.bincount(
                    current[:, :, :, token, :].reshape(-1).astype(np.int64),
                    minlength=N_EXPERTS,
                ).astype(np.float32)
                result[band, group, token] = counts / max(float(counts.sum()), 1.0)
    return result


def extract_episode_features(
    raw_probability: np.ndarray,
    ids: np.ndarray,
    raw_as_probability: np.ndarray,
    as_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    probability = normalize_probability(raw_probability)
    as_probability = normalize_probability(raw_as_probability)
    expected = (len(HB_LAYERS), len(DENOISE), len(TOKENS), N_EXPERTS)
    if probability.ndim != 5 or probability.shape[1:] != expected:
        raise ValueError(f"unexpected HB probability shape {probability.shape}")
    if ids.shape != (*probability.shape[:-1], TOP_K):
        raise ValueError(f"unexpected HB expert-id shape {ids.shape}")
    if as_probability.shape != (len(probability), 4, 3):
        raise ValueError(f"unexpected AS probability shape {as_probability.shape}")
    if as_ids.shape != (len(probability), 4):
        raise ValueError(f"unexpected AS expert-id shape {as_ids.shape}")

    entropy = normalized_entropy(probability)
    top1 = probability.max(axis=-1)
    top4 = top_mass(probability, TOP_K)
    token_stats = grouped_layer_mean(
        np.stack((entropy, top1, top4), axis=-1)
    )
    token_stats_phase = regular_resample(token_stats)
    within = np.stack(
        (
            token_stats_phase[:4].mean(axis=0),
            token_stats_phase[4:7].mean(axis=0),
            token_stats_phase[7:].mean(axis=0),
        ),
        axis=0,
    ).astype(np.float32)

    action = probability[..., 1:, :]
    action_mean = action.mean(axis=3)
    token_dispersion = hellinger(action, action_mean[..., None, :])
    state_action_gap = hellinger(probability[..., 0, :], action_mean)
    denoise_soft = hellinger(action[:, :, 1:], action[:, :, :-1])
    denoise_hard = top4_churn(ids[:, :, 1:, 1:], ids[:, :, :-1, 1:])

    query_soft = prepend_first_transition(
        hellinger(action[1:], action[:-1]), len(probability)
    )
    query_hard = prepend_first_transition(
        top4_churn(ids[1:, :, :, 1:], ids[:-1, :, :, 1:]), len(probability)
    )
    state_query_soft = prepend_first_transition(
        hellinger(probability[1:, :, :, 0], probability[:-1, :, :, 0]),
        len(probability),
    )

    trajectory_columns = []
    for group_layers in (slice(0, 4), slice(4, 8)):
        trajectory_columns.append(
            np.stack(
                (
                    entropy[:, group_layers, :, 1:].mean(axis=(1, 2, 3)),
                    top1[:, group_layers, :, 1:].mean(axis=(1, 2, 3)),
                    top4[:, group_layers, :, 1:].mean(axis=(1, 2, 3)),
                    token_dispersion[:, group_layers].mean(axis=(1, 2, 3)),
                    state_action_gap[:, group_layers].mean(axis=(1, 2)),
                    denoise_soft[:, group_layers].mean(axis=(1, 2, 3)),
                    denoise_hard[:, group_layers].mean(axis=(1, 2, 3)),
                    query_soft[:, group_layers].mean(axis=(1, 2, 3)),
                    query_hard[:, group_layers].mean(axis=(1, 2, 3)),
                    state_query_soft[:, group_layers].mean(axis=(1, 2)),
                ),
                axis=-1,
            )
        )
    trajectory = regular_resample(np.stack(trajectory_columns, axis=1))

    if len(probability) > 1:
        cross_soft = hellinger(probability[1:, 4:], probability[:-1, 4:]).mean(
            axis=(1, 2)
        )
        cross_hard = top4_churn(ids[1:, 4:], ids[:-1, 4:]).mean(axis=(1, 2))
        cross_source = np.arange(1, len(probability), dtype=np.float64) / (
            len(probability) - 1
        )
        cross_soft = resample_at(cross_soft, cross_source, CROSS_PHASE_POINTS)
        cross_hard = resample_at(cross_hard, cross_source, CROSS_PHASE_POINTS)
        boundary_soft = hellinger(
            probability[1:, 4:, :, 1], probability[:-1, 4:, :, 10]
        ).mean(axis=(1, 2))
        boundary_hard = top4_churn(
            ids[1:, 4:, :, 1], ids[:-1, 4:, :, 10]
        ).mean(axis=(1, 2))
        boundary = resample_at(
            np.stack((boundary_soft, boundary_hard), axis=-1),
            cross_source,
            CROSS_PHASE_POINTS,
        )
    else:
        cross_soft = np.zeros((len(CROSS_PHASE_POINTS), len(TOKENS)), np.float32)
        cross_hard = np.zeros_like(cross_soft)
        boundary = np.zeros((len(CROSS_PHASE_POINTS), len(BOUNDARY_METRICS)), np.float32)

    if len(probability) > 2:
        lag2 = hellinger(probability[2:, 4:], probability[:-2, 4:]).mean(axis=(1, 2))
        lag2_source = np.arange(2, len(probability), dtype=np.float64) / (
            len(probability) - 1
        )
        lag2 = resample_at(lag2, lag2_source, CROSS_PHASE_POINTS)
    else:
        lag2 = np.zeros_like(cross_soft)
    cross = np.stack((cross_soft, cross_hard, lag2), axis=-1).astype(np.float32)

    as_entropy = normalized_entropy(as_probability).mean(axis=1)
    as_top1 = as_probability.max(axis=-1).mean(axis=1)
    as_soft = prepend_first_transition(
        hellinger(as_probability[1:], as_probability[:-1]).mean(axis=1),
        len(probability),
    )
    as_hard = prepend_first_transition(
        (as_ids[1:] != as_ids[:-1]).mean(axis=1), len(probability)
    )
    as_phase = regular_resample(
        np.stack((as_entropy, as_top1, as_soft, as_hard), axis=-1)
    )

    state_denoise_span = float(
        np.max(np.abs(probability[:, :, :, 0] - probability[:, :, :1, 0]))
    )
    diagnostics = {
        "state_denoise_span": state_denoise_span,
        "as_probability_span": float(
            np.max(np.abs(as_probability - as_probability[:1]))
        ),
        "as_hard_change_rate": float(
            (as_ids[1:] != as_ids[:-1]).mean() if len(as_ids) > 1 else 0.0
        ),
    }
    return (
        {
            "trajectory": trajectory.astype(np.float32),
            "within": within,
            "cross": cross,
            "boundary": boundary.astype(np.float32),
            "as_phase": as_phase.astype(np.float32),
            "expert_load": expert_load(ids),
        },
        diagnostics,
    )


def run_regime(run_id: str) -> str:
    if run_id == "pin-base":
        return "duplicate_pin_base"
    if run_id in ("pin-on", "pin-off"):
        return "front_layer_intervention"
    return "observational"


def load_failure_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(row["source"]["run"], int(row["episode_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("physical failure records are not unique")
    if any(bool(row["recorded_success"]) for row in rows):
        raise ValueError("failure file contains a successful rollout")
    return rows


def build_cohort(path: Path) -> tuple[list[EpisodeSpec], dict[str, Any]]:
    failures = load_failure_rows(path)
    failure_lookup = {
        (row["source"]["run"], int(row["episode_index"])): row for row in failures
    }
    failures_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failures:
        failures_by_run[row["source"]["run"]].append(row)

    provisional: list[dict[str, Any]] = []
    integrity_runs = []
    for source_run in sorted(failures_by_run):
        run = HUB / source_run
        summaries = sorted(
            read_json(run / "client/summaries.json"),
            key=lambda row: int(row["episode_index"]),
        )
        metadata = read_json(run / "meta.json")
        server_metadata = read_json(run / "client/server_metadata.json")
        lengths = np.asarray([int(row["inference_calls"]) for row in summaries])
        offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        rows = int(store["episode_id"].shape[0])
        if rows != int(lengths.sum()):
            raise ValueError(f"{source_run}: summary and route row counts differ")
        episode_axis = np.asarray(store["episode_id"][:], dtype=np.int64)
        expected_axis = np.repeat(
            np.asarray([int(row["episode_index"]) for row in summaries]), lengths
        )
        if not np.array_equal(episode_axis, expected_axis):
            raise ValueError(f"{source_run}: episode route alignment failed")
        failure_states = {
            int(row["init_state_id"]) for row in failures_by_run[source_run]
        }
        summary_by_episode = {int(row["episode_index"]): row for row in summaries}
        for failure in failures_by_run[source_run]:
            summary = summary_by_episode[int(failure["episode_index"])]
            if bool(summary["success"]):
                raise ValueError(f"{source_run}: physical failure aligns to success")
        for summary, offset, length in zip(summaries, offsets, lengths):
            key = (source_run, int(summary["episode_index"]))
            physical = failure_lookup.get(key)
            selected = physical is not None or (
                bool(summary["success"])
                and int(summary["init_state_id"]) in failure_states
            )
            if not selected:
                continue
            current_run_id = str(metadata["run_id"])
            provisional.append(
                {
                    "source_run": source_run,
                    "episode_index": int(summary["episode_index"]),
                    "offset": int(offset),
                    "inference_calls": int(length),
                    "init_state_id": int(summary["init_state_id"]),
                    "flow_noise_seed": int(summary["flow_noise_seed"]),
                    "suite": str(metadata["suite"]),
                    "task_name": str(metadata["task_name"]),
                    "run_id": current_run_id,
                    "checkpoint_sha256": str(server_metadata["checkpoint_sha256"]),
                    "success": bool(summary["success"]),
                    "primary_failure_reason": (
                        "success" if physical is None else str(physical["primary_failure_reason"])
                    ),
                    "completion_state": (
                        "success"
                        if physical is None
                        else str(physical["completion_state_at_last_checkpoint"])
                    ),
                    "confidence": (
                        "success" if physical is None else str(physical["failure_reason_confidence"])
                    ),
                    "run_regime": run_regime(current_run_id),
                }
            )
        integrity_runs.append(
            {
                "source_run": source_run,
                "episodes": len(summaries),
                "route_rows": rows,
                "failures": len(failures_by_run[source_run]),
            }
        )

    specs = [EpisodeSpec(feature_row=index, **row) for index, row in enumerate(provisional)]
    failure_specs = [spec for spec in specs if not spec.success]
    if len(failure_specs) != len(failures):
        raise ValueError("not every physical failure entered the feature cohort")
    reference_counts = Counter(
        (spec.source_run, spec.init_state_id) for spec in specs if spec.success
    )
    matched = sum(
        reference_counts[(spec.source_run, spec.init_state_id)] > 0
        for spec in failure_specs
    )
    primary_matched = sum(
        spec.run_regime == "observational"
        and reference_counts[(spec.source_run, spec.init_state_id)] > 0
        for spec in failure_specs
    )
    return specs, {
        "failure_records": len(failures),
        "selected_success_controls": sum(spec.success for spec in specs),
        "selected_episodes": len(specs),
        "source_runs": len(failures_by_run),
        "aligned_source_runs": len(integrity_runs),
        "matched_failures": int(matched),
        "unmatched_failures": int(len(failures) - matched),
        "primary_observational_failures": sum(
            not spec.success and spec.run_regime == "observational" for spec in specs
        ),
        "primary_matched_failures": int(primary_matched),
        "run_regime_failures": dict(
            Counter(spec.run_regime for spec in failure_specs)
        ),
        "runs": integrity_runs,
    }


def process_run(payload: tuple[str, list[EpisodeSpec]]) -> dict[str, Any]:
    source_run, specs = payload
    run = HUB / source_run
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    rows = []
    diagnostics = []
    for spec in sorted(specs, key=lambda item: item.offset):
        start = spec.offset
        stop = start + spec.inference_calls
        episode_axis = np.asarray(store["episode_id"][start:stop], dtype=np.int64)
        if not np.all(episode_axis == spec.episode_index):
            raise ValueError(f"{source_run}: episode {spec.episode_index} slice mismatch")
        features, audit = extract_episode_features(
            np.asarray(store["hb_router_probs"][start:stop], dtype=np.float32),
            np.asarray(store["hb_expert_ids"][start:stop], dtype=np.uint8),
            np.asarray(store["as_probs"][start:stop], dtype=np.float32),
            np.asarray(store["as_expert_ids"][start:stop], dtype=np.uint8),
        )
        rows.append(spec.feature_row)
        for name, value in features.items():
            arrays[name].append(value)
        diagnostics.append(audit)
    return {
        "source_run": source_run,
        "rows": np.asarray(rows, dtype=np.int64),
        "arrays": {name: np.stack(values) for name, values in arrays.items()},
        "state_denoise_spans": np.asarray(
            [item["state_denoise_span"] for item in diagnostics], dtype=np.float32
        ),
        "as_probability_span": max(item["as_probability_span"] for item in diagnostics),
        "as_hard_change_rate": float(
            np.mean([item["as_hard_change_rate"] for item in diagnostics])
        ),
    }


def extract_all(
    specs: list[EpisodeSpec], workers: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    shapes = {
        "trajectory": (len(PHASE_POINTS), 2, len(TRAJECTORY_METRICS)),
        "within": (3, 2, 10, 11, len(WITHIN_METRICS)),
        "cross": (len(CROSS_PHASE_POINTS), 11, len(CROSS_METRICS)),
        "boundary": (len(CROSS_PHASE_POINTS), len(BOUNDARY_METRICS)),
        "as_phase": (len(PHASE_POINTS), len(AS_METRICS)),
        "expert_load": (3, 2, 11, N_EXPERTS),
    }
    output = {
        name: np.full((len(specs), *shape), np.nan, dtype=np.float32)
        for name, shape in shapes.items()
    }
    by_run: dict[str, list[EpisodeSpec]] = defaultdict(list)
    for spec in specs:
        by_run[spec.source_run].append(spec)
    payloads = sorted(by_run.items())
    audits = []
    if workers <= 1:
        iterator: Iterable[dict[str, Any]] = map(process_run, payloads)
        for position, result in enumerate(iterator, start=1):
            rows = result["rows"]
            for name in output:
                output[name][rows] = result["arrays"][name]
            audits.append(result)
            print(f"  extracted run {position}/{len(payloads)}: {result['source_run']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_run, payload): payload[0] for payload in payloads}
            for position, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                rows = result["rows"]
                for name in output:
                    output[name][rows] = result["arrays"][name]
                audits.append(result)
                print(
                    f"  extracted run {position}/{len(payloads)}: {result['source_run']}",
                    flush=True,
                )
    for name, values in output.items():
        if np.any(~np.isfinite(values)):
            raise RuntimeError(f"feature extraction left non-finite cells in {name}")
    state_spans = np.concatenate([item["state_denoise_spans"] for item in audits])
    maximum_audit = max(audits, key=lambda item: float(item["state_denoise_spans"].max()))
    return output, {
        "state_token_denoise_median_max_abs_difference_per_episode": float(
            np.median(state_spans)
        ),
        "state_token_denoise_p95_max_abs_difference_per_episode": float(
            np.quantile(state_spans, 0.95)
        ),
        "state_token_denoise_max_abs_difference": float(state_spans.max()),
        "state_token_denoise_max_source_run": maximum_audit["source_run"],
        "as_probability_max_within_episode_span": max(
            item["as_probability_span"] for item in audits
        ),
        "as_hard_adjacent_change_rate_mean_by_run": float(
            np.mean([item["as_hard_change_rate"] for item in audits])
        ),
    }


def reference_map(specs: list[EpisodeSpec]) -> dict[tuple[str, int], np.ndarray]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for spec in specs:
        if spec.success:
            grouped[(spec.source_run, spec.init_state_id)].append(spec.feature_row)
    return {key: np.asarray(value, dtype=np.int64) for key, value in grouped.items()}


def matched_comparison(
    values: np.ndarray,
    failure_specs: list[EpisodeSpec],
    references: dict[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(failure_specs), *values.shape[1:])
    percentiles = np.full(shape, np.nan, dtype=np.float32)
    reference_means = np.full(shape, np.nan, dtype=np.float32)
    counts = np.zeros(len(failure_specs), dtype=np.int16)
    for row, spec in enumerate(failure_specs):
        indices = references.get((spec.source_run, spec.init_state_id))
        if indices is None or not len(indices):
            continue
        current = values[spec.feature_row]
        controls = values[indices]
        tied = np.isclose(
            controls,
            current,
            rtol=MATCH_RTOL,
            atol=MATCH_ATOL,
            equal_nan=False,
        )
        percentiles[row] = (
            ((controls < current) & ~tied).mean(axis=0)
            + 0.5 * tied.mean(axis=0)
        )
        reference_means[row] = controls.mean(axis=0)
        counts[row] = len(indices)
    return percentiles, reference_means, counts


def bootstrap_task_macro(
    values: np.ndarray, tasks: np.ndarray, draws: int, seed: int
) -> tuple[float, float, float, int]:
    finite = np.isfinite(values)
    values = np.asarray(values, dtype=np.float64)[finite]
    tasks = np.asarray(tasks)[finite]
    unique = np.unique(tasks)
    task_means = np.asarray([values[tasks == task].mean() for task in unique])
    center = float(task_means.mean())
    if len(unique) < 2 or draws <= 0:
        return center, float("nan"), float("nan"), len(unique)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(task_means), size=(draws, len(task_means)))
    boot = task_means[sampled].mean(axis=1)
    return (
        center,
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
        len(unique),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subset_failure_indices(
    failure_specs: list[EpisodeSpec], reason: str | None
) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, spec in enumerate(failure_specs)
            if spec.run_regime == "observational"
            and (reason is None or spec.primary_failure_reason == reason)
        ],
        dtype=np.int64,
    )


def aggregate_outputs(
    output: Path,
    specs: list[EpisodeSpec],
    arrays: dict[str, np.ndarray],
    failure_specs: list[EpisodeSpec],
    comparisons: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    reasons = [reason for reason, _ in Counter(
        spec.primary_failure_reason
        for spec in failure_specs
        if spec.run_regime == "observational"
    ).most_common()]
    groups: list[tuple[str, np.ndarray]] = [("all_failures", subset_failure_indices(failure_specs, None))]
    groups.extend((reason, subset_failure_indices(failure_specs, reason)) for reason in reasons)

    curve_rows: list[dict[str, Any]] = []
    trajectory_pct, trajectory_ref, _ = comparisons["trajectory"]
    for label, indices in groups:
        valid = indices[np.isfinite(trajectory_pct[indices, 0, 0, 0])]
        for phase_axis, phase in enumerate(PHASE_POINTS):
            for layer_axis, layer_group in enumerate(LAYER_GROUPS):
                for metric_axis, metric in enumerate(TRAJECTORY_METRICS):
                    pct = trajectory_pct[valid, phase_axis, layer_axis, metric_axis]
                    raw_rows = np.asarray([failure_specs[i].feature_row for i in valid])
                    curve_rows.append(
                        {
                            "failure_group": label,
                            "phase": round(float(phase), 3),
                            "layer_group": layer_group,
                            "metric": metric,
                            "failures": len(valid),
                            "matched_percentile_mean": float(np.mean(pct)),
                            "failure_raw_mean": float(
                                arrays["trajectory"][raw_rows, phase_axis, layer_axis, metric_axis].mean()
                            ),
                            "same_run_init_success_mean": float(
                                trajectory_ref[valid, phase_axis, layer_axis, metric_axis].mean()
                            ),
                        }
                    )
    write_csv(output / "trajectory_phase_curves.csv", curve_rows)

    within_rows: list[dict[str, Any]] = []
    within_pct, within_ref, _ = comparisons["within"]
    for label, indices in groups:
        valid = indices[np.isfinite(within_pct[indices, 0, 0, 0, 0, 0])]
        raw_rows = np.asarray([failure_specs[i].feature_row for i in valid])
        for band_axis, band in enumerate(PHASE_BANDS):
            for layer_axis, layer_group in enumerate(LAYER_GROUPS):
                for denoise in DENOISE:
                    for token_axis, token in enumerate(TOKENS):
                        for metric_axis, metric in enumerate(WITHIN_METRICS):
                            within_rows.append(
                                {
                                    "failure_group": label,
                                    "phase_band": band,
                                    "layer_group": layer_group,
                                    "denoise_step": denoise,
                                    "token": token,
                                    "metric": metric,
                                    "failures": len(valid),
                                    "matched_percentile_mean": float(
                                        within_pct[
                                            valid,
                                            band_axis,
                                            layer_axis,
                                            denoise,
                                            token_axis,
                                            metric_axis,
                                        ].mean()
                                    ),
                                    "failure_raw_mean": float(
                                        arrays["within"][
                                            raw_rows,
                                            band_axis,
                                            layer_axis,
                                            denoise,
                                            token_axis,
                                            metric_axis,
                                        ].mean()
                                    ),
                                    "same_run_init_success_mean": float(
                                        within_ref[
                                            valid,
                                            band_axis,
                                            layer_axis,
                                            denoise,
                                            token_axis,
                                            metric_axis,
                                        ].mean()
                                    ),
                                }
                            )
    write_csv(output / "within_chunk_profiles.csv", within_rows)

    cross_rows: list[dict[str, Any]] = []
    cross_pct, cross_ref, _ = comparisons["cross"]
    for label, indices in groups:
        valid = indices[np.isfinite(cross_pct[indices, 0, 0, 0])]
        raw_rows = np.asarray([failure_specs[i].feature_row for i in valid])
        for phase_axis, phase in enumerate(CROSS_PHASE_POINTS):
            for token_axis, token in enumerate(TOKENS):
                for metric_axis, metric in enumerate(CROSS_METRICS):
                    cross_rows.append(
                        {
                            "failure_group": label,
                            "transition_phase": round(float(phase), 3),
                            "layer_group": "back_L12_15",
                            "token": token,
                            "metric": metric,
                            "failures": len(valid),
                            "matched_percentile_mean": float(
                                cross_pct[valid, phase_axis, token_axis, metric_axis].mean()
                            ),
                            "failure_raw_mean": float(
                                arrays["cross"][raw_rows, phase_axis, token_axis, metric_axis].mean()
                            ),
                            "same_run_init_success_mean": float(
                                cross_ref[valid, phase_axis, token_axis, metric_axis].mean()
                            ),
                        }
                    )
    write_csv(output / "back_token_cross_chunk.csv", cross_rows)

    expert_rows: list[dict[str, Any]] = []
    load_pct, load_ref, _ = comparisons["expert_load"]
    del load_pct
    for label, indices in groups[1:]:
        valid = indices[np.isfinite(load_ref[indices, 0, 0, 0, 0])]
        checkpoints = sorted({failure_specs[index].checkpoint_sha256 for index in valid})
        for checkpoint in checkpoints:
            local = np.asarray(
                [index for index in valid if failure_specs[index].checkpoint_sha256 == checkpoint],
                dtype=np.int64,
            )
            if not len(local):
                continue
            raw_rows = np.asarray([failure_specs[index].feature_row for index in local])
            failure_mean = arrays["expert_load"][raw_rows].mean(axis=0)
            reference_mean = load_ref[local].mean(axis=0)
            suite = failure_specs[int(local[0])].suite
            for band_axis, band in enumerate(PHASE_BANDS):
                for layer_axis, layer_group in enumerate(LAYER_GROUPS):
                    for token_axis, token in enumerate(TOKENS):
                        delta = failure_mean[band_axis, layer_axis, token_axis] - reference_mean[
                            band_axis, layer_axis, token_axis
                        ]
                        rank = np.empty(N_EXPERTS, dtype=np.int64)
                        rank[np.argsort(-np.abs(delta))] = np.arange(1, N_EXPERTS + 1)
                        for expert in range(N_EXPERTS):
                            expert_rows.append(
                                {
                                    "failure_group": label,
                                    "suite": suite,
                                    "checkpoint_sha256": checkpoint,
                                    "phase_band": band,
                                    "layer_group": layer_group,
                                    "token": token,
                                    "expert": expert,
                                    "failures": len(local),
                                    "failure_selection_load": float(
                                        failure_mean[band_axis, layer_axis, token_axis, expert]
                                    ),
                                    "same_run_init_success_load": float(
                                        reference_mean[band_axis, layer_axis, token_axis, expert]
                                    ),
                                    "load_delta": float(delta[expert]),
                                    "absolute_delta_rank": int(rank[expert]),
                                }
                            )
    write_csv(output / "checkpoint_expert_load_shifts.csv", expert_rows)

    metric_lookup = {name: axis for axis, name in enumerate(TRAJECTORY_METRICS)}
    within_lookup = {name: axis for axis, name in enumerate(WITHIN_METRICS)}
    cross_lookup = {name: axis for axis, name in enumerate(CROSS_METRICS)}
    boundary_pct = comparisons["boundary"][0]
    failure_feature_rows = np.asarray(
        [spec.feature_row for spec in failure_specs], dtype=np.int64
    )
    trajectory_raw = arrays["trajectory"][failure_feature_rows]
    within_raw = arrays["within"][failure_feature_rows]
    cross_raw = arrays["cross"][failure_feature_rows]
    boundary_raw = arrays["boundary"][failure_feature_rows]
    within_ref = comparisons["within"][1]
    cross_ref = comparisons["cross"][1]
    boundary_ref = comparisons["boundary"][1]

    def triple(percentile: np.ndarray, raw: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return percentile, raw, reference

    headline_extractors = {
        "late_front_action_entropy": triple(
            trajectory_pct[:, 7:, 0, metric_lookup["action_entropy"]].mean(axis=1),
            trajectory_raw[:, 7:, 0, metric_lookup["action_entropy"]].mean(axis=1),
            trajectory_ref[:, 7:, 0, metric_lookup["action_entropy"]].mean(axis=1),
        ),
        "late_back_action_entropy": triple(
            trajectory_pct[:, 7:, 1, metric_lookup["action_entropy"]].mean(axis=1),
            trajectory_raw[:, 7:, 1, metric_lookup["action_entropy"]].mean(axis=1),
            trajectory_ref[:, 7:, 1, metric_lookup["action_entropy"]].mean(axis=1),
        ),
        "late_front_action_top1": triple(
            trajectory_pct[:, 7:, 0, metric_lookup["action_top1_mass"]].mean(axis=1),
            trajectory_raw[:, 7:, 0, metric_lookup["action_top1_mass"]].mean(axis=1),
            trajectory_ref[:, 7:, 0, metric_lookup["action_top1_mass"]].mean(axis=1),
        ),
        "late_back_action_top1": triple(
            trajectory_pct[:, 7:, 1, metric_lookup["action_top1_mass"]].mean(axis=1),
            trajectory_raw[:, 7:, 1, metric_lookup["action_top1_mass"]].mean(axis=1),
            trajectory_ref[:, 7:, 1, metric_lookup["action_top1_mass"]].mean(axis=1),
        ),
        "late_front_token_dispersion": triple(
            trajectory_pct[:, 7:, 0, metric_lookup["action_token_dispersion"]].mean(axis=1),
            trajectory_raw[:, 7:, 0, metric_lookup["action_token_dispersion"]].mean(axis=1),
            trajectory_ref[:, 7:, 0, metric_lookup["action_token_dispersion"]].mean(axis=1),
        ),
        "late_back_token_dispersion": triple(
            trajectory_pct[:, 7:, 1, metric_lookup["action_token_dispersion"]].mean(axis=1),
            trajectory_raw[:, 7:, 1, metric_lookup["action_token_dispersion"]].mean(axis=1),
            trajectory_ref[:, 7:, 1, metric_lookup["action_token_dispersion"]].mean(axis=1),
        ),
        "late_front_state_action_gap": triple(
            trajectory_pct[:, 7:, 0, metric_lookup["state_action_gap"]].mean(axis=1),
            trajectory_raw[:, 7:, 0, metric_lookup["state_action_gap"]].mean(axis=1),
            trajectory_ref[:, 7:, 0, metric_lookup["state_action_gap"]].mean(axis=1),
        ),
        "late_back_state_action_gap": triple(
            trajectory_pct[:, 7:, 1, metric_lookup["state_action_gap"]].mean(axis=1),
            trajectory_raw[:, 7:, 1, metric_lookup["state_action_gap"]].mean(axis=1),
            trajectory_ref[:, 7:, 1, metric_lookup["state_action_gap"]].mean(axis=1),
        ),
        "late_front_denoise_soft_change": triple(
            trajectory_pct[:, 7:, 0, metric_lookup["denoise_soft_change"]].mean(axis=1),
            trajectory_raw[:, 7:, 0, metric_lookup["denoise_soft_change"]].mean(axis=1),
            trajectory_ref[:, 7:, 0, metric_lookup["denoise_soft_change"]].mean(axis=1),
        ),
        "late_back_denoise_soft_change": triple(
            trajectory_pct[:, 7:, 1, metric_lookup["denoise_soft_change"]].mean(axis=1),
            trajectory_raw[:, 7:, 1, metric_lookup["denoise_soft_change"]].mean(axis=1),
            trajectory_ref[:, 7:, 1, metric_lookup["denoise_soft_change"]].mean(axis=1),
        ),
        "late_front_query_soft_change": triple(
            trajectory_pct[:, 7:, 0, metric_lookup["query_soft_change"]].mean(axis=1),
            trajectory_raw[:, 7:, 0, metric_lookup["query_soft_change"]].mean(axis=1),
            trajectory_ref[:, 7:, 0, metric_lookup["query_soft_change"]].mean(axis=1),
        ),
        "late_back_query_soft_change": triple(
            trajectory_pct[:, 7:, 1, metric_lookup["query_soft_change"]].mean(axis=1),
            trajectory_raw[:, 7:, 1, metric_lookup["query_soft_change"]].mean(axis=1),
            trajectory_ref[:, 7:, 1, metric_lookup["query_soft_change"]].mean(axis=1),
        ),
        "late_front_query_hard_churn": triple(
            trajectory_pct[:, 7:, 0, metric_lookup["query_hard_churn"]].mean(axis=1),
            trajectory_raw[:, 7:, 0, metric_lookup["query_hard_churn"]].mean(axis=1),
            trajectory_ref[:, 7:, 0, metric_lookup["query_hard_churn"]].mean(axis=1),
        ),
        "late_back_query_hard_churn": triple(
            trajectory_pct[:, 7:, 1, metric_lookup["query_hard_churn"]].mean(axis=1),
            trajectory_raw[:, 7:, 1, metric_lookup["query_hard_churn"]].mean(axis=1),
            trajectory_ref[:, 7:, 1, metric_lookup["query_hard_churn"]].mean(axis=1),
        ),
        "late_d9_front_action_top1": triple(
            within_pct[:, 2, 0, 9, 1:, within_lookup["top1_mass"]].mean(axis=1),
            within_raw[:, 2, 0, 9, 1:, within_lookup["top1_mass"]].mean(axis=1),
            within_ref[:, 2, 0, 9, 1:, within_lookup["top1_mass"]].mean(axis=1),
        ),
        "late_d9_back_action_top1": triple(
            within_pct[:, 2, 1, 9, 1:, within_lookup["top1_mass"]].mean(axis=1),
            within_raw[:, 2, 1, 9, 1:, within_lookup["top1_mass"]].mean(axis=1),
            within_ref[:, 2, 1, 9, 1:, within_lookup["top1_mass"]].mean(axis=1),
        ),
        "back_cross_chunk_soft_change": triple(
            cross_pct[:, :, :, cross_lookup["lag1_soft_change"]].mean(axis=(1, 2)),
            cross_raw[:, :, :, cross_lookup["lag1_soft_change"]].mean(axis=(1, 2)),
            cross_ref[:, :, :, cross_lookup["lag1_soft_change"]].mean(axis=(1, 2)),
        ),
        "back_cross_chunk_hard_churn": triple(
            cross_pct[:, :, :, cross_lookup["lag1_hard_churn"]].mean(axis=(1, 2)),
            cross_raw[:, :, :, cross_lookup["lag1_hard_churn"]].mean(axis=(1, 2)),
            cross_ref[:, :, :, cross_lookup["lag1_hard_churn"]].mean(axis=(1, 2)),
        ),
        "back_boundary_T10_T1_soft_change": triple(
            boundary_pct[:, :, 0].mean(axis=1),
            boundary_raw[:, :, 0].mean(axis=1),
            boundary_ref[:, :, 0].mean(axis=1),
        ),
        "back_boundary_T10_T1_hard_churn": triple(
            boundary_pct[:, :, 1].mean(axis=1),
            boundary_raw[:, :, 1].mean(axis=1),
            boundary_ref[:, :, 1].mean(axis=1),
        ),
    }
    headline_rows = []
    for group_axis, (label, indices) in enumerate(groups):
        for metric, (values, raw_values, reference_values) in headline_extractors.items():
            current = values[indices]
            finite = np.isfinite(current)
            selected = indices[finite]
            current = current[finite]
            raw_current = raw_values[selected]
            reference_current = reference_values[selected]
            tasks = np.asarray(
                [f"{failure_specs[index].suite}/{failure_specs[index].task_name}" for index in selected]
            )
            macro, low, high, task_count = bootstrap_task_macro(
                current, tasks, bootstrap, seed + group_axis * 1000 + len(headline_rows)
            )
            headline_rows.append(
                {
                    "failure_group": label,
                    "metric": metric,
                    "failures": len(current),
                    "tasks": task_count,
                    "episode_weighted_mean_percentile": float(current.mean()),
                    "task_macro_mean_percentile": macro,
                    "task_cluster_bootstrap_ci_low": low,
                    "task_cluster_bootstrap_ci_high": high,
                    "failure_raw_mean": float(raw_current.mean()),
                    "same_run_init_success_raw_mean": float(reference_current.mean()),
                    "raw_mean_delta": float((raw_current - reference_current).mean()),
                }
            )
    write_csv(output / "headline_effects.csv", headline_rows)

    special_rows = []
    special_groups = (
        ("pin-base", np.asarray([i for i, spec in enumerate(failure_specs) if spec.run_id == "pin-base"])),
        ("pin-off", np.asarray([i for i, spec in enumerate(failure_specs) if spec.run_id == "pin-off"])),
        ("pin-on", np.asarray([i for i, spec in enumerate(failure_specs) if spec.run_id == "pin-on"])),
    )
    for label, indices in special_groups:
        for metric, (values, raw_values, reference_values) in headline_extractors.items():
            finite = np.isfinite(values[indices])
            selected = indices[finite]
            special_rows.append(
                {
                    "run_id": label,
                    "metric": metric,
                    "failures": len(selected),
                    "matched_percentile_mean": float(values[selected].mean()),
                    "failure_raw_mean": float(raw_values[selected].mean()),
                    "same_run_init_success_raw_mean": float(reference_values[selected].mean()),
                    "raw_mean_delta": float((raw_values[selected] - reference_values[selected]).mean()),
                }
            )
    write_csv(output / "special_pin_run_effects.csv", special_rows)

    return {
        "reasons": reasons,
        "reason_counts_observational": dict(
            Counter(
                spec.primary_failure_reason
                for spec in failure_specs
                if spec.run_regime == "observational"
            )
        ),
        "headline_effects": headline_rows,
        "special_pin_run_effects": special_rows,
    }


def plot_results(
    output: Path,
    failure_specs: list[EpisodeSpec],
    comparisons: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    natural = subset_failure_indices(failure_specs, None)
    colors = ("#2F6B5F", "#C44E52")
    trajectory = comparisons["trajectory"][0]
    metric_axes = {name: axis for axis, name in enumerate(TRAJECTORY_METRICS)}
    panels = (
        ("action_entropy", "action entropy percentile"),
        ("action_top1_mass", "Top-1 mass percentile"),
        ("action_token_dispersion", "token dispersion percentile"),
        ("state_action_gap", "state/action gap percentile"),
        ("denoise_soft_change", "within-chunk denoise change"),
        ("query_soft_change", "between-chunk route change"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for axis, (metric, title) in zip(axes.ravel(), panels):
        for group, name, color in zip(range(2), ("front L2-5", "back L12-15"), colors):
            values = trajectory[natural, :, group, metric_axes[metric]]
            axis.plot(PHASE_POINTS, np.nanmean(values, axis=0), label=name, color=color, lw=2)
        axis.axhline(0.5, color="#777777", lw=1, ls="--")
        axis.set_title(title)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    for axis in axes[1]:
        axis.set_xlabel("relative episode phase")
    fig.suptitle("Physical failures relative to same-run, same-init successes")
    fig.tight_layout()
    fig.savefig(output / "trajectory_overview.png", dpi=160)
    plt.close(fig)

    within = comparisons["within"][0]
    within_axes = {name: axis for axis, name in enumerate(WITHIN_METRICS)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for column, metric in enumerate(WITHIN_METRICS):
        for row, (band, band_name) in enumerate(((0, "early phase"), (2, "late phase"))):
            axis = axes[row, column]
            for group, name, color in zip(range(2), ("front L2-5", "back L12-15"), colors):
                values = within[natural, band, group, :, 1:, within_axes[metric]].mean(axis=-1)
                axis.plot(DENOISE, np.nanmean(values, axis=0), label=name, color=color, lw=2)
            axis.axhline(0.5, color="#777777", lw=1, ls="--")
            axis.set_ylim(0.0, 1.0)
            axis.set_title(f"{band_name}: {metric}")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    for axis in axes[1]:
        axis.set_xlabel("denoise forward d0 (noisy) -> d9")
    fig.suptitle("Within-chunk HB routing, action tokens T1-T10")
    fig.tight_layout()
    fig.savefig(output / "within_chunk_front_back.png", dpi=160)
    plt.close(fig)

    cross = comparisons["cross"][0]
    cross_axes = {name: axis for axis, name in enumerate(CROSS_METRICS)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, metric, title in (
        (axes[0], "lag1_soft_change", "soft-route Hellinger percentile"),
        (axes[1], "lag1_hard_churn", "Top-4 churn percentile"),
    ):
        matrix = np.nanmean(cross[natural, :, :, cross_axes[metric]], axis=0).T
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(CROSS_PHASE_POINTS)), [f"{x:.1f}" for x in CROSS_PHASE_POINTS])
        axis.set_yticks(range(len(TOKENS)), TOKENS)
        axis.set_xlabel("transition endpoint phase")
        axis.set_title(title)
        fig.colorbar(image, ax=axis)
    fig.suptitle("Back HB layers L12-L15: each token across adjacent chunks")
    fig.tight_layout()
    fig.savefig(output / "back_token_cross_chunk.png", dpi=160)
    plt.close(fig)


def format_percentile(value: float) -> str:
    return f"{value:.3f}"


def build_report(
    output: Path,
    integrity: dict[str, Any],
    extraction: dict[str, Any],
    aggregate: dict[str, Any],
    failure_specs: list[EpisodeSpec],
    comparisons: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    headlines = aggregate["headline_effects"]
    all_head = {row["metric"]: row for row in headlines if row["failure_group"] == "all_failures"}
    within = comparisons["within"][0]
    cross = comparisons["cross"][0]
    natural = subset_failure_indices(failure_specs, None)
    valid = natural[np.isfinite(within[natural, 0, 0, 0, 0, 0])]
    within_metric = {name: axis for axis, name in enumerate(WITHIN_METRICS)}
    late_d = {}
    for metric in WITHIN_METRICS:
        late_d[metric] = {
            group: np.nanmean(
                within[valid, 2, group, :, 1:, within_metric[metric]], axis=(0, 2)
            ).tolist()
            for group in range(2)
        }
    cross_metric = {name: axis for axis, name in enumerate(CROSS_METRICS)}
    token_soft = np.nanmean(
        cross[valid, :, :, cross_metric["lag1_soft_change"]], axis=(0, 1)
    )
    token_hard = np.nanmean(
        cross[valid, :, :, cross_metric["lag1_hard_churn"]], axis=(0, 1)
    )

    reason_rows = []
    for reason, count in sorted(
        aggregate["reason_counts_observational"].items(), key=lambda item: -item[1]
    ):
        candidates = [
            row
            for row in headlines
            if row["failure_group"] == reason and row["failures"] > 0
        ]
        strongest = max(
            candidates,
            key=lambda row: abs(row["task_macro_mean_percentile"] - 0.5),
        )
        reason_rows.append(
            f"| {REASON_ZH.get(reason, reason)} | {count} | {strongest['failures']} | "
            f"`{strongest['metric']}` | "
            f"{strongest['task_macro_mean_percentile']:.3f} | {strongest['tasks']} |"
        )

    special_lookup = {
        (row["run_id"], row["metric"]): row
        for row in aggregate["special_pin_run_effects"]
    }
    special_rows = []
    for run_id in ("pin-base", "pin-off", "pin-on"):
        entropy = special_lookup[(run_id, "late_front_action_entropy")]
        front_hard = special_lookup[(run_id, "late_front_query_hard_churn")]
        back_hard = special_lookup[(run_id, "late_back_query_hard_churn")]
        special_rows.append(
            f"| `{run_id}` | {entropy['failures']} | "
            f"{entropy['matched_percentile_mean']:.3f} | "
            f"{front_hard['matched_percentile_mean']:.3f} | "
            f"{back_hard['matched_percentile_mean']:.3f} |"
        )

    lines = [
        "# 物理失败轨迹的 HB-MoE 全程变化",
        "",
        "## 覆盖与比较口径",
        "",
        f"共对齐 `{integrity['failure_records']}` 条物理失败、`{integrity['selected_success_controls']}` 条同 run/同初始状态候选成功对照，覆盖 `{integrity['source_runs']}` 个源 run。",
        f"其中 `{integrity['matched_failures']}` 条失败存在同 run、同初始状态成功对照；`{integrity['unmatched_failures']}` 条只进入原始描述，不进入 success-normalized 主比较。",
        f"主观察性描述使用 `{integrity['primary_observational_failures']}` 条失败，其中 `{integrity['primary_matched_failures']}` 条进入 success-normalized 比较；`pin-base` 的逐字节重复失败和 `pin-on/off` 前层干预失败单列，不作为独立自然样本。",
        "",
        "每个百分位都先在同 source run、同 init-state 的成功轨迹内计算，再以 episode 为单位汇总。`0.5` 表示与匹配成功无差异，越大表示该指标在失败中越高。专家编号只在相同 checkpoint 内比较。",
        "",
        "## 全程主结果",
        "",
        "最稳定的结构不是简单的全局塌缩，而是两个时间尺度方向相反：同一 chunk 的 d0→d9 路由变化更大，但相邻 chunk 的同位置路由、尤其后层 Top-4，反而更粘滞。后层的 state/action gap 同时升高。",
        "",
        "| 指标（末段或全程） | 失败原始均值 | 匹配成功原始均值 | episode 百分位 | task 宏平均 | task-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in (
        "late_front_action_entropy",
        "late_back_action_entropy",
        "late_front_token_dispersion",
        "late_back_token_dispersion",
        "late_front_denoise_soft_change",
        "late_back_denoise_soft_change",
        "late_front_query_soft_change",
        "late_back_query_soft_change",
        "late_front_query_hard_churn",
        "late_back_query_hard_churn",
        "back_boundary_T10_T1_soft_change",
    ):
        row = all_head[metric]
        low = row["task_cluster_bootstrap_ci_low"]
        high = row["task_cluster_bootstrap_ci_high"]
        lines.append(
            f"| `{metric}` | {row['failure_raw_mean']:.4f} | "
            f"{row['same_run_init_success_raw_mean']:.4f} | "
            f"{row['episode_weighted_mean_percentile']:.3f} | "
            f"{row['task_macro_mean_percentile']:.3f} | [{low:.3f}, {high:.3f}] |"
        )

    lines.extend(
        [
            "",
            "完整 0%–100% 相位曲线在 `trajectory_phase_curves.csv`；末端点仍会包含成功提前终止、失败运行到 horizon 的语义差异，因此不能把末端分离直接解释为早期预警。",
            "",
            "## Chunk 内：前层与后层",
            "",
            "下面是失败相对匹配成功的 late-phase、动作 token T1–T10 平均百分位，列顺序为 d0（最噪）到 d9（最后一次前向）：",
            "",
        ]
    )
    for metric in WITHIN_METRICS:
        lines.append(f"- `{metric}` front: " + ", ".join(format_percentile(v) for v in late_d[metric][0]))
        lines.append(f"- `{metric}` back:  " + ", ".join(format_percentile(v) for v in late_d[metric][1]))
    lines.extend(
        [
            "",
            "state token 按因果位置应基本不随 d 改变。实测每 episode 最大差的中位数为 "
            f"`{extraction['state_token_denoise_median_max_abs_difference_per_episode']:.3g}`，p95 为 "
            f"`{extraction['state_token_denoise_p95_max_abs_difference_per_episode']:.3g}`，全局最大为 "
            f"`{extraction['state_token_denoise_max_abs_difference']:.3g}`；因此逐 d 的 state 值不作为十次独立证据。",
            "",
            "## Chunk 间：后层逐 token",
            "",
            "| token | 相邻 soft-route 变化百分位 | Top-4 churn 百分位 |",
            "|---:|---:|---:|",
        ]
    )
    for token, soft, hard in zip(TOKENS, token_soft, token_hard):
        lines.append(f"| {token} | {soft:.3f} | {hard:.3f} |")
    lines.extend(
        [
            "",
            "这里比较相邻 query 的同位置 token；`T10(q) -> T1(q+1)` 的真实 chunk 边界另见 `boundary` 指标。soft-route 使用完整 32-way 概率，hard churn 使用权威保存的 Top-4 集合。",
            "",
            "## 不同物理失败原因",
            "",
            "下表列出预设 headline 指标中偏离 0.5 最大的一项。百分位只使用有同 run、同初始状态成功对照的失败；它是探索性摘要，不是从多重扫描中得到的确认性因果结论，样本少或只覆盖少数任务的原因尤其要谨慎。",
            "",
            "| 物理失败原因 | 总数 | 已匹配 | 最强 headline 指标 | task 宏平均百分位 | 支持任务 |",
            "|---|---:|---:|---|---:|---:|",
            *reason_rows,
            "",
            "这里的九类是物理 outcome 标签，不等同于停滞、来回摆或周期运动标签。原始 1442 条物理标签没有 `stagnation`、`oscillation` 或 `periodicity` 字段，因此本表不把这些运动模式强行映射到物理原因。独立的 Scene8 运动学对照只覆盖 512 条轨迹（216 条失败，其中 197 条为预定义停滞型陷入）：它支持末期低变化/粘滞，不支持校正后的 period-2 到 period-5 MoE 周期；该结论只作为子集证据，不能外推成 1442 条的周期计数。详见 `../../moe-token-dynamics/results/CONCLUSION.zh.md`。",
            "",
            "## Pin 数据单列",
            "",
            "这些数只描述各 arm 内失败相对该 arm 成功的路由位置，不估计 pin 干预的因果效应。`pin-base` 又与对应 `right-16x32` 逐字节重复。",
            "",
            "| run | 失败 | late 前层 entropy 百分位 | late 前层 hard churn 百分位 | late 后层 hard churn 百分位 |",
            "|---|---:|---:|---:|---:|",
            *special_rows,
            "",
            "## AS 与完整性检查",
            "",
            f"- AS 概率在单 episode 内的最大跨度为 `{extraction['as_probability_max_within_episode_span']:.3g}`；平均相邻 hard route 变化率为 `{extraction['as_hard_adjacent_change_rate_mean_by_run']:.3g}`。",
            "- 所有 Zarr 行均通过 episode id、summary inference_calls 和 offset 三重对齐。",
            "- `pin-on/off` 的 soft router 概率是干预前 gate 分布，而保存的 hard Top-4 是干预后实际执行选择；两者只在干预附表中解释。",
            "- 这些结果说明路由状态与失败模式相关，不证明 MoE 路由是物理失败的原因。视觉 token、attention、expert hidden/output 和实际动作几何不在本分析中。",
            "",
            "## 产物",
            "",
            "- `cohort.csv`: episode 对齐、失败标签、run regime 和匹配对照数。",
            "- `trajectory_phase_curves.csv`: 0%–100% 全程前/后层曲线。",
            "- `within_chunk_profiles.csv`: early/middle/late × d0–d9 × state/T1–T10 × 前/后层。",
            "- `back_token_cross_chunk.csv`: 后层各 token 的 lag-1/lag-2 soft 与 Top-4 变化。",
            "- `checkpoint_expert_load_shifts.csv`: checkpoint 内逐 expert 的失败-成功 load 差。",
            "- `special_pin_run_effects.csv`: 三个 pin run 的失败子集描述。",
            "- `episode_features.npz` / `matched_failure_features.npz`: 可复核的 episode 级数组。",
        ]
    )
    (output / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_test() -> None:
    same = np.asarray([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
    assert np.allclose(hellinger(same, same), 0.0)
    left = np.asarray([[1.0, 0.0]], dtype=np.float32)
    right = np.asarray([[0.0, 1.0]], dtype=np.float32)
    assert np.allclose(hellinger(left, right), 1.0)
    ids = np.asarray([[0, 1, 2, 3]], dtype=np.uint8)
    assert np.allclose(top4_churn(ids, ids), 0.0)
    assert np.allclose(top4_churn(ids, np.asarray([[4, 5, 6, 7]], np.uint8)), 1.0)
    curve = np.asarray([[0.0], [2.0]], dtype=np.float32)
    assert np.allclose(regular_resample(curve)[[0, 5, 10], 0], [0.0, 1.0, 2.0])

    rng = np.random.default_rng(7)
    raw = rng.uniform(size=(6, 8, 10, 11, 32)).astype(np.float32)
    raw[:, :, :, 0] = raw[:, :, :1, 0]
    probability = normalize_probability(raw)
    ids = np.argpartition(probability, -4, axis=-1)[..., -4:].astype(np.uint8)
    as_raw = rng.uniform(size=(6, 4, 3)).astype(np.float32)
    as_ids = as_raw.argmax(axis=-1).astype(np.uint8)
    features, audit = extract_episode_features(raw, ids, as_raw, as_ids)
    assert features["trajectory"].shape == (11, 2, len(TRAJECTORY_METRICS))
    assert features["within"].shape == (3, 2, 10, 11, len(WITHIN_METRICS))
    assert features["cross"].shape == (10, 11, len(CROSS_METRICS))
    assert features["expert_load"].shape == (3, 2, 11, 32)
    assert audit["state_denoise_span"] == 0.0
    assert np.allclose(features["expert_load"].sum(axis=-1), 1.0)
    print("self-test passed")


def load_reusable_features(
    output: Path, expected_episodes: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    feature_path = output / "episode_features.npz"
    summary_path = output / "summary.json"
    if not feature_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("--reuse-features requires existing episode features and summary")
    with np.load(feature_path, allow_pickle=False) as stored:
        feature_row = np.asarray(stored["feature_row"], dtype=np.int64)
        if not np.array_equal(feature_row, np.arange(expected_episodes)):
            raise ValueError("reusable feature rows do not match the current aligned cohort")
        arrays = {
            "trajectory": np.asarray(stored["trajectory"]),
            "within": np.asarray(stored["within_chunk"]),
            "cross": np.asarray(stored["back_cross_chunk"]),
            "boundary": np.asarray(stored["back_boundary"]),
            "as_phase": np.asarray(stored["as_phase"]),
            "expert_load": np.asarray(stored["expert_load"]),
        }
    previous = read_json(summary_path)
    extraction = dict(previous["extraction"])
    return arrays, extraction


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    print("building aligned failure/control cohort", flush=True)
    specs, integrity = build_cohort(args.failures)
    print(
        f"  {integrity['failure_records']} failures, "
        f"{integrity['selected_success_controls']} success controls, "
        f"{integrity['source_runs']} runs",
        flush=True,
    )
    if args.reuse_features:
        print("loading validated episode features", flush=True)
        arrays, extraction = load_reusable_features(args.output, len(specs))
    else:
        print("extracting HB/AS episode features", flush=True)
        arrays, extraction = extract_all(specs, args.workers)

    failure_specs = [spec for spec in specs if not spec.success]
    references = reference_map(specs)
    comparisons = {
        name: matched_comparison(values, failure_specs, references)
        for name, values in arrays.items()
    }
    control_counts = comparisons["trajectory"][2]

    cohort_rows = []
    failure_position = {(spec.source_run, spec.episode_index): index for index, spec in enumerate(failure_specs)}
    for spec in specs:
        row = asdict(spec)
        if spec.success:
            row["matched_success_controls"] = ""
            row["primary_analysis"] = False
        else:
            index = failure_position[(spec.source_run, spec.episode_index)]
            row["matched_success_controls"] = int(control_counts[index])
            row["primary_analysis"] = spec.run_regime == "observational"
        cohort_rows.append(row)
    write_csv(args.output / "cohort.csv", cohort_rows)

    if not args.reuse_features:
        np.savez_compressed(
            args.output / "episode_features.npz",
            feature_row=np.arange(len(specs), dtype=np.int32),
            trajectory=arrays["trajectory"],
            within_chunk=arrays["within"],
            back_cross_chunk=arrays["cross"],
            back_boundary=arrays["boundary"],
            as_phase=arrays["as_phase"],
            expert_load=arrays["expert_load"],
            trajectory_metrics=np.asarray(TRAJECTORY_METRICS),
            within_metrics=np.asarray(WITHIN_METRICS),
            cross_metrics=np.asarray(CROSS_METRICS),
            layer_groups=np.asarray(LAYER_GROUPS),
            tokens=np.asarray(TOKENS),
            phase_points=PHASE_POINTS,
            cross_phase_points=CROSS_PHASE_POINTS,
        )
    np.savez_compressed(
        args.output / "matched_failure_features.npz",
        failure_feature_row=np.asarray([spec.feature_row for spec in failure_specs], dtype=np.int32),
        matched_success_controls=control_counts,
        trajectory_percentile=comparisons["trajectory"][0],
        trajectory_reference_mean=comparisons["trajectory"][1],
        within_chunk_percentile=comparisons["within"][0],
        within_chunk_reference_mean=comparisons["within"][1],
        back_cross_chunk_percentile=comparisons["cross"][0],
        back_cross_chunk_reference_mean=comparisons["cross"][1],
        back_boundary_percentile=comparisons["boundary"][0],
        back_boundary_reference_mean=comparisons["boundary"][1],
        as_phase_percentile=comparisons["as_phase"][0],
        expert_load_reference_mean=comparisons["expert_load"][1],
    )

    print("aggregating physical failure modes", flush=True)
    aggregate = aggregate_outputs(
        args.output,
        specs,
        arrays,
        failure_specs,
        comparisons,
        args.bootstrap,
        args.seed,
    )
    plot_results(args.output, failure_specs, comparisons)
    build_report(args.output, integrity, extraction, aggregate, failure_specs, comparisons)
    summary = {
        "schema": "himoe.physical_failure_moe_dynamics.v1",
        "axes": {
            "hb_layers": HB_LAYERS,
            "layer_groups": LAYER_GROUPS,
            "denoise_order": "d0 flow time 1.0/noisiest; d9 flow time 0.1/last recorded forward",
            "tokens": TOKENS,
            "phase_points": PHASE_POINTS,
            "cross_phase_points": CROSS_PHASE_POINTS,
        },
        "method": {
            "statistical_unit": "episode",
            "success_reference": "same source run and init_state_id",
            "primary_regime": "observational; excludes duplicate pin-base and intervened pin-on/off",
            "matched_effect_scale": "midrank percentile among matched successes; 0.5 is null",
            "midrank_tolerance": {"rtol": MATCH_RTOL, "atol": MATCH_ATOL},
            "expert_identity_pooling": "checkpoint-local only",
            "bootstrap": "task-cluster macro mean",
            "bootstrap_draws": args.bootstrap,
            "seed": args.seed,
        },
        "integrity": integrity,
        "extraction": extraction,
        "aggregate": aggregate,
    }
    (args.output / "summary.json").write_text(
        json.dumps(plain(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output / 'report.zh.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
