#!/usr/bin/env python3
"""Build and seal the external high-precision routing alarm replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import online_closed_loop_alarm_v2 as v2
import online_multihead_alarm as v1
from precision_cascade_monitor import (
    BASE_DETECTORS,
    DETECTORS,
    GATED_DETECTORS,
    PRIMARY_DETECTOR,
    PRIMARY_QUANTILE,
    QUANTILES,
    ROUTE_HEADS,
    mobility_detector_scores,
    typed_gates,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_CACHE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
DEFAULT_MAIN_CACHE = HERE / "results/online_multihead_hub/unlabeled_query_features.npz"
DEFAULT_MAIN_EXTERNAL_CACHE = (
    HERE / "results/online_multihead_hub_external/unlabeled_query_features.npz"
)
DEFAULT_OUTPUT = HERE / "results/online_precision_cascade_external"
PROTOCOL = HERE / "ONLINE_PRECISION_CASCADE_PROTOCOL.md"
TEST_RUN_ID = "right-50x8b-20260903"
REFERENCE_RUN_ID = "right-50x8-20260903"
EXPECTED_TASKS = 39
EXPECTED_EPISODES = 15_600
INCOMPLETE_TASK = (
    "libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--main-cache", type=Path, default=DEFAULT_MAIN_CACHE)
    parser.add_argument(
        "--main-external-cache", type=Path, default=DEFAULT_MAIN_EXTERNAL_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-test-features", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
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


def discover_test_runs(cache_root: Path) -> list[Path]:
    runs: list[Path] = []
    excluded: list[str] = []
    pattern = f"libero_*/*/{TEST_RUN_ID}/client/summaries.json"
    for summary_path in sorted(cache_root.glob(pattern)):
        run = summary_path.parents[1]
        meta_path = run / "meta.json"
        route_path = run / "server/routes.zarr"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sampling = meta.get("sampling", {})
        complete = (
            meta.get("status") == "complete"
            and not meta.get("error")
            and sampling.get("complete") is True
            and int(sampling.get("actual_episodes", -1)) == 400
            and int(sampling.get("designed_episodes", -2)) == 400
            and route_path.exists()
        )
        task = v1.task_key(run, cache_root)
        if complete:
            runs.append(run)
        else:
            excluded.append(task)
    if len(runs) != EXPECTED_TASKS:
        raise RuntimeError(
            f"external cohort mismatch: {len(runs)} complete tasks, "
            f"expected {EXPECTED_TASKS}"
        )
    if excluded != [INCOMPLETE_TASK]:
        raise RuntimeError(f"unexpected incomplete external tasks: {excluded}")
    return runs


def combine_feature_caches(
    caches: list[dict[str, np.ndarray]], wanted_tasks: set[str]
) -> dict[str, np.ndarray]:
    task_names: list[str] = []
    task_index: list[np.ndarray] = []
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "episode",
            "init_state_id",
            "flow_noise_seed",
            "length",
            "valid",
            "features",
        )
    }
    seen: set[str] = set()
    for cache in caches:
        names = cache["task_names"].astype(str)
        source_task = cache["task_index"].astype(int)
        for source_position, task in enumerate(names):
            if task not in wanted_tasks:
                continue
            if task in seen:
                raise ValueError(f"duplicate reference task: {task}")
            seen.add(task)
            take = np.flatnonzero(source_task == source_position)
            if len(take) != 400:
                raise ValueError(f"expected 400 reference trajectories for {task}")
            target_position = len(task_names)
            task_names.append(task)
            task_index.append(np.full(len(take), target_position, dtype=np.int16))
            for name in arrays:
                arrays[name].append(np.asarray(cache[name][take]))
    missing = wanted_tasks - seen
    if missing:
        raise ValueError(f"missing historical reference tasks: {sorted(missing)}")
    order = np.argsort(np.asarray(task_names))
    if not np.array_equal(order, np.arange(len(order))):
        # Rebuild recursively in deterministic task order without changing rows.
        task_to_block = {
            task: {name: arrays[name][position] for name in arrays}
            for position, task in enumerate(task_names)
        }
        task_names = sorted(task_names)
        task_index = []
        arrays = {name: [] for name in arrays}
        for position, task in enumerate(task_names):
            block = task_to_block[task]
            count = len(block["episode"])
            task_index.append(np.full(count, position, dtype=np.int16))
            for name in arrays:
                arrays[name].append(block[name])
    return {
        "schema": np.asarray("himoe.online_multihead.features.v1"),
        "feature_names": np.asarray(v1.FEATURES),
        "task_names": np.asarray(task_names),
        "task_index": np.concatenate(task_index),
        **{name: np.concatenate(blocks, axis=0) for name, blocks in arrays.items()},
    }


def normalized_route_heads(
    reference_features: np.ndarray,
    reference_valid: np.ndarray,
    test_features: np.ndarray,
    test_valid: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    features = np.concatenate((reference_features, test_features), axis=0)
    valid = np.concatenate((reference_valid, test_valid), axis=0)
    reference = np.zeros(len(features), dtype=bool)
    reference[: len(reference_features)] = True
    feature_index = {name: index for index, name in enumerate(v1.FEATURES)}
    ranks: dict[tuple[str, int], np.ndarray] = {}
    members = {member for head_members in v1.HEADS.values() for member in head_members}
    for name, direction in sorted(members):
        ranks[(name, direction)] = v2.directed_rank(
            features[:, :, feature_index[name]],
            direction,
            reference,
            valid,
            v1.RAW_SCORE_START_QUERY,
        )
    heads: dict[str, np.ndarray] = {}
    for head, head_members in v1.HEADS.items():
        instant = v2.complete_mean(
            np.stack([ranks[member] for member in head_members], axis=-1)
        )
        heads[head] = v2.trailing_mean(instant)
        heads[head][~valid] = np.nan
    reference_heads = {name: values[: len(reference_features)] for name, values in heads.items()}
    test_heads = {name: values[len(reference_features) :] for name, values in heads.items()}
    return reference_heads, test_heads


def detector_thresholds(
    reference_scores: dict[str, np.ndarray], reference_valid: np.ndarray
) -> np.ndarray:
    output = np.full((len(DETECTORS), len(QUANTILES)), np.nan, dtype=np.float32)
    for detector_position, detector in enumerate(BASE_DETECTORS):
        maxima = v2.nan_row_max(reference_scores[detector])
        for quantile_position, quantile in enumerate(QUANTILES):
            output[detector_position, quantile_position] = v2.quantile_higher(
                maxima, quantile
            )
    primary_position = DETECTORS.index(PRIMARY_DETECTOR)
    for detector in GATED_DETECTORS:
        output[DETECTORS.index(detector)] = output[primary_position]
    primary = output[DETECTORS.index(PRIMARY_DETECTOR)]
    if not np.isfinite(primary).all():
        raise ValueError("non-finite primary precision-cascade threshold")
    return output


def calibrate_and_replay(
    reference_cache: dict[str, np.ndarray],
    test_cache: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    reference_tasks = reference_cache["task_names"].astype(str)
    test_tasks = test_cache["task_names"].astype(str)
    if not np.array_equal(reference_tasks, test_tasks):
        raise ValueError("historical and external task order differs")

    n_episode, n_query = test_cache["valid"].shape
    n_detector = len(DETECTORS)
    n_quantile = len(QUANTILES)
    scores = np.full((n_episode, n_query, n_detector), np.nan, dtype=np.float32)
    route_head_scores = np.full(
        (n_episode, n_query, len(ROUTE_HEADS)), np.nan, dtype=np.float32
    )
    gate_values = np.zeros(
        (n_episode, n_query, len(GATED_DETECTORS)), dtype=bool
    )
    thresholds = np.full(
        (n_episode, n_detector, n_quantile), np.nan, dtype=np.float32
    )
    alarms = np.zeros(
        (n_episode, n_query, n_detector, n_quantile), dtype=bool
    )
    profile_thresholds = np.full(
        (len(test_tasks), n_detector, n_quantile), np.nan, dtype=np.float32
    )
    threshold_rows: list[dict[str, Any]] = []
    feature_index = {name: index for index, name in enumerate(v1.FEATURES)}

    reference_blocks: list[np.ndarray] = []
    reference_valid_blocks: list[np.ndarray] = []
    reference_task_index: list[np.ndarray] = []

    for task_position, task in enumerate(test_tasks):
        ref_take = np.flatnonzero(reference_cache["task_index"] == task_position)
        test_take = np.flatnonzero(test_cache["task_index"] == task_position)
        if len(ref_take) != 400 or len(test_take) != 400:
            raise ValueError(f"expected 400 historical and external rows for {task}")
        ref_features = np.asarray(reference_cache["features"][ref_take], dtype=np.float32)
        ref_valid = np.asarray(reference_cache["valid"][ref_take], dtype=bool)
        ext_features = np.asarray(test_cache["features"][test_take], dtype=np.float32)
        ext_valid = np.asarray(test_cache["valid"][test_take], dtype=bool)
        reference_blocks.append(ref_features)
        reference_valid_blocks.append(ref_valid)
        reference_task_index.append(
            np.full(len(ref_take), task_position, dtype=np.int16)
        )

        ref_mobility = ref_features[:, :, feature_index["route_mobility"]]
        ext_mobility = ext_features[:, :, feature_index["route_mobility"]]
        ref_scores = mobility_detector_scores(ref_mobility, ref_valid)
        ext_scores = mobility_detector_scores(ext_mobility, ext_valid)
        _, ext_heads = normalized_route_heads(
            ref_features, ref_valid, ext_features, ext_valid
        )
        ext_gates = typed_gates(ext_heads)
        task_thresholds = detector_thresholds(ref_scores, ref_valid)
        profile_thresholds[task_position] = task_thresholds

        for head_position, head in enumerate(ROUTE_HEADS):
            route_head_scores[test_take, :, head_position] = ext_heads[head]
        for gate_position, gate in enumerate(GATED_DETECTORS):
            gate_values[test_take, :, gate_position] = ext_gates[gate]

        primary_score = ext_scores[PRIMARY_DETECTOR]
        for detector_position, detector in enumerate(DETECTORS):
            detector_score = (
                ext_scores[detector]
                if detector in BASE_DETECTORS
                else np.where(ext_gates[detector], primary_score, np.nan)
            )
            scores[test_take, :, detector_position] = detector_score
            thresholds[test_take, detector_position] = task_thresholds[
                detector_position
            ]
            for quantile_position, quantile in enumerate(QUANTILES):
                threshold = task_thresholds[detector_position, quantile_position]
                trigger = (
                    np.isfinite(detector_score)
                    & (detector_score > threshold)
                    & ext_valid
                )
                alarms[
                    test_take, :, detector_position, quantile_position
                ] = np.maximum.accumulate(trigger, axis=1) & ext_valid
                threshold_rows.append(
                    {
                        "task": task,
                        "detector": detector,
                        "threshold_source": (
                            PRIMARY_DETECTOR
                            if detector in GATED_DETECTORS
                            else detector
                        ),
                        "quantile": quantile,
                        "threshold": float(threshold),
                        "historical_reference_episodes": len(ref_take),
                        "external_calibration_episodes": 0,
                    }
                )
        print(
            f"[external replay {task_position + 1}/{len(test_tasks)}] {task}",
            flush=True,
        )

    primary_thresholds = thresholds[:, DETECTORS.index(PRIMARY_DETECTOR)]
    if not np.isfinite(primary_thresholds).all():
        raise ValueError("external replay contains non-finite primary thresholds")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "sealed_online_scores.npz",
        schema=np.asarray("himoe.precision_cascade.sealed.v1"),
        task_names=test_tasks,
        task_index=np.asarray(test_cache["task_index"], dtype=np.int16),
        episode=np.asarray(test_cache["episode"], dtype=np.int16),
        init_state_id=np.asarray(test_cache["init_state_id"], dtype=np.int16),
        flow_noise_seed=np.asarray(test_cache["flow_noise_seed"], dtype=np.int16),
        length=np.asarray(test_cache["length"], dtype=np.int16),
        valid=np.asarray(test_cache["valid"], dtype=bool),
        detector_names=np.asarray(DETECTORS),
        route_head_names=np.asarray(ROUTE_HEADS),
        gate_names=np.asarray(GATED_DETECTORS),
        quantiles=np.asarray(QUANTILES, dtype=np.float32),
        scores=scores,
        route_head_scores=route_head_scores,
        gates=gate_values,
        thresholds=thresholds,
        alarms=alarms,
    )
    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "unlabeled_thresholds.csv", index=False
    )
    np.savez_compressed(
        output_dir / "deployment_profiles.npz",
        schema=np.asarray("himoe.precision_cascade.profile.v1"),
        task_names=test_tasks,
        reference_task_index=np.concatenate(reference_task_index),
        reference_valid=np.concatenate(reference_valid_blocks, axis=0),
        route_feature_names=np.asarray(v1.FEATURES),
        route_reference=np.concatenate(reference_blocks, axis=0),
        detector_names=np.asarray(DETECTORS),
        quantiles=np.asarray(QUANTILES, dtype=np.float32),
        thresholds=profile_thresholds,
    )

    primary_position = DETECTORS.index(PRIMARY_DETECTOR)
    primary_quantile = QUANTILES.index(PRIMARY_QUANTILE)
    rows: list[dict[str, Any]] = []
    for row in range(n_episode):
        item: dict[str, Any] = {
            "task": test_tasks[int(test_cache["task_index"][row])],
            "episode": int(test_cache["episode"][row]),
            "init_state_id": int(test_cache["init_state_id"][row]),
            "flow_noise_seed": int(test_cache["flow_noise_seed"][row]),
            "length": int(test_cache["length"][row]),
        }
        for detector_position, detector in enumerate(DETECTORS):
            for quantile_position, quantile in enumerate(QUANTILES):
                where = np.flatnonzero(
                    alarms[row, :, detector_position, quantile_position]
                )
                suffix = str(quantile).replace("0.", "q")
                item[f"first_{detector}_{suffix}"] = (
                    int(where[0]) if len(where) else -1
                )
        primary = np.flatnonzero(
            alarms[row, :, primary_position, primary_quantile]
        )
        if len(primary):
            query = int(primary[0])
            active = [
                GATED_DETECTORS[position].replace("_confirmed", "")
                for position in range(len(GATED_DETECTORS) - 1)
                if gate_values[row, query, position]
            ]
            item["primary_phenotype"] = "+".join(active) if active else "untyped"
        else:
            item["primary_phenotype"] = "none"
        rows.append(item)
    pd.DataFrame(rows).to_csv(
        output_dir / "sealed_episode_alarms.csv", index=False
    )


def self_test() -> None:
    route = np.full((2, 14), np.nan, dtype=np.float32)
    route[0, 1:] = 0.02
    route[1, 1:] = np.linspace(0.02, 0.08, 13)
    valid = np.ones_like(route, dtype=bool)
    scores = mobility_detector_scores(route, valid)
    assert np.isfinite(scores["mobility_w4_k4"][0, 7])
    assert np.isclose(scores["mobility_w4_k4"][0, 7], -0.02)
    assert scores["mobility_w4_k4"][1, 7] < scores["mobility_w4_k4"][0, 7]
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if (args.output / "sealed_manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite sealed result: {args.output}")

    runs = discover_test_runs(args.cache_root)
    test_tasks = {v1.task_key(run, args.cache_root) for run in runs}
    main_cache = v1.load_feature_cache(args.main_cache)
    external_cache = v1.load_feature_cache(args.main_external_cache)
    reference_cache = combine_feature_caches(
        [main_cache, external_cache], test_tasks
    )

    feature_path = args.output / "unlabeled_query_features.npz"
    if args.reuse_test_features and feature_path.exists():
        test_cache = v1.load_feature_cache(feature_path)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        test_cache = v1.build_feature_cache(runs, args.cache_root, feature_path)
    if len(test_cache["episode"]) != EXPECTED_EPISODES:
        raise RuntimeError("unexpected external feature-cache episode count")
    if set(test_cache["task_names"].astype(str)) != test_tasks:
        raise RuntimeError("external feature-cache task set mismatch")

    calibrate_and_replay(reference_cache, test_cache, args.output)
    artifacts = {
        "protocol_sha256": sha256(PROTOCOL),
        "scorer_sha256": sha256(Path(__file__)),
        "runtime_monitor_sha256": sha256(HERE / "precision_cascade_monitor.py"),
        "runtime_extractor_sha256": sha256(HERE / "route_derivative_monitor.py"),
        "runtime_base_extractor_sha256": sha256(HERE / "single_rollout_monitor.py"),
        "feature_extractor_sha256": sha256(HERE / "online_multihead_alarm.py"),
        "main_reference_cache_sha256": sha256(args.main_cache),
        "main_external_reference_cache_sha256": sha256(args.main_external_cache),
        "external_feature_cache_sha256": sha256(feature_path),
        "sealed_scores_sha256": sha256(args.output / "sealed_online_scores.npz"),
        "thresholds_sha256": sha256(args.output / "unlabeled_thresholds.csv"),
        "episode_alarms_sha256": sha256(args.output / "sealed_episode_alarms.csv"),
        "deployment_profiles_sha256": sha256(args.output / "deployment_profiles.npz"),
    }
    manifest = {
        "schema": "himoe.precision_cascade.manifest.v1",
        "reference_run_id": REFERENCE_RUN_ID,
        "external_run_id": TEST_RUN_ID,
        "tasks": int(len(test_cache["task_names"])),
        "episodes": int(len(test_cache["episode"])),
        "excluded_incomplete_tasks": [INCOMPLETE_TASK],
        "reference_episodes_per_task": 400,
        "test_episodes_used_for_calibration": 0,
        "runtime_api": "PrecisionCascadeMonitor.update(router_prob, expert_ids)",
        "runtime_inputs": ["current_router_prob", "current_expert_ids"],
        "runtime_cross_rollout_access": False,
        "runtime_future_access": False,
        "runtime_final_length_access": False,
        "duration_detector_feature": False,
        "termination_or_censoring_filter": False,
        "summary_fields_whitelisted": [
            "episode_index",
            "init_state_id",
            "flow_noise_seed",
            "inference_calls",
        ],
        "detectors": list(DETECTORS),
        "primary_detector": PRIMARY_DETECTOR,
        "primary_quantile": PRIMARY_QUANTILE,
        "balanced_quantile": 0.975,
        "operating_quantiles": list(QUANTILES),
        "query_causal": True,
        "labels_used": [],
        "sim_state_used": False,
        "method_selection_outcome_informed_on_prior_cohort": True,
        "external_outcomes_read_before_seal": False,
        "artifacts": artifacts,
    }
    (args.output / "sealed_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"sealed {manifest['episodes']} external rollouts at {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
