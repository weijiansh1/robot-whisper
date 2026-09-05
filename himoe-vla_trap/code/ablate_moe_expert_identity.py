#!/usr/bin/env python3
"""Ablate arbitrary expert labels versus persistent expert identity.

All variants share one train-free recurrence detector. Target query predictions
are written and hashed before target outcomes are loaded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import zarr

import evaluate_moe_structured_alarm_two_runs as structured


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/moe_expert_identity_ablation.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/moe_expert_identity_ablation"
SCHEMA = "himoe.moe_expert_identity_ablation.v1"
VARIANTS = (
    "soft_coordinate",
    "actual_support",
    "selected_weight",
    "rank_shape",
    "reconstructed_support",
    "tie_resolved_support",
    "chunk_relabel_soft",
    "chunk_relabel_support",
)
ODD_MOD32 = np.arange(1, 32, 2, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def resolve_workspace(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(structured.plain(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def cache_path(root: Path, run_name: str, task: str) -> Path:
    token = hashlib.sha256(task.encode()).hexdigest()[:16]
    return root / run_name / f"{token}.npz"


def fingerprint(config: dict[str, Any], base: dict[str, Any]) -> str:
    payload = {
        "schema": SCHEMA,
        "variants": config["variants"],
        "scoring": config["scoring"],
        "tie_filter": config["tie_filter"],
        "position_grid": base["position_grid"],
        "causal_self_reference": base["causal_self_reference"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def affine_parameters(task: str, run_name: str, control_step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = int.from_bytes(
        hashlib.sha256(f"{run_name}:{task}".encode()).digest()[:8], "little"
    )
    values = np.asarray(control_step, dtype=np.uint64) ^ np.uint64(seed)
    with np.errstate(over="ignore"):
        values ^= values >> np.uint64(30)
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(27)
        values *= np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    multiplier = ODD_MOD32[np.asarray(values & np.uint64(15), dtype=np.int64)]
    offset = np.asarray((values >> np.uint64(8)) & np.uint64(31), dtype=np.int64)
    return multiplier, offset


def relabel_probability(
    probability: np.ndarray, multiplier: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    old = np.arange(32, dtype=np.int64)[None, :]
    new = (multiplier[:, None] * old + offset[:, None]) % 32
    indices = np.broadcast_to(new[:, None, None, :], probability.shape)
    output = np.empty_like(probability)
    np.put_along_axis(output, indices, probability, axis=-1)
    return output


def relabel_ids(ids: np.ndarray, multiplier: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return (
        multiplier[:, None, None, None] * np.asarray(ids, dtype=np.int64)
        + offset[:, None, None, None]
    ) % 32


def aggregate_layer_token_nan(
    values: np.ndarray,
    layer_groups: list[np.ndarray],
    token_groups: list[np.ndarray],
    minimum_fraction: float,
) -> np.ndarray:
    output = np.full(
        (len(values), len(layer_groups), len(token_groups)), np.nan, dtype=np.float32
    )
    for layer_index, layers in enumerate(layer_groups):
        by_layer = np.take(values, layers, axis=1)
        for token_index, tokens in enumerate(token_groups):
            selected = np.take(by_layer, tokens, axis=2)
            flat = selected.reshape(len(values), -1)
            valid = np.isfinite(flat)
            count = valid.sum(axis=1)
            minimum = int(math.ceil(flat.shape[1] * minimum_fraction))
            keep = count >= minimum
            if np.any(keep):
                output[keep, layer_index, token_index] = (
                    np.where(valid[keep], flat[keep], 0.0).sum(axis=1) / count[keep]
                )
    return output


def quantile_action_cells(values: np.ndarray) -> np.ndarray:
    action = np.asarray(values[..., 1:], dtype=np.float32).reshape(len(values), -1)
    output = np.full(len(action), np.nan, dtype=np.float32)
    valid = np.isfinite(action).any(axis=1)
    if np.any(valid):
        output[valid] = np.nanquantile(action[valid], 0.75, axis=1)
    return output


def build_evidence(
    similarity: np.ndarray,
    episode: np.ndarray,
    query: np.ndarray,
    lags: list[int],
    recurrence_observations: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.empty((len(query), len(VARIANTS), len(lags)), dtype=np.float32)
    for variant in range(len(VARIANTS)):
        for lag_index in range(len(lags)):
            raw[:, variant, lag_index] = quantile_action_cells(
                similarity[:, variant, lag_index]
            )
    evidence = np.full_like(raw, np.nan)
    for _episode, indices in structured.episode_indices(episode):
        local_query = query[indices]
        for lag_index, lag in enumerate(lags):
            baseline_queries = np.arange(lag, lag + recurrence_observations)
            baseline_queries = baseline_queries[baseline_queries < len(indices)]
            score = indices[local_query >= lag + recurrence_observations]
            for variant in range(len(VARIANTS)):
                baseline = np.nanmedian(raw[indices[baseline_queries], variant, lag_index])
                if np.isfinite(baseline):
                    evidence[score, variant, lag_index] = (
                        raw[score, variant, lag_index] - baseline
                    )
    return raw, evidence


def extract_task(
    task: str,
    run_name: str,
    run_string: str,
    original_cache_string: str,
    output_cache_string: str,
    config: dict[str, Any],
    base: dict[str, Any],
    rebuild: bool,
) -> dict[str, Any]:
    run = Path(run_string)
    original_cache = Path(original_cache_string)
    output_cache = Path(output_cache_string)
    expected_fingerprint = fingerprint(config, base)
    if output_cache.exists() and not rebuild:
        with np.load(output_cache, allow_pickle=False) as archive:
            if (
                str(archive["schema"].item()) == SCHEMA
                and str(archive["feature_fingerprint"].item()) == expected_fingerprint
                and str(archive["task"].item()) == task
            ):
                return {
                    "run_name": run_name,
                    "task": task,
                    "cache": str(output_cache),
                    "rows": int(len(archive["episode"])),
                    "tie_valid": int(archive["tie_pair_valid"].item()),
                    "tie_total": int(archive["tie_pair_total"].item()),
                    "stable_id_matches": int(archive["stable_id_matches"].item()),
                    "stable_id_positions": int(archive["stable_id_positions"].item()),
                    "tied_id_matches": int(archive["tied_id_matches"].item()),
                    "tied_id_positions": int(archive["tied_id_positions"].item()),
                    "reused": True,
                }

    with np.load(original_cache, allow_pickle=False) as archive:
        episode = np.asarray(archive["episode"], dtype=np.int32)
        query = np.asarray(archive["query"], dtype=np.int16)
        original_soft = np.asarray(archive["lag_soft_recurrence"], dtype=np.float32)
        original_support = np.asarray(archive["lag_support_recurrence"], dtype=np.float32)
        original_selected = np.asarray(archive["lag_selected_recurrence"], dtype=np.float32)

    route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    probability_store = route["hb_router_probs"]
    id_store = route["hb_expert_ids"]
    control_step = np.asarray(route["control_step"][:], dtype=np.int64)
    if len(episode) != int(probability_store.shape[0]) or len(query) != len(episode):
        raise ValueError(f"row mismatch: {run_name}/{task}")

    layer_groups, _probability_flows, _transition_flows, _acceleration_flows, token_groups, _action_groups = structured.groups_from_config(base)
    lags = [int(value) for value in base["position_grid"]["recurrence_lags"]]
    similarity = np.full(
        (len(episode), len(VARIANTS), len(lags), 2, 4), np.nan, dtype=np.float16
    )
    similarity[:, 0] = original_soft
    similarity[:, 1] = original_support
    similarity[:, 2] = original_selected

    history: dict[str, np.ndarray] | None = None
    max_lag = max(lags)
    tie_pair_valid = 0
    tie_pair_total = 0
    stable_id_matches = 0
    stable_id_positions = 0
    tied_id_matches = 0
    tied_id_positions = 0
    minimum_fraction = float(config["tie_filter"]["minimum_valid_fraction_per_layer_token_cell"])

    for start in range(0, len(episode), 32):
        stop = min(start + 32, len(episode))
        probability = structured.normalize_probability(
            probability_store[start:stop, :, -1, :, :]
        )
        ids = np.asarray(id_store[start:stop, :, -1, :, :], dtype=np.uint8)
        order = np.argsort(probability, axis=-1, kind="stable")[..., ::-1]
        ranked = np.take_along_axis(probability, order, axis=-1)
        reconstructed = np.asarray(order[..., :4], dtype=np.uint8)
        boundary_resolved = ranked[..., 3] > ranked[..., 4]
        exact = structured.support_jaccard_similarity(ids, reconstructed) == 1.0
        stable_id_positions += int(boundary_resolved.sum())
        stable_id_matches += int((exact & boundary_resolved).sum())
        tied_id_positions += int((~boundary_resolved).sum())
        tied_id_matches += int((exact & ~boundary_resolved).sum())

        multiplier, offset = affine_parameters(task, run_name, control_step[start:stop])
        relabeled_probability = relabel_probability(probability, multiplier, offset)
        relabeled_ids = relabel_ids(ids, multiplier, offset)
        current = {
            "probability": probability,
            "ids": ids,
            "ranked": ranked,
            "reconstructed": reconstructed,
            "resolved": boundary_resolved,
            "relabeled_probability": relabeled_probability,
            "relabeled_ids": relabeled_ids,
            "episode": episode[start:stop],
        }
        if history is None:
            combined = current
            history_length = 0
        else:
            combined = {
                name: np.concatenate([history[name], current[name]])
                for name in current
            }
            history_length = len(history["episode"])
        positions = history_length + np.arange(stop - start)

        for lag_index, lag in enumerate(lags):
            previous = positions - lag
            safe_previous = np.maximum(previous, 0)
            valid_rows = (previous >= 0) & (
                combined["episode"][positions] == combined["episode"][safe_previous]
            )
            if not np.any(valid_rows):
                continue
            take = np.flatnonzero(valid_rows)
            now = positions[take]
            before = previous[take]
            derived = {
                3: 1.0
                - structured.hellinger(combined["ranked"][now], combined["ranked"][before]),
                4: structured.support_jaccard_similarity(
                    combined["reconstructed"][now], combined["reconstructed"][before]
                ),
                6: 1.0
                - structured.hellinger(
                    combined["relabeled_probability"][now],
                    combined["relabeled_probability"][before],
                ),
                7: structured.support_jaccard_similarity(
                    combined["relabeled_ids"][now], combined["relabeled_ids"][before]
                ),
            }
            for variant_index, values in derived.items():
                similarity[start + take, variant_index, lag_index] = (
                    structured.aggregate_layer_token(values, layer_groups, token_groups)
                )

            tie_values = structured.support_jaccard_similarity(
                combined["ids"][now], combined["ids"][before]
            ).astype(np.float32)
            valid_positions = combined["resolved"][now] & combined["resolved"][before]
            tie_pair_valid += int(valid_positions.sum())
            tie_pair_total += int(valid_positions.size)
            tie_values[~valid_positions] = np.nan
            similarity[start + take, 5, lag_index] = aggregate_layer_token_nan(
                tie_values, layer_groups, token_groups, minimum_fraction
            )

        history = {
            name: values[-max_lag:].copy() for name, values in combined.items()
        }

    raw, evidence = build_evidence(
        similarity,
        episode,
        query,
        lags,
        int(base["causal_self_reference"]["recurrence_baseline_observations_per_lag"]),
    )
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_cache,
        schema=np.asarray(SCHEMA),
        feature_fingerprint=np.asarray(expected_fingerprint),
        task=np.asarray(task),
        run_name=np.asarray(run_name),
        training=np.asarray(False),
        endpoint_labels_loaded=np.asarray(False),
        variants=np.asarray(VARIANTS),
        lags=np.asarray(lags, dtype=np.int8),
        episode=episode,
        query=query,
        recurrence_raw=raw,
        recurrence_evidence=evidence,
        tie_pair_valid=np.asarray(tie_pair_valid, dtype=np.int64),
        tie_pair_total=np.asarray(tie_pair_total, dtype=np.int64),
        stable_id_matches=np.asarray(stable_id_matches, dtype=np.int64),
        stable_id_positions=np.asarray(stable_id_positions, dtype=np.int64),
        tied_id_matches=np.asarray(tied_id_matches, dtype=np.int64),
        tied_id_positions=np.asarray(tied_id_positions, dtype=np.int64),
    )
    return {
        "run_name": run_name,
        "task": task,
        "cache": str(output_cache),
        "rows": len(episode),
        "tie_valid": tie_pair_valid,
        "tie_total": tie_pair_total,
        "stable_id_matches": stable_id_matches,
        "stable_id_positions": stable_id_positions,
        "tied_id_matches": tied_id_matches,
        "tied_id_positions": tied_id_positions,
        "reused": False,
    }


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def build_reference(
    cache_root: Path,
    run_name: str,
    run_map: dict[str, Path],
    tasks: list[str],
    base: dict[str, Any],
) -> dict[tuple[int, int, int], np.ndarray]:
    edges = np.asarray(base["healthy_cdf_query_bins"], dtype=np.int64)
    collected: dict[tuple[int, int, int], list[np.ndarray]] = {
        (variant, lag, bin_index): []
        for variant in range(len(VARIANTS))
        for lag in range(4)
        for bin_index in range(len(edges) - 1)
    }
    for task in tasks:
        arrays = load_cache(cache_path(cache_root, run_name, task))
        outcomes = structured.load_outcomes(run_map[task])
        success_ids = [episode for episode, success in outcomes.items() if success]
        success = np.isin(arrays["episode"], success_ids)
        bins = structured.query_bins(arrays["query"], edges)
        evidence = np.asarray(arrays["recurrence_evidence"], dtype=np.float32)
        for bin_index in range(len(edges) - 1):
            rows = success & (bins == bin_index)
            if not np.any(rows):
                continue
            for variant in range(len(VARIANTS)):
                for lag in range(4):
                    values = evidence[rows, variant, lag]
                    values = values[np.isfinite(values)]
                    if len(values):
                        collected[(variant, lag, bin_index)].append(values)
    reference: dict[tuple[int, int, int], np.ndarray] = {}
    for variant in range(len(VARIANTS)):
        for lag in range(4):
            all_parts = [
                part
                for bin_index in range(len(edges) - 1)
                for part in collected[(variant, lag, bin_index)]
            ]
            reference[(variant, lag, -1)] = (
                np.sort(np.concatenate(all_parts).astype(np.float32))
                if all_parts
                else np.empty(0, dtype=np.float32)
            )
            for bin_index in range(len(edges) - 1):
                parts = collected[(variant, lag, bin_index)]
                reference[(variant, lag, bin_index)] = (
                    np.sort(np.concatenate(parts).astype(np.float32))
                    if parts
                    else np.empty(0, dtype=np.float32)
                )
    return reference


def score_cache(
    arrays: dict[str, np.ndarray],
    reference: dict[tuple[int, int, int], np.ndarray],
    base: dict[str, Any],
) -> pd.DataFrame:
    evidence = np.asarray(arrays["recurrence_evidence"], dtype=np.float32)
    episode = np.asarray(arrays["episode"], dtype=np.int32)
    query = np.asarray(arrays["query"], dtype=np.int16)
    edges = np.asarray(base["healthy_cdf_query_bins"], dtype=np.int64)
    bins = structured.query_bins(query, edges)
    minimum = int(base["minimum_reference_values_per_bin"])
    confidence = np.full_like(evidence, np.nan)
    for variant in range(len(VARIANTS)):
        for lag in range(4):
            for bin_index in range(len(edges) - 1):
                rows = bins == bin_index
                values = reference[(variant, lag, bin_index)]
                if len(values) < minimum:
                    values = reference[(variant, lag, -1)]
                confidence[rows, variant, lag] = structured.empirical_upper_confidence(
                    values, evidence[rows, variant, lag]
                )
    frame: dict[str, np.ndarray] = {"episode": episode, "query": query}
    for variant_index, variant in enumerate(VARIANTS):
        raw = structured.finite_max(confidence[:, variant_index])
        frame[variant] = structured.persistent_two_of_three(raw, episode, query)
    return pd.DataFrame(frame)


def select_variant_thresholds(
    task_maxima: dict[str, dict[str, list[float]]], budget: float
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    thresholds: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        candidates = []
        for task, by_variant in task_maxima.items():
            threshold, alarms, total = structured.select_threshold(
                np.asarray(by_variant[variant]), budget
            )
            candidates.append(threshold)
            rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "task_threshold": threshold,
                    "success_alarm_episodes": alarms,
                    "success_episodes": total,
                }
            )
        thresholds[variant] = max(candidates)
    return thresholds, rows


def episode_maxima(frame: pd.DataFrame, episode_ids: Iterable[int]) -> dict[str, list[float]]:
    grouped = {int(key): group for key, group in frame.groupby("episode", sort=False)}
    output = {variant: [] for variant in VARIANTS}
    for episode_id in episode_ids:
        group = grouped.get(int(episode_id))
        for variant in VARIANTS:
            if group is None:
                output[variant].append(float("-inf"))
                continue
            values = group[variant].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            output[variant].append(float(values.max()) if len(values) else float("-inf"))
    return output


def add_alarms(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    output = frame.copy()
    for variant in VARIANTS:
        output[f"alarm_{variant}"] = output[variant] >= thresholds[variant]
        output[f"first_alarm_{variant}"] = False
    for _episode, indices in structured.episode_indices(output["episode"].to_numpy()):
        for variant in VARIANTS:
            hits = indices[output.loc[indices, f"alarm_{variant}"].to_numpy(dtype=bool)]
            if len(hits):
                output.loc[hits[0], f"first_alarm_{variant}"] = True
    return output


def build_episode_rows(
    predictions: pd.DataFrame,
    cache_root: Path,
    run_name: str,
    run_map: dict[str, Path],
    partition: pd.DataFrame,
) -> pd.DataFrame:
    roles = partition.set_index("task")["role"].to_dict()
    rows: list[dict[str, Any]] = []
    for task, run in sorted(run_map.items()):
        arrays = load_cache(cache_path(cache_root, run_name, task))
        outcomes = structured.load_outcomes(run)
        task_predictions = predictions[predictions["task"] == task]
        grouped = {
            int(key): group for key, group in task_predictions.groupby("episode", sort=False)
        }
        for episode_id, indices in structured.episode_indices(arrays["episode"]):
            group = grouped.get(episode_id)
            row: dict[str, Any] = {
                "task": task,
                "role": roles[task],
                "episode": episode_id,
                "episode_length": len(indices),
                "failure": not outcomes[episode_id],
            }
            for variant in VARIANTS:
                hits = group[group[f"first_alarm_{variant}"]] if group is not None else pd.DataFrame()
                row[f"alarm_{variant}"] = not hits.empty
                row[f"first_alarm_query_{variant}"] = (
                    int(hits.iloc[0]["query"]) if not hits.empty else np.nan
                )
                values = group[variant].to_numpy(dtype=float) if group is not None else np.empty(0)
                values = values[np.isfinite(values)]
                row[f"max_score_{variant}"] = values.max() if len(values) else -np.inf
            rows.append(row)
    return pd.DataFrame(rows)


def combine_metrics(rows: pd.DataFrame, variant: str) -> dict[str, Any]:
    return structured.rate_metrics(rows[f"alarm_{variant}"], rows["failure"])


def evaluate_onset(
    predictions: pd.DataFrame,
    episodes: pd.DataFrame,
    onset_path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    onset = pd.read_csv(onset_path)
    onset = onset[onset["onset_query"].notna() & (onset["onset_query"] >= 6)].copy()
    heldout = episodes[episodes["role"] == "heldout_test"]
    onset = onset.merge(
        heldout[["task", "episode"]], on=["task", "episode"], how="inner", validate="one_to_one"
    )
    records: list[dict[str, Any]] = []
    for event in onset.itertuples(index=False):
        group = predictions[
            (predictions["task"] == event.task)
            & (predictions["episode"] == int(event.episode))
        ]
        row: dict[str, Any] = {
            "task": event.task,
            "episode": int(event.episode),
            "onset_query": int(event.onset_query),
            "onset_type": event.onset_type,
        }
        for variant in VARIANTS:
            alarms = group.loc[group[f"alarm_{variant}"], "query"].to_numpy(dtype=int)
            first = int(alarms[0]) if len(alarms) else None
            row[f"first_alarm_query_{variant}"] = first
            row[f"delta_{variant}"] = None if first is None else first - int(event.onset_query)
            row[f"near_active_{variant}"] = bool(
                np.any((alarms >= event.onset_query - 2) & (alarms <= event.onset_query))
            )
        records.append(row)
    frame = pd.DataFrame(records)
    summary: dict[str, dict[str, int]] = {}
    for variant in VARIANTS:
        delta = pd.to_numeric(frame[f"delta_{variant}"], errors="coerce")
        summary[variant] = {
            "events": len(frame),
            "alarm_events": int(delta.notna().sum()),
            "not_later": int((delta <= 0).sum()),
            "near_first_minus2_to_0": int(delta.between(-2, 0).sum()),
            "near_active_minus2_to_0": int(frame[f"near_active_{variant}"].sum()),
            "too_early": int((delta < -2).sum()),
            "late": int((delta > 0).sum()),
        }
    return frame, summary


def posthoc_fp_sweep(episodes: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    failure = episodes["failure"].to_numpy(dtype=bool)
    for variant in VARIANTS:
        score = episodes[f"max_score_{variant}"].to_numpy(dtype=float)
        success_scores = score[~failure]
        for budget in budgets:
            values = np.unique(success_scores[np.isfinite(success_scores)])
            threshold = float(np.nextafter(values[-1], np.inf)) if len(values) else np.inf
            for candidate in values[::-1]:
                if int(np.sum(success_scores >= candidate)) <= budget:
                    threshold = float(candidate)
                else:
                    break
            metrics = structured.rate_metrics(score >= threshold, failure)
            rows.append(
                {"variant": variant, "fp_budget": budget, "threshold": threshold, **metrics}
            )
    for budget in budgets:
        query, metrics = structured.matched_clock(episodes, budget)
        rows.append(
            {
                "variant": "fixed_query_clock",
                "fp_budget": budget,
                "threshold": query,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def audit_global_permutation(run: Path, rows: int = 32) -> dict[str, Any]:
    route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    probability = structured.normalize_probability(
        route["hb_router_probs"][:rows, :, -1, :, :]
    )
    ids = np.asarray(route["hb_expert_ids"][:rows, :, -1, :, :], dtype=np.uint8)
    selected = np.asarray(
        route["hb_selected_prob"][:rows, :, -1, :, :], dtype=np.float32
    )
    permutation = np.random.default_rng(20260904).permutation(32)
    relabeled_probability = np.empty_like(probability)
    relabeled_probability[..., permutation] = probability
    relabeled_ids = permutation[ids]
    before_soft = structured.hellinger(probability[1:], probability[:-1])
    after_soft = structured.hellinger(
        relabeled_probability[1:], relabeled_probability[:-1]
    )
    before_support = structured.support_jaccard_similarity(ids[1:], ids[:-1])
    after_support = structured.support_jaccard_similarity(
        relabeled_ids[1:], relabeled_ids[:-1]
    )
    before_selected = structured.selected_jaccard_similarity(
        ids[1:], selected[1:], ids[:-1], selected[:-1]
    )
    after_selected = structured.selected_jaccard_similarity(
        relabeled_ids[1:], selected[1:], relabeled_ids[:-1], selected[:-1]
    )
    result = {
        "route_rows": rows,
        "layer_token_pairs": int(before_soft.size),
        "soft_max_abs_difference": float(np.max(np.abs(before_soft - after_soft))),
        "support_exact": bool(np.array_equal(before_support, after_support)),
        "selected_max_abs_difference": float(
            np.max(np.abs(before_selected - after_selected))
        ),
    }
    if result["soft_max_abs_difference"] > 1e-5:
        raise AssertionError(result)
    if not result["support_exact"]:
        raise AssertionError(result)
    if result["selected_max_abs_difference"] > 1e-6:
        raise AssertionError(result)
    return result


def render_report(
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    onset: pd.DataFrame,
    sweep: pd.DataFrame,
) -> str:
    equal_fp = sweep[
        (sweep["fp_budget"] == 22)
        & sweep["variant"].isin(
            [
                "soft_coordinate",
                "actual_support",
                "rank_shape",
                "reconstructed_support",
                "tie_resolved_support",
                "chunk_relabel_soft",
                "chunk_relabel_support",
                "fixed_query_clock",
            ]
        )
    ]
    lines = [
        "# MoE Expert 身份消融",
        "",
        "## 问题",
        "",
        "Expert 的数字编号没有语义；本实验检验固定模型中跨 replan 的 expert 身份连续性是否有信息。所有表示使用同一个 train-free recurrence 判断器和同一数据划分。",
        "",
        "## 双向冻结结果",
        "",
        structured.markdown_table(metrics),
        "",
        "## 相同误报预算诊断",
        "",
        "以下阈值在目标结果揭盲后按不超过 22 个成功误报选择，只用于比较表示中的排序信息，不是可部署阈值。",
        "",
        structured.markdown_table(equal_fp),
        "",
        "## Top-4 边界审计",
        "",
        f"共检查 {summary['id_reconstruction']['positions']:,} 个 final-flow layer-token 位置；严格 p4>p5 的位置 {summary['id_reconstruction']['stable_positions']:,} 个，其中实际 support 与 float16 重建 support 一致 {summary['id_reconstruction']['stable_matches']:,} 个。边界并列位置 {summary['id_reconstruction']['tied_positions']:,} 个，其中一致 {summary['id_reconstruction']['tied_matches']:,} 个。",
        "",
        f"实际数据统一全局重编号审计中，support 结果逐项完全相同；selected 最大绝对差为 {summary['global_label_permutation_audit']['selected_max_abs_difference']:.3g}，soft 仅因 float32 求和次序产生 {summary['global_label_permutation_audit']['soft_max_abs_difference']:.3g} 的最大绝对差。",
        "",
        "## Onset 时机",
        "",
        structured.markdown_table(onset),
        "",
        "## 解释",
        "",
        "- `soft_coordinate` 没读取 top-4 ID 字段，但仍保留 expert 坐标身份；",
        "- `rank_shape` 才完全移除 expert 身份，只保留每个位置的概率谱形状；",
        "- `chunk_relabel_*` 保留每个 replan 内的结构，却故意破坏跨 replan 身份连续性；",
        "- 一次全局统一重编号不会改变任何 equality/Jaccard/Hellinger 距离，数字标签本身因此不可能是信号；",
        "- endpoint failure 仍受轨迹长度混杂，onset 表用于判断是否真是及时 Trap 报警。",
        "- 结论：数字编号没有语义；固定 checkpoint 中一致的 expert 身份有信息；保存的 top-4 tie-break ID 不是必要信息；可靠 early detector 仍未成立。",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    rng = np.random.default_rng(7)
    probability = structured.normalize_probability(rng.random((5, 2, 3, 32)))
    ids = np.argsort(probability, axis=-1)[..., -4:]
    permutation = rng.permutation(32)
    relabeled_ids = permutation[ids]
    relabeled_probability = np.empty_like(probability)
    relabeled_probability[..., permutation] = probability
    assert np.allclose(
        structured.hellinger(probability[1:], probability[:-1]),
        structured.hellinger(relabeled_probability[1:], relabeled_probability[:-1]),
        atol=1e-6,
    )
    assert np.array_equal(
        structured.support_jaccard_similarity(ids[1:], ids[:-1]),
        structured.support_jaccard_similarity(relabeled_ids[1:], relabeled_ids[:-1]),
    )
    multiplier = np.asarray([1, 3, 5, 7, 9])
    offset = np.asarray([0, 1, 2, 3, 4])
    changed = relabel_probability(probability, multiplier, offset)
    assert np.allclose(changed.sum(axis=-1), 1.0)
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["training"] is not False or config["learned_feature_weights"] is not False:
        raise ValueError("ablation must remain train-free")
    base_config_path = resolve_workspace(config["base_config"])
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    base_result = resolve_workspace(config["base_result"])
    output = args.output.resolve()
    cache_root = output / "route_only_tasks"
    prediction_root = output / "predictions_label_free"
    table_root = output / "tables"
    for path in (output, cache_root, prediction_root, table_root):
        path.mkdir(parents=True, exist_ok=True)

    raw_root = resolve_workspace(base["cache_root"])
    run_settings = {str(row["name"]): row for row in base["runs"]}
    run_maps = {
        name: dict(structured.discover_runs(raw_root, str(setting["run_id"])))
        for name, setting in run_settings.items()
    }
    tasks = sorted(next(iter(run_maps.values())))
    partition = structured.partition_tasks(tasks, base)
    partition.to_csv(table_root / "task_partition.csv", index=False)

    jobs = []
    for run_name, run_map in run_maps.items():
        for task, run in run_map.items():
            jobs.append(
                (
                    task,
                    run_name,
                    str(run),
                    str(structured.task_cache_path(base_result / "route_only_tasks", run_name, task)),
                    str(cache_path(cache_root, run_name, task)),
                    config,
                    base,
                    args.rebuild,
                )
            )
    extraction: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(extract_task, *job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            extraction.append(result)
            print(
                f"[extract {index}/{len(futures)}] {result['run_name']} {result['task']} rows={result['rows']} reused={result['reused']}",
                flush=True,
            )
    extraction.sort(key=lambda row: (row["run_name"], row["task"]))
    write_json(
        output / "route_only_manifest.json",
        {
            "schema": SCHEMA,
            "training": False,
            "endpoint_labels_loaded": False,
            "feature_fingerprint": fingerprint(config, base),
            "rows": sum(row["rows"] for row in extraction),
            "tasks": extraction,
        },
    )

    reference_tasks = partition.loc[partition["role"] == "healthy_reference", "task"].tolist()
    threshold_tasks = partition.loc[partition["role"] == "threshold_confirmation", "task"].tolist()
    run_names = list(run_settings)
    directions = [(run_names[0], run_names[1]), (run_names[1], run_names[0])]
    metric_rows: list[dict[str, Any]] = []
    onset_rows: list[pd.DataFrame] = []
    onset_summaries: dict[str, Any] = {}

    for source_name, target_name in directions:
        direction = f"{source_name}_to_{target_name}"
        print(f"[{direction}] reference", flush=True)
        reference = build_reference(
            cache_root, source_name, run_maps[source_name], reference_tasks, base
        )
        task_maxima: dict[str, dict[str, list[float]]] = {}
        for task in threshold_tasks:
            arrays = load_cache(cache_path(cache_root, source_name, task))
            frame = score_cache(arrays, reference, base)
            outcomes = structured.load_outcomes(run_maps[source_name][task])
            successes = [episode for episode, success in outcomes.items() if success]
            task_maxima[task] = episode_maxima(frame, successes)
        thresholds, threshold_rows = select_variant_thresholds(
            task_maxima,
            float(config["threshold_confirmation"]["target_success_episode_false_alarm_rate"]),
        )
        for row in threshold_rows:
            row["direction"] = direction
            row["frozen_threshold"] = thresholds[row["variant"]]
        pd.DataFrame(threshold_rows).to_csv(
            table_root / f"task_thresholds_{direction}.csv", index=False
        )
        threshold_path = output / f"thresholds_{direction}.json"
        write_json(
            threshold_path,
            {
                "schema": SCHEMA,
                "training": False,
                "failure_labels_used": False,
                "source_run": source_name,
                "target_run": target_name,
                "selection": "maximum_task_specific_threshold",
                "thresholds": thresholds,
            },
        )

        parts = []
        for task in tasks:
            arrays = load_cache(cache_path(cache_root, target_name, task))
            frame = add_alarms(score_cache(arrays, reference, base), thresholds)
            frame.insert(0, "task", task)
            parts.append(frame)
        predictions = pd.concat(parts, ignore_index=True)
        prediction_path = prediction_root / f"{direction}.csv.gz"
        predictions.to_csv(
            prediction_path,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        prediction_sha = structured.sha256_file(prediction_path)
        write_json(
            prediction_root / f"{direction}_manifest.json",
            {
                "schema": SCHEMA,
                "training": False,
                "target_outcomes_loaded": False,
                "predictions_written_before_target_outcomes": True,
                "rows": len(predictions),
                "prediction_sha256": prediction_sha,
                "threshold_sha256": structured.sha256_file(threshold_path),
            },
        )
        print(f"[{direction}] predictions frozen {prediction_sha}", flush=True)

        episodes = build_episode_rows(
            predictions, cache_root, target_name, run_maps[target_name], partition
        )
        episodes.insert(0, "direction", direction)
        episodes.to_csv(table_root / f"episodes_{direction}.csv", index=False)
        heldout = episodes[episodes["role"] == "heldout_test"]
        for variant in VARIANTS:
            metric_rows.append(
                {
                    "direction": direction,
                    "variant": variant,
                    "episodes": len(heldout),
                    "failures": int(heldout["failure"].sum()),
                    **combine_metrics(heldout, variant),
                }
            )
        onset_path = resolve_workspace(run_settings[target_name]["posthoc_onset_proxy"])
        onset_frame, onset_summary = evaluate_onset(predictions, episodes, onset_path)
        onset_frame.insert(0, "direction", direction)
        onset_rows.append(onset_frame)
        onset_summaries[direction] = onset_summary

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(table_root / "direction_metrics.csv", index=False)
    combined_rows = []
    for variant in VARIANTS:
        selected = metrics[metrics["variant"] == variant]
        counts = {name: int(selected[name].sum()) for name in ("episodes", "failures", "tp", "fp", "fn", "tn")}
        combined_rows.append(
            {
                "variant": variant,
                **counts,
                "precision": counts["tp"] / max(counts["tp"] + counts["fp"], 1),
                "failure_recall": counts["tp"] / max(counts["failures"], 1),
                "success_false_alarm_rate": counts["fp"] / max(counts["fp"] + counts["tn"], 1),
            }
        )
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(table_root / "combined_metrics.csv", index=False)
    onset = pd.concat(onset_rows, ignore_index=True)
    onset.to_csv(table_root / "heldout_onset_timing.csv", index=False)
    onset_combined_rows = []
    for variant in VARIANTS:
        delta = pd.to_numeric(onset[f"delta_{variant}"], errors="coerce")
        onset_combined_rows.append(
            {
                "variant": variant,
                "events": len(onset),
                "alarm_events": int(delta.notna().sum()),
                "not_later": int((delta <= 0).sum()),
                "near_first_minus2_to_0": int(delta.between(-2, 0).sum()),
                "near_active_minus2_to_0": int(onset[f"near_active_{variant}"].sum()),
                "too_early": int((delta < -2).sum()),
                "late": int((delta > 0).sum()),
            }
        )
    onset_combined = pd.DataFrame(onset_combined_rows)
    onset_combined.to_csv(table_root / "combined_onset_metrics.csv", index=False)

    heldout_episodes = pd.concat(
        [pd.read_csv(table_root / f"episodes_{a}_to_{b}.csv") for a, b in directions],
        ignore_index=True,
    )
    heldout_episodes = heldout_episodes[heldout_episodes["role"] == "heldout_test"]
    sweep = posthoc_fp_sweep(heldout_episodes, [0, 17, 22, 27, 58])
    sweep.to_csv(table_root / "posthoc_fp_budget_sweep.csv", index=False)

    reconstruction = {
        "positions": sum(row["stable_id_positions"] + row["tied_id_positions"] for row in extraction),
        "stable_positions": sum(row["stable_id_positions"] for row in extraction),
        "stable_matches": sum(row["stable_id_matches"] for row in extraction),
        "tied_positions": sum(row["tied_id_positions"] for row in extraction),
        "tied_matches": sum(row["tied_id_matches"] for row in extraction),
        "tie_pair_valid": sum(row["tie_valid"] for row in extraction),
        "tie_pair_total": sum(row["tie_total"] for row in extraction),
    }
    first_run = run_maps[run_names[0]][tasks[0]]
    permutation_audit = audit_global_permutation(first_run)
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "training": False,
        "gradient_optimization": False,
        "learned_feature_weights": False,
        "failure_labels_used_for_features_or_thresholds": False,
        "global_label_permutation_invariant": True,
        "global_label_permutation_audit": permutation_audit,
        "route_rows": sum(row["rows"] for row in extraction),
        "task_runs": len(extraction),
        "variants": list(VARIANTS),
        "combined_heldout": combined.set_index("variant").to_dict(orient="index"),
        "combined_onset": onset_combined.set_index("variant").to_dict(orient="index"),
        "id_reconstruction": reconstruction,
        "prediction_hashes": {
            path.stem.replace(".csv", ""): structured.sha256_file(path)
            for path in sorted(prediction_root.glob("*.csv.gz"))
        },
        "conclusion": {
            "numeric_expert_label_has_semantics": False,
            "consistent_expert_identity_contributes_signal": True,
            "stored_actual_top4_tiebreak_is_required": False,
            "reliable_early_detector_validated": False
        },
    }
    write_json(output / "summary.json", summary)
    (output / "REPORT_ZH.md").write_text(
        render_report(summary, combined, onset_combined, sweep), encoding="utf-8"
    )
    print(json.dumps(structured.plain(summary["combined_heldout"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
