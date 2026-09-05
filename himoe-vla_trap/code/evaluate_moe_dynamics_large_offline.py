#!/usr/bin/env python3
"""Strict offline replay of the frozen MoE-dynamics alarm.

Prediction and evaluation are deliberately separated.  The prediction pass opens
only HB router probabilities, episode-boundary ids, the frozen healthy route bank,
and its healthy-only calibration.  Outcome and physical labels are loaded only
after every prediction has been produced.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import zarr

from moe_dynamics_online_selector import (
    DynamicsCalibration,
    MoeDynamicsOnlineAlarm,
    SELECTOR_VERSION,
)
from moe_only_online_selector import HealthySequenceBank


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
REFERENCE = (
    PACKAGE_ROOT
    / "results/moe_only_online_alarm/h20_healthy_route_sequences_seed20260905.npz"
)
CALIBRATION = (
    PACKAGE_ROOT / "results/moe_dynamics_online_alarm/calibration_v2/calibration.json"
)
OUTPUT = PACKAGE_ROOT / "results/moe_dynamics_online_alarm/large_offline_replay"

A_ROOT = (
    WORKSPACE_ROOT
    / "himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
)
A_STORE = A_ROOT / "formal/server/routes.zarr"
A_LABELS = A_ROOT / "analysis/candidate_physical_labels.csv"

B_ROOT = (
    WORKSPACE_ROOT
    / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
)
B_STORE = B_ROOT / "server/routes.zarr"
B_LABELS = B_ROOT / "client/summaries.json"

C_ROOT = (
    WORKSPACE_ROOT
    / "VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_long/"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-50x8-20260903"
)
C_STORE = C_ROOT / "server/routes.zarr"
C_LABELS = C_ROOT / "client/summaries.json"

P_ROOT = (
    PACKAGE_ROOT
    / "results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908"
)
P_LABELS = (
    PACKAGE_ROOT
    / "results/moe_dynamics_online_alarm/prospective_audit/episode_audit.csv"
)

ROUTE_SHAPE = (8, 10, 11, 32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_list(values: Iterable[int]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def contiguous_episode_rows(episode_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Return temporally ordered row indices for every episode id."""

    order = np.argsort(episode_ids, kind="stable")
    ordered_ids = episode_ids[order]
    unique, starts = np.unique(ordered_ids, return_index=True)
    stops = np.r_[starts[1:], len(order)]
    return {
        int(identity): np.sort(order[start:stop])
        for identity, start, stop in zip(unique, starts, stops)
    }


def replay_routes(
    corpus: str,
    episodes: Iterable[tuple[int, np.ndarray]],
    bank: HealthySequenceBank,
    calibration: DynamicsCalibration,
    query_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    workers: int,
) -> None:
    """Run prediction from route tensors; this function has no label argument."""

    def score_episode(
        item: tuple[int, np.ndarray],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        episode_id, routes = item
        routes = np.asarray(routes)
        if routes.ndim != 5 or tuple(routes.shape[1:]) != ROUTE_SHAPE:
            raise RuntimeError(
                f"{corpus}/{episode_id}: unexpected route shape {routes.shape}"
            )
        selector = MoeDynamicsOnlineAlarm(bank, calibration)
        local_query_rows: list[dict[str, Any]] = []
        raw_queries: list[int] = []
        alarm_queries: list[int] = []
        scores: list[float] = []
        for route in routes:
            decision = selector.update(route)
            if decision is None:
                continue
            record = asdict(decision)
            record["matched_reference_queries"] = json_list(
                record["matched_reference_queries"]
            )
            local_query_rows.append(
                {"corpus": corpus, "episode_id": episode_id, **record}
            )
            scores.append(decision.normalized_acceleration_excess)
            if decision.raw_reject:
                raw_queries.append(decision.query)
            if decision.alarm:
                alarm_queries.append(decision.query)
        episode_row = {
            "corpus": corpus,
            "episode_id": episode_id,
            "queries": len(routes),
            "score_max": max(scores, default=float("nan")),
            "raw_reject_queries": json_list(raw_queries),
            "alarm_queries": json_list(alarm_queries),
            "any_raw_reject": bool(raw_queries),
            "formal_alarm": bool(alarm_queries),
        }
        return local_query_rows, episode_row

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        scored = executor.map(score_episode, episodes)
        for ordinal, (local_queries, episode_row) in enumerate(scored, start=1):
            query_rows.extend(local_queries)
            episode_rows.append(episode_row)
            if ordinal % 25 == 0:
                print(
                    f"{corpus}: replayed {ordinal} episodes in "
                    f"{time.time() - started:.1f}s",
                    flush=True,
                )


def replay_zarr_corpus(
    corpus: str,
    store_path: Path,
    bank: HealthySequenceBank,
    calibration: DynamicsCalibration,
    query_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    workers: int,
) -> None:
    """Open only route telemetry and episode boundaries during prediction."""

    store = zarr.open_group(str(store_path), mode="r")
    episode_ids = np.asarray(store["episode_id"][:], dtype=np.int64)
    rows_by_episode = contiguous_episode_rows(episode_ids)
    # Corpus A has 22 known one-row interrupted captures.  A selector requiring
    # persistence=2 cannot evaluate them; the complete corpus has >= 33 queries.
    eligible = [
        (identity, rows)
        for identity, rows in sorted(rows_by_episode.items())
        if len(rows) >= 3
    ]
    router = store["hb_router_probs"]

    def episodes() -> Iterable[tuple[int, np.ndarray]]:
        for identity, rows in eligible:
            # Zarr v3 basic indexing is reliable for contiguous blocks; corpus A
            # is interleaved, so gather its rows explicitly in temporal order.
            if np.all(np.diff(rows) == 1):
                route = np.asarray(router[int(rows[0]) : int(rows[-1]) + 1])
            else:
                route = np.stack([np.asarray(router[int(row)]) for row in rows])
            yield identity, route

    replay_routes(
        corpus, episodes(), bank, calibration, query_rows, episode_rows, workers
    )


def prospective_id(path: Path) -> int:
    try:
        return int(path.name.split("_", 2)[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"cannot parse episode index from {path.name}") from error


def replay_prospective(
    bank: HealthySequenceBank,
    calibration: DynamicsCalibration,
    query_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    workers: int,
) -> None:
    """Replay saved NPZ files while opening only their routing array."""

    def episodes() -> Iterable[tuple[int, np.ndarray]]:
        for episode_dir in sorted(P_ROOT.glob("episode_*")):
            with np.load(
                episode_dir / "trajectory_and_routes.npz", allow_pickle=False
            ) as archive:
                routes = np.asarray(archive["hb_router_probs"])
            yield prospective_id(episode_dir), routes

    replay_routes(
        "prospective",
        episodes(),
        bank,
        calibration,
        query_rows,
        episode_rows,
        workers,
    )


def load_labels() -> dict[tuple[str, int], dict[str, Any]]:
    """Load evaluation-only labels after prediction has completely finished."""

    labels: dict[tuple[str, int], dict[str, Any]] = {}
    with A_LABELS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            episode_id = int(row["episode_id"])
            labels[("A", episode_id)] = {
                "success": parse_bool(row["success"]),
                "failure_family": row["primary_failure_type"],
                "physical_labels": row["physical_labels"],
                "label_loop_or_cycling": parse_bool(row["label_loop_or_cycling"]),
                "label_stagnation": parse_bool(row["label_stagnation"]),
                "label_goal_contact_near_miss": parse_bool(
                    row["label_goal_contact_near_miss"]
                ),
                "init_state_id": int(row["init_state_id"]),
                "flow_noise_seed": None,
                "condition_present_in_old_B": None,
            }
    b_rows = json.loads(B_LABELS.read_text(encoding="utf-8"))
    old_b_conditions = {
        (int(row["init_state_id"]), int(row["flow_noise_seed"]))
        for row in b_rows
    }
    for row in b_rows:
        episode_id = int(row["episode_index"])
        success = bool(row["success"])
        labels[("B", episode_id)] = {
            "success": success,
            "failure_family": "success" if success else "outcome_failure",
            "physical_labels": "not_available",
            "label_loop_or_cycling": None,
            "label_stagnation": None,
            "label_goal_contact_near_miss": None,
            "init_state_id": int(row["init_state_id"]),
            "flow_noise_seed": int(row["flow_noise_seed"]),
            "condition_present_in_old_B": True,
        }
    c_rows = json.loads(C_LABELS.read_text(encoding="utf-8"))
    for row in c_rows:
        episode_id = int(row["episode_index"])
        success = bool(row["success"])
        condition = (int(row["init_state_id"]), int(row["flow_noise_seed"]))
        labels[("C_cache_new", episode_id)] = {
            "success": success,
            "failure_family": "success" if success else "outcome_failure",
            "physical_labels": "not_available",
            "label_loop_or_cycling": None,
            "label_stagnation": None,
            "label_goal_contact_near_miss": None,
            "init_state_id": condition[0],
            "flow_noise_seed": condition[1],
            "condition_present_in_old_B": condition in old_b_conditions,
        }
    with P_LABELS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            episode_id = int(row["episode_index"])
            labels[("prospective", episode_id)] = {
                "success": parse_bool(row["success"]),
                "failure_family": row["failure_family"],
                "physical_labels": row["generic_labels"],
                "label_loop_or_cycling": "loop_or_cycling" in row["generic_labels"],
                "label_stagnation": "stagnation" in row["generic_labels"],
                "label_goal_contact_near_miss": (
                    row["failure_family"] == "target_missed_grasp_proxy"
                ),
                "init_state_id": int(row["init_state_id"]),
                "flow_noise_seed": None,
                "condition_present_in_old_B": None,
            }
    return labels


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for corpus in ("A", "B", "C_cache_new", "prospective"):
        selected = [row for row in rows if row["corpus"] == corpus]
        successes = [row for row in selected if row["success"]]
        failures = [row for row in selected if not row["success"]]
        result[corpus] = {
            "episodes": len(selected),
            "queries": sum(int(row["queries"]) for row in selected),
            "successes": len(successes),
            "failures": len(failures),
            "failure_episode_recall": ratio(
                sum(bool(row["formal_alarm"]) for row in failures), len(failures)
            ),
            "success_episode_false_alarm": ratio(
                sum(bool(row["formal_alarm"]) for row in successes), len(successes)
            ),
            "failure_any_raw_reject": ratio(
                sum(bool(row["any_raw_reject"]) for row in failures), len(failures)
            ),
            "success_any_raw_reject": ratio(
                sum(bool(row["any_raw_reject"]) for row in successes), len(successes)
            ),
        }
    return result


def aggregate_cache_new_conditions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["corpus"] == "C_cache_new"]
    output: dict[str, Any] = {}
    groups = {
        "all": selected,
        "condition_not_in_old_B": [
            row for row in selected if not row["condition_present_in_old_B"]
        ],
        "condition_also_in_old_B": [
            row for row in selected if row["condition_present_in_old_B"]
        ],
    }
    for name, group in groups.items():
        successes = [row for row in group if row["success"]]
        failures = [row for row in group if not row["success"]]
        output[name] = {
            "episodes": len(group),
            "successes": len(successes),
            "failures": len(failures),
            "failure_episode_recall": ratio(
                sum(bool(row["formal_alarm"]) for row in failures), len(failures)
            ),
            "success_episode_false_alarm": ratio(
                sum(bool(row["formal_alarm"]) for row in successes), len(successes)
            ),
        }
    return output


def aggregate_families(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["corpus"], row["failure_family"])].append(row)
    output: dict[str, Any] = {}
    for (corpus, family), selected in sorted(grouped.items()):
        output[f"{corpus}/{family}"] = {
            "episodes": len(selected),
            "formal_alarms": sum(bool(row["formal_alarm"]) for row in selected),
            "alarm_rate": sum(bool(row["formal_alarm"]) for row in selected)
            / len(selected),
        }
    return output


def prospective_online_parity(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {
        int(row["episode_id"]): row
        for row in predictions
        if row["corpus"] == "prospective"
    }
    if not by_id:
        return {
            "episodes": 0,
            "exact_raw_and_formal_alarm_sequences": 0,
            "mismatch_episode_ids": [],
        }
    exact = 0
    mismatches: list[int] = []
    for episode_dir in sorted(P_ROOT.glob("episode_*")):
        episode_id = prospective_id(episode_dir)
        online = json.loads(
            (episode_dir / "summary.json").read_text(encoding="utf-8")
        )
        offline = by_id[episode_id]
        offline_raw = json.loads(offline["raw_reject_queries"])
        offline_alarm = json.loads(offline["alarm_queries"])
        matches = (
            offline_raw == [int(value) for value in online["raw_reject_queries"]]
            and offline_alarm == [int(value) for value in online["alarm_queries"]]
        )
        exact += int(matches)
        if not matches:
            mismatches.append(episode_id)
    return {
        "episodes": len(by_id),
        "exact_raw_and_formal_alarm_sequences": exact,
        "mismatch_episode_ids": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=("A", "B", "C_cache_new", "prospective"),
        default=("A", "B", "C_cache_new", "prospective"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    reference = REFERENCE.resolve()
    bank = HealthySequenceBank.load(reference)
    calibration = DynamicsCalibration.load(CALIBRATION.resolve(), reference, bank)

    # Phase 1: prediction.  No outcome, physical, action, state, reward, or
    # gripper file is opened before all selected corpora have been scored.
    query_predictions: list[dict[str, Any]] = []
    episode_predictions: list[dict[str, Any]] = []
    if "A" in args.corpora:
        replay_zarr_corpus(
            "A",
            A_STORE,
            bank,
            calibration,
            query_predictions,
            episode_predictions,
            args.workers,
        )
    if "B" in args.corpora:
        replay_zarr_corpus(
            "B",
            B_STORE,
            bank,
            calibration,
            query_predictions,
            episode_predictions,
            args.workers,
        )
    if "C_cache_new" in args.corpora:
        replay_zarr_corpus(
            "C_cache_new",
            C_STORE,
            bank,
            calibration,
            query_predictions,
            episode_predictions,
            args.workers,
        )
    if "prospective" in args.corpora:
        replay_prospective(
            bank,
            calibration,
            query_predictions,
            episode_predictions,
            args.workers,
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "query_predictions_moe_only.csv", query_predictions)
    write_csv(output / "episode_predictions_moe_only.csv", episode_predictions)

    # Phase 2: evaluation.  Labels cannot affect already materialized decisions.
    labels = load_labels()
    evaluated: list[dict[str, Any]] = []
    for prediction in episode_predictions:
        key = (str(prediction["corpus"]), int(prediction["episode_id"]))
        if key not in labels:
            raise RuntimeError(f"missing post-hoc label for {key}")
        evaluated.append({**prediction, **labels[key]})
    write_csv(output / "episode_evaluation_posthoc.csv", evaluated)

    parity = prospective_online_parity(episode_predictions)
    summary = {
        "schema": "himoe.moe_dynamics_large_offline_replay.v1",
        "selector_version": SELECTOR_VERSION,
        "training": False,
        "gradient_optimization": False,
        "runtime_selector_dynamic_input": "hb_router_probs only",
        "runtime_selector_reference": "five known-healthy HB route sequences",
        "runtime_selector_uses_action_values": False,
        "runtime_selector_uses_robot_or_sim_state": False,
        "runtime_selector_uses_physical_distance": False,
        "runtime_selector_uses_gripper_events": False,
        "runtime_selector_uses_reward_or_success": False,
        "episode_id_role": "episode-boundary reset only",
        "labels_loaded_after_predictions_written": True,
        "healthy_reference_selection_uses_known_success": True,
        "failure_labels_used_to_set_threshold": False,
        "feature_design_informed_by_prior_failure_analysis": True,
        "retrospective_corpora": [
            corpus for corpus in args.corpora if corpus in {"A", "B", "C_cache_new"}
        ],
        "independent_prospective_corpus": (
            "prospective" if "prospective" in args.corpora else None
        ),
        "threshold": calibration.threshold,
        "persistence": calibration.persistence,
        "aggregate": aggregate(evaluated),
        "cache_new_condition_breakdown": aggregate_cache_new_conditions(evaluated),
        "failure_family_breakdown": aggregate_families(evaluated),
        "prospective_offline_vs_online_parity": parity,
        "interpretation_guardrails": [
            "A and B are large retrospective stress tests, not independent validation.",
            "The prospective set was collected after v2 was frozen and is the independent test.",
            "Outcome and physical labels are post-hoc evaluation data, never selector inputs.",
            "Known-success metadata was used to construct the healthy reference bank.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
