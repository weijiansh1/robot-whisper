#!/usr/bin/env python3
"""Replay the MoE-only selector and run healthy leave-one-out checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from moe_only_online_selector import HealthySequenceBank, MoeOnlyOnlineAlarm


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/moe_only_online_alarm.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/moe_only_online_alarm/offline_replay"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_sequence(
    routes: np.ndarray,
    bank: HealthySequenceBank,
    persistence: int,
    identity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    selector = MoeOnlyOnlineAlarm(bank, persistence=persistence, max_advance=2)
    rows = []
    raw_queries = []
    alarm_queries = []
    for route in routes:
        decision = selector.update(route)
        if decision is None:
            continue
        row = {**identity, **decision.to_dict()}
        row["matched_reference_queries"] = json.dumps(
            list(decision.matched_reference_queries), separators=(",", ":")
        )
        rows.append(row)
        if decision.raw_reject:
            raw_queries.append(decision.query)
        if decision.alarm:
            alarm_queries.append(decision.query)
    return rows, raw_queries, alarm_queries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert config["training"] is False
    assert config["physical_inputs_to_selector"] is False
    reference_path = (PACKAGE_ROOT / config["healthy_reference"]).resolve()
    persistence = int(config["alarm_persistence"])
    output = args.output.resolve()

    query_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    bank = HealthySequenceBank.load(reference_path)
    for run_spec in config["retrospective_runs"]:
        run_path = (PACKAGE_ROOT / run_spec["path"]).resolve()
        manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
        for episode in manifest["episodes"]:
            with np.load(
                run_path / episode["trajectory_npz"], allow_pickle=False
            ) as archive:
                routes = np.asarray(archive["hb_router_probs"], dtype=np.float32)
            identity = {
                "evaluation_role": "online_route_replay",
                "run_role": run_spec["role"],
                "episode_index": int(episode["episode_index"]),
                "init_state_id": int(episode["init_state_id"]),
                "success": bool(episode["success"]),
            }
            rows, raw_queries, alarm_queries = evaluate_sequence(
                routes, bank, persistence, identity
            )
            query_rows.extend(rows)
            episode_rows.append({
                **identity,
                "queries": len(routes),
                "raw_reject_queries": json.dumps(raw_queries),
                "alarm_queries": json.dumps(alarm_queries),
                "alarm": bool(alarm_queries),
                "near_pot_closure_in_prior_experiment": episode["closure"] is not None,
                "prior_experiment_alarm": int(episode["alarm_count"]) > 0,
            })

    with np.load(reference_path, allow_pickle=False) as archive:
        padded = np.asarray(archive["routes"], dtype=np.float32)
        lengths = np.asarray(archive["lengths"], dtype=np.int64)
        candidates = np.asarray(archive["candidates"], dtype=np.int64)
    healthy_alarm_count = 0
    for index, candidate in enumerate(candidates):
        loo_bank = HealthySequenceBank.load(reference_path, exclude_identity=int(candidate))
        identity = {
            "evaluation_role": "healthy_leave_one_out",
            "run_role": "healthy_reference",
            "episode_index": int(candidate),
            "init_state_id": 0,
            "success": True,
        }
        rows, raw_queries, alarm_queries = evaluate_sequence(
            padded[index, : lengths[index]], loo_bank, persistence, identity
        )
        query_rows.extend(rows)
        healthy_alarm_count += int(bool(alarm_queries))
        episode_rows.append({
            **identity,
            "queries": int(lengths[index]),
            "raw_reject_queries": json.dumps(raw_queries),
            "alarm_queries": json.dumps(alarm_queries),
            "alarm": bool(alarm_queries),
            "near_pot_closure_in_prior_experiment": True,
            "prior_experiment_alarm": False,
        })

    online = [
        row for row in episode_rows if row["evaluation_role"] == "online_route_replay"
    ]
    fresh = [
        row for row in online if row["run_role"] == "fresh_random_preliminary_set"
    ]
    summary = {
        "schema": "himoe.moe_only_selector_replay.summary.v1",
        "training": False,
        "physical_inputs_to_selector": False,
        "failure_labels_used_to_set_thresholds": False,
        "healthy_reference_trajectories": len(bank.routes),
        "matcher": config["matcher"],
        "alarm_persistence": persistence,
        "healthy_leave_one_out": {
            "trajectories": len(candidates),
            "trajectories_with_alarm": healthy_alarm_count,
        },
        "all_saved_online_routes": {
            "episodes": len(online),
            "episodes_with_alarm": sum(bool(row["alarm"]) for row in online),
            "successful_episodes_with_alarm": sum(
                bool(row["alarm"]) and bool(row["success"]) for row in online
            ),
            "failed_episodes_with_alarm": sum(
                bool(row["alarm"]) and not bool(row["success"]) for row in online
            ),
        },
        "fresh_random_preliminary_set": {
            "episodes": len(fresh),
            "episodes_with_alarm": sum(bool(row["alarm"]) for row in fresh),
            "successful_episodes_with_alarm": sum(
                bool(row["alarm"]) and bool(row["success"]) for row in fresh
            ),
            "failed_episodes_with_alarm": sum(
                bool(row["alarm"]) and not bool(row["success"]) for row in fresh
            ),
            "alarm_episode_init_states": [
                int(row["init_state_id"]) for row in fresh if row["alarm"]
            ],
            "alarm_queries": [
                json.loads(row["alarm_queries"]) for row in fresh if row["alarm"]
            ],
        },
        "interpretation": (
            "Persistence removes formal alarms on the two saved successful "
            "trajectories and all seven healthy LOO trajectories. It alarms one "
            "saved failed trajectory (init 20, q36), but misses the previously "
            "confirmed init-0 failed-grasp episode. This is a preliminary replay, "
            "not held-out online validation."
        ),
    }
    write_csv(output / "query_decisions.csv", query_rows)
    write_csv(output / "episode_decisions.csv", episode_rows)
    (output / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
