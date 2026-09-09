#!/usr/bin/env python3
"""Replay terminal MuJoCo states and evaluate each original LIBERO goal predicate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from himoe_libero_bridge.libero_runtime import EpisodeConfig, _load_task


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def unwrap(environment: Any) -> Any:
    current = environment
    seen = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return current


def predicate_key(state: list[str]) -> str:
    return "__".join(str(value).lower() for value in state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--environment-seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    config = EpisodeConfig(
        task_suite=args.task_suite,
        task_id=args.task_id,
        init_state_id=0,
        seed=args.environment_seed,
        host="127.0.0.1",
        port=1,
        libero_root=str(args.libero_root.resolve()),
        output_root=str(run_root / "formal" / "physical_audit"),
        settle_steps=10,
        max_steps=520,
        replan_steps=10,
        inference_timeout=1.0,
    )
    environment, _observation, task, prompt = _load_task(config)
    core = unwrap(environment)
    goals = [list(state) for state in core.parsed_problem["goal_state"]]
    rows = []
    try:
        manifests = sorted((run_root / "formal").glob("worker*/snapshot_*/manifest.json"))
        if not manifests:
            raise RuntimeError("no committed snapshot manifest found")
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            worker_config = json.loads(
                (manifest_path.parents[1] / "experiment_config.json").read_text(
                    encoding="utf-8"
                )
            )
            for summary in manifest["candidate_summaries"]:
                candidate = int(summary["candidate"])
                npz_path = manifest_path.parent / f"candidate_{candidate:02d}.npz"
                with np.load(npz_path, allow_pickle=False) as archive:
                    terminal = np.asarray(
                        archive["control_sim_state"][-1], dtype=np.float64
                    )
                # Predicate evaluation needs only MuJoCo state.  The wrapper's
                # set_init_state() also regenerates and renders every camera,
                # which is orders of magnitude slower and irrelevant here.
                core.sim.set_state_from_flattened(terminal)
                core.sim.forward()
                values = {
                    predicate_key(goal): bool(core._eval_predicate(goal))
                    for goal in goals
                }
                replayed = bool(all(values.values()))
                recorded = bool(summary["success"])
                if replayed != recorded:
                    raise RuntimeError(
                        f"terminal replay mismatch for {npz_path}: "
                        f"recorded={recorded} replayed={replayed} values={values}"
                    )
                rows.append(
                    {
                        "worker": int(worker_config["worker_id"]),
                        "init_state_id": int(worker_config["init_state_id"]),
                        "snapshot": int(manifest["snapshot_index"]),
                        "candidate": candidate,
                        "episode_id": int(summary["episode_id"]),
                        "recorded_success": recorded,
                        "replayed_success": replayed,
                        **values,
                    }
                )
    finally:
        environment.close()

    fieldnames = list(rows[0])
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    out = run_root / "formal" / "physical_audit"
    atomic_text(out / "terminal_predicates.csv", buffer.getvalue())

    patterns = Counter(
        tuple(bool(row[predicate_key(goal)]) for goal in goals) for row in rows
    )
    summary = {
        "schema": "himoe.rolling_star.terminal_predicates.v1",
        "task_name": str(task.name),
        "prompt": prompt,
        "goal_predicates": goals,
        "branches": len(rows),
        "successes": int(sum(bool(row["recorded_success"]) for row in rows)),
        "failures": int(sum(not bool(row["recorded_success"]) for row in rows)),
        "recorded_replay_mismatches": 0,
        "predicate_patterns": {
            "|".join("1" if value else "0" for value in key): count
            for key, count in sorted(patterns.items())
        },
    }
    atomic_text(
        out / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
