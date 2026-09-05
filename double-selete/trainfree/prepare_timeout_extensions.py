#!/usr/bin/env python3
"""Freeze the exact source-failure list for the timeout-extension experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
HUB = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
MAIN_LABELS = HERE / "results/hub_binary_audit/episode_physical_labels.csv"
SEALED = HERE / "results/online_precision_cascade_external/sealed_online_scores.npz"
PROTOCOL = HERE / "TIMEOUT_EXTENSION_PROTOCOL.md"
DEFAULT_OUTPUT = HERE / "results/timeout_extension_plus10"
RUNS = {
    "development_main": "right-50x8-20260903",
    "external_8b": "right-50x8b-20260903",
}
EXPECTED = {"development_main": 487, "external_8b": 564}
SUITE_MAX_ACTION_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_long": 520,
}
SUITE_SERVER_NAME = {
    "libero_spatial": "spatial",
    "libero_object": "object",
    "libero_goal": "goal",
    "libero_long": "long",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def source_summary(task: str, run_id: str, episode: int) -> tuple[Path, dict[str, Any]]:
    client = HUB / task / run_id / "client"
    summaries = read_json(client / "summaries.json")
    lookup = {int(row["episode_index"]): row for row in summaries}
    if episode not in lookup:
        raise ValueError(f"{task}/{run_id}: missing episode {episode}")
    return client, lookup[episode]


def main_failures() -> list[tuple[str, int]]:
    with MAIN_LABELS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = [
        (row["task"], int(row["episode"]))
        for row in rows
        if row["failure"].strip().lower() == "true"
    ]
    if len(rows) != 14_800 or len(output) != EXPECTED["development_main"]:
        raise ValueError("development cohort does not match the audited label set")
    return output


def external_failures() -> list[tuple[str, int]]:
    with np.load(SEALED, allow_pickle=False) as archive:
        task_names = archive["task_names"].astype(str)
        task_index = archive["task_index"].astype(int)
        episodes = archive["episode"].astype(int)
    if len(episodes) != 15_600 or len(task_names) != 39:
        raise ValueError("external seal does not contain the frozen 15,600 rows")
    summary_cache: dict[str, dict[int, dict[str, Any]]] = {}
    output: list[tuple[str, int]] = []
    for position, episode in zip(task_index, episodes):
        task = str(task_names[position])
        if task not in summary_cache:
            client = HUB / task / RUNS["external_8b"] / "client"
            summary_cache[task] = {
                int(row["episode_index"]): row
                for row in read_json(client / "summaries.json")
            }
        row = summary_cache[task][int(episode)]
        if not bool(row["success"]):
            output.append((task, int(episode)))
    if len(output) != EXPECTED["external_8b"]:
        raise ValueError("external failure count differs from the sealed evaluation")
    return output


def build_case(
    ordinal: int, cohort: str, task: str, episode: int
) -> dict[str, Any]:
    run_id = RUNS[cohort]
    client, summary = source_summary(task, run_id, episode)
    hub_suite = task.split("/", 1)[0]
    if hub_suite not in SUITE_MAX_ACTION_STEPS:
        raise ValueError(f"unsupported suite in {task}")
    expected_steps = SUITE_MAX_ACTION_STEPS[hub_suite]
    if bool(summary["success"]) or int(summary["action_steps"]) != expected_steps:
        raise ValueError(f"{cohort}/{task}/{episode}: source is not a horizon failure")
    if int(summary["inference_calls"]) * 10 != expected_steps:
        raise ValueError(f"{cohort}/{task}/{episode}: source chunk count is inconsistent")
    source_npz = client / f"episode_{episode:02d}.npz"
    with np.load(source_npz, allow_pickle=False) as archive:
        required = {"state", "actions", "sim_state"}
        if set(archive.files) != required:
            raise ValueError(f"{source_npz}: arrays are {archive.files}")
        query_count = int(summary["inference_calls"])
        if archive["state"].shape != (query_count, 8):
            raise ValueError(f"{source_npz}: invalid policy-state shape")
        if archive["actions"].shape != (query_count, 10, 7):
            raise ValueError(f"{source_npz}: invalid action shape")
        if archive["sim_state"].shape[0] != query_count:
            raise ValueError(f"{source_npz}: invalid simulator-state length")
    meta = read_json(client.parent / "meta.json")
    expected_benchmark = "libero_10" if hub_suite == "libero_long" else hub_suite
    if meta["benchmark"] != expected_benchmark or meta["status"] != "complete":
        raise ValueError(f"{client.parent}: invalid source metadata")
    return {
        "case_id": f"{ordinal:04d}",
        "cohort": cohort,
        "task": task,
        "hub_suite": hub_suite,
        "server_suite": SUITE_SERVER_NAME[hub_suite],
        "benchmark": expected_benchmark,
        "task_id": int(summary["task_id"]),
        "episode": episode,
        "init_state_id": int(summary["init_state_id"]),
        "environment_seed": int(summary["seed"]),
        "flow_noise_seed": int(summary["flow_noise_seed"]),
        "original_action_steps": int(summary["action_steps"]),
        "original_inference_calls": int(summary["inference_calls"]),
        "source_run_id": run_id,
        "source_npz": str(source_npz.resolve()),
        "source_npz_bytes": source_npz.stat().st_size,
        "source_npz_sha256": sha256(source_npz),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "source_failures.csv").exists():
        raise FileExistsError("frozen source manifest already exists")

    sources = {
        "development_main": main_failures(),
        "external_8b": external_failures(),
    }
    cases: list[dict[str, Any]] = []
    for cohort in RUNS:
        for task, episode in sorted(sources[cohort]):
            cases.append(build_case(len(cases), cohort, task, episode))
    if len(cases) != sum(EXPECTED.values()):
        raise ValueError("combined continuation count is inconsistent")

    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(cases[0])
    lines: list[str] = []
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
        handle.seek(0)
        lines.append(handle.read())
    atomic_text(output / "source_failures.csv", "".join(lines))
    cohort_counts = {
        cohort: sum(case["cohort"] == cohort for case in cases) for cohort in RUNS
    }
    suite_counts = {
        suite: sum(case["server_suite"] == suite for case in cases)
        for suite in SUITE_SERVER_NAME.values()
    }
    manifest = {
        "schema": "himoe.timeout_extension.source_manifest.v1",
        "status": "frozen_before_continuation",
        "cases": len(cases),
        "cohort_counts": cohort_counts,
        "suite_counts": suite_counts,
        "source_runs": RUNS,
        "extra_control_queries": 10,
        "actions_per_query": 10,
        "maximum_extra_low_level_actions": 100,
        "source_manifest_sha256": sha256(output / "source_failures.csv"),
        "protocol_sha256": sha256(PROTOCOL),
        "main_labels_sha256": sha256(MAIN_LABELS),
        "external_seal_sha256": sha256(SEALED),
        "source_files_hashed_individually": True,
        "source_data_modified": False,
    }
    atomic_text(
        output / "source_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
