#!/usr/bin/env python3
"""Validate timeout continuations and write cleaned failure labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
HUB = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
MAIN_LABELS = HERE / "results/hub_binary_audit/episode_physical_labels.csv"
SEALED_ROOT = HERE / "results/online_precision_cascade_external"
SHARED_ROOT = HERE / "results/online_precision_cascade_shared"
DEFAULT_OUTPUT = HERE / "results/timeout_extension_plus10"
RUNS = {
    "development_main": "right-50x8-20260903",
    "external_8b": "right-50x8b-20260903",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        atomic_text(path, handle.read())


def result_paths(output: Path, case: dict[str, str]) -> tuple[Path, Path]:
    root = output / "episodes" / case["cohort"] / case["server_suite"]
    stem = "%s_task%02d_ep%03d" % (
        case["case_id"],
        int(case["task_id"]),
        int(case["episode"]),
    )
    return root / (stem + ".json"), root / (stem + ".npz")


def validate_results(
    output: Path, cases: list[dict[str, str]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    problems: list[str] = []
    for case in cases:
        json_path, npz_path = result_paths(output, case)
        if not json_path.is_file():
            problems.append(f"missing result {case['case_id']}")
            continue
        result = json.loads(json_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            problems.append(
                f"case {case['case_id']} status={result.get('status')}: "
                f"{result.get('error', '')}"
            )
            continue
        if (
            result.get("source_npz_sha256") != case["source_npz_sha256"]
            or result.get("case_id") != case["case_id"]
        ):
            problems.append(f"case {case['case_id']} source identity mismatch")
            continue
        source_server_path = Path(case["source_npz"]).parent / "server_metadata.json"
        source_server = json.loads(source_server_path.read_text(encoding="utf-8"))
        if result.get("checkpoint_sha256") != source_server.get("checkpoint_sha256"):
            problems.append(f"case {case['case_id']} checkpoint identity mismatch")
            continue
        if result.get("libero_wrist_layout") != source_server.get(
            "libero_wrist_layout"
        ):
            problems.append(f"case {case['case_id']} wrist layout mismatch")
            continue
        if not result.get("exact_source_prefix_replay"):
            problems.append(f"case {case['case_id']} prefix was not exact")
            continue
        if not npz_path.is_file() or sha256(npz_path) != result.get("arrays_sha256"):
            problems.append(f"case {case['case_id']} continuation arrays mismatch")
            continue
        success = bool(result["success_after_extension"])
        success_query = int(result["first_success_extra_query"])
        success_action = int(result["first_success_extra_action"])
        if success != (1 <= success_query <= 10 and 1 <= success_action <= 100):
            problems.append(f"case {case['case_id']} success fields disagree")
            continue
        results.append(result)
    if problems:
        preview = "\n".join(problems[:20])
        raise RuntimeError(
            f"continuation is incomplete or invalid ({len(problems)} cases):\n{preview}"
        )
    if len(results) != len(cases):
        raise RuntimeError("validated result count differs from source manifest")
    return results


def result_table(
    output: Path,
    cases: list[dict[str, str]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_by_id = {row["case_id"]: row for row in results}
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = result_by_id[case["case_id"]]
        rows.append(
            {
                **case,
                "exact_source_prefix_replay": True,
                "success_after_extension": bool(result["success_after_extension"]),
                "success_within_one_extra_query": bool(
                    result["success_within_one_extra_query"]
                ),
                "first_success_extra_query": int(
                    result["first_success_extra_query"]
                ),
                "first_success_extra_action": int(
                    result["first_success_extra_action"]
                ),
                "extra_queries_executed": int(result["extra_queries_executed"]),
                "extra_action_steps_executed": int(
                    result["extra_action_steps_executed"]
                ),
                "max_source_sim_state_error": float(
                    result["max_source_sim_state_error"]
                ),
                "max_source_policy_state_error": float(
                    result["max_source_policy_state_error"]
                ),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "server_instance_id": result["server_instance_id"],
                "result_json": str(result_paths(output, case)[0]),
                "result_arrays": result["arrays"],
                "result_arrays_sha256": result["arrays_sha256"],
                "wall_s": float(result["wall_s"]),
            }
        )
    return rows


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def clean_main_labels(
    output: Path, lookup: dict[tuple[str, str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = read_csv(MAIN_LABELS)
    fields = list(rows[0])
    original_failure_position = fields.index("failure")
    fields.insert(original_failure_position, "original_failure")
    fields[fields.index("failure") + 1 : fields.index("failure") + 1] = [
        "late_success_plus10_queries",
        "late_success_first_extra_query",
        "late_success_first_extra_action",
    ]
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        original_failure = row["failure"].lower() == "true"
        continuation = lookup.get(
            ("development_main", row["task"], int(row["episode"]))
        )
        if original_failure and continuation is None:
            raise RuntimeError("main failure has no continuation result")
        late = bool(continuation and continuation["success_after_extension"])
        updated: dict[str, Any] = dict(row)
        updated["original_failure"] = bool_text(original_failure)
        updated["failure"] = bool_text(original_failure and not late)
        updated["late_success_plus10_queries"] = bool_text(late)
        updated["late_success_first_extra_query"] = (
            continuation["first_success_extra_query"] if late else ""
        )
        updated["late_success_first_extra_action"] = (
            continuation["first_success_extra_action"] if late else ""
        )
        cleaned.append(updated)
    write_csv(output / "development_main_clean_labels.csv", cleaned, fields)
    return cleaned


def clean_external_labels(
    output: Path, lookup: dict[tuple[str, str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    sealed_path = SEALED_ROOT / "sealed_online_scores.npz"
    with np.load(sealed_path, allow_pickle=False) as archive:
        task_names = archive["task_names"].astype(str)
        task_index = archive["task_index"].astype(int)
        episodes = archive["episode"].astype(int)
        init_states = archive["init_state_id"].astype(int)
        flow_seeds = archive["flow_noise_seed"].astype(int)
        lengths = archive["length"].astype(int)
    summary_cache: dict[str, dict[int, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for position, episode, init_state, flow_seed, length in zip(
        task_index, episodes, init_states, flow_seeds, lengths
    ):
        task = str(task_names[position])
        if task not in summary_cache:
            path = HUB / task / RUNS["external_8b"] / "client/summaries.json"
            summary_cache[task] = {
                int(row["episode_index"]): row
                for row in json.loads(path.read_text(encoding="utf-8"))
            }
        source = summary_cache[task][int(episode)]
        if (
            int(source["init_state_id"]) != init_state
            or int(source["flow_noise_seed"]) != flow_seed
            or int(source["inference_calls"]) != length
        ):
            raise RuntimeError("external seal and source summary differ")
        original_failure = not bool(source["success"])
        continuation = lookup.get(("external_8b", task, int(episode)))
        if original_failure and continuation is None:
            raise RuntimeError("external failure has no continuation result")
        late = bool(continuation and continuation["success_after_extension"])
        rows.append(
            {
                "task": task,
                "episode": int(episode),
                "init_state_id": init_state,
                "flow_noise_seed": flow_seed,
                "episode_length": length,
                "original_failure": original_failure,
                "failure": original_failure and not late,
                "late_success_plus10_queries": late,
                "late_success_first_extra_query": (
                    continuation["first_success_extra_query"] if late else ""
                ),
                "late_success_first_extra_action": (
                    continuation["first_success_extra_action"] if late else ""
                ),
            }
        )
    fields = list(rows[0])
    write_csv(output / "external_8b_clean_labels.csv", rows, fields)
    return rows


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def summaries(
    output: Path, table: list[dict[str, Any]]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    overall: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    by_suite: list[dict[str, Any]] = []
    by_task: list[dict[str, Any]] = []
    for cohort in [*RUNS, "all"]:
        block = [
            row
            for row in table
            if cohort == "all" or row["cohort"] == cohort
        ]
        late = [row for row in block if row["success_after_extension"]]
        within_one = [row for row in block if row["success_within_one_extra_query"]]
        overall.append(
            {
                "cohort": cohort,
                "original_failures": len(block),
                "late_successes_plus10_queries": len(late),
                "late_success_rate": ratio(len(late), len(block)),
                "late_successes_plus1_query": len(within_one),
                "late_success_plus1_query_rate": ratio(len(within_one), len(block)),
                "remaining_failures": len(block) - len(late),
            }
        )
        for query in range(1, 11):
            count = sum(
                row["success_after_extension"]
                and int(row["first_success_extra_query"]) <= query
                for row in block
            )
            curve.append(
                {
                    "cohort": cohort,
                    "extra_queries": query,
                    "maximum_extra_low_level_actions": query * 10,
                    "late_successes": count,
                    "late_success_rate": ratio(count, len(block)),
                    "remaining_failures": len(block) - count,
                }
            )
    suites = sorted({row["server_suite"] for row in table})
    for cohort in [*RUNS, "all"]:
        for suite in suites:
            block = [
                row
                for row in table
                if row["server_suite"] == suite
                and (cohort == "all" or row["cohort"] == cohort)
            ]
            late = sum(row["success_after_extension"] for row in block)
            within_one = sum(row["success_within_one_extra_query"] for row in block)
            by_suite.append(
                {
                    "cohort": cohort,
                    "suite": suite,
                    "original_failures": len(block),
                    "late_successes_plus10_queries": late,
                    "late_success_rate": ratio(late, len(block)),
                    "late_successes_plus1_query": within_one,
                    "late_success_plus1_query_rate": ratio(within_one, len(block)),
                    "remaining_failures": len(block) - late,
                }
            )
    groups = sorted({(row["cohort"], row["task"]) for row in table})
    for cohort, task in groups:
        block = [
            row for row in table if row["cohort"] == cohort and row["task"] == task
        ]
        late = sum(row["success_after_extension"] for row in block)
        within_one = sum(row["success_within_one_extra_query"] for row in block)
        by_task.append(
            {
                "cohort": cohort,
                "task": task,
                "original_failures": len(block),
                "late_successes_plus10_queries": late,
                "late_success_rate": ratio(late, len(block)),
                "late_successes_plus1_query": within_one,
                "late_success_plus1_query_rate": ratio(within_one, len(block)),
                "remaining_failures": len(block) - late,
            }
        )
    write_csv(output / "summary_by_cohort.csv", overall, list(overall[0]))
    write_csv(output / "late_success_curve.csv", curve, list(curve[0]))
    write_csv(output / "summary_by_suite.csv", by_suite, list(by_suite[0]))
    write_csv(output / "summary_by_task.csv", by_task, list(by_task[0]))
    return overall, curve, by_suite, by_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    cases = read_csv(output / "source_failures.csv")
    results = validate_results(output, cases)
    table = result_table(output, cases, results)
    fields = list(table[0])
    write_csv(output / "continuation_results.csv", table, fields)
    late = [row for row in table if row["success_after_extension"]]
    remaining = [row for row in table if not row["success_after_extension"]]
    write_csv(output / "late_successes.csv", late, fields)
    write_csv(output / "remaining_failures.csv", remaining, fields)
    lookup = {
        (row["cohort"], row["task"], int(row["episode"])): row for row in table
    }
    main_labels = clean_main_labels(output, lookup)
    external_labels = clean_external_labels(output, lookup)
    overall, _, by_suite, _ = summaries(output, table)
    summary = {
        "schema": "himoe.timeout_extension.summary.v1",
        "status": "complete",
        "all_source_failures_completed": True,
        "cases": len(cases),
        "exact_prefix_replays": sum(
            row["exact_source_prefix_replay"] for row in table
        ),
        "cohorts": overall,
        "suites": by_suite,
        "development_clean_rows": len(main_labels),
        "external_clean_rows": len(external_labels),
        "source_manifest_sha256": sha256(output / "source_failures.csv"),
        "continuation_results_sha256": sha256(output / "continuation_results.csv"),
        "late_successes_sha256": sha256(output / "late_successes.csv"),
        "remaining_failures_sha256": sha256(output / "remaining_failures.csv"),
        "source_data_modified": False,
    }
    atomic_text(
        output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
