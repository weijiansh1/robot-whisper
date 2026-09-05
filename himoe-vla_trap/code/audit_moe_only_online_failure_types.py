#!/usr/bin/env python3
"""Post-hoc failure taxonomy for the MoE-only online alarm runs.

Physical trajectories are used only here, after rollout.  They are never passed
to the online selector.  The missed-grasp label is deliberately kinematic: it
does not claim access to contact, force, or the policy's latent belief.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import zarr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
TAXONOMY_PATH = WORKSPACE_ROOT / "himoe-route-capture" / "analyze_rolling_star_experiment.py"
DEFAULT_RUNS = (
    ("preliminary_gpu4", PACKAGE_ROOT / "results/moe_only_online_alarm/gpu4_fresh_random_seed20260905"),
    ("heldout_gpu4", PACKAGE_ROOT / "results/moe_only_online_alarm/gpu4_heldout_random_seed20260906"),
    ("gpu5_extended", PACKAGE_ROOT / "results/moe_only_online_alarm/gpu5_extended_random_seed20260907"),
)
DEFAULT_SERVERS = {
    "preliminary_gpu4": PACKAGE_ROOT / "results/moe_only_online_alarm/gpu4_server_seed20260905",
    "heldout_gpu4": PACKAGE_ROOT / "results/moe_only_online_alarm/gpu4_server_seed20260906",
    "gpu5_extended": PACKAGE_ROOT / "results/moe_only_online_alarm/gpu5_server_seed20260907",
}
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/moe_only_online_alarm/failure_type_audit"

CLOSE_THRESHOLD = 0.05
NEAR_THRESHOLD_M = 0.16
EEF_DEPARTURE_M = 0.10
OBJECT_STATIONARY_M = 0.01
SEPARATION_M = 0.15


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_taxonomy_module() -> Any:
    spec = importlib.util.spec_from_file_location("route_capture_taxonomy", TAXONOMY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load taxonomy module: {TAXONOMY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def target_joints(layout: dict[str, Any]) -> list[dict[str, Any]]:
    targets = [
        joint
        for joint in layout["joints"]
        if not bool(joint["is_robot"])
        and int(joint["state_hi"]) - int(joint["state_lo"]) == 7
    ]
    if len(targets) != 2:
        raise ValueError(f"expected two free-joint targets, found {len(targets)}")
    return targets


def missed_grasp_events(
    arrays: dict[str, np.ndarray], targets: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    sim = np.asarray(arrays["control_sim_state"], dtype=np.float32)
    eef = np.asarray(arrays["control_eef_position"], dtype=np.float32)
    gripper = np.asarray(arrays["control_gripper_qpos"], dtype=np.float32)
    query_index = np.asarray(arrays["control_query_index"], dtype=np.int32)
    aperture = np.abs(gripper).sum(axis=1)
    crossings = np.flatnonzero(
        (aperture[:-1] >= CLOSE_THRESHOLD) & (aperture[1:] < CLOSE_THRESHOLD)
    ) + 1
    events: list[dict[str, Any]] = []
    for step_value in crossings:
        step = int(step_value)
        for target in targets:
            lo = int(target["state_lo"])
            target_position = sim[step, lo : lo + 3]
            distance = float(np.linalg.norm(eef[step] - target_position))
            if distance >= NEAR_THRESHOLD_M:
                continue
            future_eef = eef[step:]
            future_target = sim[step:, lo : lo + 3]
            eef_displacement = np.linalg.norm(future_eef - eef[step], axis=1)
            target_displacement = np.linalg.norm(
                future_target - target_position, axis=1
            )
            separation = np.linalg.norm(future_eef - future_target, axis=1)
            query = int(query_index[step - 1])
            is_missed = bool(
                eef_displacement.max() >= EEF_DEPARTURE_M
                and target_displacement.max() < OBJECT_STATIONARY_M
                and separation.max() >= SEPARATION_M
            )
            is_borderline_15mm = bool(
                eef_displacement.max() >= EEF_DEPARTURE_M
                and OBJECT_STATIONARY_M <= target_displacement.max() < 0.015
                and (future_target[:, 2] - target_position[2]).max() < 0.005
                and separation.max() >= SEPARATION_M
            )
            events.append(
                {
                    "target": str(target["joint"]),
                    "query": query,
                    "control_step": step,
                    "eef_target_distance_at_closure_m": distance,
                    "eef_max_displacement_after_m": float(eef_displacement.max()),
                    "target_max_displacement_after_m": float(target_displacement.max()),
                    "target_max_lift_after_m": float(
                        (future_target[:, 2] - target_position[2]).max()
                    ),
                    "eef_target_max_separation_after_m": float(separation.max()),
                    "kinematic_missed_grasp_then_departure": is_missed,
                    "missed_grasp_sensitivity_only_15mm": is_borderline_15mm,
                }
            )
    return events


def nearest_signed_delta(queries: list[int], anchors: list[int]) -> int | None:
    if not queries or not anchors:
        return None
    return min(
        (query - anchor for query in queries for anchor in anchors),
        key=lambda value: (abs(value), value),
    )


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def render(value: Any) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if value is None:
            return "-"
        return str(value).replace("|", "\\|")

    output = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(render(row[column]) for column in columns) + " |")
    return "\n".join(output)


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
    }


def aggregate(
    rows: list[dict[str, Any]], family_key: str = "failure_family"
) -> dict[str, Any]:
    cohorts = {
        "target_missed_grasp_proxy": [
            row for row in rows if row[family_key] == "target_missed_grasp_proxy"
        ],
        "other_failure": [
            row for row in rows if row[family_key] == "other_failure"
        ],
        "success": [row for row in rows if row[family_key] == "success"],
    }
    output: dict[str, Any] = {}
    for name, selected in cohorts.items():
        output[name] = {
            "episodes": len(selected),
            "episodes_with_any_raw_reject": ratio(
                sum(bool(row["any_raw_reject"]) for row in selected), len(selected)
            ),
            "episodes_with_formal_alarm": ratio(
                sum(bool(row["formal_alarm"]) for row in selected), len(selected)
            ),
        }
    failures = [row for row in rows if not bool(row["success"])]
    output["all_failures_descriptive_only"] = {
        "episodes": len(failures),
        "generic_primary_counts": dict(Counter(row["generic_primary"] for row in failures)),
    }
    return output


def route_capture_audit(
    loaded: list[dict[str, Any]], run_role: str
) -> dict[str, Any]:
    episodes = [item for item in loaded if item["run_role"] == run_role]
    client_routes = np.concatenate(
        [item["arrays"]["hb_router_probs"] for item in episodes], axis=0
    )
    client_episode_ids = np.concatenate(
        [
            np.full(
                item["arrays"]["hb_router_probs"].shape[0],
                int(item["summary"]["episode_id"]),
                dtype=np.int32,
            )
            for item in episodes
        ]
    )
    server_root = DEFAULT_SERVERS[run_role]
    store = zarr.open_group(str(server_root / "routes.zarr"), mode="r")
    server_routes = np.asarray(store["hb_router_probs"][:], dtype=np.float16)
    control_steps = np.asarray(store["control_step"][:], dtype=np.int32)
    capture = json.loads(
        (server_root / "capture_summary.json").read_text(encoding="utf-8")
    )
    return {
        "client_rows": int(client_routes.shape[0]),
        "server_rows": int(server_routes.shape[0]),
        "route_shape": list(server_routes.shape),
        "client_server_full_routes_exact": bool(
            np.array_equal(client_routes.astype(np.float16), server_routes)
        ),
        "client_server_episode_ids_exact": bool(
            np.array_equal(client_episode_ids, np.asarray(store["episode_id"][:]))
        ),
        "server_control_steps_strictly_sequential": bool(
            np.array_equal(control_steps, np.arange(len(control_steps), dtype=np.int32))
        ),
        "store_full_probs": bool(capture["store_full_probs"]),
        "return_full_probs": bool(capture["return_full_probs"]),
        "hook_verified_calls": int(capture["hook_verified_calls"]),
        "hook_verify_failures": list(capture["hook_verify_failures"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy_module()

    loaded: list[dict[str, Any]] = []
    canonical_layout: dict[str, Any] | None = None
    for run_role, run_root in DEFAULT_RUNS:
        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["training"] is not False:
            raise ValueError(f"incomplete or non-train-free run: {run_root}")
        for episode_dir in sorted(run_root.glob("episode_*")):
            summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
            layout = json.loads((episode_dir / "sim_layout.json").read_text(encoding="utf-8"))
            if canonical_layout is None:
                canonical_layout = layout
            elif layout != canonical_layout:
                raise ValueError(f"simulator layout differs: {episode_dir}")
            with np.load(episode_dir / "trajectory_and_routes.npz", allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            loaded.append(
                {
                    "run_role": run_role,
                    "run_root": run_root,
                    "episode_dir": episode_dir,
                    "summary": summary,
                    "arrays": arrays,
                }
            )
    if canonical_layout is None:
        raise RuntimeError("no episodes found")
    targets = target_joints(canonical_layout)

    successful = [item for item in loaded if bool(item["summary"]["success"])]
    if not successful:
        raise RuntimeError("goal-reference construction needs successful episodes")
    pooled_terminal_positions = []
    for target in targets:
        lo = int(target["state_lo"])
        pooled_terminal_positions.extend(
            item["arrays"]["control_sim_state"][-1, lo : lo + 3]
            for item in successful
        )
    shared_goal_reference = np.stack(pooled_terminal_positions).astype(np.float32)
    references = {
        str(target["joint"]): shared_goal_reference.copy() for target in targets
    }

    rows: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for global_index, item in enumerate(loaded):
        summary = item["summary"]
        candidate = taxonomy.Candidate(
            worker=0 if item["run_role"] == "preliminary_gpu4" else 1,
            init_state=int(summary["init_state_id"]),
            snapshot=0,
            candidate=int(summary["episode_index"]),
            episode_id=int(summary["episode_id"]),
            success=bool(summary["success"]),
            inference_calls=int(summary["inference_calls"]),
            snapshot_key=str(item["run_role"]),
            npz_path=item["episode_dir"] / "trajectory_and_routes.npz",
            json_path=item["episode_dir"] / "summary.json",
        )
        generic = taxonomy.physical_metrics(
            candidate, item["arrays"], targets, references
        )
        events = missed_grasp_events(item["arrays"], targets)
        for event in events:
            all_events.append(
                {
                    "run_role": item["run_role"],
                    "episode_index": int(summary["episode_index"]),
                    "init_state_id": int(summary["init_state_id"]),
                    "success": bool(summary["success"]),
                    **event,
                }
            )
        missed = [
            event
            for event in events
            if bool(event["kinematic_missed_grasp_then_departure"])
        ]
        borderline = [
            event
            for event in events
            if bool(event["missed_grasp_sensitivity_only_15mm"])
        ]
        raw_queries = [int(value) for value in summary["raw_reject_queries"]]
        alarm_queries = [int(value) for value in summary["alarm_queries"]]
        missed_queries = [int(event["query"]) for event in missed]
        borderline_queries = [int(event["query"]) for event in borderline]
        success = bool(summary["success"])
        if success:
            failure_family = "success"
        elif missed:
            failure_family = "target_missed_grasp_proxy"
        else:
            failure_family = "other_failure"
        sensitivity_family = (
            "success"
            if success
            else "target_missed_grasp_proxy"
            if missed or borderline
            else "other_failure"
        )
        closest_delta = nearest_signed_delta(raw_queries, missed_queries)
        rows.append(
            {
                "run_role": item["run_role"],
                "episode_index": int(summary["episode_index"]),
                "init_state_id": int(summary["init_state_id"]),
                "success": success,
                "failure_family": failure_family,
                "failure_family_sensitivity_15mm": sensitivity_family,
                "generic_primary": str(generic["primary_failure_type"]),
                "generic_labels": str(generic["physical_labels"]),
                "missed_grasp_targets": ";".join(
                    sorted({str(event["target"]) for event in missed})
                ),
                "missed_grasp_queries": json.dumps(missed_queries, separators=(",", ":")),
                "borderline_15mm_queries": json.dumps(
                    borderline_queries, separators=(",", ":")
                ),
                "raw_reject_queries": json.dumps(raw_queries, separators=(",", ":")),
                "formal_alarm_queries": json.dumps(alarm_queries, separators=(",", ":")),
                "any_raw_reject": bool(raw_queries),
                "formal_alarm": bool(alarm_queries),
                "nearest_raw_minus_missed_query": closest_delta,
                "raw_in_5q_before_or_at_miss": bool(
                    any(-5 <= raw - miss <= 0 for raw in raw_queries for miss in missed_queries)
                ),
                "raw_in_5q_after_miss": bool(
                    any(0 < raw - miss <= 5 for raw in raw_queries for miss in missed_queries)
                ),
                "placed_end_count": int(generic["placed_end_count"]),
                "lift_count": int(generic["lift_count"]),
                "drop_count": int(generic["drop_count"]),
                "loop_return_count": int(generic["loop_return_count"]),
                "late_static_window_fraction": float(generic["late_static_window_fraction"]),
                "trajectory": str(
                    (item["episode_dir"] / "trajectory_and_routes.npz").relative_to(PACKAGE_ROOT)
                ),
                "audit_index": global_index,
            }
        )

    columns = list(rows[0])
    with (output / "episode_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "grasp_events.jsonl").open("w", encoding="utf-8") as stream:
        for event in all_events:
            stream.write(json.dumps(plain(event), sort_keys=True) + "\n")

    by_run = {
        role: aggregate([row for row in rows if row["run_role"] == role])
        for role, _ in DEFAULT_RUNS
    }
    capture_audits = {
        role: route_capture_audit(loaded, role) for role, _ in DEFAULT_RUNS
    }
    pooled = aggregate(rows)
    pooled_sensitivity_15mm = aggregate(
        rows, family_key="failure_family_sensitivity_15mm"
    )
    target_rows = [row for row in rows if row["failure_family"] == "target_missed_grasp_proxy"]
    target_stats = pooled["target_missed_grasp_proxy"]
    other_stats = pooled["other_failure"]
    success_stats = pooled["success"]
    target_count = target_stats["episodes"]
    other_count = other_stats["episodes"]
    success_count = success_stats["episodes"]
    target_alarm = target_stats["episodes_with_formal_alarm"]["numerator"]
    other_alarm = other_stats["episodes_with_formal_alarm"]["numerator"]
    success_alarm = success_stats["episodes_with_formal_alarm"]["numerator"]
    target_raw = target_stats["episodes_with_any_raw_reject"]["numerator"]
    other_raw = other_stats["episodes_with_any_raw_reject"]["numerator"]
    success_raw = success_stats["episodes_with_any_raw_reject"]["numerator"]
    formal_total = target_alarm + other_alarm + success_alarm
    summary = {
        "schema": "himoe.moe_only_online_failure_type_audit.v1",
        "training": False,
        "selector_inputs": [
            "current and previous HB router probabilities",
            "successful HB route sequence reference bank",
        ],
        "selector_uses_physical_state": False,
        "physical_data_role": "posthoc_failure_typing_only",
        "runs": [
            {"role": role, "path": str(path.relative_to(PACKAGE_ROOT))}
            for role, path in DEFAULT_RUNS
        ],
        "thresholds": {
            "closure_aperture": CLOSE_THRESHOLD,
            "near_target_m": NEAR_THRESHOLD_M,
            "eef_departure_m": EEF_DEPARTURE_M,
            "object_stationary_m": OBJECT_STATIONARY_M,
            "eef_target_separation_m": SEPARATION_M,
        },
        "pooled": pooled,
        "pooled_sensitivity_15mm": pooled_sensitivity_15mm,
        "by_run": by_run,
        "route_capture_audit": capture_audits,
        "target_timing": {
            "episodes": len(target_rows),
            "raw_within_5_queries_before_or_at_miss": sum(
                bool(row["raw_in_5q_before_or_at_miss"]) for row in target_rows
            ),
            "raw_within_5_queries_after_miss": sum(
                bool(row["raw_in_5q_after_miss"]) for row in target_rows
            ),
            "nearest_raw_minus_missed_query": [
                row["nearest_raw_minus_missed_query"] for row in target_rows
            ],
        },
        "sensitivity_analysis": {
            "object_stationary_threshold_m": 0.015,
            "primary_threshold_unchanged_m": OBJECT_STATIONARY_M,
            "additional_borderline_failure_episodes": sum(
                row["failure_family"] == "other_failure"
                and row["failure_family_sensitivity_15mm"]
                == "target_missed_grasp_proxy"
                for row in rows
            ),
            "role": "posthoc_threshold_sensitivity_only_not_primary_label",
        },
        "interpretation": [
            "Failure recall must be reported separately for the missed-grasp target family and other traps.",
            (
                f"The formal persistence-2 MoE-only alarm fired on {formal_total}/{len(rows)} "
                f"episodes: target {target_alarm}/{target_count}, other failure "
                f"{other_alarm}/{other_count}, success {success_alarm}/{success_count}."
            ),
            (
                f"Single raw rejects cover target {target_raw}/{target_count}, other failure "
                f"{other_raw}/{other_count}, and success {success_raw}/{success_count}; "
                "they are not a deployable alarm."
            ),
            "Kinematic labels are post-hoc proxies and do not prove contact state or the policy's latent belief.",
        ],
    }
    write_json(output / "summary.json", summary)

    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "run": row["run_role"].replace("_gpu4", ""),
                "ep/init": f"{row['episode_index']}/{row['init_state_id']}",
                "outcome": "success" if row["success"] else "failure",
                "family": row["failure_family"],
                "generic": row["generic_primary"],
                "miss q": row["missed_grasp_queries"],
                "raw q": row["raw_reject_queries"],
                "alarm": row["formal_alarm"],
            }
        )
    report = f"""# MoE-only 在线报警：失败类型事后审计

## 核心结论

三批共 {len(rows)} 条在线轨迹（{success_count} 成功、{target_count + other_count} 失败）。失败不能视为同一种 Trap：{target_count} 条满足保守的 `missed-grasp-then-departure` 运动学代理，另外 {other_count} 条表现为其他失败。

正式规则是连续两次 raw reject（persistence=2）。合并三批后：目标漏抓型 {target_alarm}/{target_count}、其他失败 {other_alarm}/{other_count}、成功误报 {success_alarm}/{success_count}。GPU5 扩展批次首次出现正式报警，但目标型召回仍低，不能声称已经可用。

单次 raw reject 在目标漏抓型上是 {target_raw}/{target_count}，成功轨迹也有 {success_raw}/{success_count} 出现；所以它只是诊断信号，不能直接当成报警结果。

## 分轨迹结果

{markdown_table(display_rows, ['run', 'ep/init', 'outcome', 'family', 'generic', 'miss q', 'raw q', 'alarm'])}

## 边界

- 在线 selector 只读 HB MoE routing 和成功 routing reference；不读物理距离、机器人状态、动作值、reward 或 success。
- 本审计中的物理位置只在 rollout 完成后用于区分失败类型，不参与阈值或在线决策。
- 三批客户端与服务端保存的完整 HB 路由均逐元素一致；hook failure 均为 0。
- `missed-grasp-then-departure` 是运动学代理：闭爪时靠近目标，随后 EEF 离开而目标近似静止。数据没有接触力，不能把它升级成真实 contact 或模型 belief 的直接观测。
- 通用 taxonomy 描述最终行为表型；例如漏抓造成后续 loop 时，`loop_or_cycling` 是结果，不一定是根因。
"""
    (output / "report_zh.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
