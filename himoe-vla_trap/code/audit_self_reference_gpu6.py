#!/usr/bin/env python3
"""Audit the task-free self-reference selector's GPU6 prospective runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import zarr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
RESULT_ROOT = PACKAGE_ROOT / "results/task_free_self_reference_selector"
DEFAULT_OUTPUT = RESULT_ROOT / "gpu6_audit"
PROSPECTIVE_RUNS = (
    RESULT_ROOT / "gpu6_prospective_init09_seeds1000_1007",
    RESULT_ROOT / "gpu6_prospective_init30_seeds1000_1007",
)
REPLAY_RUN = RESULT_ROOT / "gpu6_exact_failure_replay_retry3"
OLD_TASK_RUN = (
    WORKSPACE_ROOT
    / "VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_goal"
    / "push_the_plate_to_the_front_of_the_stove/right-50x8-20260903"
)
OFFLINE_SUMMARY = RESULT_ROOT / "summary.json"
SELECTOR_VERSION = "self_reference_coupling_collapse_v3"
PLATE_JOINT = "plate_1_joint0"
PLATEAU_INITIAL_DISPLACEMENT_M = 0.05
PLATEAU_REMAINING_EXCURSION_M = 0.001
MEANINGFUL_STEP_M = 0.001
HELDOUT_CLOCK_PHASE = 0.81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def prospective_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    for run in PROSPECTIVE_RUNS:
        manifest = load_json(run / "manifest.json")
        config = load_json(run / "experiment_config.json")
        assert isinstance(manifest, dict) and isinstance(config, dict)
        assert manifest["status"] == "complete"
        assert manifest["training"] is False
        assert manifest["selector_version"] == SELECTOR_VERSION
        horizon_queries = int(math.ceil(config["max_steps"] / config["replan_steps"]))
        clock_query = int(math.ceil(HELDOUT_CLOCK_PHASE * horizon_queries) - 1)
        for episode in manifest["episodes"]:
            assert episode["selector_version"] == SELECTOR_VERSION
            assert episode["training"] is False
            for flag in (
                "selector_uses_action_values",
                "selector_uses_gripper_event",
                "selector_uses_normal_trajectory_bank",
                "selector_uses_physical_state",
                "selector_uses_reward_or_success",
                "selector_uses_task_identity",
            ):
                assert episode[flag] is False

            archive_path = run / episode["trajectory_npz"]
            assert sha256(archive_path) == episode["trajectory_npz_sha256"]
            first_alarm = (
                int(episode["alarm_queries"][0]) if episode["alarm_queries"] else None
            )
            with np.load(archive_path, allow_pickle=False) as archive:
                online_alarm = np.asarray(archive["selector_alarm"], dtype=bool)
                expected = np.zeros(len(online_alarm), dtype=bool)
                expected[np.asarray(episode["alarm_queries"], dtype=int)] = True
                assert np.array_equal(online_alarm, expected)
                route_shape = tuple(archive["hb_router_probs"].shape[1:])
                assert route_shape == (8, 10, 11, 32)

            videos = []
            for video in episode["videos"]:
                video_path = run / video["path"]
                assert video_path.stat().st_size == int(video["bytes"])
                assert sha256(video_path) == video["sha256"]
                videos.append(str(video_path.relative_to(PACKAGE_ROOT)))
            rows.append(
                {
                    "run": run.name,
                    "episode_index": int(episode["episode_index"]),
                    "init_state_id": int(episode["init_state_id"]),
                    "flow_noise_seed": int(episode["flow_noise_seed"]),
                    "success": bool(episode["success"]),
                    "failure": not bool(episode["success"]),
                    "inference_calls": int(episode["inference_calls"]),
                    "action_steps": int(episode["action_steps"]),
                    "alarm": first_alarm is not None,
                    "first_alarm_query": first_alarm,
                    "alarm_query_count": int(episode["alarm_count"]),
                    "post_alarm_action_steps": int(episode["post_alarm_action_steps"]),
                    "fixed_clock_query": clock_query,
                    "fixed_clock_would_alarm": int(episode["inference_calls"]) > clock_query,
                    "video_count": len(videos),
                    "trajectory_npz_sha256": episode["trajectory_npz_sha256"],
                }
            )
            if not episode["success"]:
                physical_rows.append(
                    physical_timing(run, episode, horizon_queries, clock_query)
                )
    return rows, physical_rows


def object_position_slice(layout_path: Path, joint_name: str) -> slice:
    layout = load_json(layout_path)
    assert isinstance(layout, dict)
    matching = [item for item in layout["joints"] if item["joint"] == joint_name]
    if len(matching) != 1:
        raise RuntimeError(f"expected one {joint_name!r} in {layout_path}")
    joint = matching[0]
    if int(joint["state_hi"]) - int(joint["state_lo"]) < 3:
        raise RuntimeError(f"{joint_name!r} has no xyz position")
    return slice(int(joint["state_lo"]), int(joint["state_lo"]) + 3)


def physical_timing(
    run: Path,
    episode: dict[str, Any],
    horizon_queries: int,
    clock_query: int,
) -> dict[str, Any]:
    episode_dir = run / Path(episode["trajectory_npz"]).parent
    position_slice = object_position_slice(episode_dir / "sim_layout.json", PLATE_JOINT)
    with np.load(episode_dir / "trajectory_and_routes.npz", allow_pickle=False) as archive:
        control_position = np.asarray(
            archive["control_sim_state"][:, position_slice], dtype=np.float64
        )
        query_steps = np.asarray(archive["action_steps_at_query"], dtype=int)
        query_position = np.asarray(
            archive["query_sim_state"][:, position_slice], dtype=np.float64
        )
    initial = control_position[0]
    step_motion = np.linalg.norm(np.diff(control_position, axis=0), axis=1)
    moving = np.flatnonzero(step_motion > MEANINGFUL_STEP_M)
    last_motion_transition = int(moving[-1]) if len(moving) else None

    plateau_query = None
    plateau_remaining_excursion = None
    for query, step in enumerate(query_steps):
        initial_displacement = float(np.linalg.norm(control_position[step] - initial))
        remaining = float(
            np.linalg.norm(control_position[step:] - control_position[step], axis=1).max()
        )
        if (
            initial_displacement >= PLATEAU_INITIAL_DISPLACEMENT_M
            and remaining <= PLATEAU_REMAINING_EXCURSION_M
        ):
            plateau_query = query
            plateau_remaining_excursion = remaining
            break
    if plateau_query is None:
        raise RuntimeError(f"no post-movement plateau found for {episode_dir}")

    first_alarm = int(episode["alarm_queries"][0])
    first_payload = episode["alarms"][0]
    return {
        "run": run.name,
        "episode_index": int(episode["episode_index"]),
        "init_state_id": int(episode["init_state_id"]),
        "flow_noise_seed": int(episode["flow_noise_seed"]),
        "object_joint": PLATE_JOINT,
        "plateau_definition": (
            f"initial displacement >= {PLATEAU_INITIAL_DISPLACEMENT_M} m and "
            f"all remaining excursion <= {PLATEAU_REMAINING_EXCURSION_M} m"
        ),
        "plateau_query": plateau_query,
        "last_motion_transition_control_step": last_motion_transition,
        "plateau_remaining_excursion_m": plateau_remaining_excursion,
        "initial_to_final_displacement_m": float(
            np.linalg.norm(control_position[-1] - initial)
        ),
        "query_initial_to_final_displacement_m": float(
            np.linalg.norm(query_position[-1] - query_position[0])
        ),
        "first_moe_alarm_query": first_alarm,
        "moe_alarm_delay_after_plateau_queries": first_alarm - plateau_query,
        "fixed_clock_phase": HELDOUT_CLOCK_PHASE,
        "fixed_clock_query": clock_query,
        "moe_lead_over_fixed_clock_queries": clock_query - first_alarm,
        "horizon_queries": horizon_queries,
        "state_response_ratio_at_alarm": float(first_payload["state_response_ratio"]),
        "planning_churn_ratio_at_alarm": float(first_payload["planning_churn_ratio"]),
        "state_action_gap_ratio_at_alarm": float(first_payload["state_action_gap_ratio"]),
        "marker_is_semantic_failure_onset": False,
        "marker_scope": "posthoc object-motion plateau; never used by selector",
    }


def same_seed_replay_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_summaries = load_json(OLD_TASK_RUN / "client/summaries.json")
    replay = load_json(REPLAY_RUN / "manifest.json")
    old_server = load_json(OLD_TASK_RUN / "client/server_metadata.json")
    new_server = load_json(REPLAY_RUN / "server_metadata.json")
    assert isinstance(old_summaries, list)
    assert isinstance(replay, dict) and isinstance(old_server, dict)
    assert isinstance(new_server, dict)
    identity_fields = (
        "checkpoint_sha256",
        "himoe_upstream_commit",
        "himoe_working_tree_diff_sha256",
        "libero_wrist_layout",
    )
    identity_match = all(old_server[key] == new_server[key] for key in identity_fields)
    old_group = zarr.open_group(str(OLD_TASK_RUN / "server/routes.zarr"), mode="r")
    old_ids = np.asarray(old_group["episode_id"][:], dtype=np.int64)

    rows: list[dict[str, Any]] = []
    old_episode_by_condition = {
        (int(item["init_state_id"]), int(item["flow_noise_seed"])): item
        for item in old_summaries
    }
    for new_episode in replay["episodes"]:
        condition = (
            int(new_episode["init_state_id"]),
            int(new_episode["flow_noise_seed"]),
        )
        old_episode = old_episode_by_condition[condition]
        old_index = int(old_episode["episode_index"])
        old_archive_path = OLD_TASK_RUN / "client" / f"episode_{old_index:02d}.npz"
        new_archive_path = REPLAY_RUN / new_episode["trajectory_npz"]
        old_route_rows = np.flatnonzero(old_ids == old_index)
        assert len(old_route_rows) == int(old_episode["inference_calls"])
        old_route_q0 = np.asarray(
            old_group["hb_router_probs"][int(old_route_rows[0])], dtype=np.float32
        )
        with np.load(old_archive_path, allow_pickle=False) as old_archive, np.load(
            new_archive_path, allow_pickle=False
        ) as new_archive:
            old_action = np.asarray(old_archive["actions"][0], dtype=np.float32)
            new_action = np.asarray(new_archive["action_chunks"][0], dtype=np.float32)
            new_route_q0 = np.asarray(new_archive["hb_router_probs"][0], dtype=np.float32)
            action_abs = np.abs(old_action - new_action)
            route_abs = np.abs(old_route_q0 - new_route_q0)
            state_exact = np.array_equal(old_archive["state"][0], new_archive["policy_state"][0])
            sim_exact = np.array_equal(
                old_archive["sim_state"][0], new_archive["query_sim_state"][0]
            )
        rows.append(
            {
                "init_state_id": condition[0],
                "flow_noise_seed": condition[1],
                "old_gpu": int(old_server["cuda_visible_devices"]),
                "new_gpu": int(new_server["cuda_visible_devices"]),
                "old_episode_index": old_index,
                "old_success": bool(old_episode["success"]),
                "new_success": bool(new_episode["success"]),
                "q0_policy_state_exact": state_exact,
                "q0_sim_state_exact": sim_exact,
                "q0_action_mae": float(action_abs.mean()),
                "q0_action_max_abs": float(action_abs.max()),
                "q0_route_mae": float(route_abs.mean()),
                "q0_route_max_abs": float(route_abs.max()),
                "runtime_identity_fields_match": identity_match,
            }
        )
    identity = {
        "fields_compared": list(identity_fields),
        "all_match": identity_match,
        "old_gpu": old_server["cuda_visible_devices"],
        "new_gpu": new_server["cuda_visible_devices"],
        "checkpoint_sha256": old_server["checkpoint_sha256"],
    }
    return rows, identity


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(bool(row["failure"] and row["alarm"]) for row in rows)
    fn = sum(bool(row["failure"] and not row["alarm"]) for row in rows)
    fp = sum(bool(not row["failure"] and row["alarm"]) for row in rows)
    tn = sum(bool(not row["failure"] and not row["alarm"]) for row in rows)
    return {
        "episodes": len(rows),
        "failures": tp + fn,
        "successes": fp + tn,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "failure_recall": tp / max(tp + fn, 1),
        "success_false_alarm_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def make_figure(
    output: Path,
    episode_rows: list[dict[str, Any]],
    physical: dict[str, Any],
) -> None:
    failure_run = PACKAGE_ROOT / "results/task_free_self_reference_selector" / physical["run"]
    failure_dir = failure_run / (
        f"episode_{physical['episode_index']:03d}_init_{physical['init_state_id']:02d}_"
        f"flowseed_{physical['flow_noise_seed']}"
    )
    with np.load(failure_dir / "trajectory_and_routes.npz", allow_pickle=False) as archive:
        names = [str(item) for item in archive["selector_metric_names"]]
        metrics = np.asarray(archive["selector_metrics"], dtype=np.float64)
        query_steps = np.asarray(archive["action_steps_at_query"], dtype=int)
        position_slice = object_position_slice(failure_dir / "sim_layout.json", PLATE_JOINT)
        control_position = np.asarray(
            archive["control_sim_state"][:, position_slice], dtype=np.float64
        )
    lookup = {name: metrics[:, index] for index, name in enumerate(names)}
    query = np.arange(len(metrics))
    plate_displacement = np.linalg.norm(
        control_position[query_steps] - control_position[0], axis=1
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    labels = [f"i{row['init_state_id']}/s{row['flow_noise_seed']}" for row in episode_rows]
    colors = ["#c83e4d" if row["failure"] else "#2a9d8f" for row in episode_rows]
    axes[0].bar(np.arange(len(labels)), [row["inference_calls"] for row in episode_rows], color=colors)
    for index, row in enumerate(episode_rows):
        if row["alarm"]:
            axes[0].scatter(index, row["first_alarm_query"], marker="x", s=80, color="black", zorder=4)
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axes[0].set_ylabel("queries")
    axes[0].set_title("GPU6 prospective: green=success, red=failure, x=first MoE alarm")

    ratio_specs = (
        ("state_response_ratio", 0.25, "state response / prefix", "#2878b5"),
        ("planning_churn_ratio", 1.10, "action churn / prefix", "#e07a1f"),
        ("state_action_gap_ratio", 1.20, "state-action gap / prefix", "#7a5195"),
    )
    for name, threshold, label, color in ratio_specs:
        axes[1].plot(query, lookup[name], label=label, color=color, linewidth=2)
        axes[1].axhline(threshold, color=color, linestyle="--", alpha=0.45)
    axes[1].axvline(physical["plateau_query"], color="gray", linestyle=":", label="plate plateau")
    axes[1].axvline(physical["first_moe_alarm_query"], color="black", linestyle="-.", label="MoE alarm")
    axes[1].set_ylabel("self-reference ratio")
    axes[1].set_title("Failure trajectory: three route-only gates")
    axes[1].legend(ncol=2, fontsize=9)

    axes[2].plot(query, plate_displacement, color="#264653", linewidth=2.5)
    axes[2].axvline(physical["plateau_query"], color="gray", linestyle=":", label="posthoc plateau")
    axes[2].axvline(physical["first_moe_alarm_query"], color="black", linestyle="-.", label="MoE alarm")
    axes[2].axvline(physical["fixed_clock_query"], color="#d1495b", linestyle="--", label="phase-0.81 clock")
    axes[2].set_xlabel("query")
    axes[2].set_ylabel("plate displacement (m)")
    axes[2].set_title("Physical marker is posthoc only; it is not a selector input")
    axes[2].legend(fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def report_text(summary: dict[str, Any]) -> str:
    online = summary["gpu6_prospective"]
    heldout = summary["offline_heldout_39_tasks"]
    timing = summary["posthoc_physical_timing"][0]
    replay = summary["cross_gpu_same_seed_replay"]
    return f"""# 无任务先验 MoE 自参照报警：GPU6 审计

## 核心结论

这是一条 **train-free、运行时只读当前 episode 自身 HB MoE 路由前缀** 的窄型报警规则。它不读取任务 ID、正常轨迹库、动作、物理距离、reward、success 或 timeout。阈值查看过一个开发任务，因此不是开发阶段 label-free。

GPU6 前瞻性固定扫描共 {online['episodes']} 条：{online['successes']} 条成功全部不报警，{online['failures']} 条失败中 {online['tp']} 条报警；唯一失败在 q{timing['first_moe_alarm_query']} 首次报警并继续执行 {summary['gpu6_post_alarm_action_steps']} 个 control steps，未自行恢复。样本只有 1 个失败，不能把 100% recall/precision 当成总体性能。

## 报警到底发生在什么时候

失败轨迹中的盘子并非完全没动。按事后物理轨迹定义：盘子已移动至少 5 cm，且此后总 excursion 不超过 1 mm，则 plateau 从 q{timing['plateau_query']} 开始。MoE 在 q{timing['first_moe_alarm_query']} 报警，晚 {timing['moe_alarm_delay_after_plateau_queries']} 个 query；同 FPR 档的 phase-0.81 固定时钟在 q{timing['fixed_clock_query']} 才响，因此 MoE 比时钟早 {timing['moe_lead_over_fixed_clock_queries']} 个 query。

所以当前正确表述是：**错误物理状态稳定后，MoE 出现持续的 internal response collapse**。它不是这条轨迹的事前 precursor，也不能仅凭 routing 判定具体失败原因。

报警瞬间三个无量纲量为：state response `{timing['state_response_ratio_at_alarm']:.3f}`（阈值 <=0.25）、action planning churn `{timing['planning_churn_ratio_at_alarm']:.3f}`（阈值 >=1.10）、state/action gap `{timing['state_action_gap_ratio_at_alarm']:.3f}`（阈值 >=1.20）。

## 大规模离线边界

严格排除开发任务后，39 个未参与阈值设定的任务共有 {heldout['episodes']} 条：failure recall `{heldout['tp']}/{heldout['failures']} = {heldout['tpr']:.2%}`，success FPR `{heldout['fp']}/{heldout['successes']} = {heldout['fpr']:.3%}`，precision `{heldout['precision']:.2%}`。这是一个高精度、低召回的 phenotype head，不是通用失败检测器。

更重要的是，endpoint success/timeout 标签被 horizon 严重混杂：phase-0.81 固定时钟在 held-out 数据上达到 100% failure recall，FPR 约 0.21%。因此后续主评估必须改成物理或语义事件 onset，而不能继续只比较最终 success。

## 同 seed 跨 GPU 复现边界

旧 GPU3 的两条失败条件在 GPU6 上都成功。两端 checkpoint、代码 commit、diff hash 和 wrist layout 完全一致，q0 policy/sim state 也逐值相同；但 q0 action MAE 为 `{replay['action_mae_min']:.6f}`--`{replay['action_mae_max']:.6f}`，q0 route MAE 为 `{replay['route_mae_min']:.6f}`--`{replay['route_mae_max']:.6f}`。这些小数值差异在接触闭环中累积，改变了 outcome。因此这里应称为 **same-seed cross-GPU divergence audit**，不能称为精确失败复现。

## 文件

- `online_episodes.csv`：GPU6 16 条逐 episode 在线结果
- `physical_timing.csv`：物体 plateau、MoE alarm 与时钟的事后对齐
- `same_seed_cross_gpu.csv`：GPU3/GPU6 两个同 seed 条件的 q0 差异
- `gpu6_self_reference_audit.png`：在线结果、三项 MoE 比率和物理时序
- `summary.json`：机器可读汇总

报警视频位于：

- `../gpu6_prospective_init09_seeds1000_1007/episode_000_init_09_flowseed_1000/videos/alarm_clean_full.mp4`
- `../gpu6_prospective_init09_seeds1000_1007/episode_000_init_09_flowseed_1000/videos/alarm_annotated_full.mp4`
- `../gpu6_prospective_init09_seeds1000_1007/episode_000_init_09_flowseed_1000/videos/alarm_annotated_clip.mp4`
"""


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    online_rows, physical_rows = prospective_rows()
    replay_rows, runtime_identity = same_seed_replay_rows()
    offline = load_json(OFFLINE_SUMMARY)
    assert isinstance(offline, dict)
    online_counts = confusion(online_rows)
    clock_counts = confusion(
        [{**row, "alarm": row["fixed_clock_would_alarm"]} for row in online_rows]
    )
    action_mae = [float(row["q0_action_mae"]) for row in replay_rows]
    route_mae = [float(row["q0_route_mae"]) for row in replay_rows]
    summary = {
        "schema": "himoe.self_reference_gpu6_audit.v1",
        "training": False,
        "selector_version": SELECTOR_VERSION,
        "runtime_selector_inputs": [
            "current HB router probabilities",
            "own episode route prefix",
        ],
        "runtime_selector_excluded_inputs": [
            "task identity",
            "normal trajectory bank",
            "action values",
            "robot or simulator state",
            "physical distance",
            "reward",
            "success",
            "timeout",
        ],
        "development_outcomes_inspected": True,
        "gpu6_prospective": online_counts,
        "gpu6_fixed_clock_phase_0_81": clock_counts,
        "gpu6_post_alarm_action_steps": sum(
            int(row["post_alarm_action_steps"]) for row in online_rows
        ),
        "posthoc_physical_timing": physical_rows,
        "offline_heldout_39_tasks": offline["heldout_39_task_evaluation"],
        "offline_clock_controls_heldout": offline["clock_controls_heldout"],
        "cross_gpu_same_seed_replay": {
            "conditions": len(replay_rows),
            "old_failures": sum(not row["old_success"] for row in replay_rows),
            "new_successes": sum(row["new_success"] for row in replay_rows),
            "q0_policy_state_exact_conditions": sum(
                row["q0_policy_state_exact"] for row in replay_rows
            ),
            "q0_sim_state_exact_conditions": sum(
                row["q0_sim_state_exact"] for row in replay_rows
            ),
            "runtime_identity": runtime_identity,
            "action_mae_min": min(action_mae),
            "action_mae_max": max(action_mae),
            "route_mae_min": min(route_mae),
            "route_mae_max": max(route_mae),
            "interpretation": (
                "same seed is not bitwise closed-loop replay across GPUs; small q0 "
                "route/action differences compound after contact"
            ),
        },
        "invalid_startup_attempts_excluded": [
            "gpu6_exact_failure_replay",
            "gpu6_exact_failure_replay_retry1",
            "gpu6_exact_failure_replay_retry2",
        ],
        "scientific_scope": (
            "high-precision internal response-collapse phenotype head; not a universal "
            "failure detector and not a failure-cause classifier"
        ),
    }

    write_csv(args.out / "online_episodes.csv", online_rows)
    write_csv(args.out / "physical_timing.csv", physical_rows)
    write_csv(args.out / "same_seed_cross_gpu.csv", replay_rows)
    make_figure(args.out / "gpu6_self_reference_audit.png", online_rows, physical_rows[0])
    (args.out / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "REPORT_ZH.md").write_text(report_text(summary), encoding="utf-8")
    print(json.dumps(plain(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
