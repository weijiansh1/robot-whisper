#!/usr/bin/env python3
"""Post-hoc audit for a prospective MoE dynamics online-alarm run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from audit_moe_only_online_failure_types import (
    aggregate,
    load_taxonomy_module,
    markdown_table,
    missed_grasp_events,
    plain,
    target_joints,
    write_json,
)
from moe_only_online_selector import HealthySequenceBank, MoeOnlyOnlineAlarm


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    PACKAGE_ROOT / "results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908"
)
DEFAULT_SERVER = (
    PACKAGE_ROOT / "results/moe_dynamics_online_alarm/gpu5_server_seed20260908"
)
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/moe_dynamics_online_alarm/prospective_audit"
V1_AUDIT = PACKAGE_ROOT / "results/moe_only_online_alarm/failure_type_audit/summary.json"
HEALTHY_REFERENCE = (
    PACKAGE_ROOT
    / "results/moe_only_online_alarm/h20_healthy_route_sequences_seed20260905.npz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(hits: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return []
    rate = hits / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [float(center - half), float(center + half)]


def exact_mcnemar(v2_only: int, v1_only: int) -> float:
    discordant = v2_only + v1_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(v2_only, v1_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    both = sum(row["formal_alarm"] and row["v1_formal_alarm"] for row in rows)
    v2_only = sum(row["formal_alarm"] and not row["v1_formal_alarm"] for row in rows)
    v1_only = sum(not row["formal_alarm"] and row["v1_formal_alarm"] for row in rows)
    neither = len(rows) - both - v2_only - v1_only
    return {
        "episodes": len(rows),
        "both": both,
        "v2_only": v2_only,
        "v1_only": v1_only,
        "neither": neither,
        "exact_mcnemar_two_sided_p": exact_mcnemar(v2_only, v1_only),
    }


def capture_audit(
    loaded: list[dict[str, Any]], server_root: Path
) -> dict[str, Any]:
    client_routes = np.concatenate(
        [item["arrays"]["hb_router_probs"] for item in loaded], axis=0
    ).astype(np.float16)
    client_episode_ids = np.concatenate(
        [
            np.full(
                len(item["arrays"]["hb_router_probs"]),
                int(item["summary"]["episode_id"]),
                dtype=np.int32,
            )
            for item in loaded
        ]
    )
    store = zarr.open_group(str(server_root / "routes.zarr"), mode="r")
    server_routes = np.asarray(store["hb_router_probs"][:], dtype=np.float16)
    server_episode_ids = np.asarray(store["episode_id"][:], dtype=np.int32)
    control_steps = np.asarray(store["control_step"][:], dtype=np.int32)
    capture = json.loads(
        (server_root / "capture_summary.json").read_text(encoding="utf-8")
    )
    return {
        "client_rows": len(client_routes),
        "server_rows": len(server_routes),
        "route_shape": list(server_routes.shape),
        "client_server_full_routes_exact": bool(
            np.array_equal(client_routes, server_routes)
        ),
        "client_server_episode_ids_exact": bool(
            np.array_equal(client_episode_ids, server_episode_ids)
        ),
        "server_control_steps_strictly_sequential": bool(
            np.array_equal(control_steps, np.arange(len(control_steps), dtype=np.int32))
        ),
        "store_full_probs": bool(capture["store_full_probs"]),
        "return_full_probs": bool(capture["return_full_probs"]),
        "hook_verify_failures": list(capture["hook_verify_failures"]),
    }


def replay_v1(routes: np.ndarray, bank: HealthySequenceBank) -> tuple[list[int], list[int]]:
    selector = MoeOnlyOnlineAlarm(bank, persistence=2, max_advance=2)
    raw_queries: list[int] = []
    alarm_queries: list[int] = []
    for route in routes:
        decision = selector.update(route)
        if decision is None:
            continue
        if decision.raw_reject:
            raw_queries.append(decision.query)
        if decision.alarm:
            alarm_queries.append(decision.query)
    return raw_queries, alarm_queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run = args.run.resolve()
    server = args.server.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["training"] is not False:
        raise RuntimeError("prospective run is incomplete or declares training")
    if manifest["selector_version"] != "back_front_route_acceleration_v2":
        raise RuntimeError("unexpected selector version")

    taxonomy = load_taxonomy_module()
    loaded: list[dict[str, Any]] = []
    layout: dict[str, Any] | None = None
    for episode_dir in sorted(run.glob("episode_*")):
        summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
        current_layout = json.loads(
            (episode_dir / "sim_layout.json").read_text(encoding="utf-8")
        )
        if layout is None:
            layout = current_layout
        elif current_layout != layout:
            raise RuntimeError("simulator layouts differ")
        with np.load(episode_dir / "trajectory_and_routes.npz", allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        loaded.append(
            {
                "episode_dir": episode_dir,
                "summary": summary,
                "arrays": arrays,
            }
        )
    if layout is None or len(loaded) != manifest["completed_episodes"]:
        raise RuntimeError("episode set does not match manifest")
    targets = target_joints(layout)
    v1_bank = HealthySequenceBank.load(HEALTHY_REFERENCE)

    successes = [item for item in loaded if item["summary"]["success"]]
    if not successes:
        raise RuntimeError("generic taxonomy requires successful goal references")
    shared_goals = np.stack(
        [
            item["arrays"]["control_sim_state"][-1, int(target["state_lo"]) : int(target["state_lo"]) + 3]
            for target in targets
            for item in successes
        ]
    ).astype(np.float32)
    references = {str(target["joint"]): shared_goals.copy() for target in targets}

    rows: list[dict[str, Any]] = []
    events_out: list[dict[str, Any]] = []
    for item in loaded:
        summary = item["summary"]
        arrays = item["arrays"]
        candidate = taxonomy.Candidate(
            worker=5,
            init_state=int(summary["init_state_id"]),
            snapshot=0,
            candidate=int(summary["episode_index"]),
            episode_id=int(summary["episode_id"]),
            success=bool(summary["success"]),
            inference_calls=int(summary["inference_calls"]),
            snapshot_key="gpu5_prospective_v2",
            npz_path=item["episode_dir"] / "trajectory_and_routes.npz",
            json_path=item["episode_dir"] / "summary.json",
        )
        generic = taxonomy.physical_metrics(candidate, arrays, targets, references)
        events = missed_grasp_events(arrays, targets)
        for event in events:
            events_out.append(
                {
                    "episode_index": int(summary["episode_index"]),
                    "init_state_id": int(summary["init_state_id"]),
                    "success": bool(summary["success"]),
                    **event,
                }
            )
        missed = [
            event
            for event in events
            if event["kinematic_missed_grasp_then_departure"]
        ]
        borderline = [
            event
            for event in events
            if event["missed_grasp_sensitivity_only_15mm"]
        ]
        success = bool(summary["success"])
        family = (
            "success"
            if success
            else "target_missed_grasp_proxy"
            if missed
            else "other_failure"
        )
        family_15mm = (
            "success"
            if success
            else "target_missed_grasp_proxy"
            if missed or borderline
            else "other_failure"
        )
        raw_queries = [int(value) for value in summary["raw_reject_queries"]]
        alarm_queries = [int(value) for value in summary["alarm_queries"]]
        v1_raw_queries, v1_alarm_queries = replay_v1(
            arrays["hb_router_probs"], v1_bank
        )
        missed_queries = [int(event["query"]) for event in missed]
        deltas = [
            alarm - miss
            for alarm in alarm_queries
            for miss in missed_queries
        ]
        rows.append(
            {
                "episode_index": int(summary["episode_index"]),
                "init_state_id": int(summary["init_state_id"]),
                "success": success,
                "failure_family": family,
                "failure_family_sensitivity_15mm": family_15mm,
                "generic_primary": str(generic["primary_failure_type"]),
                "generic_labels": str(generic["physical_labels"]),
                "missed_grasp_queries": json.dumps(missed_queries, separators=(",", ":")),
                "borderline_15mm_queries": json.dumps(
                    [int(event["query"]) for event in borderline], separators=(",", ":")
                ),
                "raw_reject_queries": json.dumps(raw_queries, separators=(",", ":")),
                "formal_alarm_queries": json.dumps(alarm_queries, separators=(",", ":")),
                "any_raw_reject": bool(raw_queries),
                "formal_alarm": bool(alarm_queries),
                "v1_raw_reject_queries": json.dumps(
                    v1_raw_queries, separators=(",", ":")
                ),
                "v1_formal_alarm_queries": json.dumps(
                    v1_alarm_queries, separators=(",", ":")
                ),
                "v1_any_raw_reject": bool(v1_raw_queries),
                "v1_formal_alarm": bool(v1_alarm_queries),
                "nearest_alarm_minus_missed_query": (
                    None
                    if not deltas
                    else min(deltas, key=lambda value: (abs(value), value))
                ),
                "alarm_before_or_at_missed_grasp": bool(
                    any(-5 <= delta <= 0 for delta in deltas)
                ),
                "alarm_within_5_after_missed_grasp": bool(
                    any(0 < delta <= 5 for delta in deltas)
                ),
                "trajectory": str(
                    (item["episode_dir"] / "trajectory_and_routes.npz").relative_to(
                        PACKAGE_ROOT
                    )
                ),
            }
        )

    with (output / "episode_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "grasp_events.jsonl").open("w", encoding="utf-8") as stream:
        for event in events_out:
            stream.write(json.dumps(plain(event), sort_keys=True) + "\n")

    primary = aggregate(rows)
    sensitivity = aggregate(rows, family_key="failure_family_sensitivity_15mm")
    v1_rows = [
        {
            **row,
            "any_raw_reject": row["v1_any_raw_reject"],
            "formal_alarm": row["v1_formal_alarm"],
        }
        for row in rows
    ]
    v1_paired = aggregate(v1_rows)
    target = primary["target_missed_grasp_proxy"]
    other = primary["other_failure"]
    successful = primary["success"]
    failure_hits = (
        target["episodes_with_formal_alarm"]["numerator"]
        + other["episodes_with_formal_alarm"]["numerator"]
    )
    failure_total = target["episodes"] + other["episodes"]
    success_hits = successful["episodes_with_formal_alarm"]["numerator"]
    v1_failure_hits = sum(
        row["v1_formal_alarm"] for row in rows if not row["success"]
    )
    v1 = json.loads(V1_AUDIT.read_text(encoding="utf-8"))["by_run"]["gpu5_extended"]
    config = json.loads((run / "experiment_config.json").read_text(encoding="utf-8"))
    calibration_path = Path(config["dynamics_calibration"]).resolve()
    calibration_digest = sha256_file(calibration_path)
    if calibration_digest != config["dynamics_calibration_sha256"]:
        raise RuntimeError("calibration changed after prospective collection")
    summary = {
        "schema": "himoe.moe_dynamics_prospective_audit.v1",
        "training": False,
        "selector_version": manifest["selector_version"],
        "selector_uses_physical_state": False,
        "physical_data_role": "posthoc_failure_typing_only",
        "fixed_episode_count": bool(manifest["fixed_episode_count"]),
        "episodes": len(rows),
        "successes": int(manifest["successes"]),
        "failures": int(manifest["failures"]),
        "calibration_sha256": calibration_digest,
        "calibration_unchanged_after_prospective_run": True,
        "primary": primary,
        "sensitivity_15mm": sensitivity,
        "v1_paired_same_trajectories": v1_paired,
        "all_failure_formal_alarm": {
            "numerator": failure_hits,
            "denominator": failure_total,
            "rate": failure_hits / failure_total,
            "wilson_95": wilson(failure_hits, failure_total),
        },
        "target_formal_alarm": {
            "numerator": target["episodes_with_formal_alarm"]["numerator"],
            "denominator": target["episodes"],
            "rate": target["episodes_with_formal_alarm"]["rate"],
            "wilson_95": wilson(
                target["episodes_with_formal_alarm"]["numerator"],
                target["episodes"],
            ),
        },
        "v1_all_failure_formal_alarm": {
            "numerator": v1_failure_hits,
            "denominator": failure_total,
            "rate": v1_failure_hits / failure_total,
            "wilson_95": wilson(v1_failure_hits, failure_total),
        },
        "successful_episode_false_alarm": {
            "numerator": success_hits,
            "denominator": successful["episodes"],
            "rate": success_hits / successful["episodes"],
            "wilson_95": wilson(success_hits, successful["episodes"]),
        },
        "paired_v2_vs_v1": {
            "target_missed_grasp_proxy": paired_counts(
                [row for row in rows if row["failure_family"] == "target_missed_grasp_proxy"]
            ),
            "all_failures": paired_counts(
                [row for row in rows if not row["success"]]
            ),
            "success": paired_counts(
                [row for row in rows if row["success"]]
            ),
        },
        "generic_failure_counts": dict(
            Counter(row["generic_primary"] for row in rows if not row["success"])
        ),
        "target_alarm_timing": [
            {
                "episode_index": row["episode_index"],
                "missed_grasp_queries": json.loads(row["missed_grasp_queries"]),
                "alarm_queries": json.loads(row["formal_alarm_queries"]),
                "nearest_alarm_minus_missed_query": row[
                    "nearest_alarm_minus_missed_query"
                ],
            }
            for row in rows
            if row["failure_family"] == "target_missed_grasp_proxy"
        ],
        "v1_gpu5_reference": v1,
        "capture_audit": capture_audit(loaded, server),
        "online_alarm_video_files": len(manifest["videos"]),
        "feature_design_informed_by_prior_gpu5_batch": True,
        "threshold_changed_during_prospective_run": False,
    }
    write_json(output / "summary.json", summary)

    display = [
        {
            "ep/init": f"{row['episode_index']}/{row['init_state_id']}",
            "outcome": "success" if row["success"] else "failure",
            "family": row["failure_family"],
            "generic": row["generic_primary"],
            "miss q": row["missed_grasp_queries"],
            "alarm q": row["formal_alarm_queries"],
        }
        for row in rows
    ]
    report = f"""# MoE dynamics v2 prospective 在线审计

固定 {len(rows)} 条：{manifest['successes']} 成功、{manifest['failures']} 失败。v2 严格目标漏抓报警 {target['episodes_with_formal_alarm']['numerator']}/{target['episodes']}，其他失败 {other['episodes_with_formal_alarm']['numerator']}/{other['episodes']}，成功误报 {success_hits}/{successful['episodes']}。全部失败召回 {failure_hits}/{failure_total}。旧 v1 规则在完全相同轨迹上的结果保存在 `v1_paired_same_trajectories`。

阈值仅由旧健康 reference 的 leave-one-out 序列上界确定。物理量只在 rollout 完成后用于本审计；特征设计受上一批 GPU5 posthoc 结果启发，所以本批才是 v2 的第一次 prospective test。

{markdown_table(display, ['ep/init', 'outcome', 'family', 'generic', 'miss q', 'alarm q'])}
"""
    (output / "report_zh.md").write_text(report, encoding="utf-8")
    print(json.dumps(plain(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
