#!/usr/bin/env python3
"""Analyze the frozen paired short-horizon Trap-recovery pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "himoe.moe_horizon_recovery.v1"
ANALYSIS_SCHEMA = "himoe.moe_horizon_recovery_analysis.v1"
ALARM_THRESHOLD = 0.2945967216491699
BOOTSTRAP_SEED = 20260906
BOOTSTRAP_RESAMPLES = 20_000
EPS = 1e-12


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _t_statistic(means: np.ndarray, sum_squares: np.ndarray, n: int) -> np.ndarray:
    variances = (sum_squares[None, :] - n * np.square(means)) / max(n - 1, 1)
    standard_errors = np.sqrt(np.maximum(variances, 0.0) / n)
    result = np.divide(
        means,
        standard_errors,
        out=np.zeros_like(means),
        where=standard_errors > EPS,
    )
    degenerate = standard_errors <= EPS
    result[degenerate & (means > EPS)] = np.inf
    result[degenerate & (means < -EPS)] = -np.inf
    return result


def exact_joint_sign_flip_max_t(
    values: np.ndarray, chunk_size: int = 65_536
) -> Dict[str, Any]:
    """Exact paired sign-flip inference with one sign per pair across cells."""
    differences = np.asarray(values, np.float64)
    if differences.ndim != 2 or differences.shape[0] < 2:
        raise ValueError("values must be pair x cell with at least two pairs")
    n, cells = differences.shape
    if n > 24:
        raise ValueError("exact enumeration is intentionally capped at 24 pairs")
    sum_squares = np.square(differences).sum(axis=0)
    observed_mean = differences.mean(axis=0)
    observed_t = _t_statistic(observed_mean[None, :], sum_squares, n)[0]
    total = 1 << n
    raw_counts = np.zeros(cells, np.int64)
    max_counts = np.zeros(cells, np.int64)
    bit_positions = np.arange(n, dtype=np.uint64)
    for left in range(0, total, chunk_size):
        right = min(left + chunk_size, total)
        integers = np.arange(left, right, dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((integers >> bit_positions[None, :]) & 1).astype(
            np.float64
        )
        means = signs @ differences / n
        statistics = _t_statistic(means, sum_squares, n)
        absolute = np.abs(statistics)
        maximum = absolute.max(axis=1)
        raw_counts += np.sum(absolute >= np.abs(observed_t)[None, :], axis=0)
        max_counts += np.sum(maximum[:, None] >= np.abs(observed_t)[None, :], axis=0)
    return {
        "permutations": total,
        "observed_t": observed_t.tolist(),
        "raw_p": (raw_counts / total).tolist(),
        "max_t_p": (max_counts / total).tolist(),
    }


def paired_summary(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
) -> Dict[str, Any]:
    sample = np.asarray(values, np.float64)
    indices = rng.integers(0, len(sample), size=(n_bootstrap, len(sample)))
    boot = sample[indices].mean(axis=1)
    low, high = np.quantile(boot, (0.025, 0.975))
    return {
        "n_pairs": len(sample),
        "mean": float(sample.mean()),
        "median": float(np.median(sample)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "rescues": int(np.count_nonzero(sample > 0)),
        "harms": int(np.count_nonzero(sample < 0)),
        "ties": int(np.count_nonzero(sample == 0)),
    }


def goal_distance(objects: np.ndarray, references: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        objects[:, :, None, :] - references[None, None, :, :], axis=-1
    ).min(axis=-1)
    return distances.max(axis=1)


def loop_onset_index(
    eef: np.ndarray,
    objects: np.ndarray,
    gripper: np.ndarray,
    references: np.ndarray,
) -> Tuple[int, int]:
    goal = goal_distance(objects, references)
    step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(step)]
    for right in range(3, len(eef)):
        for left in range(0, right - 2):
            if (
                np.linalg.norm(eef[right] - eef[left]) <= 0.045
                and np.linalg.norm(objects[right] - objects[left], axis=1).max()
                <= 0.030
                and abs(gripper[right] - gripper[left]) <= 0.012
                and cumulative[right] - cumulative[left] >= 0.120
                and goal[left] - goal[right] <= 0.035
            ):
                return right, left
    return -1, -1


def _arm_metrics(
    npz_path: Path, summary: Mapping[str, Any], references: np.ndarray
) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    success_by_step = arrays["success_by_step"].astype(bool)
    fixed_grid = np.arange(0, len(success_by_step), 10, dtype=np.int32)
    sim = arrays["sim_state"][fixed_grid]
    if sim.shape[1] < 20:
        raise RuntimeError("task-8 simulator state is too short")
    objects = np.stack((sim[:, 10:13], sim[:, 17:20]), axis=1).astype(np.float64)
    eef = arrays["eef_position"][fixed_grid].astype(np.float64)
    gripper = arrays["gripper_qpos"][fixed_grid].mean(axis=1).astype(np.float64)
    onset, partner = loop_onset_index(eef, objects, gripper, references)
    distances = goal_distance(objects, references)
    metrics = {
        "success": bool(summary["success"]),
        "success_action": summary["success_action"],
        "actions_executed": int(summary["actions_executed"]),
        "queries": int(summary["queries"]),
        "inference_seconds": float(summary["inference_seconds"]),
        "physical_recurrence": bool(onset >= 0),
        "physical_recurrence_action": None if onset < 0 else int(fixed_grid[onset]),
        "physical_recurrence_partner_action": None
        if partner < 0
        else int(fixed_grid[partner]),
        "goal_distance_start": float(distances[0]),
        "goal_distance_final_grid": float(distances[-1]),
        "goal_distance_best": float(distances.min()),
    }
    for horizon in (50, 100, 150, 200):
        metrics["success_by_%d" % horizon] = bool(
            success_by_step[: min(horizon + 1, len(success_by_step))].any()
        )
    return metrics


def random_subset_targeting(benefits: np.ndarray, alarm: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(benefits, np.float64)
    selected = np.asarray(alarm, bool)
    n = len(values)
    k = int(selected.sum())
    observed_sum = float(values[selected].sum())
    subset_sums = np.asarray(
        [
            values[list(indices)].sum()
            for indices in itertools.combinations(range(n), k)
        ],
        np.float64,
    )
    return {
        "n": n,
        "selected": k,
        "subsets": len(subset_sums),
        "alarm_gated_success_gain": observed_sum / n,
        "selected_mean_benefit": observed_sum / max(k, 1),
        "random_subset_gain_mean": float(subset_sums.mean() / n),
        "random_subset_gain_ci95_low": float(np.quantile(subset_sums / n, 0.025)),
        "random_subset_gain_ci95_high": float(np.quantile(subset_sums / n, 0.975)),
        "one_sided_p_random_at_least_alarm": float(
            np.count_nonzero(subset_sums >= observed_sum - EPS) / len(subset_sums)
        ),
    }


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    protocol_sha256 = _sha256_file(args.protocol)
    definitions = _load_json(args.physical_definitions)
    references = np.asarray(
        definitions["success_terminal_references"]["moka_pot_1_joint0"], np.float64
    )
    manifests = []
    states = []
    for shard in range(args.num_shards):
        directory = args.run_root / ("shard%d" % shard)
        manifest = _load_json(directory / "manifest.json")
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
            raise RuntimeError("shard %d is incomplete" % shard)
        if manifest.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(
                "shard %d protocol hash differs from the frozen file" % shard
            )
        if not manifest.get("oom_kill_unchanged"):
            raise RuntimeError("shard %d observed an OOM kill" % shard)
        for phase in ("preflight_power", "postflight_power"):
            power = manifest[phase]
            if power["power_limit_w"] != power["power_max_limit_w"]:
                raise RuntimeError("shard %d GPU power was not maximal" % shard)
        manifests.append(manifest)
        states.extend(sorted(directory.glob("pair_*_event.json")))
        states.extend(sorted(directory.glob("pair_*_control.json")))
    if len(states) != 46:
        raise RuntimeError("expected 46 complete states, found %d" % len(states))

    rows = []
    seen = set()
    for state_path in states:
        state = _load_json(state_path)
        key = (int(state["pair_id"]), str(state["role"]))
        if key in seen:
            raise RuntimeError("duplicate state %s" % (key,))
        seen.add(key)
        if not all(state["hard_checks"].values()):
            raise RuntimeError("state %s failed a paired validity gate" % (key,))
        row = {
            "pair_id": key[0],
            "role": key[1],
            "score": float(state["score"]),
            "alarm": bool(state["alarm"]),
        }
        for arm in ("h10", "h2_burst10"):
            summary = state[arm]
            npz_path = state_path.parent / "arms" / summary["npz"]
            metrics = _arm_metrics(npz_path, summary, references)
            for name, value in metrics.items():
                row[arm + "_" + name] = value
        row["success_effect"] = int(row["h2_burst10_success"]) - int(row["h10_success"])
        row["recurrence_effect"] = int(row["h2_burst10_physical_recurrence"]) - int(
            row["h10_physical_recurrence"]
        )
        row["query_cost"] = int(row["h2_burst10_queries"]) - int(row["h10_queries"])
        rows.append(row)
    rows.sort(key=lambda row: (int(row["pair_id"]), 0 if row["role"] == "event" else 1))
    if seen != {
        (pair, role) for pair in range(24) if pair != 4 for role in ("event", "control")
    }:
        raise RuntimeError("complete state keys differ from the frozen eligible set")

    pairs = []
    for pair_id in sorted({int(row["pair_id"]) for row in rows}):
        event = next(
            row for row in rows if row["pair_id"] == pair_id and row["role"] == "event"
        )
        control = next(
            row
            for row in rows
            if row["pair_id"] == pair_id and row["role"] == "control"
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "event_effect": int(event["success_effect"]),
                "control_effect": int(control["success_effect"]),
                "difference_in_differences": int(event["success_effect"])
                - int(control["success_effect"]),
                "event_alarm": bool(event["alarm"]),
                "control_alarm": bool(control["alarm"]),
            }
        )
    event_effect = np.asarray([pair["event_effect"] for pair in pairs], np.float64)
    control_effect = np.asarray([pair["control_effect"] for pair in pairs], np.float64)
    did = event_effect - control_effect
    primary_matrix = np.column_stack((event_effect, did))
    exact = exact_joint_sign_flip_max_t(primary_matrix)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary = []
    for index, (name, values) in enumerate(
        (("event_state_effect", event_effect), ("difference_in_differences", did))
    ):
        cell = paired_summary(values, rng)
        cell.update(
            {
                "estimand": name,
                "exact_t": exact["observed_t"][index],
                "exact_p_raw": exact["raw_p"][index],
                "exact_p_max_t": exact["max_t_p"][index],
            }
        )
        primary.append(cell)

    event_rows = [row for row in rows if row["role"] == "event"]
    control_rows = [row for row in rows if row["role"] == "control"]
    event_alarm = np.asarray([row["alarm"] for row in event_rows], bool)
    control_alarm = np.asarray([row["alarm"] for row in control_rows], bool)
    event_h10 = np.asarray([row["h10_success"] for row in event_rows], bool)
    event_h2 = np.asarray([row["h2_burst10_success"] for row in event_rows], bool)
    control_h10 = np.asarray([row["h10_success"] for row in control_rows], bool)
    control_h2 = np.asarray([row["h2_burst10_success"] for row in control_rows], bool)
    event_gated = np.where(event_alarm, event_h2, event_h10)
    control_gated = np.where(control_alarm, control_h2, control_h10)
    targeting = random_subset_targeting(event_effect, event_alarm)
    targeting.update(
        {
            "event_alarms": int(event_alarm.sum()),
            "control_false_alarms": int(control_alarm.sum()),
            "event_h10_successes": int(event_h10.sum()),
            "event_alarm_gated_successes": int(event_gated.sum()),
            "event_alarm_gated_success_rate": float(event_gated.mean()),
            "control_h10_successes": int(control_h10.sum()),
            "control_alarm_gated_successes": int(control_gated.sum()),
            "control_alarm_gated_success_rate": float(control_gated.mean()),
            "control_alarm_gated_success_gain": float(
                np.mean(control_effect * control_alarm)
            ),
            "event_always_short_success_gain": float(event_effect.mean()),
            "control_always_short_success_gain": float(control_effect.mean()),
        }
    )

    descriptive = {}
    for role, role_rows in (("event", event_rows), ("control", control_rows)):
        descriptive[role] = {}
        for arm in ("h10", "h2_burst10"):
            successes = np.asarray([row[arm + "_success"] for row in role_rows], bool)
            success_actions = [
                int(row[arm + "_success_action"])
                for row in role_rows
                if row[arm + "_success_action"] is not None
            ]
            success_queries = [
                int(row[arm + "_queries"]) for row in role_rows if row[arm + "_success"]
            ]
            descriptive[role][arm] = {
                "successes": int(successes.sum()),
                "n": len(role_rows),
                "success_rate": float(successes.mean()),
                "physical_recurrences": int(
                    sum(bool(row[arm + "_physical_recurrence"]) for row in role_rows)
                ),
                "physical_recurrence_rate": float(
                    np.mean([row[arm + "_physical_recurrence"] for row in role_rows])
                ),
                "mean_queries": float(
                    np.mean([row[arm + "_queries"] for row in role_rows])
                ),
                "mean_actions": float(
                    np.mean([row[arm + "_actions_executed"] for row in role_rows])
                ),
                "mean_actions_to_success": (
                    float(np.mean(success_actions)) if success_actions else None
                ),
                "mean_queries_to_success": (
                    float(np.mean(success_queries)) if success_queries else None
                ),
                "mean_inference_seconds": float(
                    np.mean([row[arm + "_inference_seconds"] for row in role_rows])
                ),
            }
            for horizon in (50, 100, 150, 200):
                values = [row[arm + "_success_by_%d" % horizon] for row in role_rows]
                descriptive[role][arm]["successes_by_%d" % horizon] = int(sum(values))
                descriptive[role][arm]["success_rate_by_%d" % horizon] = float(
                    np.mean(values)
                )
        descriptive[role]["mean_query_cost"] = float(
            np.mean([row["query_cost"] for row in role_rows])
        )
        descriptive[role]["mean_recurrence_effect"] = float(
            np.mean([row["recurrence_effect"] for row in role_rows])
        )
        descriptive[role]["recurrence_prevented"] = int(
            sum(row["recurrence_effect"] < 0 for row in role_rows)
        )
        descriptive[role]["recurrence_induced"] = int(
            sum(row["recurrence_effect"] > 0 for row in role_rows)
        )

    result = {
        "schema": ANALYSIS_SCHEMA,
        "passed": True,
        "run_root": str(args.run_root.resolve()),
        "protocol_sha256": protocol_sha256,
        "n_pairs": len(pairs),
        "n_states": len(rows),
        "alarm_threshold": ALARM_THRESHOLD,
        "primary": primary,
        "exact_joint_sign_flip": exact,
        "targeting": targeting,
        "descriptive": descriptive,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "shard_manifests": manifests,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "horizon_recovery_rows.csv", rows)
    _write_csv(args.out_dir / "horizon_recovery_pairs.csv", pairs)
    (args.out_dir / "horizon_recovery_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "horizon_recovery_report.md").write_text(
        render_report(result), encoding="utf-8"
    )
    return result


def render_report(result: Mapping[str, Any]) -> str:
    primary = {row["estimand"]: row for row in result["primary"]}
    event = primary["event_state_effect"]
    did = primary["difference_in_differences"]
    desc = result["descriptive"]
    target = result["targeting"]
    lines = [
        "# MoE 触发的短时域 Trap 恢复 pilot",
        "",
        "## 主要结果",
        "",
        "- Trap 前 event 状态：h10 成功 %d/%d，h2-burst10 成功 %d/%d；配对净变化 %.1f 个百分点（95%% bootstrap CI %.1f 至 %.1f；maxT p=%.4g）。"
        % (
            desc["event"]["h10"]["successes"],
            desc["event"]["h10"]["n"],
            desc["event"]["h2_burst10"]["successes"],
            desc["event"]["h2_burst10"]["n"],
            100 * event["mean"],
            100 * event["ci95_low"],
            100 * event["ci95_high"],
            event["exact_p_max_t"],
        ),
        "- 健康匹配状态：h10 成功 %d/%d，h2-burst10 成功 %d/%d。event-minus-control 的差分差分为 %.1f 个百分点（95%% CI %.1f 至 %.1f；maxT p=%.4g）。"
        % (
            desc["control"]["h10"]["successes"],
            desc["control"]["h10"]["n"],
            desc["control"]["h2_burst10"]["successes"],
            desc["control"]["h2_burst10"]["n"],
            100 * did["mean"],
            100 * did["ci95_low"],
            100 * did["ci95_high"],
            did["exact_p_max_t"],
        ),
        "- 这里正的差分差分完全来自 control 的成功率下降，不是 event rescue：event effect=0/23，control 中有 0 次 rescue、2 次 harm。",
        "",
        "## 检测到恢复",
        "",
        "- 固定 cosine 报警命中 event %d/%d，健康误报 %d/%d。"
        % (
            target["event_alarms"],
            result["n_pairs"],
            target["control_false_alarms"],
            result["n_pairs"],
        ),
        "- 报警后才启用短 burst 的 event 成功率增量为 %.1f 个百分点；等数量随机触发的均值 %.1f 个百分点，单侧精确子集 p=%.4g。"
        % (
            100 * target["alarm_gated_success_gain"],
            100 * target["random_subset_gain_mean"],
            target["one_sided_p_random_at_least_alarm"],
        ),
        "- 报警门控策略的绝对结果：event %d/%d（always-h10 为 %d/%d），control %d/%d（always-h10 为 %d/%d）。"
        % (
            target["event_alarm_gated_successes"],
            result["n_pairs"],
            target["event_h10_successes"],
            result["n_pairs"],
            target["control_alarm_gated_successes"],
            result["n_pairs"],
            target["control_h10_successes"],
            result["n_pairs"],
        ),
        "- 平均额外查询成本：event %.2f，control %.2f。"
        % (desc["event"]["mean_query_cost"], desc["control"]["mean_query_cost"]),
        "",
        "## 时间与 recurrence",
        "",
        "| 状态 | arm | success@50 | @100 | @150 | @200 | recurrence | mean queries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("event", "control"):
        for arm in ("h10", "h2_burst10"):
            cell = desc[role][arm]
            lines.append(
                "| %s | %s | %d/%d | %d/%d | %d/%d | %d/%d | %d/%d | %.2f |"
                % (
                    role,
                    arm,
                    cell["successes_by_50"],
                    cell["n"],
                    cell["successes_by_100"],
                    cell["n"],
                    cell["successes_by_150"],
                    cell["n"],
                    cell["successes_by_200"],
                    cell["n"],
                    cell["physical_recurrences"],
                    cell["n"],
                    cell["mean_queries"],
                )
            )
    lines.extend(
        [
            "",
            "- 短 burst 相对 h10 防止/诱发 recurrence：event %d/%d，control %d/%d。"
            % (
                desc["event"]["recurrence_prevented"],
                desc["event"]["recurrence_induced"],
                desc["control"]["recurrence_prevented"],
                desc["control"]["recurrence_induced"],
            ),
            "",
            "## 解释边界",
            "",
            "这是同一重建状态上的因果 A/B，但仍是单任务、单个未来噪声流的 pilot。它否定的是前十个物理步采用 h=2 的局部 burst，不是否定更长 cooldown、持续 receding horizon 或能改变物理状态的恢复动作。检测阈值来自同一批健康控制，必须在新 rollout 上做前瞻验证。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument(
        "--physical-definitions",
        type=Path,
        default=ROOT
        / "trap-recovery-depth-20260904/configs/task08_physical_trap_definitions.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "analysis_moe_execution_signals/RECOVERY_HORIZON_PROTOCOL.md",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "analysis_moe_execution_signals"
    )
    return parser


def main() -> int:
    result = analyze(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
