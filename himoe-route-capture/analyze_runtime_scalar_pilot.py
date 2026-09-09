"""Descriptive fresh-state runtime pilot for frozen HB5 scalar signals.

Five K=16 activation-flow-v2 captures each contain one task-specific fresh
state.  The frozen raw/rebound checks use equal-stratum Kendall-style pair
concordance.  Per-round raw, D_abs, and C_abs checks against the flow trajectory
are descriptive scans only; this script deliberately computes no p-values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from himoe_activation_store import SUPPORTED_FORMATS


TASK_RUNS = (
    ("goal-middle", "activation-raw-goal-t0s24-k16"),
    ("goal-top", "runtime-raw-goal-top-k16"),
    ("long-t08", "runtime-raw-long-t08-k16"),
    ("spatial-ramekin", "runtime-raw-spatial-ramekin-k16"),
    ("spatial-stove", "runtime-raw-spatial-stove-k16"),
)
FROZEN_FEATURES = ("raw.d6", "raw.d7", "raw.d8", "weighted.rebound")
FROZEN_ORIENTATION = {
    "raw.d6": -1.0,
    "raw.d7": -1.0,
    "raw.d8": -1.0,
    "weighted.rebound": 1.0,
}
N_EXPERTS = 32
N_DENOISE = 10
LIVE_DIMS = 7


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=here / "runs")
    parser.add_argument(
        "--offline-summary",
        type=Path,
        default=here / "analysis" / "fixed-expert-flow-displacement" / "summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "analysis" / "runtime-scalar-pilot",
    )
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--min-selected", type=int, default=4)
    return parser.parse_args()


def _pair_concordance(feature: np.ndarray, target: np.ndarray) -> float:
    """Kendall-style pair concordance with zero contribution from ties."""
    feature = np.asarray(feature, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if feature.ndim != 1 or target.shape != feature.shape:
        raise ValueError("feature and target must be aligned one-dimensional arrays")
    if len(feature) < 2:
        raise ValueError("pair concordance needs at least two candidates")
    pair_i, pair_j = np.triu_indices(len(feature), 1)
    return float(
        np.mean(
            np.sign(feature[pair_i] - feature[pair_j])
            * np.sign(target[pair_i] - target[pair_j])
        )
    )


def fixed_expert_round_concordance(
    expert_ids: np.ndarray,
    raw_rms: np.ndarray,
    target: np.ndarray,
    denoise: int,
    min_selected: int,
) -> dict[str, Any]:
    """Equal token/expert-stratum concordance for one denoise round."""
    ids = np.asarray(expert_ids, dtype=np.int64)
    raw = np.asarray(raw_rms, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if ids.shape != raw.shape or ids.ndim != 4:
        raise ValueError("expert IDs and raw RMS must share [K,D,A,topk]")
    if target.shape != (ids.shape[0],):
        raise ValueError("target must contain one scalar per candidate")
    if not 0 <= denoise < ids.shape[1]:
        raise ValueError("denoise index is out of range")
    if not 2 <= min_selected <= ids.shape[0]:
        raise ValueError("min_selected is outside the candidate axis")

    effects = []
    selected_counts = []
    for token in range(ids.shape[2]):
        token_ids = ids[:, denoise, token]
        token_raw = raw[:, denoise, token]
        for expert in range(N_EXPERTS):
            matches = token_ids == expert
            selected = np.any(matches, axis=-1)
            index = np.flatnonzero(selected)
            if len(index) < min_selected:
                continue
            value = np.sum(token_raw * matches, axis=-1)[index]
            effects.append(_pair_concordance(value, target[index]))
            selected_counts.append(len(index))
    return {
        "pair_concordance": float(np.mean(effects)) if effects else None,
        "valid_token_expert_strata": len(effects),
        "median_candidates_per_stratum": (
            float(np.median(selected_counts)) if selected_counts else None
        ),
        "minimum_candidates": min_selected,
    }


def persistent_weighted_rebound_concordance(
    expert_ids: np.ndarray,
    raw_rms: np.ndarray,
    gate_weight: np.ndarray,
    target: np.ndarray,
    min_selected: int,
) -> dict[str, Any]:
    """Concordance for experts selected for the same candidate in all rounds."""
    ids = np.asarray(expert_ids, dtype=np.int64)
    raw = np.asarray(raw_rms, dtype=np.float64)
    weight = np.asarray(gate_weight, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if ids.shape != raw.shape or ids.shape != weight.shape or ids.ndim != 4:
        raise ValueError("expert arrays must share [K,D,A,topk]")
    if ids.shape[1] != N_DENOISE:
        raise ValueError("persistent rebound requires ten denoise rounds")
    if target.shape != (ids.shape[0],):
        raise ValueError("target must contain one scalar per candidate")

    effects = []
    selected_counts = []
    for token in range(ids.shape[2]):
        token_ids = ids[:, :, token]
        token_raw = raw[:, :, token]
        token_weight = weight[:, :, token]
        for expert in range(N_EXPERTS):
            matches = token_ids == expert
            persistent = np.all(np.any(matches, axis=-1), axis=-1)
            index = np.flatnonzero(persistent)
            if len(index) < min_selected:
                continue
            raw_curve = np.sum(token_raw * matches, axis=-1)[index]
            weight_curve = np.sum(token_weight * matches, axis=-1)[index]
            weighted_curve = raw_curve * weight_curve
            rebound = weighted_curve[:, -1] - weighted_curve.min(axis=-1)
            effects.append(_pair_concordance(rebound, target[index]))
            selected_counts.append(len(index))
    return {
        "pair_concordance": float(np.mean(effects)) if effects else None,
        "valid_persistent_token_expert_strata": len(effects),
        "median_candidates_per_stratum": (
            float(np.median(selected_counts)) if selected_counts else None
        ),
        "minimum_candidates": min_selected,
        "rebound_definition": "weighted[d9] - min_d weighted[d]",
    }


def scalar_token_concordance(value: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Equal-token concordance for a mixture scalar [K, action_token]."""
    value = np.asarray(value, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if value.ndim != 2 or target.shape != (value.shape[0],):
        raise ValueError("value must be [K,A] and target must be [K]")
    effects = [_pair_concordance(value[:, token], target) for token in range(value.shape[1])]
    return {
        "pair_concordance": float(np.mean(effects)),
        "valid_token_strata": len(effects),
    }


def flow_targets(x_traj: np.ndarray) -> dict[str, np.ndarray]:
    """Live-7 chunk RMS targets aligned to each denoise forward."""
    x = np.asarray(x_traj, dtype=np.float64)
    if x.ndim != 4 or x.shape[1] != N_DENOISE + 1 or x.shape[-1] < LIVE_DIMS:
        raise ValueError("x_traj must be [K,11,action_token,D>=7]")
    live = x[..., :LIVE_DIMS]
    immediate = np.sqrt(np.mean(np.square(np.diff(live, axis=1)), axis=(2, 3)))
    remaining = np.sqrt(
        np.mean(np.square(live[:, :N_DENOISE] - live[:, -1, None]), axis=(2, 3))
    )
    total = np.sqrt(np.mean(np.square(live[:, 0] - live[:, -1]), axis=(1, 2)))
    return {
        "total_x0_x10": total,
        "immediate_xd_xd1": immediate,
        "remaining_xd_x10": remaining,
    }


def runtime_absolute_tension(
    gate_weight: np.ndarray,
    raw_rms: np.ndarray,
    routed_rms: np.ndarray,
) -> dict[str, np.ndarray]:
    """D_abs/C_abs using the actual captured top-k weights and routed RMS."""
    weight = np.asarray(gate_weight, dtype=np.float64)
    raw = np.asarray(raw_rms, dtype=np.float64)
    routed = np.asarray(routed_rms, dtype=np.float64)
    if weight.shape != raw.shape or routed.shape != weight.shape[:-1]:
        raise ValueError("runtime scalar arrays are not aligned")
    s1 = np.sum(weight * raw, axis=-1)
    s2 = np.sum(weight * np.square(raw), axis=-1)
    d_energy = s2 - np.square(routed)
    if np.min(d_energy) < -1e-6:
        raise ValueError("negative D_abs energy exceeds numerical tolerance")
    d_abs = np.sqrt(np.maximum(d_energy, 0.0))
    c_abs = s1 - routed
    if np.min(c_abs) < -1e-6:
        raise ValueError("negative C_abs exceeds numerical tolerance")
    return {"s1": s1, "s2": s2, "D_abs": d_abs, "C_abs": c_abs}


def _load_offline_frozen(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    rows = summary["global_maxT_117"]["endpoints"][
        "noise_to_final_correction"
    ]["all_features"]
    lookup = {row["feature"]: row for row in rows}
    return {
        feature: {
            "macro_pair_concordance": lookup[feature]["macro_pair_concordance"],
            "per_task": lookup[feature]["per_task"],
            "expected_orientation": (
                "negative" if FROZEN_ORIENTATION[feature] < 0 else "positive"
            ),
        }
        for feature in FROZEN_FEATURES
    }


def _load_task(
    task: str,
    run_dir: Path,
    layer: int,
    min_selected: int,
) -> dict[str, Any]:
    store_path = run_dir / "activation_flow.zarr"
    root = zarr.open_group(str(store_path), mode="r")
    attrs = dict(root.attrs)
    if attrs.get("format") not in SUPPORTED_FORMATS:
        raise ValueError(f"{run_dir} is not a supported activation-flow store")
    layers = [int(value) for value in attrs["hb_layers"]]
    if layer not in layers:
        raise ValueError(f"HB{layer} is absent from {run_dir}")
    layer_axis = layers.index(layer)

    ids = np.asarray(root["hb_selected_expert_id"][:, layer_axis], dtype=np.int64)
    weight = np.asarray(
        root["hb_selected_expert_weight"][:, layer_axis], dtype=np.float64
    )
    raw = np.asarray(
        root["hb_selected_expert_raw_rms"][:, layer_axis], dtype=np.float64
    )
    routed = np.asarray(root["hb_routed_rms"][:, layer_axis], dtype=np.float64)
    x_traj = np.asarray(root["x_traj"][:], dtype=np.float64)
    if ids.shape != (16, 10, 10, 4):
        raise ValueError(f"{task}: expected K16 [16,10,10,4], got {ids.shape}")
    targets = flow_targets(x_traj)
    absolute = runtime_absolute_tension(weight, raw, routed)

    frozen = {}
    for denoise in (6, 7, 8):
        frozen[f"raw.d{denoise}"] = fixed_expert_round_concordance(
            ids, raw, targets["total_x0_x10"], denoise, min_selected
        )
    frozen["weighted.rebound"] = persistent_weighted_rebound_concordance(
        ids, raw, weight, targets["total_x0_x10"], min_selected
    )

    per_round: dict[str, dict[str, list[Any]]] = {}
    for endpoint_name in ("immediate_xd_xd1", "remaining_xd_x10"):
        endpoint = targets[endpoint_name]
        rows: dict[str, list[Any]] = {"raw": [], "D_abs": [], "C_abs": []}
        for denoise in range(N_DENOISE):
            rows["raw"].append(
                fixed_expert_round_concordance(
                    ids, raw, endpoint[:, denoise], denoise, min_selected
                )
            )
            rows["D_abs"].append(
                scalar_token_concordance(
                    absolute["D_abs"][:, denoise], endpoint[:, denoise]
                )
            )
            rows["C_abs"].append(
                scalar_token_concordance(
                    absolute["C_abs"][:, denoise], endpoint[:, denoise]
                )
            )
        per_round[endpoint_name] = rows

    config = json.loads((run_dir / "experiment_config.json").read_text())
    capture = json.loads((run_dir / "capture_summary.json").read_text())
    query_records = json.loads((run_dir / "query_records.json").read_text())
    if len(query_records) != 1:
        raise ValueError(f"{task}: expected one fresh-state query record")
    return {
        "task": task,
        "run_dir": str(run_dir.resolve()),
        "config": config,
        "capture": capture,
        "query_record": query_records[0],
        "attrs": attrs,
        "validation": {
            "candidate_count": int(ids.shape[0]),
            "max_abs_weight_sum_minus_one": float(
                np.max(np.abs(weight.sum(axis=-1) - 1.0))
            ),
            "minimum_D_energy": float(
                np.min(absolute["s2"] - np.square(routed))
            ),
            "minimum_C_abs": float(np.min(absolute["C_abs"])),
            "control_step_values": np.asarray(root["control_step"][:], dtype=int).tolist(),
        },
        "target_summary": {
            name: {
                "median_by_round": np.median(value, axis=0).tolist()
                if value.ndim == 2
                else None,
                "minimum": float(np.min(value)),
                "maximum": float(np.max(value)),
            }
            for name, value in targets.items()
        },
        "frozen": frozen,
        "per_round": per_round,
        "arrays": {
            "expert_ids": ids.astype(np.uint8),
            "gate_weight": weight.astype(np.float32),
            "raw_rms": raw.astype(np.float32),
            "routed_rms": routed.astype(np.float32),
            "D_abs": absolute["D_abs"].astype(np.float32),
            "C_abs": absolute["C_abs"].astype(np.float32),
            **{name: value.astype(np.float32) for name, value in targets.items()},
        },
    }


def analyze(
    runs_root: Path,
    offline_summary: Path,
    layer: int,
    min_selected: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if min_selected < 2 or min_selected > 16:
        raise ValueError("min_selected must be in [2,16]")
    loaded = [
        _load_task(task, runs_root / run_name, layer, min_selected)
        for task, run_name in TASK_RUNS
    ]
    task_names = [row["task"] for row in loaded]
    offline = _load_offline_frozen(offline_summary)

    frozen = {}
    for feature in FROZEN_FEATURES:
        values = [row["frozen"][feature]["pair_concordance"] for row in loaded]
        expected = FROZEN_ORIENTATION[feature]
        matches = [bool(value * expected > 0) for value in values]
        frozen[feature] = {
            "expected_orientation": "negative" if expected < 0 else "positive",
            "offline": offline[feature],
            "runtime_equal_task_macro_pair_concordance": float(np.mean(values)),
            "runtime_per_task_pair_concordance": dict(zip(task_names, values)),
            "runtime_matches_expected_orientation": dict(zip(task_names, matches)),
            "matching_tasks": int(sum(matches)),
            "all_five_tasks_match": bool(all(matches)),
            "coverage": {
                task: loaded[axis]["frozen"][feature]
                for axis, task in enumerate(task_names)
            },
        }

    per_round = {}
    for endpoint in ("immediate_xd_xd1", "remaining_xd_x10"):
        per_round[endpoint] = {}
        for feature in ("raw", "D_abs", "C_abs"):
            per_task = {
                task: [
                    entry["pair_concordance"]
                    for entry in loaded[axis]["per_round"][endpoint][feature]
                ]
                for axis, task in enumerate(task_names)
            }
            matrix = np.asarray(list(per_task.values()), dtype=np.float64)
            per_round[endpoint][feature] = {
                "per_task_pair_concordance_by_round": per_task,
                "equal_task_macro_pair_concordance_by_round": matrix.mean(axis=0).tolist(),
                "positive_tasks_by_round": np.sum(matrix > 0, axis=0).tolist(),
                "coverage": {
                    task: loaded[axis]["per_round"][endpoint][feature]
                    for axis, task in enumerate(task_names)
                },
            }

    seed_bases = [int(row["config"]["noise_seed_base"]) for row in loaded]
    source_formats = sorted({row["attrs"]["format"] for row in loaded})
    summary = {
        "experiment": "runtime_scalar_fresh_state_pilot_v1",
        "status": "descriptive_single_fresh_state_no_significance_claim",
        "task_names": task_names,
        "source_format": (
            source_formats[0] if len(source_formats) == 1 else source_formats
        ),
        "fixed_hb_layer": layer,
        "candidates_per_task": 16,
        "states_per_task": 1,
        "minimum_candidates_per_fixed_stratum": min_selected,
        "comparison_unit": (
            "within task/fresh-state; raw fixes denoise/action-token/expert-ID, "
            "persistent rebound also requires the expert in all ten rounds"
        ),
        "statistic": (
            "equal-stratum mean of sign(feature_i-feature_j) * "
            "sign(target_i-target_j); ties contribute zero"
        ),
        "targets": {
            "total_x0_x10": "RMS over the complete 10x7 live chunk of x0-x10",
            "immediate_xd_xd1": "round d: RMS over the complete 10x7 live chunk of x[d+1]-x[d]",
            "remaining_xd_x10": "round d: RMS over the complete 10x7 live chunk of x[d]-x[10]",
        },
        "features": {
            "raw": "runtime RMS(E_e(h)) before gate multiplication",
            "weighted_rebound": (
                "for a persistent candidate/token/expert: "
                "w*raw at d9 minus its minimum over d0..d9"
            ),
            "D_abs": "sqrt(max(sum_e w_e raw_e^2 - routed_rms^2, 0))",
            "C_abs": "sum_e w_e raw_e - routed_rms",
        },
        "frozen_total_displacement": frozen,
        "per_round_descriptive": per_round,
        "sources": {
            row["task"]: {
                "run_dir": row["run_dir"],
                "experiment_config": row["config"],
                "capture_summary": row["capture"],
                "query_record": row["query_record"],
                "validation": row["validation"],
                "target_summary": row["target_summary"],
            }
            for row in loaded
        },
        "design_boundaries": {
            "fresh_state": (
                "init_state_id 24 is outside the earlier 16-state archive, but "
                "there is only one state per task"
            ),
            "different_rng_blocks": dict(zip(task_names, seed_bases)),
            "candidate_columns_shared_across_tasks": False,
            "power": "K=16 and one state per task give low task-level replication power",
            "selection": (
                "raw comparisons condition on the same expert being selected in "
                "at least four candidates"
            ),
            "inference": (
                "no p-values, confidence intervals, or multiple-testing claims; "
                "the 3 x 10 x 2 per-round scan is descriptive"
            ),
        },
        "external_formal_controls_pending_runtime_integration": {
            "old_K32_total_displacement_baselines": {
                "within_pool_noise_rms_spearman_macro": 0.666,
                "exact_seed_task_local_state_LOO_target_template_spearman": 0.971,
                "interpretation": (
                    "total flow displacement is strongly predictable without MoE; "
                    "runtime incremental value must be tested beyond these baselines"
                ),
            },
            "old_K32_same_seed_other_state_residual": {
                "residual_definition": (
                    "target minus the mean of the other 15 states at the same seed"
                ),
                "raw_d6_macro": -0.01554,
                "raw_d7_macro": -0.01746,
                "raw_d8_macro": -0.01813,
                "raw_d6_d7_d8_all_five_tasks_negative": True,
                "raw_d6_d7_d8_maxT4_p_upper_bound": 0.00005,
                "weighted_rebound_macro": 0.01845,
                "weighted_rebound_maxT4_p": 0.1306,
                "boundary": (
                    "external K32 formal control supplied after this runtime pilot; "
                    "it is not evidence that the K16 runtime capture replicated"
                ),
            },
        },
        "verdict": (
            "The offline frozen directions retain the expected task-equal macro "
            "sign, but none reproduces its expected direction in all five fresh-state "
            "runtime tasks. This pilot is not a confirmation."
        ),
    }
    arrays = {"task_names": np.asarray(task_names)}
    for key in loaded[0]["arrays"]:
        arrays[key] = np.stack([row["arrays"][key] for row in loaded])
    return summary, arrays


def _format_tasks(values: dict[str, float]) -> str:
    return " / ".join(f"{value:+.4f}" for value in values.values())


def _render_round_table(summary: dict[str, Any], endpoint: str) -> list[str]:
    rows = summary["per_round_descriptive"][endpoint]
    lines = [
        "| round | raw macro (five tasks) | D_abs macro (five tasks) | C_abs macro (five tasks) |",
        "|---:|---|---|---|",
    ]
    for denoise in range(N_DENOISE):
        cells = []
        for feature in ("raw", "D_abs", "C_abs"):
            row = rows[feature]
            macro = row["equal_task_macro_pair_concordance_by_round"][denoise]
            task_values = {
                task: values[denoise]
                for task, values in row["per_task_pair_concordance_by_round"].items()
            }
            cells.append(f"{macro:+.4f} ({_format_tasks(task_values)})")
        lines.append(f"| d{denoise} | " + " | ".join(cells) + " |")
    return lines


def _render_report(summary: dict[str, Any]) -> str:
    frozen = summary["frozen_total_displacement"]
    tasks = summary["task_names"]
    lines = [
        "# Fresh-state runtime scalar pilot",
        "",
        "本 pilot 读取五个 `himoe_hb_activation_flow_v2` capture；每个任务只有一个新的",
        "`init_state_id=24` 和 K=16 候选。固定 HB5，所有结果只报告 effect size，不做显著性声明。",
        "",
        "## 配对定义",
        "",
        "`raw` 比较固定 `task / state / denoise round / action token / expert ID`，",
        "同一 expert 至少出现在 4 个候选中。pair concordance 为",
        "`mean sign(delta feature) * sign(delta target)`，先等权 stratum，再看任务。",
        "`weighted.rebound` 还要求同一 candidate/token/expert 十轮持续入选。",
        "",
        "总位移目标为 model-normalized live-7 空间的完整 `10x7 RMS(x0-x10)`。",
        "",
        "## 离线冻结点的 runtime 检查",
        "",
        "| feature | offline macro | expected | runtime macro | runtime per-task | matches | all five |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for feature in FROZEN_FEATURES:
        row = frozen[feature]
        lines.append(
            "| %s | %+.4f | %s | %+.4f | %s | %d/5 | %s |"
            % (
                feature,
                row["offline"]["macro_pair_concordance"],
                row["expected_orientation"],
                row["runtime_equal_task_macro_pair_concordance"],
                _format_tasks(row["runtime_per_task_pair_concordance"]),
                row["matching_tasks"],
                "yes" if row["all_five_tasks_match"] else "no",
            )
        )
    lines += [
        "",
        (
            "任务顺序为：`%s`。四项 runtime macro 仍是离线冻结方向，但没有一项在五个任务"
            "都复现：raw.d6/d7 为 4/5，raw.d8 为 3/5，weighted.rebound 为 4/5。"
            % " / ".join(tasks)
        ),
        "因此离线五任务同向发现没有通过这次 fresh-state runtime pilot。",
        "",
        "persistent rebound 每任务只有少量有效 token/expert strata：",
        "`%s`；这是最脆弱的一项。"
        % " / ".join(
            str(frozen["weighted.rebound"]["coverage"][task]["valid_persistent_token_expert_strata"])
            for task in tasks
        ),
        "",
        "## 每轮即时更新",
        "",
        "round d 的目标是完整 live-7 chunk 的 `RMS(x[d+1]-x[d])`。括号内为五任务值，",
        "顺序同上。`raw` 固定 expert ID；`D_abs/C_abs` 是同 token 的 top-4 mixture scalar。",
        "",
    ]
    lines += _render_round_table(summary, "immediate_xd_xd1")
    lines += [
        "",
        "## 每轮剩余修正",
        "",
        "round d 的目标是完整 live-7 chunk 的 `RMS(x[d]-x[10])`。",
        "",
    ]
    lines += _render_round_table(summary, "remaining_xd_x10")
    lines += [
        "",
        "这些逐轮曲线在任务间频繁变号，只是描述性扫描，不能从中挑一轮重新宣称发现。",
        "",
        "## 数据与解释边界",
        "",
        "- 五任务 RNG seed blocks 分别为 `%s`，候选列并非跨任务共同 seed 配对。"
        % " / ".join(
            str(summary["design_boundaries"]["different_rng_blocks"][task])
            for task in tasks
        ),
        "- 每任务只有一个 fresh state；K=16 只增加该状态内候选数，不提供 state-level replication。",
        "- raw 是 exact-runtime-input 上捕获的 scalar RMS，但仍是 selected-only；它不是专家内部状态。",
        "- `D_abs/C_abs` 使用实际捕获 gate 权重和 routed RMS；各 run 的最大 weight-sum error",
        "  约 0.006，报告保留实际 runtime 数值，没有用离线值替换。",
        "- 没有 p-value、置信区间或 multiple-testing 结论。逐轮 60 个组合全部属于探索。",
        "- 免费 baseline 背景：旧 K32 中，总位移与同池 `noise_rms` 的 rho macro 为 0.666，",
        "  exact-seed、task-local、state-LOO target template 的 rho 为 0.971。因此任何 MoE",
        "  runtime 价值都必须检验超过这两个 baseline 的增量。",
        "- 外部 formal control 待补入 runtime：旧 K32 对 `y-other15states_same_seed_mean`",
        "  残差的 raw d6/d7/d8 macro 为 -0.01554/-0.01746/-0.01813，五任务均负，",
        "  maxT4 p 均 <5e-5；weighted rebound 为 +0.01845、p=0.1306。它是旧 K32 控制，",
        "  不能表述为本次 K16 runtime 已验证。",
        "",
        "完整数值见 `summary.json`，对齐后的候选数组见 `arrays.npz`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary, arrays = analyze(
        args.runs_root.resolve(),
        args.offline_summary.resolve(),
        args.layer,
        args.min_selected,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (args.out_dir / "REPORT.md").write_text(_render_report(summary))
    np.savez_compressed(args.out_dir / "arrays.npz", **arrays)
    print(args.out_dir / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
