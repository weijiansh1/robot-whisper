#!/usr/bin/env python3
"""Visualize 32 complete MoE routing trajectories from one initial state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import BoundaryNorm


HERE = Path(__file__).resolve().parent
HUB_CACHE = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
DEFAULT_RUN = (
    HUB_CACHE
    / "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
)
DEFAULT_OUT = HERE / "analysis/k32-route-trajectories"
HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
N_DENOISE = 10
N_EXPERTS = 32
ACTION_TOKENS = slice(1, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def normalize_probabilities(values: np.ndarray) -> tuple[np.ndarray, float]:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape[-1] != N_EXPERTS:
        raise ValueError("expected a 32-expert probability axis")
    if not np.all(np.isfinite(probabilities)) or probabilities.min() < -1e-6:
        raise ValueError("router probabilities must be finite and nonnegative")
    probabilities = np.maximum(probabilities, 0.0)
    mass = probabilities.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("router probability vector has zero mass")
    raw_error = float(np.max(np.abs(mass - 1.0)))
    probabilities /= mass
    if np.max(np.abs(probabilities.sum(axis=-1) - 1.0)) > 1e-12:
        raise RuntimeError("router normalization failed")
    return probabilities, raw_error


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.shape[-1] != N_EXPERTS:
        raise ValueError("Hellinger inputs are not aligned")
    return np.sqrt(0.5 * np.square(np.sqrt(left) - np.sqrt(right)).sum(axis=-1))


def trajectory_metrics(raw: np.ndarray) -> dict[str, np.ndarray | float]:
    """Reduce [Q,8,10,10,32] action-token routes without mixing layers."""

    if raw.ndim != 5 or raw.shape[1:] != (8, 10, 10, 32):
        raise ValueError("expected [query,8,10,10,32] HB action-token routes")
    probabilities, raw_error = normalize_probabilities(raw)

    sorted_probabilities = np.sort(probabilities, axis=-1)
    top4_mass = sorted_probabilities[..., -4:].sum(axis=-1).mean(axis=(1, 3))
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=-1
    )
    normalized_entropy = entropy.mean(axis=(1, 3)) / math.log(N_EXPERTS)

    token_mean = probabilities.mean(axis=3)
    token_mean /= token_mean.sum(axis=-1, keepdims=True)
    dominant_expert = np.argmax(token_mean, axis=-1).astype(np.int16)

    timeline = token_mean.transpose(0, 2, 1, 3).reshape(-1, 8, N_EXPERTS)
    change = np.full(len(timeline), np.nan, dtype=np.float64)
    change[1:] = np.sqrt(np.square(hellinger(timeline[1:], timeline[:-1])).mean(axis=1))
    return {
        "top4_mass": top4_mass.reshape(-1),
        "normalized_entropy": normalized_entropy.reshape(-1),
        "dominant_expert": dominant_expert.transpose(1, 0, 2).reshape(8, -1),
        "route_change": change,
        "token_mean": timeline,
        "raw_probability_mass_max_error": raw_error,
    }


def k_oracle_summary(summaries: list[dict[str, Any]]) -> dict[str, float | int]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault(int(row["init_state_id"]), []).append(row)
    if not grouped:
        raise ValueError("no summaries")
    baseline = []
    k8 = []
    k32 = []
    all_failure_k32 = 0
    for state, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["flow_noise_seed"]))
        if len(ordered) != 32:
            raise ValueError(f"state {state} does not have 32 candidates")
        seeds = [int(row["flow_noise_seed"]) for row in ordered]
        if len(set(seeds)) != 32:
            raise ValueError(f"state {state} has duplicate seeds")
        outcomes = np.asarray([bool(row["success"]) for row in ordered])
        baseline.extend(map(float, outcomes))
        for fold in range(4):
            k8.append(float(outcomes[fold::4].max()))
        value = float(outcomes.max())
        k32.append(value)
        all_failure_k32 += int(value == 0.0)
    return {
        "states": len(grouped),
        "random_expected_success": float(np.mean(baseline)),
        "k8_outcome_oracle_success": float(np.mean(k8)),
        "k32_outcome_oracle_success": float(np.mean(k32)),
        "all_failure_k32_states": all_failure_k32,
    }


def discover_complete_k32_runs(cache: Path) -> list[Path]:
    return sorted(
        path.parent.parent
        for path in cache.glob("**/right-16x32/client/summaries.json")
        if (path.parent.parent / "server/routes.zarr").exists()
    )


def load_selected_trajectories(run: Path, init_state: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = json.loads((run / "client/summaries.json").read_text())
    selected = sorted(
        [row for row in summaries if int(row["init_state_id"]) == init_state],
        key=lambda row: int(row["flow_noise_seed"]),
    )
    if len(selected) != 32:
        raise RuntimeError(f"expected 32 rollouts for state {init_state}, found {len(selected)}")

    group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    attrs = dict(group.attrs)
    expected_attrs = {"n_hb_layers": 8, "n_denoise": 10, "n_hb_experts": 32}
    for key, expected in expected_attrs.items():
        if int(attrs.get(key, -1)) != expected:
            raise RuntimeError(f"route store {key} is not {expected}")
    episode_axis = np.asarray(group["episode_id"][:], dtype=np.int64)
    control_axis = np.asarray(group["control_step"][:], dtype=np.int64)
    if not np.array_equal(control_axis, np.arange(len(control_axis))):
        raise RuntimeError("global control_step axis is not contiguous")

    trajectories = []
    raw_error = 0.0
    for row in selected:
        episode = int(row["episode_index"])
        indices = np.flatnonzero(episode_axis == episode)
        expected = int(row["inference_calls"])
        if len(indices) != expected or not np.array_equal(indices, np.arange(indices[0], indices[0] + expected)):
            raise RuntimeError(f"episode {episode} boundary mismatch")
        raw = np.asarray(
            group["hb_router_probs"].oindex[indices, :, :, ACTION_TOKENS, :],
            dtype=np.float64,
        )
        metrics = trajectory_metrics(raw)
        raw_error = max(raw_error, float(metrics["raw_probability_mass_max_error"]))
        trajectories.append({"summary": row, **metrics})
    return trajectories, {
        "route_store_attrs": attrs,
        "raw_probability_mass_max_error": raw_error,
    }


def padded_matrix(trajectories: list[dict[str, Any]], field: str, width: int) -> np.ndarray:
    output = np.full((len(trajectories), width), np.nan, dtype=np.float64)
    for row, trajectory in enumerate(trajectories):
        values = np.asarray(trajectory[field], dtype=np.float64)
        output[row, : len(values)] = values
    return output


def axis_labels(trajectories: list[dict[str, Any]]) -> list[str]:
    return [
        "%d  %s" % (
            int(trajectory["summary"]["flow_noise_seed"]),
            "S" if trajectory["summary"]["success"] else "F",
        )
        for trajectory in trajectories
    ]


def add_query_grid(axis: plt.Axes, max_queries: int) -> None:
    for query in range(max_queries + 1):
        axis.axvline(query * N_DENOISE - 0.5, color="white", lw=0.25, alpha=0.16)
    for query in range(0, max_queries + 1, 5):
        axis.axvline(query * N_DENOISE - 0.5, color="black", lw=0.45, alpha=0.25)


def plot_overview(trajectories: list[dict[str, Any]], out: Path) -> None:
    max_queries = max(int(row["summary"]["inference_calls"]) for row in trajectories)
    width = max_queries * N_DENOISE
    matrices = [
        ("top4_mass", "Mean top-4 routing mass", "viridis"),
        ("normalized_entropy", "Normalized routing entropy", "cividis"),
        ("route_change", "Hellinger change from previous internal step", "magma"),
    ]
    labels = axis_labels(trajectories)
    figure, axes = plt.subplots(3, 1, figsize=(19, 20), sharex=True, constrained_layout=True)
    for axis, (field, title, cmap_name) in zip(axes, matrices):
        values = padded_matrix(trajectories, field, width)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e6e7e8")
        finite = values[np.isfinite(values)]
        lower, upper = np.percentile(finite, [1, 99])
        image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap,
                            vmin=lower, vmax=upper)
        axis.set_title(title, fontsize=13)
        axis.set_yticks(np.arange(32), labels=labels, fontsize=7)
        axis.set_ylabel("flow seed / outcome")
        add_query_grid(axis, max_queries)
        figure.colorbar(image, ax=axis, pad=0.008, fraction=0.018)
    ticks = np.arange(0, max_queries + 1, 5)
    axes[-1].set_xticks(ticks * N_DENOISE + (N_DENOISE - 1) / 2, labels=ticks)
    axes[-1].set_xlabel("policy query q; each query contains denoise rounds d=0..9")
    success = sum(bool(row["summary"]["success"]) for row in trajectories)
    figure.suptitle(
        "32 complete MoE trajectories, init_state=0 (%d success / %d failure)" %
        (success, len(trajectories) - success),
        fontsize=16,
    )
    figure.savefig(out, dpi=180)
    plt.close(figure)


def plot_expert_atlas(trajectories: list[dict[str, Any]], out: Path) -> None:
    max_queries = max(int(row["summary"]["inference_calls"]) for row in trajectories)
    width = max_queries * N_DENOISE
    atlas = np.full((len(trajectories) * len(HB_LAYERS), width), np.nan)
    for rollout, trajectory in enumerate(trajectories):
        values = np.asarray(trajectory["dominant_expert"], dtype=np.float64)
        atlas[rollout * 8 : (rollout + 1) * 8, : values.shape[1]] = values
    cmap = plt.get_cmap("turbo", N_EXPERTS).copy()
    cmap.set_bad("#e6e7e8")
    norm = BoundaryNorm(np.arange(-0.5, N_EXPERTS + 0.5), N_EXPERTS)
    figure, axis = plt.subplots(figsize=(20, 24), constrained_layout=True)
    image = axis.imshow(atlas, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    for rollout in range(len(trajectories) + 1):
        axis.axhline(rollout * 8 - 0.5, color="white", lw=0.55, alpha=0.75)
    add_query_grid(axis, max_queries)
    centers = np.arange(len(trajectories)) * 8 + 3.5
    axis.set_yticks(centers, labels=axis_labels(trajectories), fontsize=7)
    ticks = np.arange(0, max_queries + 1, 5)
    axis.set_xticks(ticks * N_DENOISE + (N_DENOISE - 1) / 2, labels=ticks)
    axis.set_xlabel("policy query q; thin blocks are denoise rounds d=0..9")
    axis.set_ylabel("flow seed / outcome; each group contains HB layers 2,3,4,5,12,13,14,15")
    axis.set_title("Action-token-averaged dominant expert across 32 complete rollouts")
    colorbar = figure.colorbar(image, ax=axis, pad=0.008, fraction=0.018,
                               ticks=np.arange(0, N_EXPERTS, 4))
    colorbar.set_label("dominant expert id (layer-local)")
    figure.savefig(out, dpi=180)
    plt.close(figure)


def plot_outcomes(summaries: list[dict[str, Any]], selected_state: int, out: Path) -> None:
    states = sorted({int(row["init_state_id"]) for row in summaries})
    seeds = sorted({int(row["flow_noise_seed"]) for row in summaries})
    lookup = {
        (int(row["init_state_id"]), int(row["flow_noise_seed"])): bool(row["success"])
        for row in summaries
    }
    matrix = np.asarray([[lookup[(state, seed)] for seed in seeds] for state in states])
    figure, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(["#c94c4c", "#2c8c62"])
    axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(seeds)), labels=seeds, rotation=90, fontsize=7)
    axis.set_yticks(np.arange(len(states)), labels=states)
    axis.set_xlabel("flow-noise seed")
    axis.set_ylabel("init state")
    axis.set_title("K32 outcome grid (red=failure, green=success)")
    selected_index = states.index(selected_state)
    axis.add_patch(
        matplotlib.patches.Rectangle(
            (-0.5, selected_index - 0.5), len(seeds), 1,
            fill=False, edgecolor="#111111", linewidth=2.2,
        )
    )
    figure.savefig(out, dpi=180)
    plt.close(figure)


def summarize_trajectories(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("top4_mass", "normalized_entropy", "route_change")
    output = {}
    success_mask = np.asarray([bool(row["summary"]["success"]) for row in trajectories])
    for field in fields:
        per_rollout = np.asarray([
            float(np.nanmean(np.asarray(row[field], dtype=np.float64))) for row in trajectories
        ])
        output[field] = {
            "all_rollout_equal_mean": float(per_rollout.mean()),
            "success_rollout_mean": float(per_rollout[success_mask].mean()),
            "failure_rollout_mean": float(per_rollout[~success_mask].mean()),
            "success_minus_failure": float(
                per_rollout[success_mask].mean() - per_rollout[~success_mask].mean()
            ),
        }
    within = []
    seams = []
    for row in trajectories:
        change = np.asarray(row["route_change"], dtype=np.float64)
        index = np.arange(len(change))
        within.extend(change[(index % N_DENOISE) != 0])
        seams.extend(change[(index % N_DENOISE) == 0][1:])
    output["change_by_boundary"] = {
        "within_query_adjacent_denoise": float(np.nanmean(within)),
        "cross_query_d9_to_next_d0": float(np.nanmean(seams)),
    }
    profile = {}
    for field in fields:
        per_rollout = []
        for row in trajectories:
            values = np.asarray(row[field], dtype=np.float64).reshape(-1, N_DENOISE)
            per_rollout.append(np.nanmean(values, axis=0))
        profile[field] = np.nanmean(per_rollout, axis=0).tolist()
    output["denoise_profile"] = profile
    return output


def plot_denoise_profile(metrics: dict[str, Any], out: Path) -> None:
    profile = metrics["denoise_profile"]
    denoise = np.arange(N_DENOISE)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    specifications = [
        ("top4_mass", "Mean top-4 routing mass", "#167d70"),
        ("normalized_entropy", "Normalized routing entropy", "#3766a0"),
        ("route_change", "Change from previous internal step", "#b5433f"),
    ]
    for axis, (field, title, color) in zip(axes, specifications):
        values = np.asarray(profile[field], dtype=np.float64)
        axis.plot(denoise, values, marker="o", color=color, lw=2)
        axis.set_xticks(denoise)
        axis.set_xlabel("denoise round")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    axes[2].annotate(
        "cross-query seam\n(previous d9 to new d0)",
        xy=(0, profile["route_change"][0]),
        xytext=(1.4, profile["route_change"][0] * 0.88),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=8,
    )
    figure.suptitle("Route concentration and movement across the 10 flow-denoise passes")
    figure.savefig(out, dpi=180)
    plt.close(figure)


def render_report(summary: dict[str, Any]) -> str:
    selected = summary["selected_state"]
    long_oracle = summary["k_oracle"][summary["source_task"]]
    macro = summary["k_oracle_informative_macro"]
    change = summary["trajectory_metrics"]["change_by_boundary"]
    profile = summary["trajectory_metrics"]["denoise_profile"]
    contrast = summary["trajectory_metrics"]
    lines = [
        "# K32 完整 rollout 的 MoE 激活轨迹",
        "",
        "## 实验思路",
        "",
        "- 选择 long/moka 的同一初始状态 `%d`，一次性使用 seed 1000-1031 的 32 条完整 rollout；不再拆 K8。" % selected["init_state"],
        "- 横轴严格拼接每条 rollout 的全部 policy query；每个 query 再展开为 10 个 denoise round。提前成功结束后的区域留白，不做时间拉伸。",
        "- 每个路由概率先在 32 专家上重新归一化。图一显示 top-4 概率质量、归一化熵和相邻内部步 Hellinger 变化；图二保留 8 个 HB 层各自的主导专家编号。",
        "",
        "## 结果",
        "",
        "该状态 32 条 rollout 中 `%d` 条成功、`%d` 条失败，每条有 `%d-%d` 个 policy query。" % (
            selected["successes"], selected["failures"], selected["min_queries"], selected["max_queries"]
        ),
        "同一次 query 内相邻 denoise 的平均路由变化为 `%.4f`；从上一 query 的 d9 跨到下一 query 的 d0 为 `%.4f`。这两种边界已在图中按真实顺序连接。" % (
            change["within_query_adjacent_denoise"], change["cross_query_d9_to_next_d0"]
        ),
        "跨 query 的大跳变同时包含新观测和新动作初始噪声，不能只解释成环境状态变化。",
        "",
        "![32 rollout activation overview](activation_overview.png)",
        "",
        "![32 rollout dominant expert atlas](dominant_expert_atlas.png)",
        "",
        "10 轮内部存在稳定的收紧过程：top-4 概率质量从 d0 的 `%.4f` 增到 d9 的 `%.4f`，归一化熵从 `%.6f` 降到 `%.6f`；后一次内部更新的路由变化从 d1 的 `%.4f` 增到 d9 的 `%.4f`。" % (
            profile["top4_mass"][0], profile["top4_mass"][-1],
            profile["normalized_entropy"][0], profile["normalized_entropy"][-1],
            profile["route_change"][1], profile["route_change"][-1],
        ),
        "",
        "![10 denoise route profile](denoise_profile.png)",
        "",
        "成功与失败整条轨迹的平均 top-4 mass 只差 `%+.6f`，平均变化速度只差 `%+.6f`；当前图没有显示一个可直接按成败分开的激活模式。" % (
            contrast["top4_mass"]["success_minus_failure"],
            contrast["route_change"]["success_minus_failure"],
        ),
        "",
        "完整 long/moka 数据上，随机选 1 的成功率为 `%.3f`，K8 偷看结果上限为 `%.3f`，K32 偷看结果上限为 `%.3f`。K32 仍不到 1，是因为 16 个状态中有 `%d` 个状态的 32 条全部失败。" % (
            long_oracle["random_expected_success"],
            long_oracle["k8_outcome_oracle_success"],
            long_oracle["k32_outcome_oracle_success"],
            long_oracle["all_failure_k32_states"],
        ),
        "四个非天花板任务等权后，K8 偷看上限是 `%.3f`，直接看 K32 后是 `%.3f`。之前使用 K8 是为了固定 8 次候选推理预算；K32 用 4 倍候选推理换来了更高上限，做上限分析时应当看 K32。" % (
            macro["k8_outcome_oracle_success"], macro["k32_outcome_oracle_success"]
        ),
        "",
        "![K32 outcome grid](k32_outcome_grid.png)",
        "",
        "## 10 次去噪是什么意思",
        "",
        "是的。一次 policy query 先生成一个动作噪声，然后做 10 次 flow denoise；每次都会重新执行 action suffix 的 18 层 transformer，其中 8 层是 HB-MoE、4 层是 AS-MoE、6 层是 dense。也就是每次 query 有 `10 x 8 = 80` 次 HB 层级路由和 `10 x 4 = 40` 次 AS 层级路由。HB 每次还同时处理 1 个 state token 和 10 个 action token。AS 路由因为只看恒定 data mask，10 次结果相同，所以存储时折叠成一次。",
        "",
        "它不是把同一个动作执行 10 次：这是模型内部把噪声动作逐步变成最终动作块的 10 次前向更新。环境只拿最终动作块执行。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    trajectories, validation = load_selected_trajectories(run, args.init_state)
    summaries = json.loads((run / "client/summaries.json").read_text())
    selected_rows = [row for row in summaries if int(row["init_state_id"]) == args.init_state]

    k_oracle = {}
    for candidate_run in discover_complete_k32_runs(HUB_CACHE):
        rows = json.loads((candidate_run / "client/summaries.json").read_text())
        key = str(candidate_run.relative_to(HUB_CACHE).parent)
        k_oracle[key] = k_oracle_summary(rows)
    source_task = str(run.relative_to(HUB_CACHE).parent)
    summary = {
        "experiment": "k32_contiguous_moe_route_visualization",
        "source_task": source_task,
        "protocol": {
            "time_axis": "policy query q, then denoise d=0..9 within q",
            "rollouts": 32,
            "hb_layers": list(HB_LAYERS),
            "action_tokens": 10,
            "experts": 32,
            "normalization": "each layer/denoise/token 32-way router vector renormalized before reduction",
            "outcome_use": "labels annotate rows only; no ordering, fitting, or feature uses success",
        },
        "selected_state": {
            "init_state": args.init_state,
            "episodes": [int(row["episode_index"]) for row in selected_rows],
            "seeds": [int(row["flow_noise_seed"]) for row in sorted(selected_rows, key=lambda row: int(row["flow_noise_seed"]))],
            "successes": sum(bool(row["success"]) for row in selected_rows),
            "failures": sum(not bool(row["success"]) for row in selected_rows),
            "min_queries": min(int(row["inference_calls"]) for row in selected_rows),
            "max_queries": max(int(row["inference_calls"]) for row in selected_rows),
            "total_queries": sum(int(row["inference_calls"]) for row in selected_rows),
        },
        "validation": validation,
        "trajectory_metrics": summarize_trajectories(trajectories),
        "k_oracle": k_oracle,
        "k_oracle_informative_macro": {},
        "model_execution": {
            "flow_denoise_steps_per_query": 10,
            "hb_moe_layers": 8,
            "as_moe_layers": 4,
            "dense_layers": 6,
            "hb_layer_invocations_per_query": 80,
            "as_layer_invocations_per_query": 40,
            "suffix_tokens": 11,
            "source": "MoEVLA.sample_actions while loop and recorder-verified route axes",
        },
    }
    informative = [
        row for row in k_oracle.values()
        if 0.0 < row["random_expected_success"] < 1.0
    ]
    summary["k_oracle_informative_macro"] = {
        "tasks": len(informative),
        "random_expected_success": float(np.mean([
            row["random_expected_success"] for row in informative
        ])),
        "k8_outcome_oracle_success": float(np.mean([
            row["k8_outcome_oracle_success"] for row in informative
        ])),
        "k32_outcome_oracle_success": float(np.mean([
            row["k32_outcome_oracle_success"] for row in informative
        ])),
        "all_failure_k32_states": int(sum(
            row["all_failure_k32_states"] for row in informative
        )),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(trajectories, args.out_dir / "activation_overview.png")
    plot_expert_atlas(trajectories, args.out_dir / "dominant_expert_atlas.png")
    plot_denoise_profile(
        summary["trajectory_metrics"], args.out_dir / "denoise_profile.png"
    )
    plot_outcomes(summaries, args.init_state, args.out_dir / "k32_outcome_grid.png")
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    print(json.dumps({
        "selected_state": summary["selected_state"],
        "long_k_oracle": k_oracle[source_task],
        "output": str(args.out_dir / "report.md"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
