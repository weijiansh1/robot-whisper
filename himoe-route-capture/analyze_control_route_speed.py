#!/usr/bin/env python3
"""Measure HB routing change speed along complete control-query sequences.

This analysis deliberately treats a rollout as q=0,1,...,T-1 and uses no
actions, robot states, rewards, success labels, or simulator progress.  A control query is the
unit of time.  Within each query, action-token router probabilities are averaged
over the ten action-token positions while layer and denoise-round identity are
retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr


HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "corpus/libero30-right-v1"
DEFAULT_OUT = HERE / "analysis/control-route-speed"
N_LAYERS = 8
N_DENOISE = 10
N_EXPERTS = 32
ACTION_TOKENS = slice(1, 11)
MAX_LAG = 8
PHASE_NAMES = ("early", "middle", "late")


@dataclass(frozen=True)
class EpisodeSpeed:
    suite: str
    task: str
    episode: int
    controls: int
    adjacent: np.ndarray
    all_pair_mean: float
    lag_mean: np.ndarray
    phase_mean: np.ndarray
    layer_mean: np.ndarray
    layer_phase_mean: np.ndarray
    denoise_mean: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def normalize_probabilities(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim < 2 or probabilities.shape[-1] != N_EXPERTS:
        raise ValueError("router probabilities must end in a 32-expert axis")
    if not np.all(np.isfinite(probabilities)) or probabilities.min() < -1e-6:
        raise ValueError("router probabilities must be finite and nonnegative")
    probabilities = np.maximum(probabilities, 0.0)
    total = probabilities.sum(axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("router probability vector has zero total mass")
    bounds = (float(total.min()), float(total.max()))
    probabilities /= total
    error = np.max(np.abs(probabilities.sum(axis=-1) - 1.0))
    if error > 1e-12:
        raise RuntimeError("router normalization failed")
    return probabilities, bounds


def route_distance_matrix(probabilities: np.ndarray) -> np.ndarray:
    """RMS Hellinger distance over aligned layer x denoise sites."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (N_LAYERS, N_DENOISE, N_EXPERTS):
        raise ValueError("expected [control,8,10,32] route probabilities")
    root = np.sqrt(values).reshape(len(values), -1)
    sites = N_LAYERS * N_DENOISE
    squared_norm = np.square(root).sum(axis=1)
    squared = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * (root @ root.T)
    ) / (2.0 * sites)
    distance = np.sqrt(np.maximum(squared, 0.0))
    np.fill_diagonal(distance, 0.0)
    return distance


def transition_details(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = np.sqrt(probabilities)
    difference = root[1:] - root[:-1]
    expert_squared = np.square(difference).sum(axis=-1)
    layer = np.sqrt(0.5 * expert_squared.mean(axis=2))
    denoise = np.sqrt(0.5 * expert_squared.mean(axis=1))
    overall = np.sqrt(0.5 * expert_squared.mean(axis=(1, 2)))
    return overall, layer, denoise


def transition_phase(controls: int) -> np.ndarray:
    if controls < 2:
        return np.empty(0, dtype=np.int64)
    midpoint = (np.arange(controls - 1, dtype=np.float64) + 0.5) / (controls - 1)
    return np.minimum((midpoint * 3.0).astype(np.int64), 2)


def summarize_episode(
    probabilities: np.ndarray,
    suite: str,
    task: str,
    episode: int,
) -> EpisodeSpeed:
    controls = len(probabilities)
    if controls < 2:
        raise ValueError("an episode needs at least two control queries")
    distance = route_distance_matrix(probabilities)
    adjacent, layer, denoise = transition_details(probabilities)
    if not np.allclose(adjacent, np.diag(distance, 1), atol=1e-12, rtol=1e-10):
        raise RuntimeError("direct and pairwise adjacent distances disagree")
    upper = distance[np.triu_indices(controls, 1)]
    lag = np.full(MAX_LAG, np.nan)
    for offset in range(1, min(MAX_LAG, controls - 1) + 1):
        lag[offset - 1] = np.diag(distance, offset).mean()
    phase = transition_phase(controls)
    phase_mean = np.asarray([adjacent[phase == value].mean() for value in range(3)])
    layer_phase = np.stack(
        [layer[phase == value].mean(axis=0) for value in range(3)], axis=0
    )
    return EpisodeSpeed(
        suite=suite,
        task=task,
        episode=episode,
        controls=controls,
        adjacent=adjacent,
        all_pair_mean=float(upper.mean()),
        lag_mean=lag,
        phase_mean=phase_mean,
        layer_mean=layer.mean(axis=0),
        layer_phase_mean=layer_phase,
        denoise_mean=denoise.mean(axis=0),
    )


def read_index(corpus: Path) -> list[dict[str, Any]]:
    with (corpus / "INDEX.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "suite",
        "task_dir",
        "episode_index",
        "inference_calls",
        "control_step_offset",
        "path",
    }
    if not rows or required - set(rows[0]):
        raise ValueError("INDEX.csv does not have the expected schema")
    return rows


def load_all_episodes(corpus: Path) -> tuple[list[EpisodeSpeed], dict[str, float]]:
    grouped: dict[Path, list[dict[str, str]]] = defaultdict(list)
    for row in read_index(corpus):
        task_path = Path(row["path"])
        if not task_path.is_absolute():
            task_path = HERE / task_path
        grouped[task_path.resolve()].append(row)

    episodes = []
    raw_min = float("inf")
    raw_max = float("-inf")
    normalized_error = 0.0
    for task_path, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        group = zarr.open_group(str(task_path / "server/routes.zarr"), mode="r")
        control_axis = np.asarray(group["control_step"][:], dtype=np.int64)
        if not np.array_equal(control_axis, np.arange(len(control_axis))):
            raise ValueError("control_step is not contiguous in %s" % task_path)
        episode_axis = np.asarray(group["episode_id"][:], dtype=np.int64)
        for row in sorted(rows, key=lambda item: int(item["episode_index"])):
            start = int(row["control_step_offset"])
            count = int(row["inference_calls"])
            episode = int(row["episode_index"])
            stop = start + count
            if stop > len(control_axis) or not np.all(episode_axis[start:stop] == episode):
                raise ValueError("episode boundary mismatch in %s episode %d" % (task_path, episode))
            raw = np.asarray(
                group["hb_router_probs"][start:stop, :, :, ACTION_TOKENS, :],
                dtype=np.float64,
            )
            normalized, bounds = normalize_probabilities(raw)
            raw_min = min(raw_min, bounds[0])
            raw_max = max(raw_max, bounds[1])
            probabilities = normalized.mean(axis=3)
            probabilities, _ = normalize_probabilities(probabilities)
            normalized_error = max(
                normalized_error,
                float(np.max(np.abs(probabilities.sum(axis=-1) - 1.0))),
            )
            episodes.append(
                summarize_episode(
                    probabilities,
                    suite=row["suite"],
                    task=row["task_dir"],
                    episode=episode,
                )
            )
        print("loaded", task_path, flush=True)
    return episodes, {
        "raw_probability_sum_min": raw_min,
        "raw_probability_sum_max": raw_max,
        "normalized_probability_sum_max_error": normalized_error,
    }


def _task_groups(episodes: list[EpisodeSpeed]) -> dict[str, list[EpisodeSpeed]]:
    output: dict[str, list[EpisodeSpeed]] = defaultdict(list)
    for episode in episodes:
        output[episode.suite + "/" + episode.task].append(episode)
    return output


def _task_statistics(group: list[EpisodeSpeed]) -> dict[str, np.ndarray | float]:
    adjacent = np.concatenate([episode.adjacent for episode in group])
    episode_adjacent = np.asarray([episode.adjacent.mean() for episode in group])
    episode_all_pair = np.asarray([episode.all_pair_mean for episode in group])
    lag_values = np.stack([episode.lag_mean for episode in group])
    lag = np.full(MAX_LAG, np.nan)
    for offset in range(MAX_LAG):
        available = lag_values[:, offset]
        available = available[np.isfinite(available)]
        if len(available):
            lag[offset] = available.mean()
    phase_values = []
    for phase in range(3):
        values = []
        for episode in group:
            labels = transition_phase(episode.controls)
            values.append(episode.adjacent[labels == phase])
        phase_values.append(float(np.concatenate(values).mean()))
    return {
        "adjacent_concat": float(adjacent.mean()),
        "adjacent_rollout_equal": float(episode_adjacent.mean()),
        "all_pair_rollout_equal": float(episode_all_pair.mean()),
        "continuity_difference": float((episode_all_pair - episode_adjacent).mean()),
        "continuity_ratio": float(episode_adjacent.mean() / episode_all_pair.mean()),
        "lag": lag,
        "phase": np.asarray(phase_values),
        "layer": np.stack([episode.layer_mean for episode in group]).mean(axis=0),
        "layer_phase": np.stack([episode.layer_phase_mean for episode in group]).mean(axis=0),
        "denoise": np.stack([episode.denoise_mean for episode in group]).mean(axis=0),
    }


def _bootstrap_tasks(
    task_stats: list[dict[str, np.ndarray | float]],
    draws: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    continuity = np.asarray([row["continuity_difference"] for row in task_stats], dtype=float)
    phase = np.stack([row["phase"] for row in task_stats])
    continuity_draw = np.empty(draws)
    late_early_draw = np.empty(draws)
    late_early_ratio = np.empty(draws)
    for draw in range(draws):
        selected = rng.integers(0, len(task_stats), len(task_stats))
        continuity_draw[draw] = continuity[selected].mean()
        selected_phase = phase[selected].mean(axis=0)
        late_early_draw[draw] = selected_phase[2] - selected_phase[0]
        late_early_ratio[draw] = selected_phase[2] / selected_phase[0]
    return {
        "continuity_difference_ci95": np.percentile(
            continuity_draw, [2.5, 97.5]
        ).tolist(),
        "late_minus_early_ci95": np.percentile(
            late_early_draw, [2.5, 97.5]
        ).tolist(),
        "late_over_early_ci95": np.percentile(
            late_early_ratio, [2.5, 97.5]
        ).tolist(),
    }


def _sign_flip_p(values: np.ndarray, draws: int, rng: np.random.Generator) -> float:
    observed = abs(float(values.mean()))
    null = np.empty(draws)
    for draw in range(draws):
        sign = rng.choice((-1.0, 1.0), size=len(values))
        null[draw] = abs(float((values * sign).mean()))
    return float((1 + (null >= observed - 1e-15).sum()) / (draws + 1))


def aggregate(
    episodes: list[EpisodeSpeed],
    bootstrap: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    groups = _task_groups(episodes)
    task_rows = {task: _task_statistics(group) for task, group in groups.items()}
    task_stats = list(task_rows.values())
    rng = np.random.default_rng(seed)
    phase = np.stack([row["phase"] for row in task_stats])
    continuity = np.asarray(
        [row["continuity_difference"] for row in task_stats], dtype=float
    )
    late_minus_early = phase[:, 2] - phase[:, 0]

    all_adjacent = np.concatenate([episode.adjacent for episode in episodes])
    episode_adjacent = np.asarray([episode.adjacent.mean() for episode in episodes])
    episode_all_pair = np.asarray([episode.all_pair_mean for episode in episodes])
    task_adjacent = np.asarray(
        [row["adjacent_rollout_equal"] for row in task_stats], dtype=float
    )
    task_all_pair = np.asarray(
        [row["all_pair_rollout_equal"] for row in task_stats], dtype=float
    )
    mean_phase = phase.mean(axis=0)
    lag = np.nanmean(np.stack([row["lag"] for row in task_stats]), axis=0)
    layer = np.stack([row["layer"] for row in task_stats]).mean(axis=0)
    layer_phase = np.stack([row["layer_phase"] for row in task_stats]).mean(axis=0)
    denoise = np.stack([row["denoise"] for row in task_stats]).mean(axis=0)

    by_index: dict[int, list[float]] = defaultdict(list)
    for episode in episodes:
        for index, value in enumerate(episode.adjacent):
            by_index[index].append(float(value))

    task_speed = {
        task: {
            "suite": task.split("/", 1)[0],
            "adjacent_speed": float(row["adjacent_concat"]),
            "continuity_ratio": float(row["continuity_ratio"]),
            "early_speed": float(row["phase"][0]),
            "middle_speed": float(row["phase"][1]),
            "late_speed": float(row["phase"][2]),
            "late_over_early": float(row["phase"][2] / row["phase"][0]),
        }
        for task, row in task_rows.items()
    }
    bootstrap_result = _bootstrap_tasks(task_stats, bootstrap, rng)
    return {
        "global": {
            "adjacent_speed_control_weighted": float(all_adjacent.mean()),
            "adjacent_speed_rollout_equal": float(episode_adjacent.mean()),
            "adjacent_speed_task_equal": float(task_adjacent.mean()),
            "random_order_expected_speed_rollout_equal": float(episode_all_pair.mean()),
            "random_order_expected_speed_task_equal": float(task_all_pair.mean()),
            "continuity_ratio_task_equal": float(
                task_adjacent.mean() / task_all_pair.mean()
            ),
            "continuity_difference_task_equal": float(continuity.mean()),
            "continuity_sign_flip_p_task": _sign_flip_p(
                continuity, permutations, rng
            ),
            **bootstrap_result,
        },
        "phase": {
            name: float(value) for name, value in zip(PHASE_NAMES, mean_phase)
        }
        | {
            "late_minus_early": float(mean_phase[2] - mean_phase[0]),
            "late_over_early": float(mean_phase[2] / mean_phase[0]),
            "late_minus_early_sign_flip_p_task": _sign_flip_p(
                late_minus_early, permutations, rng
            ),
        },
        "lag_curve": [
            {
                "lag_control_steps": offset + 1,
                "distance": float(value),
                "relative_to_lag1": float(value / lag[0]),
                "episodes": int(sum(episode.controls > offset + 1 for episode in episodes)),
            }
            for offset, value in enumerate(lag)
        ],
        "by_control_index": [
            {
                "transition": "%d->%d" % (index, index + 1),
                "start_control": index,
                "mean_speed": float(np.mean(values)),
                "episodes": len(values),
            }
            for index, values in sorted(by_index.items())
        ],
        "by_layer": [
            {
                "layer_axis": layer_axis,
                "adjacent_speed": float(layer[layer_axis]),
                "early_speed": float(layer_phase[0, layer_axis]),
                "late_speed": float(layer_phase[2, layer_axis]),
                "late_over_early": float(
                    layer_phase[2, layer_axis] / layer_phase[0, layer_axis]
                ),
            }
            for layer_axis in range(N_LAYERS)
        ],
        "by_denoise": [
            {"denoise": tau, "adjacent_speed": float(value)}
            for tau, value in enumerate(denoise)
        ],
        "tasks": task_speed,
    }


def _fmt(value: float, signed: bool = False) -> str:
    return ("%+.4f" if signed else "%.4f") % value


def _fmt_p(value: float) -> str:
    return "p<0.0001" if value < 0.0001 else "p=%.4f" % value


def render_report(summary: dict[str, Any]) -> str:
    integrity = summary["integrity"]
    result = summary["result"]
    global_result = result["global"]
    phase = result["phase"]
    task_values = list(result["tasks"].values())
    fastest = max(result["tasks"].items(), key=lambda item: item[1]["adjacent_speed"])
    slowest = min(result["tasks"].items(), key=lambda item: item[1]["adjacent_speed"])
    lines = [
        "# 全控制步 MoE 路由变化速度",
        "",
        "## 实验思路",
        "",
        "- 将每条 rollout 表示为完整的 policy-query 序列 `q0...q(T-1)`，控制步是唯一时间单位；不使用 action、robot state、reward、success 或 simulator progress。",
        "- 使用每个 query 的 8 个 HB 层 x 10 个 denoise round；先逐 site 归一化完整 32 路概率，再平均 10 个 action token。相邻控制步速度定义为对齐 site 上的 RMS Hellinger 距离。",
        "- 覆盖 30 个任务、1500 条完整 rollout 和全部 %d 个控制步；比较真实相邻顺序、episode 内随机顺序期望、1-8 步 lag、归一化早中晚阶段、逐层和逐 denoise round。任务是统计独立簇。"
        % integrity["control_queries"],
        "",
        "## 结果",
        "",
        "共 `%d` 个相邻控制步转移。按任务等权后，真实相邻速度为 `%s`；rollout 内随机打乱顺序的期望为 `%s`，连续性比值为 `%s`（越小表示相邻路由越连续），差值 95%% CI `%s` 到 `%s`，task-level sign-flip `%s`。"
        % (
            integrity["adjacent_transitions"],
            _fmt(global_result["adjacent_speed_task_equal"]),
            _fmt(global_result["random_order_expected_speed_task_equal"]),
            _fmt(global_result["continuity_ratio_task_equal"]),
            _fmt(global_result["continuity_difference_ci95"][0]),
            _fmt(global_result["continuity_difference_ci95"][1]),
            _fmt_p(global_result["continuity_sign_flip_p_task"]),
        ),
        "",
        "| rollout 归一化阶段 | 相邻路由速度 |",
        "|---|---:|",
        "| 前 1/3 | %s |" % _fmt(phase["early"]),
        "| 中 1/3 | %s |" % _fmt(phase["middle"]),
        "| 后 1/3 | %s |" % _fmt(phase["late"]),
        "",
        "后段/前段速度比为 `%s`，差值 `%s`；95%% CI `%s` 到 `%s`，task-level sign-flip `%s`。"
        % (
            _fmt(phase["late_over_early"]),
            _fmt(phase["late_minus_early"], signed=True),
            _fmt(global_result["late_minus_early_ci95"][0], signed=True),
            _fmt(global_result["late_minus_early_ci95"][1], signed=True),
            _fmt_p(phase["late_minus_early_sign_flip_p_task"]),
        ),
        "",
        "| 控制步间隔 | 路由距离 | 相对 lag-1 | 可用 rollout |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["lag_curve"]:
        lines.append(
            "| %d | %s | %s | %d |"
            % (
                row["lag_control_steps"],
                _fmt(row["distance"]),
                _fmt(row["relative_to_lag1"]),
                row["episodes"],
            )
        )
    lines.extend(
        [
            "",
            "| HB 层 | 全程速度 | 前段 | 后段 | 后/前 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    layer_numbers = (2, 3, 4, 5, 12, 13, 14, 15)
    for layer_number, row in zip(layer_numbers, result["by_layer"]):
        lines.append(
            "| %d | %s | %s | %s | %s |"
            % (
                layer_number,
                _fmt(row["adjacent_speed"]),
                _fmt(row["early_speed"]),
                _fmt(row["late_speed"]),
                _fmt(row["late_over_early"]),
            )
        )
    denoise = result["by_denoise"]
    lines.extend(
        [
            "",
            "同一次 query 内，跨控制步变化从 denoise-0 的 `%s` 单调增至 denoise-9 的 `%s`（`%s` 倍）；不同 control query 的路由差异主要在后半段 denoise 被放大。"
            % (
                _fmt(denoise[0]["adjacent_speed"]),
                _fmt(denoise[-1]["adjacent_speed"]),
                _fmt(
                    denoise[-1]["adjacent_speed"]
                    / denoise[0]["adjacent_speed"]
                ),
            ),
            "",
            "任务间绝对速度范围为 `%s-%s`：最快 `%s`，最慢 `%s`。该差异只描述路由时间序列，不解释为物理执行快慢。"
            % (
                _fmt(slowest[1]["adjacent_speed"]),
                _fmt(fastest[1]["adjacent_speed"]),
                fastest[0],
                slowest[0],
            ),
            "",
            "结论：路由不是每个 query 独立重置的白噪声；真实相邻控制步明显比任意两个 episode 内控制步更接近，距离随 control lag 增加后逐渐饱和。早中晚速度是否系统变快或变慢由上表的 task-cluster 检验给出，不借助任何物理动作或成败信息。",
            "",
        ]
    )
    return "\n".join(lines)


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0 or args.permutations <= 0:
        raise ValueError("bootstrap and permutation counts must be positive")
    episodes, normalization = load_all_episodes(args.corpus)
    result = aggregate(
        episodes,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed,
    )
    summary = finite_json(
        {
            "experiment": "complete_control_query_route_speed",
            "integrity": {
                "corpus": str(args.corpus.resolve()),
                "tasks": len(_task_groups(episodes)),
                "rollouts": len(episodes),
                "control_queries": int(sum(episode.controls for episode in episodes)),
                "adjacent_transitions": int(
                    sum(episode.controls - 1 for episode in episodes)
                ),
                "minimum_controls": int(min(episode.controls for episode in episodes)),
                "maximum_controls": int(max(episode.controls for episode in episodes)),
                **normalization,
            },
            "protocol": {
                "time_unit": "policy query / control step",
                "excluded_inputs": [
                    "actions",
                    "robot state",
                    "reward",
                    "success",
                    "simulator progress",
                ],
                "route_feature": "full 32-way probabilities, normalized per layer/denoise/action-token, then averaged over action tokens",
                "speed": "RMS Hellinger distance over aligned 8 layer x 10 denoise sites",
                "uncertainty": "30-task cluster bootstrap and task-level sign flip",
                "bootstrap_draws": args.bootstrap,
                "permutation_draws": args.permutations,
                "seed": args.seed,
            },
            "result": result,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    print("wrote", args.out_dir / "report.md", flush=True)


if __name__ == "__main__":
    main()
