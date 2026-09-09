#!/usr/bin/env python3
"""Analyze MoE routing at every action-token position in early chunks.

The analysis keeps q0-q3, all ten denoising forwards, and all ten action-token
positions.  Position-centered sharpness plus adjacent-token soft and hard route
changes remove the global router-calibration offset that dominates
cross-checkpoint comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

import analyze_libero_suite_early_moe as suite


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "analysis/chunk-token-routing-20260829"
SUITES = ("goal", "object", "spatial")
ALL_SUITES = (*SUITES, "long")
QUERIES = (0, 1, 2, 3)
LAYER_GROUPS = suite.early.LAYER_GROUPS
SEED = 20260829
ADJACENT_METRICS = (
    "adjacent_hellinger",
    "top1_switch_rate",
    "top4_turnover",
)


@dataclass(frozen=True)
class TokenFeature:
    key: str
    metric: str
    layer_group: str
    query: int
    denoise_step: int
    token_position: int
    previous_token_position: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=suite.DEFAULT_CORPUS)
    parser.add_argument("--long-run", type=Path, default=suite.DEFAULT_LONG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_rows(store: zarr.Group, episodes: np.ndarray) -> np.ndarray:
    episode_ids = np.asarray(store["episode_id"][:], dtype=np.int64)
    rows = []
    for episode in episodes:
        found = np.flatnonzero(episode_ids == episode)[: len(QUERIES)]
        if len(found) != len(QUERIES):
            raise ValueError(f"episode {episode}: missing q0-q3")
        rows.extend(found.tolist())
    return np.asarray(rows, dtype=np.int64)


def load_routes(task_path: Path, episodes: np.ndarray) -> np.ndarray:
    store = zarr.open_group(str(task_path / "server/routes.zarr"), mode="r")
    rows = route_rows(store, episodes)
    return np.asarray(store["hb_router_probs"].oindex[rows]).reshape(
        len(episodes), len(QUERIES), 8, 10, 11, 32
    )


def extract_task_profiles(routes: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    if routes.shape[1:] != (4, 8, 10, 11, 32):
        raise ValueError(f"unexpected route tensor {routes.shape}")
    profiles: dict[tuple[str, str], np.ndarray] = {}
    for group_name, layer_axes in LAYER_GROUPS.items():
        selected = np.take(routes, layer_axes, axis=2)
        action = suite.early.normalize(selected[:, :, :, :, 1:, :])
        top1 = action.max(axis=-1).mean(axis=2)
        entropy = suite.early.entropy(action).mean(axis=2)
        adjacent = suite.early.hellinger(
            action[:, :, :, :, 1:, :], action[:, :, :, :, :-1, :]
        ).mean(axis=2)
        top1_expert = np.argmax(action, axis=-1)
        top1_switch = (top1_expert[..., 1:] != top1_expert[..., :-1]).mean(axis=2)
        top4_experts = np.argpartition(action, -4, axis=-1)[..., -4:]
        left = top4_experts[..., 1:, :]
        right = top4_experts[..., :-1, :]
        overlap = (left[..., :, None] == right[..., None, :]).any(axis=-1).sum(axis=-1)
        top4_turnover = (1.0 - overlap / 4.0).mean(axis=2)
        profiles[("top1", group_name)] = top1.mean(axis=0)
        profiles[("entropy", group_name)] = entropy.mean(axis=0)
        profiles[("adjacent_hellinger", group_name)] = adjacent.mean(axis=0)
        profiles[("top1_switch_rate", group_name)] = top1_switch.mean(axis=0)
        profiles[("top4_turnover", group_name)] = top4_turnover.mean(axis=0)
    return profiles


def feature_schema() -> list[TokenFeature]:
    result = []
    for metric in ("top1_centered", "entropy_centered"):
        for group_name in LAYER_GROUPS:
            for query in QUERIES:
                for denoise in range(10):
                    for token in range(1, 11):
                        result.append(
                            TokenFeature(
                                key=(
                                    f"q{query}|{group_name}|{metric}|d{denoise}|t{token}"
                                ),
                                metric=metric,
                                layer_group=group_name,
                                query=query,
                                denoise_step=denoise,
                                token_position=token,
                                previous_token_position=None,
                            )
                        )
    for metric in ADJACENT_METRICS:
        for group_name in LAYER_GROUPS:
            for query in QUERIES:
                for denoise in range(10):
                    for token in range(2, 11):
                        result.append(
                            TokenFeature(
                                key=(
                                    f"q{query}|{group_name}|{metric}|d{denoise}|"
                                    f"t{token - 1}_to_t{token}"
                                ),
                                metric=metric,
                                layer_group=group_name,
                                query=query,
                                denoise_step=denoise,
                                token_position=token,
                                previous_token_position=token - 1,
                            )
                        )
    return result


def flatten_profiles(
    profiles: dict[tuple[str, str], np.ndarray], metadata: list[TokenFeature]
) -> np.ndarray:
    values = []
    centered = {}
    for metric in ("top1", "entropy"):
        for group_name in LAYER_GROUPS:
            raw = profiles[(metric, group_name)]
            centered[(f"{metric}_centered", group_name)] = raw - raw.mean(
                axis=-1, keepdims=True
            )
    for item in metadata:
        if item.metric in ADJACENT_METRICS:
            array = profiles[(item.metric, item.layer_group)]
            token_axis = item.token_position - 2
        else:
            array = centered[(item.metric, item.layer_group)]
            token_axis = item.token_position - 1
        values.append(array[item.query, item.denoise_step, token_axis])
    return np.asarray(values, dtype=np.float32)


def load_task_matrix(
    corpus: Path, long_run: Path
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[TokenFeature],
    list[dict[tuple[str, str], np.ndarray]],
]:
    index = suite.read_index(corpus)
    long_summaries = json.loads(
        (long_run / "client/summaries.json").read_text(encoding="utf-8")
    )
    long_seed = [row for row in long_summaries if int(row["flow_noise_seed"]) == 1000]
    long_seed = sorted(long_seed, key=lambda row: int(row["init_state_id"]))
    init_states = [int(row["init_state_id"]) for row in long_seed]
    metadata = feature_schema()
    rows = []
    values = []
    raw_profiles = []

    matched = index[index["init_state_id"].isin(init_states)]
    for (suite_name, task_dir), group in matched.groupby(
        ["suite", "task_dir"], sort=True
    ):
        group = group.sort_values("init_state_id")
        if group["init_state_id"].astype(int).tolist() != init_states:
            raise ValueError(f"{suite_name}/{task_dir}: initial states do not match")
        task_path = corpus / f"libero_{suite_name}" / task_dir
        routes = load_routes(
            task_path, group["episode_index"].to_numpy(dtype=np.int64)
        )
        profiles = extract_task_profiles(routes)
        rows.append(
            {
                "suite": suite_name,
                "task": f"{suite_name}/{task_dir}",
                "tasks_in_suite": 10,
                "episodes": len(group),
            }
        )
        values.append(flatten_profiles(profiles, metadata))
        raw_profiles.append(profiles)
        print(f"loaded {suite_name}/{task_dir}", flush=True)

    long_episodes = np.asarray(
        [int(row["episode_index"]) for row in long_seed], dtype=np.int64
    )
    long_routes = load_routes(long_run, long_episodes)
    long_profiles = extract_task_profiles(long_routes)
    rows.append(
        {
            "suite": "long",
            "task": "long/t08__two_moka_pots",
            "tasks_in_suite": 1,
            "episodes": len(long_seed),
        }
    )
    values.append(flatten_profiles(long_profiles, metadata))
    raw_profiles.append(long_profiles)
    return pd.DataFrame(rows), np.vstack(values), metadata, raw_profiles


def eta_squared(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    overall = values.mean(axis=0)
    between = np.zeros(values.shape[1], dtype=np.float64)
    within = np.zeros(values.shape[1], dtype=np.float64)
    for label in np.unique(labels):
        selected = values[labels == label]
        center = selected.mean(axis=0)
        between += len(selected) * np.square(center - overall)
        within += np.square(selected - center).sum(axis=0)
    return between / np.maximum(between + within, 1e-30)


def permutation_scan(
    values: np.ndarray, labels: np.ndarray, permutations: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = eta_squared(values, labels)
    rng = np.random.default_rng(seed)
    null = np.empty((permutations, values.shape[1]), dtype=np.float32)
    for draw in range(permutations):
        null[draw] = eta_squared(values, rng.permutation(labels))
    raw_p = (1 + np.sum(null >= observed[None, :], axis=0)) / (permutations + 1)
    maxima = null.max(axis=1)
    max_p = (1 + np.sum(maxima[:, None] >= observed[None, :], axis=0)) / (
        permutations + 1
    )
    return observed, raw_p, max_p


def aggregate_profiles(
    task_frame: pd.DataFrame,
    profiles: list[dict[tuple[str, str], np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for suite_name in ALL_SUITES:
        task_indices = np.flatnonzero(task_frame["suite"].eq(suite_name).to_numpy())
        for metric in ("top1", "entropy", *ADJACENT_METRICS):
            for group_name in LAYER_GROUPS:
                stacked = np.stack(
                    [profiles[index][(metric, group_name)] for index in task_indices]
                )
                mean = stacked.mean(axis=0)
                sd = (
                    stacked.std(axis=0, ddof=1)
                    if len(stacked) > 1
                    else np.full_like(mean, np.nan)
                )
                if metric not in ADJACENT_METRICS:
                    centered = stacked - stacked.mean(axis=-1, keepdims=True)
                    arrays = ((metric, stacked), (f"{metric}_centered", centered))
                else:
                    arrays = ((metric, stacked),)
                for output_metric, array in arrays:
                    output_mean = array.mean(axis=0)
                    output_sd = (
                        array.std(axis=0, ddof=1)
                        if len(array) > 1
                        else np.full_like(output_mean, np.nan)
                    )
                    tokens = range(2, 11) if metric in ADJACENT_METRICS else range(1, 11)
                    for query in QUERIES:
                        for denoise in range(10):
                            for token_axis, token in enumerate(tokens):
                                rows.append(
                                    {
                                        "suite": suite_name,
                                        "metric": output_metric,
                                        "layer_group": group_name,
                                        "query": query,
                                        "denoise_step": denoise,
                                        "token_position": token,
                                        "previous_token_position": (
                                            token - 1
                                            if metric in ADJACENT_METRICS
                                            else np.nan
                                        ),
                                        "task_mean": float(
                                            output_mean[query, denoise, token_axis]
                                        ),
                                        "task_sd": float(
                                            output_sd[query, denoise, token_axis]
                                        ),
                                        "tasks": len(task_indices),
                                    }
                                )
    return pd.DataFrame(rows)


def heatmap_grid(
    profiles: pd.DataFrame, metric: str, group_name: str, path: Path
) -> None:
    selected_all = profiles[
        profiles["metric"].eq(metric) & profiles["layer_group"].eq(group_name)
    ]
    values = selected_all["task_mean"].to_numpy()
    if metric.endswith("_centered"):
        limit = float(np.max(np.abs(values)))
        vmin, vmax, cmap = -limit, limit, "RdBu_r"
    else:
        vmin, vmax, cmap = float(values.min()), float(values.max()), "viridis"
    fig, axes = plt.subplots(4, 4, figsize=(13.2, 11.0), sharex=True, sharey=True)
    image = None
    for row, query in enumerate(QUERIES):
        for column, suite_name in enumerate(ALL_SUITES):
            selected = selected_all[
                selected_all["suite"].eq(suite_name)
                & selected_all["query"].eq(query)
            ]
            pivot = selected.pivot(
                index="denoise_step", columns="token_position", values="task_mean"
            ).sort_index(axis=0).sort_index(axis=1)
            image = axes[row, column].imshow(
                pivot.to_numpy(),
                aspect="auto",
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if row == 0:
                axes[row, column].set_title(suite_name.capitalize())
            if column == 0:
                axes[row, column].set_ylabel(f"q{query} denoise")
            if row == 3:
                axes[row, column].set_xlabel("Chunk token")
                axes[row, column].set_xticks(range(pivot.shape[1]))
                axes[row, column].set_xticklabels(pivot.columns.astype(int))
            axes[row, column].set_yticks(range(10))
    assert image is not None
    fig.colorbar(image, ax=axes, fraction=0.018, pad=0.02)
    fig.suptitle(f"{group_name}: {metric}", fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.07, top=0.93, wspace=0.12, hspace=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate_plot(profiles: pd.DataFrame, path: Path) -> None:
    colors = {
        "goal": "#0072B2",
        "object": "#D55E00",
        "spatial": "#009E73",
        "long": "#CC79A7",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))
    specifications = (
        (axes[0, 0], "top1_centered", None, "Top-1 concentration by token"),
        (axes[0, 1], "adjacent_hellinger", "front_2_5", "Front layers: soft route change"),
        (axes[1, 0], "adjacent_hellinger", "back_12_15", "Back layers: soft route change"),
    )
    for axis, metric, layer_group, title in specifications:
        selected = profiles[profiles["metric"].eq(metric)]
        if layer_group is not None:
            selected = selected[selected["layer_group"].eq(layer_group)]
        curves = selected.groupby(["suite", "token_position"])["task_mean"].mean()
        for suite_name in ALL_SUITES:
            curve = curves.loc[suite_name]
            axis.plot(
                curve.index,
                curve.to_numpy(),
                marker="o",
                markersize=3.5,
                linewidth=1.7,
                color=colors[suite_name],
                label=suite_name.capitalize(),
            )
        if metric.endswith("_centered"):
            axis.axhline(0, color="#777777", linewidth=0.7, alpha=0.6)
        axis.set_title(title)
        axis.set_xlabel("Chunk token position")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.2)

    axis = axes[1, 1]
    selected = profiles[profiles["metric"].eq("adjacent_hellinger")]
    curves = selected.groupby(["suite", "denoise_step"])["task_mean"].mean()
    for suite_name in ALL_SUITES:
        curve = curves.loc[suite_name]
        axis.plot(
            curve.index,
            curve.to_numpy(),
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            color=colors[suite_name],
            label=suite_name.capitalize(),
        )
    axis.set_title("Soft route change over denoising")
    axis.set_xlabel("Denoise forward")
    axis.set_ylabel("adjacent_hellinger")
    axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def token_vector(
    profiles: pd.DataFrame,
    suite_name: str,
    metric: str,
    group_name: str,
    query: int,
    denoise: int,
) -> list[float]:
    selected = profiles[
        profiles["suite"].eq(suite_name)
        & profiles["metric"].eq(metric)
        & profiles["layer_group"].eq(group_name)
        & profiles["query"].eq(query)
        & profiles["denoise_step"].eq(denoise)
    ].sort_values("token_position")
    return selected["task_mean"].astype(float).tolist()


def write_report(
    out: Path,
    scan: pd.DataFrame,
    profiles: pd.DataFrame,
    permutations: int,
) -> None:
    strongest = scan.iloc[int(np.argmax(scan["suite_eta_squared"].to_numpy()))]
    significant = int(np.sum(scan["permutation_p_max_global"] < 0.05))
    feature_count = len(scan)
    lines = [
        "# Early chunk 的逐 token MoE 路由",
        "",
        "## 正确分析轴",
        "",
        "这里保留 q0-q3、10 个去噪步和 chunk 内 10 个 action token，不再把 token 轴平均掉。",
        "主特征是每条 token 曲线去掉自身均值后的 top-1/entropy，以及相邻 token 完整 32-expert",
        "软路由的 Hellinger 距离、top-1 换专家率、top-4 集合替换率。统计检验只覆盖",
        "Goal/Object/Spatial 的 10+10+10 个任务；",
        "Long 只有一个任务，只画作描述性参照。没有使用成功、失败或超时率。",
        "",
        "## q0 / d5 / 后层的十个 token",
        "",
        "| suite | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for suite_name in ALL_SUITES:
        vector = token_vector(
            profiles, suite_name, "top1", "back_12_15", 0, 5
        )
        lines.append(
            "| %s | %s |"
            % (
                suite_name.capitalize(),
                " | ".join(f"{value:.6f}" for value in vector),
            )
        )
    lines += [
        "",
        "## 跨 q0-q3、全部去噪步的稳定形状",
        "",
        "| suite | chunk 两端比中间 top-1 高 | 前层相邻 H | 后层相邻 H | H: d0 -> d9 |",
        "|---|---:|---:|---:|---:|",
    ]
    for suite_name in ALL_SUITES:
        raw_top1 = profiles[
            profiles["suite"].eq(suite_name) & profiles["metric"].eq("top1")
        ].groupby("token_position")["task_mean"].mean()
        endpoints = 0.5 * (raw_top1.loc[1] + raw_top1.loc[10])
        middle = raw_top1.loc[4:7].mean()
        adjacent = profiles[
            profiles["suite"].eq(suite_name)
            & profiles["metric"].eq("adjacent_hellinger")
        ]
        layer_means = adjacent.groupby("layer_group")["task_mean"].mean()
        denoise_means = adjacent.groupby("denoise_step")["task_mean"].mean()
        lines.append(
            "| %s | +%.2f%% | %.5f | %.5f | %.5f -> %.5f (%+.1f%%) |"
            % (
                suite_name.capitalize(),
                100 * (endpoints / middle - 1),
                layer_means.loc["front_2_5"],
                layer_means.loc["back_12_15"],
                denoise_means.loc[0],
                denoise_means.loc[9],
                100 * (denoise_means.loc[9] / denoise_means.loc[0] - 1),
            )
        )
    lines += [
        "",
        "四类都有同一个 U 形骨架：t1/t10 的路由比 t4-t7 更集中。相邻 token 的软路由变化",
        "在前层始终大于后层；Long 的变化随去噪推进下降最明显。",
        "",
        "## 换专家率不能按字面理解",
        "",
        "| suite | top-1 概率均值 | top-1 switch 前/后层 | top-4 turnover 前/后层 |",
        "|---|---:|---:|---:|",
    ]
    for suite_name in ALL_SUITES:
        selected_suite = profiles[profiles["suite"].eq(suite_name)]
        top1_mean = selected_suite[selected_suite["metric"].eq("top1")][
            "task_mean"
        ].mean()
        switch = selected_suite[
            selected_suite["metric"].eq("top1_switch_rate")
        ].groupby("layer_group")["task_mean"].mean()
        turnover = selected_suite[
            selected_suite["metric"].eq("top4_turnover")
        ].groupby("layer_group")["task_mean"].mean()
        lines.append(
            "| %s | %.5f | %.1f%% / %.1f%% | %.1f%% / %.1f%% |"
            % (
                suite_name.capitalize(),
                top1_mean,
                100 * switch.loc["front_2_5"],
                100 * switch.loc["back_12_15"],
                100 * turnover.loc["front_2_5"],
                100 * turnover.loc["back_12_15"],
            )
        )
    lines += [
        "",
        "均匀路由的 top-1 是 0.03125；这里仅为 0.0360-0.0403。概率很接近时，轻微软分布移动",
        "就会交换 top-1/top-4 名次，所以高 switch/turnover 不能直接称为语义阶段切换。完整软分布的",
        "Hellinger 才是这次更可信的主量。",
    ]
    lines += [
        "",
        "## 位置形状是否真的区分类别",
        "",
        "去掉每条曲线的整体高低后，最强位置 cell 是 `%s`：suite 解释任务间 %.1f%% 方差，"
        "%d 次任务标签置换、%d 格 max-T `p=%.4f`。"
        % (
            strongest["key"],
            100 * strongest["suite_eta_squared"],
            permutations,
            feature_count,
            strongest["permutation_p_max_global"],
        ),
        "共有 `%d/%d` 个位置 cell 通过全局 max-T 0.05。"
        % (significant, feature_count),
        "这检验的是 chunk 内曲线形状，不是不同 checkpoint 的整体路由锐度。",
        "但四类数据使用四个不同 checkpoint，suite 与 checkpoint 效应仍然完全混杂；这个 p 值只能",
        "证明四个已保存系统的位置指纹不同，不能把差异因果归于 Goal/Object/Spatial 任务语义。",
        "",
        "## 每类在 q0/d5/后层的最大相邻跳变",
        "",
        "| suite | Hellinger 最大处 | H | top-1 switch 最大处 | rate | top-4 turnover 最大处 | rate |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    for suite_name in ALL_SUITES:
        maxima = []
        for metric in ADJACENT_METRICS:
            selected = profiles[
                profiles["suite"].eq(suite_name)
                & profiles["metric"].eq(metric)
                & profiles["layer_group"].eq("back_12_15")
                & profiles["query"].eq(0)
                & profiles["denoise_step"].eq(5)
            ]
            maxima.append(selected.iloc[int(np.argmax(selected["task_mean"].to_numpy()))])
        lines.append(
            "| %s | t%d -> t%d | %.6f | t%d -> t%d | %.4f | t%d -> t%d | %.4f |"
            % (
                suite_name.capitalize(),
                int(maxima[0]["previous_token_position"]),
                int(maxima[0]["token_position"]),
                maxima[0]["task_mean"],
                int(maxima[1]["previous_token_position"]),
                int(maxima[1]["token_position"]),
                maxima[1]["task_mean"],
                int(maxima[2]["previous_token_position"]),
                int(maxima[2]["token_position"]),
                maxima[2]["task_mean"],
            )
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "这些结果能说明模型在一个 action chunk 内给不同未来位置分配了不同路由，且这种位置曲线",
        "随 suite/独立 checkpoint 改变。它还不能单独说明哪个 token ‘更难’，也不能把 Long 的单任务",
        "曲线推广到整个 Long suite。Long 与哪一类最像也随指标而变，不能稳定归为 Goal/Object/Spatial。",
        "要把位置结构与初态难度连接起来，下一步应在同一 Long checkpoint",
        "的 16 初态 x 32 分支内，用初态失败倾向对每个 `(q,d,token)` 做相关，而不是跨 suite 用超时率排序。",
    ]
    (out / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    task_frame, matrix, metadata, raw_profiles = load_task_matrix(
        args.corpus, args.long_run
    )
    gos = task_frame["suite"].isin(SUITES).to_numpy()
    labels = task_frame.loc[gos, "suite"].to_numpy()
    observed, raw_p, max_p = permutation_scan(
        matrix[gos], labels, args.permutations, args.seed
    )
    scan = pd.DataFrame(
        [
            {
                **asdict(item),
                "suite_eta_squared": observed[index],
                "permutation_p_raw": raw_p[index],
                "permutation_p_max_global": max_p[index],
                "goal_mean": float(matrix[task_frame["suite"].eq("goal"), index].mean()),
                "object_mean": float(
                    matrix[task_frame["suite"].eq("object"), index].mean()
                ),
                "spatial_mean": float(
                    matrix[task_frame["suite"].eq("spatial"), index].mean()
                ),
                "long_value": float(
                    matrix[task_frame["suite"].eq("long"), index].mean()
                ),
            }
            for index, item in enumerate(metadata)
        ]
    )
    profiles = aggregate_profiles(task_frame, raw_profiles)

    scan.to_csv(args.out / "token_position_feature_scan.csv", index=False)
    profiles.to_csv(args.out / "suite_token_profiles.csv", index=False)
    task_frame.to_csv(args.out / "task_cohort.csv", index=False)
    task_matrix = pd.DataFrame(matrix, columns=[item.key for item in metadata])
    task_matrix.insert(0, "task", task_frame["task"].to_numpy())
    task_matrix.insert(0, "suite", task_frame["suite"].to_numpy())
    task_matrix.to_csv(args.out / "task_feature_matrix.csv", index=False)
    heatmap_grid(
        profiles,
        "top1_centered",
        "back_12_15",
        args.out / "back_top1_position_centered.png",
    )
    heatmap_grid(
        profiles,
        "adjacent_hellinger",
        "back_12_15",
        args.out / "back_adjacent_token_hellinger.png",
    )
    heatmap_grid(
        profiles,
        "top1_switch_rate",
        "back_12_15",
        args.out / "back_top1_switch_rate.png",
    )
    heatmap_grid(
        profiles,
        "top4_turnover",
        "back_12_15",
        args.out / "back_top4_turnover.png",
    )
    aggregate_plot(profiles, args.out / "aggregate_token_profiles.png")
    write_report(args.out, scan, profiles, args.permutations)

    strongest = scan.iloc[int(np.argmax(scan["suite_eta_squared"].to_numpy()))]
    summary = {
        "schema": "himoe.chunk_token_routing.v1",
        "queries": list(QUERIES),
        "denoise_steps": 10,
        "action_token_positions": 10,
        "features": len(metadata),
        "permutations": args.permutations,
        "outcome_labels_used": False,
        "strongest_feature": plain(strongest.to_dict()),
        "features_passing_max_t_005": int(
            np.sum(scan["permutation_p_max_global"] < 0.05)
        ),
    }
    (args.out / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    artifacts = sorted(
        path for path in args.out.iterdir() if path.name != "checksums.sha256"
    )
    (args.out / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="ascii",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
