#!/usr/bin/env python3
"""Compare q0 denoising routes for timeout versus success across LIBERO suites.

Goal, Object, and Spatial use the verified libero30-right-v1 corpus: ten tasks
and fifty distinct initial states per task, with one flow-noise draw per state.
Outcome labels are permuted only within task and mixed tasks are weighted
equally.  Long is kept separate because the available corpus contains one task
with sixteen initial states and thirty-two flow-noise draws per state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.stats import spearmanr

import analyze_early_chunk_denoise_timeout as early


HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "corpus/libero30-right-v1"
DEFAULT_LONG = (
    HERE.parent
    / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long"
    / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
)
DEFAULT_OUT = HERE / "analysis/libero-suite-early-moe-20260829"
FROZEN_KEY = "q0|action|back_12_15|top1|d5"
SUITES = ("goal", "object", "spatial")
MAX_STEPS = {"goal": 300, "object": 280, "spatial": 220}
DISCOVERY_TASKS = {
    "goal/t00__open_the_middle_drawer_of_the_cabinet",
    "goal/t03__open_the_top_drawer_and_put_the_bowl_inside",
    "spatial/t05__pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "spatial/t07__pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
}
SEED = 20260829


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--long-run", type=Path, default=DEFAULT_LONG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstraps", type=int, default=5000)
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


def read_index(root: Path) -> pd.DataFrame:
    with (root / "INDEX.csv").open(newline="", encoding="utf-8") as stream:
        frame = pd.DataFrame(csv.DictReader(stream))
    for column in (
        "task_id",
        "episode_index",
        "init_state_id",
        "flow_noise_seed",
        "success",
        "action_steps",
        "inference_calls",
        "control_step_offset",
    ):
        frame[column] = pd.to_numeric(frame[column])
    frame["failure"] = ~frame["success"].astype(bool)
    frame["task"] = frame["suite"] + "/" + frame["task_dir"]
    return frame.sort_values(["suite", "task_id", "episode_index"]).reset_index(drop=True)


def q0_features(
    routes: np.ndarray,
) -> tuple[np.ndarray, list[early.RouteFeature]]:
    """Extract the 118 action-route cells from [episode, 8, 10, 11, 32]."""
    if routes.ndim != 5 or routes.shape[1:] != (8, 10, 11, 32):
        raise ValueError(f"unexpected q0 route tensor {routes.shape}")
    columns: list[np.ndarray] = []
    metadata: list[early.RouteFeature] = []
    for group_name, layer_axes in early.LAYER_GROUPS.items():
        action = early.normalize(routes[:, layer_axes, :, 1:, :])
        center = action.mean(axis=3, keepdims=True)
        metrics = {
            "entropy": early.entropy(action).mean(axis=(1, 3)),
            "top1": action.max(axis=-1).mean(axis=(1, 3)),
            "top4": early.top4_mass(action).mean(axis=(1, 3)),
            "p4p5_gap": early.p4p5_gap(action).mean(axis=(1, 3)),
            "token_dispersion": early.hellinger(action, center).mean(axis=(1, 3)),
        }
        for metric, values in metrics.items():
            for denoise in range(early.N_DENOISE):
                columns.append(values[:, denoise])
                metadata.append(
                    early.RouteFeature(
                        key=f"q0|action|{group_name}|{metric}|d{denoise}",
                        family=f"level_{group_name}",
                        query=0,
                        layer_group=group_name,
                        metric=metric,
                        denoise_step=denoise,
                        feature_kind="level",
                    )
                )
        motion = early.hellinger(action[:, :, 1:], action[:, :, :-1]).mean(
            axis=(1, 3)
        )
        for denoise in range(1, early.N_DENOISE):
            columns.append(motion[:, denoise - 1])
            metadata.append(
                early.RouteFeature(
                    key=(
                        f"q0|action|{group_name}|denoise_motion|"
                        f"d{denoise - 1}_to_d{denoise}"
                    ),
                    family=f"motion_{group_name}",
                    query=0,
                    layer_group=group_name,
                    metric="denoise_motion",
                    denoise_step=denoise,
                    feature_kind="motion",
                )
            )
    return np.column_stack(columns).astype(np.float32), metadata


def first_route_rows(store: zarr.Group, episodes: np.ndarray) -> np.ndarray:
    episode_ids = np.asarray(store["episode_id"][:], dtype=np.int64)
    rows = []
    for episode in episodes:
        found = np.flatnonzero(episode_ids == episode)
        if len(found) < 1:
            raise ValueError(f"episode {episode}: no route rows")
        rows.append(int(found[0]))
    return np.asarray(rows, dtype=np.int64)


def load_libero30(
    root: Path,
) -> tuple[pd.DataFrame, np.ndarray, list[early.RouteFeature]]:
    frame = read_index(root)
    chunks = []
    schemas: list[list[early.RouteFeature]] = []
    for (suite, task_dir), rows in frame.groupby(["suite", "task_dir"], sort=True):
        task_path = root / f"libero_{suite}" / task_dir
        store = zarr.open_group(str(task_path / "server/routes.zarr"), mode="r")
        indices = first_route_rows(
            store, rows["episode_index"].to_numpy(dtype=np.int64)
        )
        routes = np.asarray(store["hb_router_probs"].oindex[indices])
        values, metadata = q0_features(routes)
        chunks.append((rows.index.to_numpy(), values))
        schemas.append(metadata)
        print(f"loaded {suite}/{task_dir}", flush=True)
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("feature schemas differ across tasks")
    values = np.empty((len(frame), len(schemas[0])), dtype=np.float32)
    for indices, chunk in chunks:
        values[indices] = chunk
    return frame, values, schemas[0]


def load_long(
    run: Path,
) -> tuple[pd.DataFrame, np.ndarray, list[early.RouteFeature]]:
    summaries = json.loads((run / "client/summaries.json").read_text(encoding="utf-8"))
    summaries = sorted(summaries, key=lambda row: int(row["episode_index"]))
    episodes = np.asarray([int(row["episode_index"]) for row in summaries])
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    indices = first_route_rows(store, episodes)
    routes = np.asarray(store["hb_router_probs"].oindex[indices])
    values, metadata = q0_features(routes)
    frame = pd.DataFrame(
        {
            "suite": "long",
            "task": "long/two_moka_pots",
            "episode_index": episodes,
            "init_state_id": [int(row["init_state_id"]) for row in summaries],
            "flow_noise_seed": [int(row["flow_noise_seed"]) for row in summaries],
            "failure": [not bool(row["success"]) for row in summaries],
            "action_steps": [int(row["action_steps"]) for row in summaries],
        }
    )
    return frame, values, metadata


def mixed_tasks(frame: pd.DataFrame) -> list[str]:
    variation = frame.groupby("task")["failure"].nunique()
    return variation.index[variation == 2].tolist()


def task_standardized_effects(
    frame: pd.DataFrame, values: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[str]]:
    effects = []
    standardized = []
    labels = []
    tasks = mixed_tasks(frame)
    for task in tasks:
        take = np.flatnonzero(frame["task"].eq(task).to_numpy())
        x = values[take].astype(np.float64)
        y = frame.iloc[take]["failure"].to_numpy(dtype=np.int8)
        scale = x.std(axis=0, ddof=1)
        z = (x - x.mean(axis=0)) / np.maximum(scale, 1e-12)
        effects.append(z[y == 1].mean(axis=0) - z[y == 0].mean(axis=0))
        standardized.append(z)
        labels.append(y)
    if not effects:
        raise ValueError("no task has both success and failure")
    return np.vstack(effects), standardized, labels, tasks


def macro_permutation_scan(
    standardized: list[np.ndarray],
    labels: list[np.ndarray],
    observed: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    null = np.zeros((permutations, len(observed)), dtype=np.float32)
    for x, y in zip(standardized, labels, strict=True):
        shuffled = np.empty((permutations, len(y)), dtype=np.float32)
        for draw in range(permutations):
            shuffled[draw] = rng.permutation(y)
        failures = int(y.sum())
        successes = len(y) - failures
        factor = 1.0 / failures + 1.0 / successes
        null += (shuffled @ x.astype(np.float32)) * factor / len(labels)
    absolute = np.abs(null)
    observed_absolute = np.abs(observed)
    raw = (1 + np.sum(absolute >= observed_absolute[None, :], axis=0)) / (
        permutations + 1
    )
    maxima = absolute.max(axis=1)
    global_p = (1 + np.sum(maxima[:, None] >= observed_absolute, axis=0)) / (
        permutations + 1
    )
    return raw, global_p


def analyze_group(
    name: str,
    frame: pd.DataFrame,
    values: np.ndarray,
    metadata: list[early.RouteFeature],
    permutations: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    task_effect, standardized, labels, tasks = task_standardized_effects(frame, values)
    observed = task_effect.mean(axis=0)
    raw_p, global_p = macro_permutation_scan(
        standardized, labels, observed, permutations, seed
    )
    rng = np.random.default_rng(seed + 1)
    picks = rng.integers(0, len(tasks), size=(bootstraps, len(tasks)))
    bootstrap = task_effect[picks].mean(axis=1)
    ci_low = np.quantile(bootstrap, 0.025, axis=0)
    ci_high = np.quantile(bootstrap, 0.975, axis=0)
    rows = []
    for index, meta in enumerate(metadata):
        rows.append(
            {
                "dataset": name,
                **asdict(meta),
                "task_macro_effect_sigma": observed[index],
                "task_bootstrap_ci95_low": ci_low[index],
                "task_bootstrap_ci95_high": ci_high[index],
                "permutation_p_raw": raw_p[index],
                "permutation_p_max_118": global_p[index],
                "tasks_positive": int(np.sum(task_effect[:, index] > 0)),
                "tasks_negative": int(np.sum(task_effect[:, index] < 0)),
                "mixed_tasks": len(tasks),
            }
        )
    task_rows = []
    for task_axis, task in enumerate(tasks):
        task_frame = frame[frame["task"].eq(task)]
        for feature_axis, meta in enumerate(metadata):
            task_rows.append(
                {
                    "dataset": name,
                    "task": task,
                    "failures": int(task_frame["failure"].sum()),
                    "successes": int((~task_frame["failure"]).sum()),
                    "key": meta.key,
                    "effect_sigma": task_effect[task_axis, feature_axis],
                }
            )
    summary = {
        "dataset": name,
        "episodes_all_tasks": len(frame),
        "failures_all_tasks": int(frame["failure"].sum()),
        "successes_all_tasks": int((~frame["failure"]).sum()),
        "tasks_all": int(frame["task"].nunique()),
        "mixed_tasks": len(tasks),
        "episodes_in_mixed_tasks": int(frame["task"].isin(tasks).sum()),
        "failures_in_mixed_tasks": int(
            frame.loc[frame["task"].isin(tasks), "failure"].sum()
        ),
    }
    return pd.DataFrame(rows), pd.DataFrame(task_rows), summary


def long_difficulty_split(
    frame: pd.DataFrame, values: np.ndarray, feature_index: int
) -> dict[str, float]:
    seeds = sorted(frame["flow_noise_seed"].unique())
    if len(seeds) != 32:
        raise ValueError(f"expected 32 Long seeds, got {len(seeds)}")
    halves = (set(seeds[:16]), set(seeds[16:]))
    result: dict[str, float] = {}
    for direction, (feature_seeds, outcome_seeds) in enumerate(
        ((halves[0], halves[1]), (halves[1], halves[0]))
    ):
        feature = []
        outcome = []
        for init_state in sorted(frame["init_state_id"].unique()):
            state = frame["init_state_id"].eq(init_state).to_numpy()
            feature_take = state & frame["flow_noise_seed"].isin(feature_seeds).to_numpy()
            outcome_take = state & frame["flow_noise_seed"].isin(outcome_seeds).to_numpy()
            feature.append(float(values[feature_take, feature_index].mean()))
            outcome.append(float(frame.loc[outcome_take, "failure"].mean()))
        correlation = spearmanr(feature, outcome).statistic
        result[f"half_{direction + 1}_to_{2 - direction}_spearman"] = float(correlation)
    return result


def within_init_branch_test(
    frame: pd.DataFrame,
    values: np.ndarray,
    metadata: list[early.RouteFeature],
    feature_index: int,
    permutations: int,
    seed: int,
) -> dict[str, float | int]:
    mixed = frame.groupby("init_state_id")["failure"].nunique()
    mixed_states = mixed.index[mixed == 2]
    take = frame["init_state_id"].isin(mixed_states).to_numpy()
    selected = values[take, feature_index : feature_index + 1]
    labels = frame.loc[take, "failure"].to_numpy(dtype=np.int64)
    groups, _ = pd.factorize(frame.loc[take, "init_state_id"], sort=True)
    _, _, effect, _ = early.layer.fixed_effect_shift(selected, labels, groups)
    raw_p, _, _ = early.layer.permutation_scan(
        selected,
        labels,
        groups,
        early.to_layer_metadata([metadata[feature_index]]),
        permutations,
        seed,
    )
    return {
        "mixed_initial_states": len(mixed_states),
        "episodes_in_mixed_initial_states": int(take.sum()),
        "failures_in_mixed_initial_states": int(labels.sum()),
        "within_init_branch_effect_sigma": float(effect[0]),
        "within_init_branch_p_raw": float(raw_p[0]),
    }


def bonferroni_frozen(
    effects: pd.DataFrame, datasets: tuple[str, ...]
) -> pd.DataFrame:
    selected = effects[
        effects["key"].eq(FROZEN_KEY) & effects["dataset"].isin(datasets)
    ].copy()
    selected["p_frozen_bonferroni_3_suites"] = np.minimum(
        1.0, selected["permutation_p_raw"] * len(SUITES)
    )
    return selected


def write_plot(profile: pd.DataFrame, path: Path) -> None:
    colors = {"goal": "#3B6FB6", "object": "#D07B2D", "spatial": "#26876E"}
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for suite in SUITES:
        subset = profile[profile["dataset"].eq(suite)].sort_values("denoise_step")
        axis.plot(
            subset["denoise_step"],
            subset["task_macro_effect_sigma"],
            marker="o",
            color=colors[suite],
            label=suite.capitalize(),
        )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.axvline(5, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Denoising step within first query")
    axis.set_ylabel("Timeout - success (task-macro SD)")
    axis.set_xticks(range(10))
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_report(
    out: Path,
    counts: pd.DataFrame,
    frozen: pd.DataFrame,
    independent_effects: pd.DataFrame,
    independent_frozen: pd.DataFrame,
    independent_counts: pd.DataFrame,
    long_summary: dict[str, Any],
) -> None:
    lines = [
        "# LIBERO suite 的第一次推理 MoE 对照",
        "",
        "## 先纠正比较口径",
        "",
        "此前列出的 2 个 Goal、1 个 Long、2 个 Spatial 是五个具体任务，",
        "不是 Goal/Object/Spatial/Long 四个 suite 的比较；其中根本没有 Object。",
        "本报告使用同协议的 `libero30-right-v1` 重算 Goal、Object、Spatial，",
        "Long 因为只有一个任务且采用 16 初态 x 32 噪声分支，严格单列。",
        "",
        "## 数据与检验",
        "",
        "- Goal/Object/Spatial 各 10 个任务 x 50 个不同初态，每个初态只有一次 rollout。",
        "- 55 个失败样本全部跑满各自 suite 的步数上限，因此这里的失败均是最终超时。",
        "- 只读取第一次推理 q0，保留后四层/前四层和 10 个去噪步；不使用时长、remaining-time 或尾部信息。",
        "- 失败标签只在同一任务内置换；只有同时含成功和失败的任务能提供成败对照。",
        "- suite 效应是任务等权的标准化差值：正数表示失败初态的该路由量更高。",
        "- `q0/back/top1/d5` 是从 Long 冻结后拿来复现的坐标；另对 118 个 q0 action 坐标做 max-T 校正。",
        "- 旧实验见过 Goal t00/t03 与 Spatial t05/t07；独立验证会整项剔除这 4 个任务。",
        "",
        "## 样本",
        "",
        "| suite | 全部失败/总数 | 可对照任务 | 对照任务内失败数 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in counts.iterrows():
        lines.append(
            "| %s | %d/%d | %d/%d | %d |"
            % (
                row["dataset"].capitalize(),
                row["failures_all_tasks"],
                row["episodes_all_tasks"],
                row["mixed_tasks"],
                row["tasks_all"],
                row["failures_in_mixed_tasks"],
            )
        )
    lines += [
        "",
        "## 含旧任务的完整三类描述",
        "",
        "这张表覆盖每类全部 10 个任务，但 Goal/Spatial 与旧实验有任务重叠，不能作为独立复现。",
        "",
        "| suite | d5 效应 (失败-成功, SD) | task-bootstrap 95% | 正方向任务 | 原始 p | 三类 Bonferroni p | 118 格 max-T p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frozen.iterrows():
        lines.append(
            "| %s | %+.3f | [%+.3f, %+.3f] | %d/%d | %.4f | %.4f | %.4f |"
            % (
                row["dataset"].capitalize(),
                row["task_macro_effect_sigma"],
                row["task_bootstrap_ci95_low"],
                row["task_bootstrap_ci95_high"],
                row["tasks_positive"],
                row["mixed_tasks"],
                row["permutation_p_raw"],
                row["p_frozen_bonferroni_3_suites"],
                row["permutation_p_max_118"],
            )
        )
    lines += [
        "",
        "## 全新任务上的冻结验证",
        "",
        "| suite | d5 效应 (失败-成功, SD) | task-bootstrap 95% | 正方向任务 | 原始 p | 三类 Bonferroni p | 118 格 max-T p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    independent_name = {
        "goal_new_tasks": "Goal",
        "object_new_tasks": "Object",
        "spatial_new_tasks": "Spatial",
    }
    for _, row in independent_frozen.iterrows():
        lines.append(
            "| %s | %+.3f | [%+.3f, %+.3f] | %d/%d | %.4f | %.4f | %.4f |"
            % (
                independent_name[row["dataset"]],
                row["task_macro_effect_sigma"],
                row["task_bootstrap_ci95_low"],
                row["task_bootstrap_ci95_high"],
                row["tasks_positive"],
                row["mixed_tasks"],
                row["permutation_p_raw"],
                row["p_frozen_bonferroni_3_suites"],
                row["permutation_p_max_118"],
            )
        )
    lines += [
        "",
        "剔除旧任务后，可用于成败对照的任务/失败数分别是 Goal `%d/%d`、Object `%d/%d`、Spatial `%d/%d`。"
        % (
            independent_counts.iloc[0]["mixed_tasks"],
            independent_counts.iloc[0]["failures_in_mixed_tasks"],
            independent_counts.iloc[1]["mixed_tasks"],
            independent_counts.iloc[1]["failures_in_mixed_tasks"],
            independent_counts.iloc[2]["mixed_tasks"],
            independent_counts.iloc[2]["failures_in_mixed_tasks"],
        ),
        "",
        "## 全新任务扫描中最强的 q0 坐标",
        "",
        "| suite | 坐标 | 效应 SD | 原始 p | 118 格 max-T p |",
        "|---|---|---:|---:|---:|",
    ]
    for suite, dataset in zip(SUITES, independent_name, strict=True):
        subset = independent_effects[independent_effects["dataset"].eq(dataset)]
        row = subset.iloc[np.argmax(np.abs(subset["task_macro_effect_sigma"].to_numpy()))]
        lines.append(
            "| %s | `%s` | %+.3f | %.4f | %.4f |"
            % (
                suite.capitalize(),
                row["key"],
                row["task_macro_effect_sigma"],
                row["permutation_p_raw"],
                row["permutation_p_max_118"],
            )
        )
    lines += [
        "",
        "## Long 为什么不能直接并表",
        "",
        "Long 现有数据只有 `two_moka_pots` 一个任务。它的 512 条不是 512 个独立初态，",
        "而是 16 个初态各重复 32 个噪声分支。冻结 d5 坐标在两个互不重叠的 16-seed half 之间",
        "预测初态失败率的 Spearman 分别为 `%s` 和 `%s`；但同一初态内区分某条分支成败的效应只有 `%s SD`。"
        % (
            fmt(long_summary["half_1_to_2_spearman"]),
            fmt(long_summary["half_2_to_1_spearman"]),
            fmt(long_summary["within_init_branch_effect_sigma"]),
        ),
        "这说明 Long 的 d5 结果是初态难度关联，不是单条 rollout 的失败读数。",
        "Goal/Object/Spatial 每个初态只有一条 rollout，所以它们只能检验‘失败初态是否不同’，",
        "无法像 Long 一样把初态难度和这次采样的偶然命运拆开。",
        "",
        "## 结论",
        "",
    ]
    replicated = independent_frozen[
        independent_frozen["p_frozen_bonferroni_3_suites"] < 0.05
    ]["dataset"].tolist()
    global_hits = independent_effects[
        independent_effects["permutation_p_max_118"] < 0.05
    ]
    if replicated:
        lines.append(
            "冻结的 Long d5 信号在全新任务的 %s 中通过三类校正。"
            % ", ".join(replicated)
        )
    else:
        lines.append(
            "冻结的 Long d5 信号没有在全新 Goal/Object/Spatial 任务中通过三类校正，不能称为跨 suite 独立复现。"
        )
    if len(global_hits):
        hit_names = ", ".join(sorted(global_hits["dataset"].unique()))
        lines.append(f"全新任务的完整 q0 扫描在 {hit_names} 中至少有一个坐标通过 118 格 max-T。")
    else:
        lines.append("全新任务上，三类的完整 q0 扫描均没有坐标通过 118 格 max-T。")
    lines += [
        "因此，当前强证据仍局限于 Long/two_moka_pots 的初态难度；",
        "不能把它泛化成 Goal/Object/Spatial 通用的早期失败信号。",
        "",
        "逐任务冻结坐标效应见 `task_frozen_effects.csv`，全新任务的完整 118 格结果见 `q0_action_effects_new_tasks.csv`。",
    ]
    (out / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame, values, metadata = load_libero30(args.corpus)
    failed = frame["failure"]
    expected_limit = frame["suite"].map(MAX_STEPS)
    if not frame.loc[failed, "action_steps"].eq(expected_limit[failed]).all():
        raise ValueError("libero30 contains a failure that did not reach its step limit")
    if len(metadata) != 118:
        raise ValueError(f"expected 118 q0 action features, got {len(metadata)}")
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite route feature")

    effects_parts = []
    task_parts = []
    summaries = []
    for suite_axis, suite in enumerate(SUITES):
        take = frame["suite"].eq(suite).to_numpy()
        effects, task_effects, summary = analyze_group(
            suite,
            frame[take].reset_index(drop=True),
            values[take],
            metadata,
            args.permutations,
            args.bootstraps,
            args.seed + suite_axis * 100,
        )
        effects_parts.append(effects)
        task_parts.append(task_effects)
        summaries.append(summary)
    effects = pd.concat(effects_parts, ignore_index=True)
    task_effects = pd.concat(task_parts, ignore_index=True)
    counts = pd.DataFrame(summaries)

    independent_parts = []
    independent_summaries = []
    independent_datasets = tuple(f"{suite}_new_tasks" for suite in SUITES)
    for suite_axis, (suite, dataset) in enumerate(
        zip(SUITES, independent_datasets, strict=True)
    ):
        take = frame["suite"].eq(suite) & ~frame["task"].isin(DISCOVERY_TASKS)
        independent_effect, _, independent_summary = analyze_group(
            dataset,
            frame[take].reset_index(drop=True),
            values[take.to_numpy()],
            metadata,
            args.permutations,
            args.bootstraps,
            args.seed + 300 + suite_axis * 100,
        )
        independent_parts.append(independent_effect)
        independent_summaries.append(independent_summary)
    independent_effects = pd.concat(independent_parts, ignore_index=True)
    independent_counts = pd.DataFrame(independent_summaries)

    long_frame, long_values, long_metadata = load_long(args.long_run)
    if long_metadata != metadata:
        raise ValueError("Long and libero30 feature schemas differ")
    frozen_index = [item.key for item in metadata].index(FROZEN_KEY)
    long_difficulty = long_difficulty_split(long_frame, long_values, frozen_index)
    long_branch = within_init_branch_test(
        long_frame,
        long_values,
        metadata,
        frozen_index,
        args.permutations,
        args.seed + 900,
    )
    long_summary = {
        **long_difficulty,
        **long_branch,
        "episodes": len(long_frame),
        "failures": int(long_frame["failure"].sum()),
        "initial_states": int(long_frame["init_state_id"].nunique()),
        "flow_noise_seeds": int(long_frame["flow_noise_seed"].nunique()),
    }

    frozen = bonferroni_frozen(effects, SUITES)
    independent_frozen = bonferroni_frozen(
        independent_effects, independent_datasets
    )
    task_frozen = task_effects[task_effects["key"].eq(FROZEN_KEY)].copy()
    profile = effects[
        effects["layer_group"].eq("back_12_15")
        & effects["metric"].eq("top1")
        & effects["feature_kind"].eq("level")
    ].copy()

    effects.to_csv(args.out / "q0_action_effects.csv", index=False)
    independent_effects.to_csv(
        args.out / "q0_action_effects_new_tasks.csv", index=False
    )
    task_frozen.to_csv(args.out / "task_frozen_effects.csv", index=False)
    frozen.to_csv(args.out / "frozen_d5_suite_tests.csv", index=False)
    independent_frozen.to_csv(
        args.out / "frozen_d5_new_task_tests.csv", index=False
    )
    profile.to_csv(args.out / "back_top1_denoise_profile.csv", index=False)
    counts.to_csv(args.out / "suite_counts.csv", index=False)
    independent_counts.to_csv(args.out / "new_task_counts.csv", index=False)
    write_plot(profile, args.out / "back_top1_denoise_profile.png")
    write_report(
        args.out,
        counts,
        frozen,
        independent_effects,
        independent_frozen,
        independent_counts,
        long_summary,
    )
    summary_path = args.out / "summary.json"
    summary_path.write_text(
        json.dumps(
            plain(
                {
                    "schema": "himoe.libero_suite_early_moe.v1",
                    "frozen_key": FROZEN_KEY,
                    "permutations": args.permutations,
                    "bootstraps": args.bootstraps,
                    "libero30": summaries,
                    "new_task_validation": independent_summaries,
                    "long": long_summary,
                    "frozen_suite_tests": frozen.to_dict(orient="records"),
                    "frozen_new_task_tests": independent_frozen.to_dict(
                        orient="records"
                    ),
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = sorted(path for path in args.out.iterdir() if path.name != "checksums.sha256")
    (args.out / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="ascii",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
