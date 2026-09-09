#!/usr/bin/env python3
"""Compare first-query MoE geometry across LIBERO task suites.

The clean comparison covers Goal, Object, and Spatial, each with ten tasks.
Every task is reduced to the same 16 initial-state IDs at flow-noise seed 1000.
Long is a descriptive projection only: the local route corpus contains one Long
task, recorded with a different checkpoint and wrist protocol.
"""

from __future__ import annotations

import argparse
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

import analyze_libero_suite_early_moe as suite


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "analysis/early-moe-suite-identity-20260829"
SUITES = ("goal", "object", "spatial")
SEED = 20260829


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


def task_centroids(
    frame: pd.DataFrame, values: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    centroids = []
    for (suite_name, task), group in frame.groupby(["suite", "task"], sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        rows.append(
            {
                "suite": suite_name,
                "task": task,
                "episodes": len(indices),
                "failures": int(group["failure"].sum()),
            }
        )
        centroids.append(values[indices].mean(axis=0))
    return pd.DataFrame(rows), np.vstack(centroids)


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
    values: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    seed: int,
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


def leave_one_task_out(
    values: np.ndarray,
    labels: np.ndarray,
    long_value: np.ndarray,
    columns: np.ndarray,
) -> dict[str, Any]:
    predictions = []
    for heldout in range(len(labels)):
        train = np.arange(len(labels)) != heldout
        mean = values[train][:, columns].mean(axis=0)
        scale = np.maximum(values[train][:, columns].std(axis=0, ddof=1), 1e-12)
        x_train = (values[train][:, columns] - mean) / scale
        x_test = (values[heldout, columns] - mean) / scale
        centers = np.stack(
            [x_train[labels[train] == name].mean(axis=0) for name in SUITES]
        )
        distances = np.mean(np.square(centers - x_test), axis=1)
        predictions.append(SUITES[int(np.argmin(distances))])

    mean = values[:, columns].mean(axis=0)
    scale = np.maximum(values[:, columns].std(axis=0, ddof=1), 1e-12)
    standardized = (values[:, columns] - mean) / scale
    long_standardized = (long_value[columns] - mean) / scale
    centers = np.stack(
        [standardized[labels == name].mean(axis=0) for name in SUITES]
    )
    long_distances = np.sqrt(np.mean(np.square(centers - long_standardized), axis=1))
    predicted = np.asarray(predictions)
    confusion = {
        truth: {
            guess: int(np.sum((labels == truth) & (predicted == guess)))
            for guess in SUITES
        }
        for truth in SUITES
    }
    return {
        "correct": int(np.sum(predicted == labels)),
        "tasks": len(labels),
        "accuracy": float(np.mean(predicted == labels)),
        "confusion": confusion,
        "long_nearest_suite": SUITES[int(np.argmin(long_distances))],
        "long_distance_goal": float(long_distances[0]),
        "long_distance_object": float(long_distances[1]),
        "long_distance_spatial": float(long_distances[2]),
    }


def checkpoint_info(corpus: Path, long_run: Path) -> dict[str, dict[str, Any]]:
    task_dirs = {
        "goal": corpus / "libero_goal/t00__open_the_middle_drawer_of_the_cabinet",
        "object": corpus
        / "libero_object/t00__pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "spatial": corpus
        / "libero_spatial/t00__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "long": long_run,
    }
    result = {}
    for name, path in task_dirs.items():
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        server = json.loads(
            (path / "client/server_metadata.json").read_text(encoding="utf-8")
        )
        result[name] = {
            "checkpoint_sha256": server["checkpoint_sha256"],
            "wrist_layout": meta["wrist_layout"],
            "mig_profile": meta["mig_profile"],
        }
    return result


def profile_frame(
    task_frame: pd.DataFrame,
    centroids: np.ndarray,
    long_value: np.ndarray,
    metadata: list[suite.early.RouteFeature],
) -> pd.DataFrame:
    keys = [item.key for item in metadata]
    signals = {
        "back_top1": [
            keys.index(f"q0|action|back_12_15|top1|d{denoise}")
            for denoise in range(10)
        ],
        "front_token_dispersion": [
            keys.index(f"q0|action|front_2_5|token_dispersion|d{denoise}")
            for denoise in range(10)
        ],
    }
    rows = []
    labels = task_frame["suite"].to_numpy()
    for signal, columns in signals.items():
        for denoise, column in enumerate(columns):
            for name in SUITES:
                selected = centroids[labels == name, column]
                rows.append(
                    {
                        "signal": signal,
                        "suite": name,
                        "denoise_step": denoise,
                        "task_mean": float(selected.mean()),
                        "task_sd": float(selected.std(ddof=1)),
                        "tasks": len(selected),
                    }
                )
            rows.append(
                {
                    "signal": signal,
                    "suite": "long",
                    "denoise_step": denoise,
                    "task_mean": float(long_value[column]),
                    "task_sd": np.nan,
                    "tasks": 1,
                }
            )
    return pd.DataFrame(rows)


def plot_profiles(frame: pd.DataFrame, path: Path) -> None:
    colors = {
        "goal": "#3B6FB6",
        "object": "#D07B2D",
        "spatial": "#26876E",
        "long": "#9A4E9E",
    }
    titles = {
        "back_top1": "Back layers: top-1 route mass",
        "front_token_dispersion": "Front layers: token route dispersion",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for axis, signal in zip(axes, titles, strict=True):
        for name in (*SUITES, "long"):
            selected = frame[
                frame["signal"].eq(signal) & frame["suite"].eq(name)
            ].sort_values("denoise_step")
            axis.plot(
                selected["denoise_step"],
                selected["task_mean"],
                marker="o",
                linewidth=2,
                color=colors[name],
                label=name.capitalize(),
            )
            if name != "long":
                low = selected["task_mean"] - selected["task_sd"]
                high = selected["task_mean"] + selected["task_sd"]
                axis.fill_between(
                    selected["denoise_step"], low, high, color=colors[name], alpha=0.12
                )
        axis.set_title(titles[signal])
        axis.set_xlabel("Denoising step in q0")
        axis.set_xticks(range(10))
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Task-centroid value")
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    out: Path,
    scan: pd.DataFrame,
    classification: pd.DataFrame,
    profiles: pd.DataFrame,
    outcome: pd.DataFrame,
    checkpoints: dict[str, dict[str, Any]],
) -> None:
    strongest = scan.iloc[int(np.argmax(scan["suite_eta_squared"].to_numpy()))]
    d5 = profiles[
        profiles["signal"].eq("back_top1") & profiles["denoise_step"].eq(5)
    ].set_index("suite")
    failure = outcome.set_index("suite")
    lines = [
        "# q0 MoE 的 LIBERO suite 类别比较",
        "",
        "## 这次比较的是什么",
        "",
        "主比较不是各 suite 内成功对失败，而是 Goal、Object、Spatial 三类任务的 q0 路由几何。",
        "每个任务固定 `flow_noise_seed=1000`，并只取 Long 数据中同样的 16 个初态编号；",
        "先把每个任务平均成一个点，再按任务留一，避免把 16 个初态伪装成 16 个独立任务。",
        "Long 只有 `two_moka_pots` 一个任务，因此只投影到三类坐标系中，不参与四类统计检验。",
        "",
        "## 结果一：类别从第一次去噪就能读出来",
        "",
        "| q0 去噪步 | 留一任务分类正确率 | Long 最靠近 |",
        "|---:|---:|---|",
    ]
    for _, row in classification.iterrows():
        lines.append(
            "| d%d | %d/%d = %.1f%% | %s |"
            % (
                row["denoise_step"],
                row["correct"],
                row["tasks"],
                100 * row["accuracy"],
                row["long_nearest_suite"].capitalize(),
            )
        )
    lines += [
        "",
        "最强的单个可对齐坐标是 `%s`：suite 标签解释了任务间 %.1f%% 的方差，"
        "10,000 次任务标签置换后的 118 格 max-T `p=%.4f`。"
        % (
            strongest["key"],
            100 * strongest["suite_eta_squared"],
            strongest["permutation_p_max_118"],
        ),
        "所以 q0 MoE 确实带有非常强的 suite/模型类别信息，而且 d0 已经存在，不需要等到 d5。",
        "",
        "## 结果二：这个类别信息不是一把难度尺",
        "",
        "| suite | 匹配样本超时率 | q0 back/top1/d5 |",
        "|---|---:|---:|",
    ]
    for name in (*SUITES, "long"):
        lines.append(
            "| %s | %d/%d = %.2f%% | %.6f |"
            % (
                name.capitalize(),
                failure.loc[name, "failures"],
                failure.loc[name, "episodes"],
                100 * failure.loc[name, "failure_rate"],
                d5.loc[name, "task_mean"],
            )
        )
    lines += [
        "",
        "难度顺序是 Goal/Object < Spatial << Long；但 d5 集中度顺序是 Goal < Long < Object < Spatial。",
        "Long 在 10 个去噪步的多指标距离里也始终最靠近 Goal，而不是形成一个‘最难’端点。",
        "因此跨 suite 的绝对路由值主要是类别/模型基线，不能直接解释为难度。",
        "",
        "## 为什么不能把四类做成干净的难度实验",
        "",
        "- 四个 suite 使用四个不同 checkpoint；expert 编号没有可靠的跨 checkpoint 对齐，所以这里只比较熵、top-k mass、token dispersion 等置换不变量。",
        "- Goal/Object/Spatial 是 `checkpoint-right`，唯一 Long 路由是 `paper-right`。",
        "- Long 只有 1/10 个任务有路由，无法估计 Long suite 的任务间变化。",
        "",
        "checkpoint SHA-256 前 12 位：Goal `%s`，Object `%s`，Spatial `%s`，Long `%s`。"
        % tuple(checkpoints[name]["checkpoint_sha256"][:12] for name in (*SUITES, "long")),
        "",
        "## 结论",
        "",
        "第一次 chunk 的 MoE 能非常清楚地区分 Goal/Object/Spatial，并把当前 Long 任务放在更接近 Goal 的路由几何区域。",
        "但这说明的是任务族/独立 checkpoint 的编码，不是模型已经算出了任务难度。",
        "此前 Long 内部 q0 d5 与初态失败率相关，仍然是同一 checkpoint 内的相对难度信号；",
        "它不能用跨 suite 的绝对数值来解释。",
    ]
    (out / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame, values, metadata = suite.load_libero30(args.corpus)
    long_frame, long_values, long_metadata = suite.load_long(args.long_run)
    if metadata != long_metadata:
        raise ValueError("route feature schemas differ")

    init_states = sorted(long_frame["init_state_id"].unique())
    selected = frame["init_state_id"].isin(init_states).to_numpy()
    frame = frame[selected].reset_index(drop=True)
    values = values[selected]
    if not frame.groupby("task").size().eq(16).all():
        raise ValueError("Goal/Object/Spatial tasks do not all have 16 matched states")
    long_selected = long_frame["flow_noise_seed"].eq(1000).to_numpy()
    long_frame = long_frame[long_selected].reset_index(drop=True)
    long_values = long_values[long_selected]
    if len(long_frame) != 16 or sorted(long_frame["init_state_id"]) != init_states:
        raise ValueError("Long seed 1000 does not contain the matched 16 states")

    task_frame, centroids = task_centroids(frame, values)
    labels = task_frame["suite"].to_numpy()
    long_centroid = long_values.mean(axis=0)
    observed, raw_p, max_p = permutation_scan(
        centroids, labels, args.permutations, args.seed
    )
    scan = pd.DataFrame(
        [
            {
                **asdict(item),
                "suite_eta_squared": observed[index],
                "permutation_p_raw": raw_p[index],
                "permutation_p_max_118": max_p[index],
                "goal_mean": float(centroids[labels == "goal", index].mean()),
                "object_mean": float(centroids[labels == "object", index].mean()),
                "spatial_mean": float(centroids[labels == "spatial", index].mean()),
                "long_value": float(long_centroid[index]),
            }
            for index, item in enumerate(metadata)
        ]
    )

    classification_rows = []
    for denoise in range(10):
        columns = np.asarray(
            [
                index
                for index, item in enumerate(metadata)
                if item.feature_kind == "level" and item.denoise_step == denoise
            ],
            dtype=np.int64,
        )
        result = leave_one_task_out(centroids, labels, long_centroid, columns)
        classification_rows.append({"denoise_step": denoise, **result})
    classification = pd.DataFrame(classification_rows)
    profiles = profile_frame(task_frame, centroids, long_centroid, metadata)

    outcome_rows = []
    for name in SUITES:
        selected_frame = frame[frame["suite"].eq(name)]
        outcome_rows.append(
            {
                "suite": name,
                "episodes": len(selected_frame),
                "failures": int(selected_frame["failure"].sum()),
                "failure_rate": float(selected_frame["failure"].mean()),
            }
        )
    outcome_rows.append(
        {
            "suite": "long",
            "episodes": len(long_frame),
            "failures": int(long_frame["failure"].sum()),
            "failure_rate": float(long_frame["failure"].mean()),
        }
    )
    outcome = pd.DataFrame(outcome_rows)
    checkpoints = checkpoint_info(args.corpus, args.long_run)

    scan.to_csv(args.out / "q0_suite_feature_scan.csv", index=False)
    classification.to_json(
        args.out / "leave_one_task_out_by_denoise.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )
    profiles.to_csv(args.out / "suite_denoise_profiles.csv", index=False)
    outcome.to_csv(args.out / "matched_outcomes.csv", index=False)
    task_frame.to_csv(args.out / "task_cohort.csv", index=False)
    plot_profiles(profiles, args.out / "suite_denoise_profiles.png")
    write_report(args.out, scan, classification, profiles, outcome, checkpoints)

    summary = {
        "schema": "himoe.early_moe_suite_identity.v1",
        "matched_initial_state_ids": init_states,
        "flow_noise_seed": 1000,
        "goal_object_spatial_tasks_each": 10,
        "long_tasks": 1,
        "permutations": args.permutations,
        "checkpoint_info": checkpoints,
        "strongest_feature": plain(
            scan.iloc[int(np.argmax(scan["suite_eta_squared"].to_numpy()))].to_dict()
        ),
        "features_passing_max_t_005": int(
            np.sum(scan["permutation_p_max_118"] < 0.05)
        ),
        "classification": plain(classification_rows),
        "outcomes": plain(outcome_rows),
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
