#!/usr/bin/env python3
"""Test whether the Long q0 within-state failure AUC replicates across tasks.

Every run has 16 initial states crossed with the same 32 flow-noise seeds.  The
primary analysis repeats the Long nested held-out-seed procedure independently
inside each task.  A second analysis freezes a complete q0 model on one task
and evaluates it on another task without reading target-task outcome labels.
Only q0 MoE router probabilities are inputs; rollout length and simulator state
are never used as features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import analyze_chunk_token_outcomes as previous
import analyze_full_chunk_outcome_auc as full
import analyze_initial_state_early_moe_signal as initial


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
DEFAULT_OUT = HERE / "analysis/q0-cross-task-outcome-auc-20260829"
SEED = 20260829
SHIFT_OFFSETS = (1, 2, 3)
LONG_TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def task_id(run: Path) -> str:
    return f"{run.parent.parent.name}/{run.parent.name}"


def discover_runs(root: Path) -> list[Path]:
    runs = sorted(root.glob("*/*/right-16x32"))
    if not runs:
        raise FileNotFoundError(f"no right-16x32 runs under {root}")
    return runs


def read_frame(run: Path) -> pd.DataFrame:
    rows = json.loads((run / "client/summaries.json").read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda item: int(item["episode_index"]))
    frame = pd.DataFrame(
        {
            "episode": [int(item["episode_index"]) for item in rows],
            "init_state_id": [int(item["init_state_id"]) for item in rows],
            "flow_noise_seed": [int(item["flow_noise_seed"]) for item in rows],
            "success": [bool(item["success"]) for item in rows],
            "inference_calls": [int(item["inference_calls"]) for item in rows],
        }
    )
    frame["failure"] = ~frame["success"]
    if len(frame) != 512 or frame["episode"].nunique() != 512:
        raise ValueError(f"{run}: expected 512 unique episodes")
    if frame["init_state_id"].nunique() != 16:
        raise ValueError(f"{run}: expected 16 initial states")
    counts = frame.groupby("init_state_id")["flow_noise_seed"].nunique()
    if not counts.eq(32).all():
        raise ValueError(f"{run}: every initial state must have 32 seeds")
    return previous.assign_folds(frame).reset_index(drop=True)


def q0_feature_schema() -> tuple[list[str], list[str], int]:
    schema = full.schemas()
    aggregate_names, aggregate_families, _ = schema["aggregate"]
    cell_names, cell_families, _ = schema["cell"]
    names = [f"aggregate/{name}" for name in aggregate_names] + [
        f"position/{name}" for name in cell_names
    ]
    families = aggregate_families + cell_families
    return names, families, len(aggregate_names)


def extract_q0_matrix(run: Path, frame: pd.DataFrame) -> np.ndarray:
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    episode_rows = np.asarray(store["episode_id"][:], dtype=np.int64)
    control_rows = np.asarray(store["control_step"][:], dtype=np.int64)
    q0_indices = []
    for episode in frame["episode"].to_numpy(dtype=np.int64):
        indices = np.flatnonzero(episode_rows == episode)
        if not len(indices):
            raise ValueError(f"{run}: episode {episode} has no route rows")
        minimum = control_rows[indices].min()
        selected = indices[control_rows[indices] == minimum]
        if len(selected) != 1:
            raise ValueError(f"{run}: episode {episode} has ambiguous q0")
        q0_indices.append(int(selected[0]))
    raw = np.asarray(store["hb_router_probs"].oindex[np.asarray(q0_indices)])
    aggregate, cells = full.current_route_features(raw)
    return np.c_[aggregate, cells].astype(np.float32)


def load_q0_matrix(
    run: Path, frame: pd.DataFrame, cache: Path, rebuild: bool
) -> np.ndarray:
    if cache.is_file() and not rebuild:
        with np.load(cache, allow_pickle=False) as archive:
            if not np.array_equal(
                archive["episode"], frame["episode"].to_numpy(dtype=np.int32)
            ):
                raise ValueError(f"{cache}: episode order mismatch")
            return np.asarray(archive["matrix"], dtype=np.float16).astype(np.float32)
    matrix = extract_q0_matrix(run, frame)
    np.savez_compressed(
        cache,
        # Match the feature-cache precision used by the original 0.601540 result.
        matrix=matrix.astype(np.float16),
        episode=frame["episode"].to_numpy(dtype=np.int32),
    )
    return matrix.astype(np.float16).astype(np.float32)


def append_rows(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    model: str,
    variant: str,
    fold: int,
) -> None:
    for position, index in enumerate(indices):
        item = frame.iloc[int(index)]
        rows.append(
            {
                "split": "heldout_seed",
                "query": 0,
                "cohort": "common",
                "model": model,
                "variant": variant,
                "fold": fold,
                "episode": int(item["episode"]),
                "init_state_id": int(item["init_state_id"]),
                "flow_noise_seed": int(item["flow_noise_seed"]),
                "failure": bool(item["failure"]),
                "score_failure": float(scores[position]),
            }
        )


def nested_oof(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    names: list[str],
    families: list[str],
    aggregate_dim: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    model_blocks = {
        "aggregate_current": (
            matrix[:, :aggregate_dim],
            names[:aggregate_dim],
            families[:aggregate_dim],
        ),
        "token_current": (matrix, names, families),
    }
    for fold in range(8):
        test_index = np.flatnonzero(frame["heldout_seed_fold"].eq(fold).to_numpy())
        train_index = np.flatnonzero(~frame["heldout_seed_fold"].eq(fold).to_numpy())
        y_train = frame.loc[train_index, "failure"].to_numpy(dtype=bool)
        train_states = frame.loc[train_index, "init_state_id"].to_numpy(dtype=int)
        test_states = frame.loc[test_index, "init_state_id"].to_numpy(dtype=int)
        train_prior, test_prior = initial.prior_scores(
            y_train, train_states, test_states, True
        )
        append_rows(rows, frame, test_index, test_prior, "state_prior", "aligned", fold)
        test_frame = frame.iloc[test_index].reset_index(drop=True)
        shift_maps = [full.shift_indices(test_frame, offset) for offset in SHIFT_OFFSETS]
        for model, (values, model_names, model_families) in model_blocks.items():
            shifted = [values[test_index][indices] for indices in shift_maps]
            score, shifted_scores, selected = full.predict_selected(
                values[train_index],
                values[test_index],
                y_train,
                train_states,
                train_prior,
                test_prior,
                model_families,
                shifted,
            )
            append_rows(rows, frame, test_index, score, model, "aligned", fold)
            for offset, shifted_score in zip(
                SHIFT_OFFSETS, shifted_scores, strict=True
            ):
                append_rows(
                    rows,
                    frame,
                    test_index,
                    shifted_score,
                    model,
                    f"shift_{offset}",
                    fold,
                )
            for rank, index in enumerate(selected, start=1):
                selections.append(
                    {
                        "fold": fold,
                        "model": model,
                        "rank": rank,
                        "family": model_families[index],
                        "feature": model_names[index],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(selections)


def task_metrics(
    predictions: pd.DataFrame, draws: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = full.state_auc_rows(predictions)
    metrics = full.auc_metrics(predictions, state, draws, seed)
    return metrics, state


def fit_source_model(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    names: list[str],
    families: list[str],
) -> dict[str, Any]:
    y = frame["failure"].to_numpy(dtype=bool)
    states = frame["init_state_id"].to_numpy(dtype=int)
    selected = full.select_by_family(matrix, y, states, families)
    columns = np.asarray(selected, dtype=np.int64)
    scaler = StandardScaler().fit(matrix[:, columns])
    train = scaler.transform(matrix[:, columns])
    train_prior, _ = initial.prior_scores(y, states, states, True)
    return {
        "columns": columns,
        "features": [names[index] for index in columns],
        "scaler": scaler,
        "train": train,
        "labels": y,
        "prior": train_prior,
        "failure_rate": float(y.mean()),
    }


def frozen_transfer_predictions(
    source: dict[str, Any], target_frame: pd.DataFrame, target_matrix: np.ndarray
) -> pd.DataFrame:
    columns = source["columns"]
    target = source["scaler"].transform(target_matrix[:, columns])
    shift_maps = [full.shift_indices(target_frame, offset) for offset in SHIFT_OFFSETS]
    blocks = [target] + [target[indices] for indices in shift_maps]
    target_prior = np.full(
        sum(len(block) for block in blocks), source["failure_rate"], dtype=float
    )
    score = previous.robust_offset_scores(
        source["train"],
        source["labels"],
        np.vstack(blocks),
        source["prior"],
        target_prior,
        0.1,
    )
    split = np.split(score, np.cumsum([len(block) for block in blocks])[:-1])
    rows: list[dict[str, Any]] = []
    for variant, values in zip(
        ["aligned"] + [f"shift_{offset}" for offset in SHIFT_OFFSETS],
        split,
        strict=True,
    ):
        append_rows(
            rows,
            target_frame,
            np.arange(len(target_frame), dtype=np.int64),
            values,
            "frozen_source_model",
            variant,
            0,
        )
    return pd.DataFrame(rows)


def fixed_score_state_aucs(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, state), group in predictions.groupby(
        ["variant", "init_state_id"], sort=True
    ):
        labels = group["failure"].to_numpy(dtype=bool)
        if len(np.unique(labels)) != 2:
            continue
        rows.append(
            {
                "variant": variant,
                "init_state_id": int(state),
                "failures": int(labels.sum()),
                "successes": int((~labels).sum()),
                "pairs": int(labels.sum() * (~labels).sum()),
                "auc": float(
                    roc_auc_score(labels, group["score_failure"].to_numpy())
                ),
            }
        )
    return pd.DataFrame(rows)


def fixed_metrics(
    predictions: pd.DataFrame, draws: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = fixed_score_state_aucs(predictions)
    rows = []
    for axis, (variant, group) in enumerate(state.groupby("variant", sort=False)):
        mean, low, high = full.bootstrap_mean(
            group["auc"].to_numpy(), draws, seed + axis
        )
        rows.append(
            {
                "variant": variant,
                "mixed_states": len(group),
                "macro_within_state_auc": mean,
                "auc_ci95_low": low,
                "auc_ci95_high": high,
                "pair_weighted_within_state_auc": float(
                    np.average(group["auc"], weights=group["pairs"])
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    shifted = metrics[metrics["variant"].str.startswith("shift_")]
    if not shifted.empty:
        rows.append(
            {
                "variant": "shift_mean",
                "mixed_states": int(shifted["mixed_states"].min()),
                "macro_within_state_auc": float(
                    shifted["macro_within_state_auc"].mean()
                ),
                "auc_ci95_low": np.nan,
                "auc_ci95_high": np.nan,
                "pair_weighted_within_state_auc": float(
                    shifted["pair_weighted_within_state_auc"].mean()
                ),
            }
        )
        metrics = pd.DataFrame(rows)
    return metrics, state


def permutation_p_fixed(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    observed: float,
    draws: int,
    seed: int,
) -> float:
    aligned = predictions[predictions["variant"].eq("aligned")].sort_values(
        "episode"
    )
    ordered = frame.sort_values("episode")
    if not np.array_equal(
        aligned["episode"].to_numpy(), ordered["episode"].to_numpy()
    ):
        raise ValueError("prediction/frame episode mismatch")
    labels = ordered["failure"].to_numpy(dtype=bool)
    scores = aligned["score_failure"].to_numpy(dtype=float)
    states = ordered["init_state_id"].to_numpy(dtype=int)
    mixed = [
        state
        for state in np.unique(states)
        if len(np.unique(labels[states == state])) == 2
    ]
    rng = np.random.default_rng(seed)
    distribution = np.zeros(draws, dtype=np.float64)
    for state in mixed:
        selected = states == state
        state_labels = labels[selected]
        state_scores = scores[selected]
        positives = int(state_labels.sum())
        negatives = len(state_labels) - positives
        ranks = rankdata(state_scores, method="average")
        random = rng.random((draws, len(ranks)))
        chosen = np.argpartition(random, positives - 1, axis=1)[:, :positives]
        rank_sum = ranks[chosen].sum(axis=1)
        distribution += (
            rank_sum - positives * (positives + 1) / 2.0
        ) / (positives * negatives)
    distribution /= len(mixed)
    greater = int(np.sum(distribution >= observed - 1e-12))
    return float((greater + 1) / (draws + 1))


def holm_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        running = max(running, min(1.0, float(value) * (count - rank)))
        result.loc[index] = running
    return result


def short_task(value: str) -> str:
    suite, name = value.split("/", 1)
    aliases = {
        "open_the_middle_drawer_of_the_cabinet": "middle drawer",
        "open_the_top_drawer_and_put_the_bowl_inside": "drawer + bowl",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": "two moka pots",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": "ramekin -> plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": "stove -> plate",
    }
    return f"{suite.removeprefix('libero_')}/{aliases.get(name, name)}"


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("NA")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.3f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_results(nested: pd.DataFrame, transfer: pd.DataFrame, out: Path) -> None:
    nested_plot = nested[
        nested["model"].eq("token_current") & nested["variant"].eq("aligned")
    ].copy()
    long_transfer = transfer[
        transfer["source"].eq(LONG_TASK) & transfer["variant"].eq("aligned")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharex=True)
    for ax, frame, title in (
        (axes[0], nested_plot, "Task-specific nested OOF"),
        (axes[1], long_transfer, "Frozen Long model transfer"),
    ):
        if frame.empty:
            continue
        y = np.arange(len(frame))
        values = frame["macro_within_state_auc"].to_numpy()
        low = frame["auc_ci95_low"].to_numpy()
        high = frame["auc_ci95_high"].to_numpy()
        ax.errorbar(
            values,
            y,
            xerr=np.vstack((values - low, high - values)),
            fmt="o",
            color="#2f6b8a",
            ecolor="#7f8c8d",
            capsize=3,
        )
        ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([short_task(task) for task in frame["task"]])
        ax.set_title(title)
        ax.set_xlabel("Macro within-state ROC AUC")
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(
    out: Path,
    support: pd.DataFrame,
    nested: pd.DataFrame,
    transfer: pd.DataFrame,
    source_models: dict[str, dict[str, Any]],
    reference: dict[str, float],
) -> None:
    external_nested = nested[
        nested["model"].eq("token_current")
        & nested["variant"].eq("aligned")
        & nested["task"].ne(LONG_TASK)
    ].sort_values("task")
    nested_description = "；".join(
        f"{short_task(row.task)} {row.macro_within_state_auc:.3f} "
        f"[{row.auc_ci95_low:.3f}, {row.auc_ci95_high:.3f}]"
        for row in external_nested.itertuples()
    )
    nested_positive = int((external_nested["auc_ci95_low"] > 0.5).sum())
    frozen_long_aligned = transfer[
        transfer["source"].eq(LONG_TASK) & transfer["variant"].eq("aligned")
    ].sort_values("task")
    frozen_description = "；".join(
        f"{short_task(row.task)} {row.macro_within_state_auc:.3f}"
        for row in frozen_long_aligned.itertuples()
    )
    frozen_positive = int(
        (frozen_long_aligned["holm_p_across_long_targets"] < 0.05).sum()
    )
    support_table = support[
        ["task", "failures", "successes", "mixed_states", "checkpoint_sha12"]
    ].copy()
    support_table["task"] = support_table["task"].map(short_task)
    support_table.columns = ["task", "fail", "success", "mixed states", "checkpoint"]

    primary = nested[
        nested["model"].eq("token_current")
        & nested["variant"].isin(("aligned", "shift_mean"))
    ].copy()
    primary = primary.pivot(
        index="task", columns="variant", values="macro_within_state_auc"
    ).reset_index()
    aligned_ci = nested[
        nested["model"].eq("token_current") & nested["variant"].eq("aligned")
    ][["task", "mixed_states", "auc_ci95_low", "auc_ci95_high"]]
    primary = primary.merge(aligned_ci, on="task", validate="one_to_one")
    primary["task"] = primary["task"].map(short_task)
    primary = primary.rename(
        columns={
            "aligned": "q0 AUC",
            "shift_mean": "shifted AUC",
            "mixed_states": "mixed states",
            "auc_ci95_low": "CI low",
            "auc_ci95_high": "CI high",
        }
    )

    long_transfer = transfer[
        transfer["source"].eq(LONG_TASK)
        & transfer["variant"].isin(("aligned", "shift_mean"))
    ].copy()
    long_transfer = long_transfer.pivot(
        index="task", columns="variant", values="macro_within_state_auc"
    ).reset_index()
    detail = transfer[
        transfer["source"].eq(LONG_TASK) & transfer["variant"].eq("aligned")
    ][
        [
            "task",
            "mixed_states",
            "auc_ci95_low",
            "auc_ci95_high",
            "permutation_p_one_sided",
            "holm_p_across_long_targets",
        ]
    ]
    long_transfer = long_transfer.merge(detail, on="task", validate="one_to_one")
    long_transfer["task"] = long_transfer["task"].map(short_task)
    long_transfer = long_transfer.rename(
        columns={
            "aligned": "frozen AUC",
            "shift_mean": "shifted AUC",
            "mixed_states": "mixed states",
            "auc_ci95_low": "CI low",
            "auc_ci95_high": "CI high",
            "permutation_p_one_sided": "perm p",
            "holm_p_across_long_targets": "Holm p",
        }
    )

    spatial = transfer[
        transfer["variant"].eq("aligned")
        & transfer["source"].str.startswith("libero_spatial/")
        & transfer["task"].str.startswith("libero_spatial/")
        & transfer["source"].ne(transfer["task"])
    ][
        [
            "source",
            "task",
            "macro_within_state_auc",
            "auc_ci95_low",
            "auc_ci95_high",
            "permutation_p_one_sided",
        ]
    ].copy()
    spatial["source"] = spatial["source"].map(short_task)
    spatial["task"] = spatial["task"].map(short_task)
    spatial.columns = ["source", "target", "AUC", "CI low", "CI high", "perm p"]

    long_features = source_models[LONG_TASK]["features"]
    lines = [
        "# q0 同初态成败信息的跨任务复现",
        "",
        "## 先说结论",
        "",
        "**不支持把 Long 的 q0 AUC=0.602 当成跨任务普遍成立的早期成败信号。**",
        "",
        f"三个可估计外部任务各自重新训练后的 AUC 为：{nested_description}。"
        f"其中 {nested_positive}/3 的 95% 区间下界高于 0.5，方向也不一致。",
        "",
        f"更严格地冻结 Long 判别器后，三个目标任务 AUC 为：{frozen_description}；"
        f"经三任务 Holm 校正后 {frozen_positive}/3 显著。共享 checkpoint 的两个 Spatial 任务互迁移也接近 0.5。",
        "",
        "## 样本",
        "",
        markdown_table(support_table),
        "",
        "`middle drawer` 为 512/512 成功，没有正负对照，因此不能计算成败 AUC。",
        "其余任务均为同一任务内 16 初态 x 32 noise seeds；AUC 只比较同一初态的成功和失败 sibling。",
        "",
        "## 每个任务内部重新训练：嵌套 held-out-seed",
        "",
        markdown_table(primary),
        "",
        "每折完整留出 4 个 noise seeds；特征坐标只在其余 28 seeds 中选择。",
        "`shifted AUC` 保留标签和初态，但换成另一个测试 sibling 的 q0 路由。",
        "q0 没有跨 chunk 历史，所以这里的 `token_current` 与此前的 `token_history` 数值完全相同。",
        "",
        "## Long 判别器完全冻结后迁移",
        "",
        markdown_table(long_transfer),
        "",
        "这里连特征坐标、标准化和回归权重都只在 Long 上拟合；目标任务标签只用于最后算 AUC。",
        "置换检验在目标任务每个初态内部打乱成败，Holm p 同时校正三个可估计目标任务。",
        "",
        "Long 全量训练选择的 5 个 q0 坐标：",
        "",
    ]
    lines.extend([f"- `{feature}`" for feature in long_features])
    lines.extend(
        [
            "",
            "## 同 checkpoint 的 Spatial 直接迁移",
            "",
            markdown_table(spatial),
            "",
            "这项对照排除了 Long 与 Spatial 使用不同 checkpoint 这一混杂，但样本中的失败仍然较少。",
            "",
            "## 复算校验",
            "",
            f"Long 旧报告 q0 AUC={reference['expected']:.6f}；本脚本从原始 zarr 独立抽 q0 后得到 {reference['observed']:.6f}；绝对差 {reference['absolute_difference']:.3g}。",
            "",
            "## 解释边界",
            "",
            "这是 q0 MoE 路由与最终成败的观察性可分性，不是因果证据。独立任务内重训若有效，只说明目标任务也存在某种 q0 信息；只有完全冻结迁移也有效，才支持 Long 中学到的是可复用的同一种失败结构。",
            "不同 suite 使用不同 checkpoint，因此跨 suite 冻结迁移是严格但带域偏移的测试；同 checkpoint 的 Spatial 互迁移用于补这个对照。",
        ]
    )
    (out / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "_q0_cache"
    cache_dir.mkdir(exist_ok=True)
    names, families, aggregate_dim = q0_feature_schema()
    frames: dict[str, pd.DataFrame] = {}
    matrices: dict[str, np.ndarray] = {}
    support_rows = []
    nested_predictions = []
    nested_state = []
    nested_metrics = []
    selection_rows = []

    for run_index, run in enumerate(discover_runs(args.root)):
        task = task_id(run)
        frame = read_frame(run)
        cache = cache_dir / f"{task.replace('/', '__')}.npz"
        matrix = load_q0_matrix(run, frame, cache, args.rebuild_cache)
        if matrix.shape != (512, len(names)):
            raise ValueError(f"{task}: unexpected q0 matrix {matrix.shape}")
        frames[task] = frame
        matrices[task] = matrix
        metadata = json.loads(
            (run / "client/server_metadata.json").read_text(encoding="utf-8")
        )
        state_outcomes = frame.groupby("init_state_id")["failure"].nunique()
        support_rows.append(
            {
                "task": task,
                "failures": int(frame["failure"].sum()),
                "successes": int(frame["success"].sum()),
                "mixed_states": int(state_outcomes.eq(2).sum()),
                "checkpoint_sha12": str(metadata["checkpoint_sha256"])[:12],
            }
        )
        if frame["failure"].nunique() != 2:
            print(f"{task}: no success/failure contrast", flush=True)
            continue
        prediction, selections = nested_oof(
            frame, matrix, names, families, aggregate_dim
        )
        metrics, state = task_metrics(
            prediction, args.bootstraps, args.seed + 1000 * run_index
        )
        for table in (prediction, selections, metrics, state):
            table.insert(0, "task", task)
        nested_predictions.append(prediction)
        selection_rows.append(selections)
        nested_metrics.append(metrics)
        nested_state.append(state)
        value = metrics[
            metrics["model"].eq("token_current")
            & metrics["variant"].eq("aligned")
        ]["macro_within_state_auc"].iloc[0]
        print(f"{task}: nested q0 AUC {value:.6f}", flush=True)

    support = pd.DataFrame(support_rows)
    nested_prediction_frame = pd.concat(nested_predictions, ignore_index=True)
    nested_selection_frame = pd.concat(selection_rows, ignore_index=True)
    nested_metric_frame = pd.concat(nested_metrics, ignore_index=True)
    nested_state_frame = pd.concat(nested_state, ignore_index=True)

    source_models = {
        task: fit_source_model(frames[task], matrices[task], names, families)
        for task in frames
        if frames[task]["failure"].nunique() == 2
    }
    transfer_predictions = []
    transfer_states = []
    transfer_metrics = []
    axis = 0
    for source_task, source_model in source_models.items():
        for target_task, target_frame in frames.items():
            if source_task == target_task or target_frame["failure"].nunique() != 2:
                continue
            same_spatial_checkpoint = source_task.startswith(
                "libero_spatial/"
            ) and target_task.startswith("libero_spatial/")
            if source_task != LONG_TASK and not same_spatial_checkpoint:
                continue
            prediction = frozen_transfer_predictions(
                source_model, target_frame, matrices[target_task]
            )
            metrics, states = fixed_metrics(
                prediction, args.bootstraps, args.seed + 100000 + axis
            )
            aligned = metrics[metrics["variant"].eq("aligned")].iloc[0]
            p_value = permutation_p_fixed(
                target_frame,
                prediction,
                float(aligned["macro_within_state_auc"]),
                args.permutations,
                args.seed + 200000 + axis,
            )
            for table in (prediction, states, metrics):
                table.insert(0, "task", target_task)
                table.insert(0, "source", source_task)
            metrics["permutation_p_one_sided"] = np.where(
                metrics["variant"].eq("aligned"), p_value, np.nan
            )
            transfer_predictions.append(prediction)
            transfer_states.append(states)
            transfer_metrics.append(metrics)
            axis += 1
            print(
                f"{source_task} -> {target_task}: frozen q0 AUC "
                f"{aligned['macro_within_state_auc']:.6f}, p={p_value:.4g}",
                flush=True,
            )

    transfer_prediction_frame = pd.concat(transfer_predictions, ignore_index=True)
    transfer_state_frame = pd.concat(transfer_states, ignore_index=True)
    transfer_metric_frame = pd.concat(transfer_metrics, ignore_index=True)
    transfer_metric_frame["holm_p_across_long_targets"] = np.nan
    long_mask = (
        transfer_metric_frame["source"].eq(LONG_TASK)
        & transfer_metric_frame["variant"].eq("aligned")
    )
    transfer_metric_frame.loc[long_mask, "holm_p_across_long_targets"] = holm_adjust(
        transfer_metric_frame.loc[long_mask, "permutation_p_one_sided"]
    )

    expected = pd.read_csv(
        HERE / "analysis/full-chunk-outcome-auc-20260829/auc_metrics.csv"
    )
    expected_auc = float(
        expected[
            expected["split"].eq("heldout_seed")
            & expected["query"].eq(0)
            & expected["model"].eq("token_current")
            & expected["variant"].eq("aligned")
        ]["macro_within_state_auc"].iloc[0]
    )
    observed_auc = float(
        nested_metric_frame[
            nested_metric_frame["task"].eq(LONG_TASK)
            & nested_metric_frame["model"].eq("token_current")
            & nested_metric_frame["variant"].eq("aligned")
        ]["macro_within_state_auc"].iloc[0]
    )
    difference = abs(expected_auc - observed_auc)
    if difference > 1e-10:
        raise RuntimeError(
            f"Long q0 reproduction failed: {observed_auc} != {expected_auc}"
        )
    reference = {
        "expected": expected_auc,
        "observed": observed_auc,
        "absolute_difference": difference,
    }

    support.to_csv(args.out / "task_support.csv", index=False)
    nested_metric_frame.to_csv(args.out / "nested_oof_metrics.csv", index=False)
    nested_state_frame.to_csv(args.out / "nested_oof_state_aucs.csv", index=False)
    nested_prediction_frame.to_csv(
        args.out / "nested_oof_predictions.csv.gz", index=False
    )
    nested_selection_frame.to_csv(args.out / "nested_feature_selections.csv", index=False)
    transfer_metric_frame.to_csv(args.out / "frozen_transfer_metrics.csv", index=False)
    transfer_state_frame.to_csv(args.out / "frozen_transfer_state_aucs.csv", index=False)
    transfer_prediction_frame.to_csv(
        args.out / "frozen_transfer_predictions.csv.gz", index=False
    )
    model_manifest = {
        task: {"features": model["features"], "failure_rate": model["failure_rate"]}
        for task, model in source_models.items()
    }
    (args.out / "frozen_source_models.json").write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "audit.json").write_text(
        json.dumps(
            {
                "estimand": "macro ROC AUC within initial state",
                "query": 0,
                "target": "eventual rollout failure",
                "features": "q0 MoE routing only",
                "folds": "8 folds, four complete noise seeds held out per fold",
                "bootstraps": args.bootstraps,
                "fixed_transfer_permutations": args.permutations,
                "long_reference": reference,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_results(nested_metric_frame, transfer_metric_frame, args.out / "q0_cross_task_auc.png")
    write_report(
        args.out,
        support,
        nested_metric_frame,
        transfer_metric_frame,
        source_models,
        reference,
    )
    outputs = sorted(
        path for path in args.out.iterdir() if path.is_file() and path.name != "checksums.sha256"
    )
    (args.out / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in outputs),
        encoding="utf-8",
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
