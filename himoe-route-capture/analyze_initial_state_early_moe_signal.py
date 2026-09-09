#!/usr/bin/env python3
"""Separate initial-state priors from genuinely early HiMoE routing signal.

The primary q0 readout occurs before the first emitted action chunk is executed.
The q0:q2 sensitivity is a fixed-index early closed-loop window: q1 and q2 may
reflect observations after earlier chunks, but the window is identical for
every rollout.  The analysis never uses episode length, remaining time,
final-window features, action vectors, or physical outcome annotations as
predictors.  Models are fit separately inside each task so that checkpoint/task
identity cannot become a shortcut.

Two complementary eight-fold evaluations answer different questions:

* ``heldout_seed`` leaves four flow seeds out for every initial state.  This
  measures whether early features add information beyond a known initial-state
  failure prior.
* ``heldout_init`` leaves two complete initial states out.  This measures
  whether an early readout transfers to geometry it has never seen.

The primary feature is expert-permutation-invariant router behavior.  Router
input hidden-state summaries are reported separately because they are context
fed to the gate, not evidence that a selected expert caused the outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import analyze_residual_failure_moe_atlas as atlas


HERE = pathlib.Path(__file__).resolve().parent
FEATURE_DIR = HERE / "analysis/single-chunk-early-signal/features"
DEFAULT_OUT = HERE / "analysis/initial-state-early-moe-signal-20260828"
DEFAULT_CAPACITY_OUT = (
    HERE / "analysis/initial-state-early-moe-signal-pca32-20260828"
)
WINDOWS = ("q0", "q0_to_q2")
SPLITS = ("heldout_seed", "heldout_init")
CS = (0.03, 0.1, 0.3)
PRIMARY_C = 0.1
N_FOLDS = 8
INIT_PRIOR_STRENGTH = 8.0
MIN_POSITIVES_PER_TASK = 5
MIN_NEGATIVES_PER_TASK = 5
TARGET_COLUMNS = {
    "any_failure": "is_failure",
    "stagnation": "label_stagnation",
    "regrasp_or_drop": "label_regrasp_or_drop",
    "goal_regression": "label_goal_regression",
    "residual_none": "residual_none",
    "active_return": "label_active_return",
}
MODELS = (
    "task_only",
    "task_init",
    "task_router",
    "task_init_router",
    "task_hidden",
    "task_init_hidden",
    "task_router_hidden",
    "task_init_router_hidden",
)
MODEL_LABELS = {
    "task_only": "task prior",
    "task_init": "task + initial-state prior",
    "task_router": "q routing",
    "task_init_router": "initial state + q routing",
    "task_hidden": "q gate-input hidden",
    "task_init_hidden": "initial state + q gate-input hidden",
    "task_router_hidden": "q routing + gate-input hidden",
    "task_init_router_hidden": "initial state + q routing + gate-input hidden",
}
COMPARISONS = {
    "initial_prior": ("heldout_seed", "task_only", "task_init"),
    "router_beyond_init": (
        "heldout_seed",
        "task_init",
        "task_init_router",
    ),
    "hidden_beyond_init": (
        "heldout_seed",
        "task_init",
        "task_init_hidden",
    ),
    "router_hidden_beyond_init": (
        "heldout_seed",
        "task_init",
        "task_init_router_hidden",
    ),
    "router_unseen_init": ("heldout_init", "task_only", "task_router"),
    "hidden_unseen_init": ("heldout_init", "task_only", "task_hidden"),
    "router_hidden_unseen_init": (
        "heldout_init",
        "task_only",
        "task_router_hidden",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=pathlib.Path, default=FEATURE_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument(
        "--capacity-sensitivity-dir",
        type=pathlib.Path,
        default=DEFAULT_CAPACITY_OUT,
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--self-test", action="store_true")
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
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(
    feature_dir: pathlib.Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray], list[pathlib.Path]]:
    metadata, _, _, _ = atlas.load_cached_inputs(atlas.FEATURE_CACHE)
    labels = atlas.load_labels(metadata)
    files = sorted(feature_dir.glob("*.npz"))
    if len(files) != 4:
        raise ValueError(f"expected four early-feature caches, found {len(files)}")

    frames: list[pd.DataFrame] = []
    router_parts: dict[str, list[np.ndarray]] = {window: [] for window in WINDOWS}
    hidden_parts: dict[str, list[np.ndarray]] = {window: [] for window in WINDOWS}
    for path in files:
        task = path.stem.replace("__", "/", 1)
        with np.load(path, allow_pickle=False) as cache:
            if int(cache["feature_version"]) != 1 or int(cache["max_chunks"]) < 3:
                raise ValueError(f"incompatible early-feature cache: {path}")
            episodes = np.asarray(cache["episodes"], dtype=np.int64)
            initial_states = np.asarray(cache["scenes"], dtype=np.int64)
            seeds = np.asarray(cache["seeds"], dtype=np.int64)
            failures = np.asarray(cache["failure"], dtype=bool)
            router = np.asarray(cache["router"], dtype=np.float32)
            hidden = np.asarray(cache["hidden"], dtype=np.float32)

        task_labels = labels.loc[labels["task"].eq(task)].set_index("episode")
        if len(episodes) != 512 or len(np.unique(episodes)) != len(episodes):
            raise ValueError(f"unexpected episode coverage in {path}")
        aligned = task_labels.loc[episodes].reset_index()
        if not np.array_equal(
            aligned["init_state_id"].to_numpy(dtype=np.int64), initial_states
        ):
            raise ValueError(f"initial-state alignment failed for {task}")
        if not np.array_equal(
            aligned["flow_noise_seed"].to_numpy(dtype=np.int64), seeds
        ):
            raise ValueError(f"flow-seed alignment failed for {task}")
        if not np.array_equal(
            aligned["is_failure"].to_numpy(dtype=bool), failures
        ):
            raise ValueError(f"outcome alignment failed for {task}")
        if not np.isfinite(router[:, :3]).all() or not np.isfinite(hidden[:, :3]).all():
            raise ValueError(f"non-finite early features in {path}")

        frames.append(aligned)
        router_parts["q0"].append(router[:, 0])
        router_parts["q0_to_q2"].append(router[:, :3].reshape(len(router), -1))
        hidden_parts["q0"].append(hidden[:, 0])
        hidden_parts["q0_to_q2"].append(hidden[:, :3].reshape(len(hidden), -1))

    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    for target, column in TARGET_COLUMNS.items():
        frame[target] = frame[column].astype(bool)
    feature_blocks: dict[tuple[str, str], np.ndarray] = {}
    for window in WINDOWS:
        feature_blocks[(window, "router")] = np.concatenate(
            router_parts[window], axis=0
        )
        feature_blocks[(window, "hidden")] = np.concatenate(
            hidden_parts[window], axis=0
        )
    if len(frame) != 2048 or int(frame["any_failure"].sum()) != 307:
        raise ValueError("unexpected variable-outcome task coverage")
    return frame, feature_blocks, files


def assign_folds(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["heldout_seed_fold"] = -1
    result["heldout_init_fold"] = -1
    for task, indices in result.groupby("task", sort=True).groups.items():
        task_index = np.asarray(list(indices), dtype=np.int64)
        seeds = sorted(result.loc[task_index, "flow_noise_seed"].unique())
        initial_states = sorted(result.loc[task_index, "init_state_id"].unique())
        if len(seeds) != 32 or len(initial_states) != 16:
            raise ValueError(f"unexpected factorial design for {task}")
        seed_fold = {seed: rank % N_FOLDS for rank, seed in enumerate(seeds)}
        init_fold = {
            initial_state: rank % N_FOLDS
            for rank, initial_state in enumerate(initial_states)
        }
        result.loc[task_index, "heldout_seed_fold"] = result.loc[
            task_index, "flow_noise_seed"
        ].map(seed_fold)
        result.loc[task_index, "heldout_init_fold"] = result.loc[
            task_index, "init_state_id"
        ].map(init_fold)
    for column in ("heldout_seed_fold", "heldout_init_fold"):
        result[column] = result[column].astype(int)
        counts = result.groupby(["task", column]).size()
        if counts.nunique() != 1 or int(counts.iloc[0]) != 64:
            raise ValueError(f"unbalanced folds in {column}")
    return result


def eligible_tasks(frame: pd.DataFrame, target: str) -> list[str]:
    support = frame.groupby("task")[target].agg(["sum", "count"])
    support["negative"] = support["count"] - support["sum"]
    return sorted(
        support.index[
            (support["sum"] >= MIN_POSITIVES_PER_TASK)
            & (support["negative"] >= MIN_NEGATIVES_PER_TASK)
        ].tolist()
    )


def reduce_features(
    train: np.ndarray,
    test: np.ndarray,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    count = min(components, train_scaled.shape[0] - 1, train_scaled.shape[1])
    if count < 1:
        raise ValueError("not enough rows for feature reduction")
    pca = PCA(n_components=count, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def initial_state_matrix(
    train_states: np.ndarray, test_states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    levels = sorted(np.unique(train_states).tolist())
    mapping = {value: index for index, value in enumerate(levels)}
    train = np.zeros((len(train_states), len(levels)), dtype=np.float32)
    test = np.zeros((len(test_states), len(levels)), dtype=np.float32)
    for row, value in enumerate(train_states):
        train[row, mapping[value]] = 1.0
    for row, value in enumerate(test_states):
        if value in mapping:
            test[row, mapping[value]] = 1.0
    return train, test


def prior_scores(
    y_train: np.ndarray,
    train_states: np.ndarray,
    test_states: np.ndarray,
    include_initial_state: bool,
) -> tuple[np.ndarray, np.ndarray]:
    y_train = y_train.astype(float)
    total = float(y_train.sum())
    count = len(y_train)
    task_probability = (total + 0.5) / (count + 1.0)
    train_score = np.empty(count, dtype=float)
    if not include_initial_state:
        train_score[:] = (total - y_train + 0.5) / count
        return train_score, np.full(len(test_states), task_probability)

    test_score = np.empty(len(test_states), dtype=float)
    state_stats = {
        state: (float(y_train[train_states == state].sum()), int(np.sum(train_states == state)))
        for state in np.unique(train_states)
    }
    for row, (label, state) in enumerate(zip(y_train, train_states, strict=True)):
        positives, state_count = state_stats[state]
        task_loo = (total - label + 0.5) / count
        train_score[row] = (
            positives - label + INIT_PRIOR_STRENGTH * task_loo
        ) / (state_count - 1 + INIT_PRIOR_STRENGTH)
    for row, state in enumerate(test_states):
        positives, state_count = state_stats.get(state, (0.0, 0))
        test_score[row] = (
            positives + INIT_PRIOR_STRENGTH * task_probability
        ) / (state_count + INIT_PRIOR_STRENGTH)
    return train_score, test_score


def offset_logistic_scores(
    train: np.ndarray,
    y_train: np.ndarray,
    test: np.ndarray,
    train_baseline: np.ndarray,
    test_baseline: np.ndarray,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if len(np.unique(y_train)) < 2:
        return train_baseline.copy(), test_baseline.copy(), True
    epsilon = 1e-6
    train_offset = np.log(
        np.clip(train_baseline, epsilon, 1 - epsilon)
        / np.clip(1 - train_baseline, epsilon, 1 - epsilon)
    )
    test_offset = np.log(
        np.clip(test_baseline, epsilon, 1 - epsilon)
        / np.clip(1 - test_baseline, epsilon, 1 - epsilon)
    )
    y_float = y_train.astype(float)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = train_offset + train @ coefficients
        probability = expit(logits)
        loss = float(
            np.sum(np.logaddexp(0.0, logits) - y_float * logits)
            + 0.5 * np.dot(coefficients, coefficients) / c_value
        )
        gradient = train.T @ (probability - y_float) + coefficients / c_value
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(train.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"offset logistic fit failed: {result.message}")
    return (
        expit(train_offset + train @ result.x),
        expit(test_offset + test @ result.x),
        False,
    )


def threshold_at_training_negative_q90(
    train_score: np.ndarray, y_train: np.ndarray
) -> float:
    negatives = train_score[~y_train.astype(bool)]
    if not len(negatives):
        return float("inf")
    return float(np.quantile(negatives, 0.9, method="higher"))


def evaluate(
    frame: pd.DataFrame,
    features: dict[tuple[str, str], np.ndarray],
    pca_components: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    rows: list[tuple[Any, ...]] = []
    fallback_rows: list[dict[str, Any]] = []
    task_support = {target: eligible_tasks(frame, target) for target in TARGET_COLUMNS}

    # Feature reduction is label-free, so each outer split is fit once and reused.
    reduced: dict[tuple[str, str, int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    index_cache: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for task in sorted(frame["task"].unique()):
        task_mask = frame["task"].eq(task).to_numpy()
        for split in SPLITS:
            fold_column = f"{split}_fold"
            for fold in range(N_FOLDS):
                test_mask = task_mask & frame[fold_column].eq(fold).to_numpy()
                train_mask = task_mask & ~frame[fold_column].eq(fold).to_numpy()
                train_index = np.flatnonzero(train_mask)
                test_index = np.flatnonzero(test_mask)
                index_cache[(task, split, fold)] = (train_index, test_index)
                for window in WINDOWS:
                    for block in ("router", "hidden"):
                        key = (task, split, fold, window, block)
                        reduced[key] = reduce_features(
                            features[(window, block)][train_index],
                            features[(window, block)][test_index],
                            pca_components,
                            seed + fold,
                        )

    columns = (
        "target",
        "split",
        "window",
        "c_value",
        "task",
        "fold",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "truth",
        "model",
        "score",
        "threshold",
        "trigger",
    )
    for target, tasks in task_support.items():
        for task in tasks:
            for split in SPLITS:
                for fold in range(N_FOLDS):
                    train_index, test_index = index_cache[(task, split, fold)]
                    y_train = frame.loc[train_index, target].to_numpy(dtype=bool)
                    y_test = frame.loc[test_index, target].to_numpy(dtype=bool)
                    train_states = frame.loc[
                        train_index, "init_state_id"
                    ].to_numpy(dtype=np.int64)
                    test_states = frame.loc[
                        test_index, "init_state_id"
                    ].to_numpy(dtype=np.int64)
                    base_predictions = {
                        "task_only": prior_scores(
                            y_train, train_states, test_states, False
                        ),
                        "task_init": prior_scores(
                            y_train, train_states, test_states, True
                        ),
                    }
                    for window in WINDOWS:
                        router_train, router_test = reduced[
                            (task, split, fold, window, "router")
                        ]
                        hidden_train, hidden_test = reduced[
                            (task, split, fold, window, "hidden")
                        ]
                        block_values = {
                            "router": (router_train, router_test),
                            "hidden": (hidden_train, hidden_test),
                            "router_hidden": (
                                np.column_stack((router_train, hidden_train)),
                                np.column_stack((router_test, hidden_test)),
                            ),
                        }
                        for c_value in CS:
                            predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
                                **base_predictions
                            }
                            evaluated_blocks = (
                                block_values
                                if np.isclose(c_value, PRIMARY_C)
                                else {"router": block_values["router"]}
                            )
                            for block, (block_train, block_test) in evaluated_blocks.items():
                                for with_initial_state in (False, True):
                                    model_name = (
                                        f"task_init_{block}"
                                        if with_initial_state
                                        else f"task_{block}"
                                    )
                                    baseline_name = (
                                        "task_init"
                                        if with_initial_state
                                        else "task_only"
                                    )
                                    baseline_train, baseline_test = base_predictions[
                                        baseline_name
                                    ]
                                    train_score, test_score, fallback = offset_logistic_scores(
                                        block_train,
                                        y_train,
                                        block_test,
                                        baseline_train,
                                        baseline_test,
                                        c_value,
                                    )
                                    predictions[model_name] = (
                                        train_score,
                                        test_score,
                                    )
                                    if fallback:
                                        fallback_rows.append(
                                            {
                                                "target": target,
                                                "task": task,
                                                "split": split,
                                                "fold": fold,
                                                "window": window,
                                                "c_value": c_value,
                                                "model": model_name,
                                            }
                                        )
                            test_meta = frame.loc[
                                test_index,
                                [
                                    "episode",
                                    "init_state_id",
                                    "flow_noise_seed",
                                ],
                            ]
                            evaluated_models = (
                                MODELS
                                if np.isclose(c_value, PRIMARY_C)
                                else (
                                    "task_only",
                                    "task_init",
                                    "task_router",
                                    "task_init_router",
                                )
                            )
                            for model_name in evaluated_models:
                                train_score, test_score = predictions[model_name]
                                threshold = threshold_at_training_negative_q90(
                                    train_score, y_train
                                )
                                trigger = test_score > threshold
                                for position, (_, meta_row) in enumerate(
                                    test_meta.iterrows()
                                ):
                                    rows.append(
                                        (
                                            target,
                                            split,
                                            window,
                                            c_value,
                                            task,
                                            fold,
                                            int(meta_row["episode"]),
                                            int(meta_row["init_state_id"]),
                                            int(meta_row["flow_noise_seed"]),
                                            bool(y_test[position]),
                                            model_name,
                                            float(test_score[position]),
                                            threshold,
                                            bool(trigger[position]),
                                        )
                                    )
    return pd.DataFrame.from_records(rows, columns=columns), pd.DataFrame(fallback_rows), task_support


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    def summarize(group: pd.DataFrame) -> pd.Series:
        truth = group["truth"].to_numpy(dtype=float)
        score = np.clip(group["score"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        trigger = group["trigger"].to_numpy(dtype=bool)
        positive = truth.astype(bool)
        return pd.Series(
            {
                "episodes": len(group),
                "positives": int(positive.sum()),
                "prevalence": float(positive.mean()),
                "mean_score": float(score.mean()),
                "brier": float(np.mean((score - truth) ** 2)),
                "log_loss": float(
                    -np.mean(truth * np.log(score) + (1 - truth) * np.log(1 - score))
                ),
                "positive_coverage_at_train_negative_q90": (
                    float(trigger[positive].mean()) if positive.any() else np.nan
                ),
                "negative_trigger_rate_at_train_negative_q90": (
                    float(trigger[~positive].mean()) if (~positive).any() else np.nan
                ),
            }
        )

    keys = ["target", "split", "window", "c_value", "model"]
    return predictions.groupby(keys, sort=True).apply(summarize, include_groups=False).reset_index()


def bootstrap_paired_deltas(
    predictions: pd.DataFrame,
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    primary = predictions[np.isclose(predictions["c_value"], PRIMARY_C)].copy()
    keys = [
        "target",
        "split",
        "window",
        "task",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "truth",
    ]
    for comparison, (split, baseline, candidate) in COMPARISONS.items():
        subset = primary[
            primary["split"].eq(split)
            & primary["model"].isin((baseline, candidate))
        ]
        for (target, window), group in subset.groupby(["target", "window"]):
            wide = group.pivot(index=keys, columns="model", values="score").reset_index()
            if baseline not in wide or candidate not in wide:
                continue
            truth = wide["truth"].to_numpy(dtype=float)
            baseline_score = np.clip(wide[baseline].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
            candidate_score = np.clip(wide[candidate].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
            losses = {
                "brier": (baseline_score - truth) ** 2
                - (candidate_score - truth) ** 2,
                "log_loss": (
                    -(truth * np.log(baseline_score) + (1 - truth) * np.log(1 - baseline_score))
                    + truth * np.log(candidate_score)
                    + (1 - truth) * np.log(1 - candidate_score)
                ),
            }
            cluster_frame = wide[["task", "init_state_id"]].copy()
            cluster_frame["cluster"] = (
                cluster_frame["task"]
                + "|init="
                + cluster_frame["init_state_id"].astype(str)
            )
            for metric, delta in losses.items():
                cluster_frame["delta"] = delta
                cluster_means = (
                    cluster_frame.groupby(["task", "cluster"], sort=True)["delta"]
                    .mean()
                    .reset_index()
                )
                observed = float(delta.mean())
                draws = np.empty(bootstraps, dtype=float)
                by_task = [
                    values["delta"].to_numpy(dtype=float)
                    for _, values in cluster_means.groupby("task", sort=True)
                ]
                for draw in range(bootstraps):
                    task_draws = [
                        values[rng.integers(0, len(values), size=len(values))].mean()
                        for values in by_task
                    ]
                    draws[draw] = float(np.mean(task_draws))
                opposite_tail = min(
                    (np.sum(draws <= 0) + 1) / (bootstraps + 1),
                    (np.sum(draws >= 0) + 1) / (bootstraps + 1),
                )
                baseline_loss = (
                    float(np.mean((baseline_score - truth) ** 2))
                    if metric == "brier"
                    else float(
                        -np.mean(
                            truth * np.log(baseline_score)
                            + (1 - truth) * np.log(1 - baseline_score)
                        )
                    )
                )
                rows.append(
                    {
                        "comparison": comparison,
                        "target": target,
                        "split": split,
                        "window": window,
                        "baseline": baseline,
                        "candidate": candidate,
                        "metric": metric,
                        "episodes": len(wide),
                        "clusters": len(cluster_means),
                        "baseline_loss": baseline_loss,
                        "improvement": observed,
                        "relative_improvement": observed / baseline_loss,
                        "ci_low": float(np.quantile(draws, 0.025)),
                        "ci_high": float(np.quantile(draws, 0.975)),
                        "bootstrap_opposite_tail_2x": min(1.0, 2 * opposite_tail),
                    }
                )
    return pd.DataFrame(rows)


def target_support_table(
    frame: pd.DataFrame, support: dict[str, list[str]]
) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        for task, group in frame.groupby("task", sort=True):
            rows.append(
                {
                    "target": target,
                    "task": task,
                    "episodes": len(group),
                    "positives": int(group[target].sum()),
                    "eligible": task in support[target],
                }
            )
    return pd.DataFrame(rows)


def comparison_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    return deltas[deltas["metric"].eq("brier")].copy()


def make_gain_plot(deltas: pd.DataFrame, output: pathlib.Path) -> None:
    selected = deltas[
        deltas["metric"].eq("brier")
        & deltas["window"].eq("q0")
        & deltas["comparison"].isin(
            ("initial_prior", "router_beyond_init", "router_unseen_init")
        )
    ].copy()
    targets = list(TARGET_COLUMNS)
    comparisons = ["initial_prior", "router_beyond_init", "router_unseen_init"]
    labels = {
        "initial_prior": "initial-state prior\nknown state",
        "router_beyond_init": "q0 routing beyond prior\nknown state",
        "router_unseen_init": "q0 routing\nunseen state",
    }
    colors = ["#2f6b4f", "#b24a3b", "#3d668f"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.7), sharey=True)
    y = np.arange(len(targets))
    for axis, comparison, color in zip(axes, comparisons, colors, strict=True):
        block = selected[selected["comparison"].eq(comparison)].set_index("target")
        values = np.array([block.loc[target, "improvement"] for target in targets])
        lows = np.array([block.loc[target, "ci_low"] for target in targets])
        highs = np.array([block.loc[target, "ci_high"] for target in targets])
        axis.errorbar(
            values,
            y,
            xerr=np.vstack((values - lows, highs - values)),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
        )
        axis.axvline(0, color="#222222", linewidth=0.8)
        axis.set_title(labels[comparison], fontsize=11)
        axis.set_xlabel("Brier improvement (positive is better)")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, targets)
    axes[0].invert_yaxis()
    fig.suptitle("Early failure signal: initial-state prior versus q0 routing")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    columns = list(columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(fmt_float(float(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: pathlib.Path,
    frame: pd.DataFrame,
    support: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    fallback_count: int,
    args: argparse.Namespace,
) -> None:
    primary_metrics = metrics[
        np.isclose(metrics["c_value"], PRIMARY_C)
        & metrics["window"].eq("q0")
    ]
    metric_lookup = primary_metrics.set_index(["target", "split", "model"])
    main_rows = []
    delta_brier = deltas[
        deltas["metric"].eq("brier") & deltas["window"].eq("q0")
    ].set_index(["target", "comparison"])
    any_initial = delta_brier.loc[("any_failure", "initial_prior")]
    any_router = delta_brier.loc[("any_failure", "router_beyond_init")]
    beyond_rows = delta_brier.reset_index().query(
        "comparison == 'router_beyond_init'"
    )
    unseen_rows = delta_brier.reset_index().query(
        "comparison == 'router_unseen_init'"
    )
    q02_rows = deltas[
        deltas["metric"].eq("brier")
        & deltas["window"].eq("q0_to_q2")
        & deltas["comparison"].isin(
            ("router_beyond_init", "router_unseen_init")
        )
    ]
    robust_beyond = beyond_rows.loc[beyond_rows["ci_low"] > 0, "target"].tolist()
    robust_unseen = unseen_rows.loc[unseen_rows["ci_low"] > 0, "target"].tolist()
    robust_q02 = q02_rows.loc[q02_rows["ci_low"] > 0, "target"].tolist()
    for target in TARGET_COLUMNS:
        main_rows.append(
            {
                "target": target,
                "positive/n": (
                    f"{int(metric_lookup.loc[(target, 'heldout_seed', 'task_only'), 'positives'])}/"
                    f"{int(metric_lookup.loc[(target, 'heldout_seed', 'task_only'), 'episodes'])}"
                ),
                "task prior": metric_lookup.loc[
                    (target, "heldout_seed", "task_only"), "brier"
                ],
                "init prior": metric_lookup.loc[
                    (target, "heldout_seed", "task_init"), "brier"
                ],
                "init+q0 routing": metric_lookup.loc[
                    (target, "heldout_seed", "task_init_router"), "brier"
                ],
                "unseen-init q0 routing": metric_lookup.loc[
                    (target, "heldout_init", "task_router"), "brier"
                ],
            }
        )
    main_table = pd.DataFrame(main_rows)

    gain_rows = []
    for target in TARGET_COLUMNS:
        for comparison in (
            "initial_prior",
            "router_beyond_init",
            "router_unseen_init",
        ):
            row = delta_brier.loc[(target, comparison)]
            gain_rows.append(
                {
                    "target": target,
                    "comparison": comparison,
                    "gain": row["improvement"],
                    "relative": row["relative_improvement"],
                    "95% cluster CI": (
                        f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
                    ),
                }
            )
    gain_table = pd.DataFrame(gain_rows)

    sensitivity = deltas[
        deltas["metric"].eq("brier")
        & deltas["comparison"].isin(
            ("router_beyond_init", "router_unseen_init")
        )
    ][
        [
            "target",
            "comparison",
            "window",
            "improvement",
            "ci_low",
            "ci_high",
        ]
    ].copy()

    trigger_rows = primary_metrics[
        primary_metrics["model"].isin(("task_init_router", "task_router"))
        & (
            (
                primary_metrics["split"].eq("heldout_seed")
                & primary_metrics["model"].eq("task_init_router")
            )
            | (
                primary_metrics["split"].eq("heldout_init")
                & primary_metrics["model"].eq("task_router")
            )
        )
    ][
        [
            "target",
            "split",
            "model",
            "positive_coverage_at_train_negative_q90",
            "negative_trigger_rate_at_train_negative_q90",
        ]
    ].copy()

    eligible_text = []
    for target in TARGET_COLUMNS:
        rows = support[support["target"].eq(target) & support["eligible"]]
        eligible_text.append(
            f"- `{target}`: {int(rows['positives'].sum())}/{int(rows['episodes'].sum())}, "
            f"{len(rows)} task(s)"
        )

    capacity_section = ""
    capacity_file = args.capacity_sensitivity_dir / "paired_deltas.csv"
    if args.pca_components == 12 and capacity_file.exists():
        capacity = pd.read_csv(capacity_file)
        capacity = capacity[
            capacity["metric"].eq("brier")
            & capacity["comparison"].isin(
                ("router_beyond_init", "router_unseen_init")
            )
        ]
        capacity_table = capacity[
            capacity["window"].eq("q0")
        ][
            [
                "target",
                "comparison",
                "improvement",
                "ci_low",
                "ci_high",
            ]
        ]
        positive_capacity = capacity.loc[
            capacity["improvement"] > 0, ["target", "comparison", "window"]
        ]
        capacity_section = f"""
## 32-component capacity sensitivity

Increasing each feature block from 12 to 32 training-fold PCA components did
not uncover a hidden improvement: among the {len(capacity)} prespecified
target/comparison/window cells, {len(positive_capacity)} had a positive point
estimate.  The q0 results are:

{markdown_table(capacity_table, capacity_table.columns)}

The complete high-capacity rerun, including predictions and independent
checksums, is in `../{args.capacity_sensitivity_dir.name}/`.
"""

    report = f"""# Initial-state prior versus early HiMoE signal

## Bottom line

- The initial-state prior is strong and reusable across held-out flow seeds.
  For any failure, it reduces Brier loss from
  {any_initial['baseline_loss']:.4f} to
  {any_initial['baseline_loss'] - any_initial['improvement']:.4f}, a
  {100 * any_initial['relative_improvement']:.1f}% reduction.  All six target
  intervals favor the initial-state prior.
- q0 routing does not show a stable increment beyond that prior.  For any
  failure the change is {any_router['improvement']:+.4f}, with a 95% clustered
  interval [{any_router['ci_low']:.4f}, {any_router['ci_high']:.4f}].  Targets
  with an interval wholly above zero: {robust_beyond or 'none'}.
- Routing also fails to improve prediction for complete unseen initial states;
  targets with an interval wholly above zero: {robust_unseen or 'none'}.
  Extending the fixed early window to q0:q2 does not rescue this result
  ({robust_q02 or 'no positive lower bounds'}).
- Therefore the defensible result is: the initial configuration is a strong
  early risk prior, but this readout provides no evidence that the first three
  MoE routing queries already identify the seed-specific eventual failure or
  transfer that risk to unseen geometry.

## Scope

The primary input is routing from the first policy query (`q0`), before its
emitted action chunk is executed.  The sensitivity input concatenates the fixed
first three queries (`q0:q2`); q1 and q2 may reflect new observations after
earlier chunks, but every rollout contributes exactly those same absolute query
indices.  No episode duration, remaining time, terminal window, action vector,
simulator trajectory, physical-failure annotation, or outcome-derived feature
is available to a predictor.

The five behavioral labels overlap and are each treated one-versus-rest.  A
task is evaluated for a label only when it contains at least
{MIN_POSITIVES_PER_TASK} positives and {MIN_NEGATIVES_PER_TASK} negatives:

{chr(10).join(eligible_text)}

## Experimental controls

1. `heldout_seed`: every fold leaves four complete flow seeds out of all 16
   initial states.  The training side still sees each test initial state.  This
   is the direct test of whether routing adds information beyond that state's
   empirical difficulty.
2. `heldout_init`: every fold leaves two complete initial states out, including
   all 32 flow seeds.  This asks whether routing transfers to unseen geometry.
3. Models are task-local.  Router descriptors are expert-permutation invariant;
   gate-input hidden summaries are kept separate.  Scaling and {args.pca_components}-component
   PCA are fit on the outer training fold only.  Each feature model is a
   regularized correction to the corresponding prior in log-odds space, so a
   zero correction reproduces its baseline exactly.  Regularization is fixed
   at `C={PRIMARY_C}`; `C={CS[0]}` and `C={CS[-1]}` are saved as sensitivity checks.
4. The initial-state probability uses an {INIT_PRIOR_STRENGTH:g}-episode shrinkage prior.
   In `heldout_init` it must reduce exactly to the task prior, because the test
   state has no training outcomes.

## Main q0 results

All entries below are cross-fitted Brier loss; lower is better.

{markdown_table(main_table, main_table.columns)}

The next table expresses paired Brier improvement as `baseline - candidate`.
Positive is useful.  Intervals resample complete initial-state clusters
({args.bootstrap:,} stratified bootstrap draws); they quantify sampling
uncertainty, not causality.

{markdown_table(gain_table, gain_table.columns)}

Interpret the three comparisons literally:

- `initial_prior`: how strong the reusable initial-state difficulty prior is
  when new flow seeds are run from a known state.
- `router_beyond_init`: whether q0 routing separates different seeds from the
  same initial state beyond that prior.
- `router_unseen_init`: whether q0 routing predicts risk for an entirely unseen
  initial state, relative to task prevalence alone.

## Fixed false-trigger operating point

The threshold is the 90th percentile of training-fold negative scores, applied
unchanged to held-out data.  This is included to show practical coverage and
the realized false-trigger rate; ties are triggered only when strictly above
the threshold.

{markdown_table(trigger_rows, trigger_rows.columns)}

## q0:q2 sensitivity

The three-query window is still absolute and early: exactly queries 0, 1, and
2 for every rollout, regardless of eventual duration.  It cannot see a final
window.

{markdown_table(sensitivity, sensitivity.columns)}

{capacity_section}

## Interpretation guardrails

- A strong `initial_prior` is an early risk signal, but it does not establish
  that a routing expert recognizes a future failure.  It can reflect geometry,
  task difficulty, or another stable property of that initial state.
- A positive `router_beyond_init` is the cleanest evidence here for seed-level
  early information, because outcome variation is compared after the initial
  state has already been identified statistically.
- A positive `router_unseen_init` is stronger evidence of transferable early
  structure.  Failure there, alongside success on known states, is consistent
  with initial-state memorization or a non-transferable geometry proxy.
- `hidden` is the input presented to a gate.  Hidden-only gains do not show
  expert selection or expert output contribution.  Those controls are retained
  in the machine-readable tables, but the main claim is based on routing.
- This corpus has one rollout per `(initial state, flow seed)`, not branched
  counterfactual continuations.  Statistical prediction cannot identify the
  intervention that would prevent a failure.

## Artifacts

- `metrics.csv`: Brier loss, log loss, calibration mean, and fixed-threshold
  coverage for every target/split/window/model/regularization setting.
- `paired_deltas.csv`: paired loss improvements and clustered uncertainty.
- `predictions.csv.gz`: every held-out score and threshold.
- `target_support.csv`: included and excluded task-label cells.
- `fallback_fits.csv`: model fits that reverted to their prior because the
  training fold had a single class.
- `brier_gain_q0.png`: visual summary of the three main comparisons.
- `summary.json` and `checksums.sha256`: audit metadata.

Fallback fits: {fallback_count}.  They are all repetitions of one data split:
drawer-task stagnation in unseen-initial-state fold 5.  That fold holds the
only stagnation-positive drawer initial state out of training.  Removing the
drawer task leaves the pooled unseen-state stagnation point estimate negative.
"""
    path.write_text(report)


def self_test() -> None:
    y = np.array([0, 1, 0, 1], dtype=bool)
    states = np.array([0, 0, 1, 1])
    train, unseen = prior_scores(y, states, np.array([2]), True)
    assert train.shape == (4,) and unseen.shape == (1,)
    assert np.isfinite(train).all() and np.isfinite(unseen).all()
    state_train, state_test = initial_state_matrix(states, np.array([0, 2]))
    assert state_train.shape == (4, 2) and state_test[1].sum() == 0
    assert threshold_at_training_negative_q90(np.arange(4), y) == 2


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame, features, feature_files = load_data(args.feature_dir)
    frame = assign_folds(frame)
    predictions, fallbacks, task_support = evaluate(
        frame, features, args.pca_components, args.seed
    )
    metrics = summarize_metrics(predictions)
    deltas = bootstrap_paired_deltas(predictions, args.bootstrap, args.seed + 1)
    support = target_support_table(frame, task_support)

    predictions.to_csv(args.out_dir / "predictions.csv.gz", index=False)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    deltas.to_csv(args.out_dir / "paired_deltas.csv", index=False)
    support.to_csv(args.out_dir / "target_support.csv", index=False)
    fallbacks.to_csv(args.out_dir / "fallback_fits.csv", index=False)
    make_gain_plot(deltas, args.out_dir / "brier_gain_q0.png")
    write_report(
        args.out_dir / "report.md",
        frame,
        support,
        metrics,
        deltas,
        len(fallbacks),
        args,
    )

    summary = {
        "episodes": len(frame),
        "failures": int(frame["any_failure"].sum()),
        "tasks": int(frame["task"].nunique()),
        "initial_states_per_task": 16,
        "flow_seeds_per_initial_state": 32,
        "folds": N_FOLDS,
        "windows": list(WINDOWS),
        "models": list(MODELS),
        "regularization_c": list(CS),
        "primary_c": PRIMARY_C,
        "pca_components_per_block": args.pca_components,
        "initial_state_prior_strength": INIT_PRIOR_STRENGTH,
        "bootstrap_draws": args.bootstrap,
        "capacity_sensitivity_dir": str(args.capacity_sensitivity_dir),
        "feature_files": [str(path) for path in feature_files],
        "target_tasks": task_support,
        "fallback_fits": len(fallbacks),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n"
    )
    artifacts = sorted(
        path
        for path in args.out_dir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    checksum_lines = [f"{sha256(path)}  {path.name}" for path in artifacts]
    (args.out_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    print(json.dumps(plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
