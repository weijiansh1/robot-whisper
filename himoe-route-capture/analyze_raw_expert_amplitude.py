"""Cross-task audit of absolute MoE expert-output amplitude.

The analysis fixes HB5/denoise-0 and reuses the five exact-observation K=32
captures from ``expert-activation-hidden-matched``.  Expert features are kept
in their absolute RMS units: no division by hidden, shared, routed, or post-MoE
amplitude is used.  Train-fold standardization is only numerical conditioning
for the linear readouts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from analyze_expert_activation_proxy import (
    load_primary_cell,
    reconstruct_primary_cell,
)


N_SEED_FOLDS = 4
RAW_METRIC_NAMES = (
    "expert_mass",
    "routed_rms",
    "input_rms",
    "shared_rms",
    "post_rms",
    "delta_rms",
    "cancellation_gap",
    "router_entropy",
    "negative_top4_margin",
)
EXPERT_METRICS = ("expert_mass", "routed_rms")
CONTROL_METRICS = ("input_rms", "shared_rms")
FRAGILITY_CONTROLS = (
    "input_rms",
    "shared_rms",
    "router_entropy",
    "negative_top4_margin",
)
MODEL_NAMES = ("baseline", "expert_raw", "control_raw", "all_raw")


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proxy-root",
        type=pathlib.Path,
        default=here / "analysis" / "expert-activation-hidden-matched",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=here / "analysis" / "raw-expert-amplitude",
    )
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--denoise", type=int, default=0)
    parser.add_argument("--perms", type=int, default=5000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def _rms(value: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(value, dtype=np.float64), axis=axis))


def _metric_axis(name: str) -> int:
    return RAW_METRIC_NAMES.index(name)


def _pool_correlation(
    predictor: np.ndarray, target: np.ndarray, rank: bool = False
) -> np.ndarray:
    """Correlation along the last (seed) axis, preserving leading pool axes."""
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} != {y.shape}")
    if rank:
        x = np.apply_along_axis(rankdata, -1, x)
        y = np.apply_along_axis(rankdata, -1, y)
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(
        np.sum(np.square(x), axis=-1) * np.sum(np.square(y), axis=-1)
    )
    return np.sum(x * y, axis=-1) / np.maximum(denominator, 1e-20)


def _hierarchical_bootstrap_ci(
    values: np.ndarray, draws: int, rng: np.random.Generator
) -> list[float]:
    """Equal-weight task -> pool bootstrap for a [task, pool] statistic."""
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("hierarchical bootstrap expects [task, pool]")
    n_task, n_pool = value.shape
    task_index = rng.integers(0, n_task, size=(draws, n_task))
    pool_index = rng.integers(0, n_pool, size=(draws, n_task, n_pool))
    sampled = value[task_index[:, :, None], pool_index]
    means = np.nanmean(sampled, axis=(1, 2))
    return [float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))]


def _pair_feature_block(value: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    """Absolute pair difference and pair level, retaining every action token."""
    left = value[pair_i]
    right = value[pair_j]
    return np.column_stack((np.abs(left - right), 0.5 * (left + right)))


def _seed_pair_masks(
    n_seed: int, pair_i: np.ndarray, pair_j: np.ndarray, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_seed % N_SEED_FOLDS:
        raise ValueError(f"{n_seed} seeds cannot be split into {N_SEED_FOLDS} equal folds")
    seed_fold = np.arange(n_seed) // (n_seed // N_SEED_FOLDS)
    train = (seed_fold[pair_i] != fold) & (seed_fold[pair_j] != fold)
    test = (seed_fold[pair_i] == fold) & (seed_fold[pair_j] == fold)
    return train, test


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) != 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _nanmean_columns(value: np.ndarray) -> np.ndarray:
    count = np.isfinite(value).sum(axis=0)
    output = np.full(value.shape[1], np.nan, dtype=np.float64)
    np.divide(np.nansum(value, axis=0), count, out=output, where=count > 0)
    return output


def _fit_logistic(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    logistic_c: float,
) -> np.ndarray:
    scaler = StandardScaler().fit(train)
    model = LogisticRegression(C=logistic_c, solver="lbfgs", max_iter=3000)
    model.fit(scaler.transform(train), labels)
    return model.predict_proba(scaler.transform(test))[:, 1]


def _reduce_noise(
    train: np.ndarray, test: np.ndarray, components: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    count = min(components, train_scaled.shape[1], train_scaled.shape[0] - 1)
    if count >= train_scaled.shape[1]:
        return train_scaled, test_scaled
    pca = PCA(n_components=count, svd_solver="full").fit(train_scaled)
    return pca.transform(train_scaled), pca.transform(test_scaled)


def _extract_task(
    name: str,
    summary_path: pathlib.Path,
    layer: int,
    denoise: int,
    threads: int,
) -> dict[str, Any]:
    old_summary = json.loads(summary_path.read_text())
    run = pathlib.Path(old_summary["run"])
    checkpoint = pathlib.Path(old_summary["checkpoint"])
    data = load_primary_cell(run, layer, denoise)
    reconstructed = reconstruct_primary_cell(data, checkpoint, layer, threads)

    input_rms = _rms(data["hidden"], axis=-1)
    routed = np.asarray(reconstructed["routed"], dtype=np.float32)
    shared = np.asarray(reconstructed["shared"], dtype=np.float32)
    routed_alt = np.asarray(reconstructed["routed_alt"], dtype=np.float32)
    post_rms = _rms(routed + shared, axis=-1)
    delta_rms = _rms(routed_alt - routed, axis=-1)
    expert_mass = np.asarray(reconstructed["expert_mass"], dtype=np.float64)
    routed_rms = np.asarray(reconstructed["routed_rms"], dtype=np.float64)
    values = np.stack(
        (
            expert_mass,
            routed_rms,
            input_rms,
            reconstructed["shared_rms"],
            post_rms,
            delta_rms,
            expert_mass - routed_rms,
            reconstructed["router_entropy"],
            -reconstructed["top4_margin"],
        ),
        axis=-1,
    ).astype(np.float32)

    order = data["order"]
    success = np.asarray(
        [bool(row["success"]) for row in data["summaries"]], dtype=np.int8
    )
    task = {
        "name": name,
        "scene_ids": np.asarray(data["scene_ids"], dtype=np.int64),
        "seed_ids": np.asarray(data["seed_ids"], dtype=np.int64),
        "raw": values[order],
        "actions": np.asarray(data["actions"][order], dtype=np.float32),
        "noise": np.asarray(data["noise"][order], dtype=np.float32),
        "success": success[order],
        "validation": reconstructed["validation"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "source_run": str(run),
    }
    if task["raw"].shape[:3] != (16, 32, 10):
        raise ValueError(f"{name}: expected 16x32x10, got {task['raw'].shape}")
    return task


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = sorted(args.proxy_root.glob("*/summary.json"))
    if not paths:
        raise FileNotFoundError(f"no task summaries below {args.proxy_root}")
    tasks = []
    for path in paths:
        name = path.parent.name
        print(f"reconstructing {name} at HB{args.layer}/d{args.denoise}", flush=True)
        tasks.append(_extract_task(name, path, args.layer, args.denoise, args.threads))
    shapes = {task["raw"].shape for task in tasks}
    if len(shapes) != 1:
        raise ValueError(f"task grids differ: {shapes}")
    return tasks


def summarize_raw_scale(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"per_task": {}}
    for task in tasks:
        raw = task["raw"]
        rows = {}
        for name in RAW_METRIC_NAMES[:7]:
            value = raw[..., _metric_axis(name)]
            candidate = value.mean(axis=2)
            pool_cv = candidate.std(axis=1) / np.maximum(candidate.mean(axis=1), 1e-20)
            rows[name] = {
                "median": float(np.median(value)),
                "q25": float(np.quantile(value, 0.25)),
                "q75": float(np.quantile(value, 0.75)),
                "median_within_pool_cv": float(np.median(pool_cv)),
            }
        result["per_task"][task["name"]] = rows
    result["cross_task_median_ratio_max_over_min"] = {
        name: float(
            max(result["per_task"][task["name"]][name]["median"] for task in tasks)
            / max(
                min(result["per_task"][task["name"]][name]["median"] for task in tasks),
                1e-20,
            )
        )
        for name in RAW_METRIC_NAMES[:7]
    }
    return result


def analyze_absolute_fragility(
    tasks: list[dict[str, Any]], args: argparse.Namespace, rng: np.random.Generator
) -> dict[str, Any]:
    names = EXPERT_METRICS + CONTROL_METRICS
    candidate = np.stack([task["raw"].mean(axis=2) for task in tasks])
    target = candidate[..., _metric_axis("delta_rms")]
    predictors = np.stack([candidate[..., _metric_axis(name)] for name in names])
    pearson_pool = np.stack(
        [_pool_correlation(value, target) for value in predictors]
    )
    spearman_pool = np.stack(
        [_pool_correlation(value, target, rank=True) for value in predictors]
    )
    observed = pearson_pool.mean(axis=(1, 2))

    null = np.empty((args.perms, len(names)), dtype=np.float64)
    for permutation in range(args.perms):
        # Seed identities recur in every exact-observation pool.  A common
        # permutation preserves that dependence under the null.
        seed_order = rng.permutation(target.shape[-1])
        permuted = target[..., seed_order]
        for axis, value in enumerate(predictors):
            null[permutation, axis] = _pool_correlation(value, permuted).mean()
    expert_axes = np.arange(len(EXPERT_METRICS))
    null_max = null[:, expert_axes].max(axis=1)

    metrics = {}
    for axis, name in enumerate(names):
        row = {
            "macro_pearson": float(observed[axis]),
            "macro_spearman": float(spearman_pool[axis].mean()),
            "pearson_hierarchical_ci95": _hierarchical_bootstrap_ci(
                pearson_pool[axis], args.bootstrap, rng
            ),
            "per_task_pearson": {
                task["name"]: float(pearson_pool[axis, task_axis].mean())
                for task_axis, task in enumerate(tasks)
            },
            "per_task_spearman": {
                task["name"]: float(spearman_pool[axis, task_axis].mean())
                for task_axis, task in enumerate(tasks)
            },
            "common_seed_permutation_p": float(
                (1 + np.sum(null[:, axis] >= observed[axis])) / (1 + args.perms)
            ),
        }
        if name in EXPERT_METRICS:
            row["maxT_fwer_p_over_raw_expert_pair"] = float(
                (1 + np.sum(null_max >= observed[axis])) / (1 + args.perms)
            )
        metrics[name] = row

    model_pool_corr = {name: [] for name in ("controls", "controls_plus_expert")}
    model_task_rows = {}
    for task in tasks:
        raw = task["raw"].mean(axis=2)
        n_pool, n_seed, _ = raw.shape
        pool_onehot = np.repeat(np.eye(n_pool, dtype=np.float64), n_seed, axis=0)
        flat = raw.reshape(n_pool * n_seed, -1)
        controls = flat[:, [_metric_axis(name) for name in FRAGILITY_CONTROLS]]
        expert = flat[:, [_metric_axis(name) for name in EXPERT_METRICS]]
        y = flat[:, _metric_axis("delta_rms")]
        seed_slot = np.tile(np.arange(n_seed), n_pool)
        predictions = {
            "controls": np.full_like(y, np.nan, dtype=np.float64),
            "controls_plus_expert": np.full_like(y, np.nan, dtype=np.float64),
        }
        for fold in range(N_SEED_FOLDS):
            test = seed_slot // (n_seed // N_SEED_FOLDS) == fold
            train = ~test
            parts = {
                "controls": np.column_stack((pool_onehot, controls)),
                "controls_plus_expert": np.column_stack(
                    (pool_onehot, controls, expert)
                ),
            }
            for model_name, x in parts.items():
                scaler = StandardScaler().fit(x[train])
                model = Ridge(alpha=args.ridge_alpha)
                model.fit(scaler.transform(x[train]), y[train])
                predictions[model_name][test] = model.predict(scaler.transform(x[test]))
        task_row = {}
        for model_name, score in predictions.items():
            score_pool = score.reshape(n_pool, n_seed)
            y_pool = y.reshape(n_pool, n_seed)
            pool_corr = _pool_correlation(score_pool, y_pool)
            model_pool_corr[model_name].append(pool_corr)
            task_row[model_name] = {
                "r2_oof": float(r2_score(y, score)),
                "rmse_oof": float(_rms(y - score)),
                "within_pool_pearson": float(pool_corr.mean()),
            }
        model_task_rows[task["name"]] = task_row

    controls = np.stack(model_pool_corr["controls"])
    augmented = np.stack(model_pool_corr["controls_plus_expert"])
    difference = augmented - controls
    incremental = {
        "per_task": model_task_rows,
        "macro_within_pool_pearson": {
            "controls": float(controls.mean()),
            "controls_plus_expert": float(augmented.mean()),
            "difference": float(difference.mean()),
            "difference_hierarchical_ci95": _hierarchical_bootstrap_ci(
                difference, args.bootstrap, rng
            ),
        },
    }
    return {
        "target": "absolute RMS(r_rank5-8 - r_top4), averaged over 10 action tokens",
        "predictors": "absolute values; no denominator normalization",
        "correlations": metrics,
        "incremental_seed_disjoint_ridge": incremental,
        "mechanistic_caveat": (
            "The target and routed-size predictors share expert outputs, so a positive "
            "association is an immediate block-sensitivity result, not outcome causality."
        ),
    }


def _build_basin_frame(task: dict[str, Any]) -> dict[str, np.ndarray]:
    n_pool, n_seed = task["raw"].shape[:2]
    pair_i, pair_j = np.triu_indices(n_seed, 1)
    n_pair = len(pair_i)
    baseline = np.empty((n_pool, n_pair, 12), dtype=np.float32)
    expert = np.empty((n_pool, n_pair, 40), dtype=np.float32)
    control = np.empty_like(expert)
    labels = np.empty((n_pool, n_pair), dtype=np.int8)
    for pool in range(n_pool):
        noise = task["noise"][pool]
        live = noise[..., :7]
        delta7 = live[pair_i] - live[pair_j]
        delta24 = noise[pair_i] - noise[pair_j]
        baseline[pool, :, 0] = _rms(delta7, axis=(1, 2))
        baseline[pool, :, 1] = _rms(delta24, axis=(1, 2))
        baseline[pool, :, 2:] = _rms(delta7, axis=2)
        raw = task["raw"][pool]
        expert[pool] = np.column_stack(
            [
                _pair_feature_block(raw[..., _metric_axis(name)], pair_i, pair_j)
                for name in EXPERT_METRICS
            ]
        )
        control[pool] = np.column_stack(
            [
                _pair_feature_block(raw[..., _metric_axis(name)], pair_i, pair_j)
                for name in CONTROL_METRICS
            ]
        )
        action_delta = task["actions"][pool, pair_i] - task["actions"][pool, pair_j]
        action_distance = _rms(action_delta, axis=(1, 2))
        labels[pool] = action_distance <= np.median(action_distance)
    return {
        "pair_i": pair_i,
        "pair_j": pair_j,
        "baseline": baseline,
        "expert": expert,
        "control": control,
        "labels": labels,
    }


def analyze_action_basin(
    tasks: list[dict[str, Any]], args: argparse.Namespace, rng: np.random.Generator
) -> dict[str, Any]:
    pool_auc = {name: [] for name in MODEL_NAMES}
    per_task = {}
    for task in tasks:
        frame = _build_basin_frame(task)
        n_pool, n_pair = frame["labels"].shape
        pool_index = np.repeat(np.arange(n_pool), n_pair)
        labels = frame["labels"].reshape(-1)
        matrices = {
            "baseline": frame["baseline"].reshape(n_pool * n_pair, -1),
            "expert_raw": np.concatenate(
                (frame["baseline"], frame["expert"]), axis=-1
            ).reshape(n_pool * n_pair, -1),
            "control_raw": np.concatenate(
                (frame["baseline"], frame["control"]), axis=-1
            ).reshape(n_pool * n_pair, -1),
            "all_raw": np.concatenate(
                (frame["baseline"], frame["expert"], frame["control"]), axis=-1
            ).reshape(n_pool * n_pair, -1),
        }
        scores = {name: np.full(len(labels), np.nan) for name in MODEL_NAMES}
        fold_pool_auc = {
            name: np.full((N_SEED_FOLDS, n_pool), np.nan) for name in MODEL_NAMES
        }
        for fold in range(N_SEED_FOLDS):
            local_train, local_test = _seed_pair_masks(
                len(task["seed_ids"]), frame["pair_i"], frame["pair_j"], fold
            )
            train = np.tile(local_train, n_pool)
            test = np.tile(local_test, n_pool)
            for name, x in matrices.items():
                scores[name][test] = _fit_logistic(
                    x[train], x[test], labels[train], args.logistic_c
                )
                for pool in range(n_pool):
                    cell = test & (pool_index == pool)
                    fold_pool_auc[name][fold, pool] = _safe_auc(
                        labels[cell], scores[name][cell]
                    )
        task_rows = {}
        for name in MODEL_NAMES:
            values = _nanmean_columns(fold_pool_auc[name])
            pool_auc[name].append(values)
            task_rows[name] = float(np.nanmean(values))
        per_task[task["name"]] = task_rows
        print(f"basin models complete: {task['name']}", flush=True)

    stacked = {name: np.stack(value) for name, value in pool_auc.items()}
    expert_delta = stacked["expert_raw"] - stacked["baseline"]
    control_delta = stacked["control_raw"] - stacked["baseline"]
    expert_minus_control = stacked["expert_raw"] - stacked["control_raw"]
    return {
        "target": "within-pool final 10x7 action distance <= pool median",
        "evaluation": "four seed-disjoint pair folds; task/pool-macro AUC",
        "feature_definition": (
            "absolute pair difference and pair mean for every action token; "
            "no amplitude ratios"
        ),
        "per_task_auc": per_task,
        "macro_auc": {name: float(np.nanmean(value)) for name, value in stacked.items()},
        "contrasts": {
            "expert_minus_baseline": {
                "difference": float(np.nanmean(expert_delta)),
                "hierarchical_ci95": _hierarchical_bootstrap_ci(
                    expert_delta, args.bootstrap, rng
                ),
            },
            "control_minus_baseline": {
                "difference": float(np.nanmean(control_delta)),
                "hierarchical_ci95": _hierarchical_bootstrap_ci(
                    control_delta, args.bootstrap, rng
                ),
            },
            "expert_minus_control": {
                "difference": float(np.nanmean(expert_minus_control)),
                "hierarchical_ci95": _hierarchical_bootstrap_ci(
                    expert_minus_control, args.bootstrap, rng
                ),
            },
        },
    }


def analyze_success(
    tasks: list[dict[str, Any]], args: argparse.Namespace, rng: np.random.Generator
) -> dict[str, Any]:
    pool_auc = {name: [] for name in MODEL_NAMES}
    per_task = {}
    per_task_log_loss = {}
    excluded_tasks = {}
    for task in tasks:
        raw = task["raw"]
        n_pool, n_seed = raw.shape[:2]
        labels = task["success"].reshape(-1)
        if len(np.unique(labels)) != 2:
            excluded_tasks[task["name"]] = "success label has only one class"
            continue
        pool_index = np.repeat(np.arange(n_pool), n_seed)
        seed_slot = np.tile(np.arange(n_seed), n_pool)
        pool_onehot = np.repeat(np.eye(n_pool), n_seed, axis=0)
        noise = task["noise"][..., :7].reshape(n_pool * n_seed, -1)
        expert = raw[..., [_metric_axis(name) for name in EXPERT_METRICS]].reshape(
            n_pool * n_seed, -1
        )
        control = raw[..., [_metric_axis(name) for name in CONTROL_METRICS]].reshape(
            n_pool * n_seed, -1
        )
        scores = {name: np.full(len(labels), np.nan) for name in MODEL_NAMES}
        fold_pool_auc = {
            name: np.full((N_SEED_FOLDS, n_pool), np.nan) for name in MODEL_NAMES
        }
        for fold in range(N_SEED_FOLDS):
            test = seed_slot // (n_seed // N_SEED_FOLDS) == fold
            train = ~test
            if len(np.unique(labels[train])) != 2:
                excluded_tasks[task["name"]] = (
                    f"training labels have one class in seed fold {fold}"
                )
                break
            noise_train, noise_test = _reduce_noise(noise[train], noise[test])
            train_parts = {
                "baseline": [pool_onehot[train], noise_train],
                "expert_raw": [pool_onehot[train], noise_train, expert[train]],
                "control_raw": [pool_onehot[train], noise_train, control[train]],
                "all_raw": [
                    pool_onehot[train],
                    noise_train,
                    expert[train],
                    control[train],
                ],
            }
            test_parts = {
                "baseline": [pool_onehot[test], noise_test],
                "expert_raw": [pool_onehot[test], noise_test, expert[test]],
                "control_raw": [pool_onehot[test], noise_test, control[test]],
                "all_raw": [
                    pool_onehot[test],
                    noise_test,
                    expert[test],
                    control[test],
                ],
            }
            for name in MODEL_NAMES:
                scores[name][test] = _fit_logistic(
                    np.column_stack(train_parts[name]),
                    np.column_stack(test_parts[name]),
                    labels[train],
                    args.logistic_c,
                )
                for pool in range(n_pool):
                    cell = test & (pool_index == pool)
                    fold_pool_auc[name][fold, pool] = _safe_auc(
                        labels[cell], scores[name][cell]
                    )
        if task["name"] in excluded_tasks:
            continue
        task_rows = {}
        task_losses = {}
        for name in MODEL_NAMES:
            values = _nanmean_columns(fold_pool_auc[name])
            pool_auc[name].append(values)
            task_rows[name] = float(np.nanmean(values))
            task_losses[name] = float(log_loss(labels, scores[name], labels=[0, 1]))
        per_task[task["name"]] = task_rows
        per_task_log_loss[task["name"]] = task_losses
        print(f"success models complete: {task['name']}", flush=True)

    stacked = {name: np.stack(value) for name, value in pool_auc.items()}
    expert_delta = stacked["expert_raw"] - stacked["baseline"]
    control_delta = stacked["control_raw"] - stacked["baseline"]
    expert_minus_control = stacked["expert_raw"] - stacked["control_raw"]
    return {
        "target": "eventual rollout success",
        "evaluation": "four seed-disjoint candidate folds; within-pool AUC",
        "status": "exploratory long-horizon endpoint",
        "eligible_tasks": list(per_task),
        "excluded_tasks": excluded_tasks,
        "informative_pools": {
            task: int(np.isfinite(pool_auc["baseline"][axis]).sum())
            for axis, task in enumerate(per_task)
        },
        "positive_rate": {
            task["name"]: float(task["success"].mean()) for task in tasks
        },
        "per_task_auc": per_task,
        "per_task_log_loss": per_task_log_loss,
        "macro_log_loss": {
            name: float(
                np.mean([per_task_log_loss[task][name] for task in per_task_log_loss])
            )
            for name in MODEL_NAMES
        },
        "macro_auc": {name: float(np.nanmean(value)) for name, value in stacked.items()},
        "contrasts": {
            "expert_minus_baseline": {
                "difference": float(np.nanmean(expert_delta)),
                "hierarchical_ci95": _hierarchical_bootstrap_ci(
                    expert_delta, args.bootstrap, rng
                ),
            },
            "control_minus_baseline": {
                "difference": float(np.nanmean(control_delta)),
                "hierarchical_ci95": _hierarchical_bootstrap_ci(
                    control_delta, args.bootstrap, rng
                ),
            },
            "expert_minus_control": {
                "difference": float(np.nanmean(expert_minus_control)),
                "hierarchical_ci95": _hierarchical_bootstrap_ci(
                    expert_minus_control, args.bootstrap, rng
                ),
            },
        },
    }


def analyze_denoise_clock(proxy_root: pathlib.Path, layer: int) -> dict[str, Any] | None:
    path = proxy_root.parent / "expert-activation-v2" / "long-t08" / "features_v2.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        scalars = np.asarray(payload["scalars"], dtype=np.float64)
        names = [str(value) for value in payload["scalar_names"]]
        layers = np.asarray(payload["layer_numbers"])
    if layer not in layers:
        return None
    layer_axis = int(np.flatnonzero(layers == layer)[0])
    result = {"source": str(path), "layer": layer, "per_metric": {}}
    for name in ("expert_mass", "routed_rms", "input_rms", "shared_rms"):
        value = scalars[:, :, layer_axis, 1:, names.index(name)].mean(axis=2)
        grand = value.mean()
        fitted = value.mean(axis=0, keepdims=True)
        total = np.sum(np.square(value - grand))
        explained = np.sum(np.square(fitted - grand)) * value.shape[0]
        result["per_metric"][name] = {
            "median_by_round": np.median(value, axis=0).tolist(),
            "round_explained_variance": float(explained / max(total, 1e-20)),
        }
    return result


def _render_report(summary: dict[str, Any]) -> str:
    scale = summary["raw_scale"]["per_task"]
    fragility = summary["absolute_block_sensitivity"]
    basin = summary["final_action_basin"]
    success = summary["eventual_success"]
    lines = [
        "# Gate 加权分支幅度审计（已被更正）",
        "",
        "> **方法范围更正：** 本脚本测量的是 gate 加权后的 branch 幅度，"
        "不是每个 selected expert 在乘 gate 前的原始输出范数。它不能回答"
        "未加权 raw expert activation 或动作承诺问题。",
        "",
        "固定单元为 HB5 / denoise 0；五个任务均为 16 个同观测池 × 32 个共同 seed。",
        "这里的 expert_mass、routed_rms 均保留绝对 RMS 单位，不除以 hidden、shared、",
        "routed 或 post-MoE 幅度。1024 维下 L2 恰为 RMS×32，因此不会改变候选排序。",
        "回归器只在训练折内做 StandardScaler，以保证数值条件，不改变实验变量定义。",
        "",
        "## 原始量级",
        "",
        "| task | expert mass | routed RMS | shared RMS | absolute swap delta | mass pool-CV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in summary["tasks"]:
        row = scale[task]
        lines.append(
            "| %s | %.4f | %.4f | %.4f | %.4f | %.3f |"
            % (
                task,
                row["expert_mass"]["median"],
                row["routed_rms"]["median"],
                row["shared_rms"]["median"],
                row["delta_rms"]["median"],
                row["expert_mass"]["median_within_pool_cv"],
            )
        )

    lines += [
        "",
        "## 绝对专家替换敏感度",
        "",
        "目标是 rank-5–8 替换 top-4 后的绝对 routed-vector RMS 变化；不使用任何分母。",
        "相关系数先在每个同观测 K=32 池内计算，再对任务和池等权平均。",
        "expert 两项的 p 值以共同 seed 置换做 maxT 校正；control 的 p 值未纳入该发现族。",
        "",
        "| predictor | Pearson | Spearman | hierarchical 95% CI | permutation p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in EXPERT_METRICS + CONTROL_METRICS:
        row = fragility["correlations"][name]
        p = row.get("maxT_fwer_p_over_raw_expert_pair", row["common_seed_permutation_p"])
        lines.append(
            "| %s | %+.3f | %+.3f | [%+.3f, %+.3f] | %.4f |"
            % (
                name,
                row["macro_pearson"],
                row["macro_spearman"],
                row["pearson_hierarchical_ci95"][0],
                row["pearson_hierarchical_ci95"][1],
                p,
            )
        )
    incremental = fragility["incremental_seed_disjoint_ridge"][
        "macro_within_pool_pearson"
    ]
    lines += [
        "",
        "按 seed 留出的线性读出：input/shared/router 控制的池内 Pearson 为 %.3f，"
        "加入原始 expert mass/routed RMS 后为 %.3f，增量 %+.3f，95%% CI [%+.3f, %+.3f]。"
        % (
            incremental["controls"],
            incremental["controls_plus_expert"],
            incremental["difference"],
            incremental["difference_hierarchical_ci95"][0],
            incremental["difference_hierarchical_ci95"][1],
        ),
        "该结果与目标共享同一 block 的专家输出，只能解释为即时机械敏感度。",
        "",
        "## 最终动作盆地",
        "",
        "| task | noise baseline | + expert raw | + control raw | expert Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in summary["tasks"]:
        row = basin["per_task_auc"][task]
        lines.append(
            "| %s | %.3f | %.3f | %.3f | %+.3f |"
            % (
                task,
                row["baseline"],
                row["expert_raw"],
                row["control_raw"],
                row["expert_raw"] - row["baseline"],
            )
        )
    b_contrast = basin["contrasts"]
    lines += [
        "",
        "五任务宏平均：baseline %.3f，+expert %.3f，+control %.3f。"
        % (
            basin["macro_auc"]["baseline"],
            basin["macro_auc"]["expert_raw"],
            basin["macro_auc"]["control_raw"],
        ),
        "expert 增量 %+.3f，95%% CI [%+.3f, %+.3f]；expert 相对 control %+.3f，"
        "95%% CI [%+.3f, %+.3f]。"
        % (
            b_contrast["expert_minus_baseline"]["difference"],
            b_contrast["expert_minus_baseline"]["hierarchical_ci95"][0],
            b_contrast["expert_minus_baseline"]["hierarchical_ci95"][1],
            b_contrast["expert_minus_control"]["difference"],
            b_contrast["expert_minus_control"]["hierarchical_ci95"][0],
            b_contrast["expert_minus_control"]["hierarchical_ci95"][1],
        ),
        "",
        "## 最终成功（探索性）",
        "",
        "| task | baseline | + expert raw | + control raw | expert Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in summary["tasks"]:
        if task in success["per_task_auc"]:
            row = success["per_task_auc"][task]
            lines.append(
                "| %s | %.3f | %.3f | %.3f | %+.3f |"
                % (
                    task,
                    row["baseline"],
                    row["expert_raw"],
                    row["control_raw"],
                    row["expert_raw"] - row["baseline"],
                )
            )
        else:
            lines.append("| %s | NA | NA | NA | NA |" % task)
    s_contrast = success["contrasts"]
    lines += [
        "",
        "四个可估计任务宏平均：baseline %.3f，+expert %.3f，+control %.3f；"
        "expert 增量 %+.3f，"
        "95%% CI [%+.3f, %+.3f]。"
        % (
            success["macro_auc"]["baseline"],
            success["macro_auc"]["expert_raw"],
            success["macro_auc"]["control_raw"],
            s_contrast["expert_minus_baseline"]["difference"],
            s_contrast["expert_minus_baseline"]["hierarchical_ci95"][0],
            s_contrast["expert_minus_baseline"]["hierarchical_ci95"][1],
        ),
    ]
    if success["excluded_tasks"]:
        lines.append(
            "未纳入任务：%s。"
            % ", ".join(
                "%s（%s）" % item for item in success["excluded_tasks"].items()
            )
        )
    lines += [
        "可计算池数：%s。"
        % ", ".join(
            "%s=%d" % item for item in success["informative_pools"].items()
        ),
        "但四个任务的 expert 模型 log loss 都比 baseline 更差；宏平均 %.4f → %.4f。"
        % (
            success["macro_log_loss"]["baseline"],
            success["macro_log_loss"]["expert_raw"],
        ),
    ]

    clock = summary.get("denoise_clock")
    if clock:
        lines += [
            "",
            "## 去噪时钟检查",
            "",
            "long-t08 的十轮原始幅度中，单独由 round 解释的变异比例：",
            "",
        ]
        for name, row in clock["per_metric"].items():
            lines.append("- %s: %.1f%%" % (name, 100 * row["round_explained_variance"]))

    lines += [
        "",
        "## 已撤回的外推",
        "",
        "此前把替换相关称为 Level-1 信号、把 median near-pair 称为动作盆地、",
        "以及据此否定动作承诺的表述均已撤回。最窄的有效结果只是：固定线性",
        "readout 下，HB5/d0 的 gate 加权聚合量没有改善“最终动作 pair 距离是否",
        "低于池中位数”的分类；这不等价于 per-expert raw norm 或时间承诺。",
    ]

    lines += [
        "",
        "## 边界",
        "",
        "- 这是从 fp16 hidden 与 checkpoint 离线重算的 HB block 代理，不是 runtime-exact 输出。",
        "- 数据没有逐轮 flow latent，因此盆地目标是最终 action chunk 相似性，不是剩余动作修正量。",
        "- rank-5–8 替换只计算即时 block 输出，没有继续完成后续 flow 或环境 rollout。",
        "- success 距离 MoE block 很远，只能作探索性关联，不能解释为动作已经承诺成败。",
        "",
        "完整数值见 `summary.json`，候选级原始数组见 `candidate_raw_amplitudes.npz`。",
    ]
    return "\n".join(lines) + "\n"


def _plot_summary(summary: dict[str, Any], output: pathlib.Path) -> None:
    tasks = summary["tasks"]
    colors = {
        "expert_mass": "#c2410c",
        "routed_rms": "#2563eb",
        "shared_rms": "#15803d",
        "delta_rms": "#7c3aed",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    x = np.arange(len(tasks))
    scale = summary["raw_scale"]["per_task"]
    for name, color in colors.items():
        axes[0].plot(
            x,
            [scale[task][name]["median"] for task in tasks],
            marker="o",
            label=name,
            color=color,
        )
    axes[0].set_xticks(x, tasks, rotation=28, ha="right")
    axes[0].set_ylabel("absolute RMS")
    axes[0].set_title("Gate-weighted HB5/d0 amplitude")
    axes[0].legend(fontsize=8)

    fragility = summary["absolute_block_sensitivity"]["correlations"]
    corr_names = EXPERT_METRICS + CONTROL_METRICS
    width = 0.17
    for task_axis, task in enumerate(tasks):
        axes[1].bar(
            np.arange(len(corr_names)) + (task_axis - 2) * width,
            [fragility[name]["per_task_pearson"][task] for name in corr_names],
            width=width,
            label=task,
        )
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(corr_names)), corr_names, rotation=25, ha="right")
    axes[1].set_ylabel("within-pool Pearson")
    axes[1].set_title("Absolute swap sensitivity")
    axes[1].legend(fontsize=6, ncol=2, loc="lower left")

    basin = summary["final_action_basin"]["macro_auc"]
    success = summary["eventual_success"]["macro_auc"]
    bx = np.arange(2)
    for model_axis, model in enumerate(("baseline", "expert_raw", "control_raw")):
        axes[2].bar(
            bx + (model_axis - 1) * 0.22,
            [basin[model], success[model]],
            width=0.22,
            label=model,
        )
    axes[2].axhline(0.5, color="#444444", linewidth=0.8, linestyle="--")
    axes[2].set_xticks(bx, ("final-action basin", "eventual success"))
    axes[2].set_ylim(0.4, max(0.75, axes[2].get_ylim()[1]))
    axes[2].set_ylabel("task/pool macro AUC")
    axes[2].set_title("Seed-disjoint readouts")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _save_candidate_arrays(tasks: list[dict[str, Any]], output: pathlib.Path) -> None:
    np.savez_compressed(
        output,
        task_names=np.asarray([task["name"] for task in tasks]),
        scene_ids=np.stack([task["scene_ids"] for task in tasks]),
        seed_ids=np.stack([task["seed_ids"] for task in tasks]),
        raw_metrics=np.stack([task["raw"] for task in tasks]),
        raw_metric_names=np.asarray(RAW_METRIC_NAMES),
        final_actions_standardized=np.stack([task["actions"] for task in tasks]),
        initial_live_noise=np.stack([task["noise"][..., :7] for task in tasks]),
        success=np.stack([task["success"] for task in tasks]),
    )


def main() -> int:
    args = parse_args()
    if (args.layer, args.denoise) != (5, 0):
        raise ValueError("the cross-task fixed cell is HB5/denoise-0")
    started = time.perf_counter()
    args.proxy_root = args.proxy_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tasks = load_tasks(args)

    summary = {
        "experiment": "raw_expert_amplitude_cross_task_v1",
        "status": "superseded_for_unweighted_raw_expert_question",
        "interpretation_scope": (
            "weighted branch amplitude only; not per-slot pre-gate expert norm"
        ),
        "known_method_errors": [
            "expert_mass and routed_rms were mislabeled as raw expert activation",
            "median-thresholded near-action pairs were mislabeled as action basins",
            "success readout held out seed columns but not states",
            "success macro pooled informative states instead of weighting tasks equally",
        ],
        "fixed_cell": {"hb_layer": args.layer, "denoise_round": args.denoise},
        "tasks": [task["name"] for task in tasks],
        "pools_per_task": int(tasks[0]["raw"].shape[0]),
        "candidates_per_pool": int(tasks[0]["raw"].shape[1]),
        "action_tokens": int(tasks[0]["raw"].shape[2]),
        "raw_metric_names": list(RAW_METRIC_NAMES),
        "raw_definition": (
            "RMS expert-output magnitude in checkpoint units; no division by "
            "hidden/shared/routed/post-MoE magnitude"
        ),
        "permutations": args.perms,
        "bootstrap_draws": args.bootstrap,
        "sources": {
            task["name"]: {
                "run": task["source_run"],
                "checkpoint_sha256": task["checkpoint_sha256"],
                "offline_reconstruction_validation": task["validation"],
            }
            for task in tasks
        },
        "raw_scale": summarize_raw_scale(tasks),
        "absolute_block_sensitivity": analyze_absolute_fragility(tasks, args, rng),
        "final_action_basin": analyze_action_basin(tasks, args, rng),
        "eventual_success": analyze_success(tasks, args, rng),
        "denoise_clock": analyze_denoise_clock(args.proxy_root, args.layer),
        "limitations": [
            "Offline fp32 reconstruction from stored fp16 hidden values is not runtime exact.",
            "No per-round flow latent or provisional action was captured.",
            "The rank-5-8 replacement is not continued through later flow blocks.",
            "Eventual success is a distant exploratory endpoint, not a causal commitment label.",
        ],
    }
    summary["elapsed_seconds"] = float(time.perf_counter() - started)
    _save_candidate_arrays(tasks, args.out_dir / "candidate_raw_amplitudes.npz")
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(_render_report(summary), encoding="utf-8")
    _plot_summary(summary, args.out_dir / "overview.png")
    print(f"wrote {args.out_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
