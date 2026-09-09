#!/usr/bin/env python3
"""Exploratory task-and-seed-held-out value probe on the five 16x32 runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from analyze_early_action_head import (  # noqa: E402
    discover_runs,
    load_action_stds,
    load_task,
    pool_array,
)


DEFAULT_HUB = REPO.parent / "VLA_MUI_HUB"
DEFAULT_OUT = HERE / "task_seed_heldout_increment.json"
SEED_FOLDS = 4
PCA_COMPONENTS = 12
RIDGE_ALPHA = 100.0
BOOTSTRAP_DRAWS = 20_000
RANDOM_SEED = 20260825
FEATURE_BLOCKS = ("noise", "action", "hidden", "route")
METHOD_PARTS = {
    "noise": ("noise",),
    "action": ("action",),
    "hidden": ("hidden",),
    "route": ("route",),
    "action_noise": ("action", "noise"),
    "action_noise_route": ("action", "noise", "route"),
    "action_noise_hidden": ("action", "noise", "hidden"),
    "action_noise_hidden_route": ("action", "noise", "hidden", "route"),
}
INCREMENT_PAIRS = {
    "route_over_action_noise": ("action_noise_route", "action_noise"),
    "route_over_action_noise_hidden": (
        "action_noise_hidden_route",
        "action_noise_hidden",
    ),
}


@dataclass(frozen=True)
class TaskGrid:
    task: str
    states: np.ndarray
    seeds: np.ndarray
    success: np.ndarray
    features: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-root", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite result: {result!r}")
    return result


def pool_standardize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=2, keepdims=True)
    scale = centered.std(axis=2, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def to_seed_folds(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return np.stack([array[:, fold::SEED_FOLDS] for fold in range(SEED_FOLDS)], axis=1)


def build_task_grid(task: Any, action_std: np.ndarray) -> TaskGrid:
    success, states, seeds = pool_array(task, "success")
    hidden, hidden_states, hidden_seeds = pool_array(task, "hidden_mean")
    route, route_states, route_seeds = pool_array(task, "router_layers")
    noise, noise_states, noise_seeds = pool_array(task, "flow_noise")
    actions, action_states, action_seeds = pool_array(task, "actions")
    for other_states, other_seeds in (
        (hidden_states, hidden_seeds),
        (route_states, route_seeds),
        (noise_states, noise_seeds),
        (action_states, action_seeds),
    ):
        if not np.array_equal(states, other_states) or not np.array_equal(seeds, other_seeds):
            raise RuntimeError(f"feature grids do not align in {task.task}")
    blocks = {
        "noise": noise[..., :7].reshape(16, 32, -1),
        "action": (actions / action_std[None, None, None, :]).reshape(16, 32, -1),
        "hidden": hidden,
        "route": route,
    }
    folded = {
        name: pool_standardize(to_seed_folds(values).astype(np.float64))
        for name, values in blocks.items()
    }
    return TaskGrid(
        task=task.task,
        states=states,
        seeds=seeds,
        success=to_seed_folds(success).astype(np.float64),
        features=folded,
    )


def reduce_block(
    train: np.ndarray,
    test: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-12)
    train_scaled = (train - mean) / scale
    test_scaled = (test - mean) / scale
    count = min(PCA_COMPONENTS, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(
        n_components=count,
        svd_solver="randomized",
        random_state=random_seed,
    )
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def fit_ridge(train: np.ndarray, target: np.ndarray, test: np.ndarray) -> np.ndarray:
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-12)
    train = (train - mean) / scale
    test = (test - mean) / scale
    design = np.column_stack([np.ones(len(train)), train])
    test_design = np.column_stack([np.ones(len(test)), test])
    penalty = RIDGE_ALPHA * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return test_design @ coefficient


def crossfit(grids: dict[str, TaskGrid], random_seed: int) -> dict[str, np.ndarray]:
    predictions = {
        task: {method: np.empty((16, 4, 8), dtype=np.float64) for method in METHOD_PARTS}
        for task in grids
    }
    tasks = sorted(grids)
    for task_index, held_task in enumerate(tasks):
        for held_fold in range(SEED_FOLDS):
            train_y = np.concatenate(
                [
                    grid.success[:, [fold for fold in range(4) if fold != held_fold]].reshape(-1)
                    for task, grid in grids.items()
                    if task != held_task
                ]
            )
            reduced = {}
            for block_index, block in enumerate(FEATURE_BLOCKS):
                train_x = np.concatenate(
                    [
                        grid.features[block][
                            :, [fold for fold in range(4) if fold != held_fold]
                        ].reshape(-1, grid.features[block].shape[-1])
                        for task, grid in grids.items()
                        if task != held_task
                    ]
                )
                test_x = grids[held_task].features[block][:, held_fold].reshape(
                    -1, grids[held_task].features[block].shape[-1]
                )
                reduced[block] = reduce_block(
                    train_x,
                    test_x,
                    random_seed + 100 * task_index + 10 * held_fold + block_index,
                )
            for method, parts in METHOD_PARTS.items():
                train_parts = [reduced[part][0] for part in parts]
                test_parts = [reduced[part][1] for part in parts]
                score = fit_ridge(
                    np.column_stack(train_parts),
                    train_y,
                    np.column_stack(test_parts),
                )
                predictions[held_task][method][:, held_fold] = score.reshape(16, 8)
            print(f"fit held task={held_task} seed_fold={held_fold}", flush=True)
    return predictions


def pool_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = float((positive[:, None] > negative[None, :]).sum())
    ties = float((positive[:, None] == negative[None, :]).sum())
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def task_method_arrays(grid: TaskGrid, score: np.ndarray) -> dict[str, Any]:
    selected_position = np.argmax(score, axis=-1)
    selected = np.take_along_axis(
        grid.success, selected_position[..., None], axis=-1
    )[..., 0]
    random = grid.success.mean(axis=-1)
    aucs = np.asarray(
        [
            pool_auc(grid.success[state, fold], score[state, fold])
            for state in range(16)
            for fold in range(4)
        ]
    )
    fold_seeds = np.stack([grid.seeds[fold::4] for fold in range(4)])
    selected_seed = np.take_along_axis(
        np.broadcast_to(fold_seeds[np.newaxis], (16, 4, 8)),
        selected_position[..., None],
        axis=-1,
    )[..., 0]
    return {
        "selected": selected,
        "random": random,
        "state_delta": (selected - random).mean(axis=1),
        "pool_auc": aucs,
        "selected_seed": selected_seed,
    }


def bootstrap_ci(
    values_by_task: dict[str, np.ndarray],
    draws: int,
    rng: np.random.Generator,
) -> tuple[dict[str, list[float]], list[float]]:
    distributions = {}
    for task, values in values_by_task.items():
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        distributions[task] = values[indices].mean(axis=1)
    macro = np.stack([distributions[task] for task in sorted(distributions)], axis=1).mean(axis=1)
    task_ci = {
        task: [finite(value) for value in np.quantile(distribution, [0.025, 0.975])]
        for task, distribution in distributions.items()
    }
    return task_ci, [finite(value) for value in np.quantile(macro, [0.025, 0.975])]


def summarize(
    grids: dict[str, TaskGrid],
    predictions: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    evaluated = {
        task: {
            method: task_method_arrays(grid, predictions[task][method])
            for method in METHOD_PARTS
        }
        for task, grid in grids.items()
    }
    rng = np.random.default_rng(args.seed)
    methods = {}
    for method in METHOD_PARTS:
        state_delta = {
            task: evaluated[task][method]["state_delta"] for task in grids
        }
        task_ci, macro_ci = bootstrap_ci(state_delta, args.bootstrap, rng)
        per_task = {}
        for task in sorted(grids):
            result = evaluated[task][method]
            seeds = result["selected_seed"].reshape(-1).astype(int).tolist()
            finite_auc = result["pool_auc"][np.isfinite(result["pool_auc"])]
            per_task[task] = {
                "selected_success_rate": finite(result["selected"].mean()),
                "random_pool_mean": finite(result["random"].mean()),
                "delta_vs_random": finite(result["state_delta"].mean()),
                "state_bootstrap_ci95": task_ci[task],
                "mean_pool_auc": finite(finite_auc.mean()) if len(finite_auc) else None,
                "evaluable_auc_pools": int(len(finite_auc)),
                "unique_selected_seeds": len(set(seeds)),
                "max_seed_share": finite(max(Counter(seeds).values()) / len(seeds)),
            }
        methods[method] = {
            "per_task": per_task,
            "task_macro_selected_success_rate": finite(
                np.mean([row["selected_success_rate"] for row in per_task.values()])
            ),
            "task_macro_random_pool_mean": finite(
                np.mean([row["random_pool_mean"] for row in per_task.values()])
            ),
            "task_macro_delta_vs_random": finite(
                np.mean([row["delta_vs_random"] for row in per_task.values()])
            ),
            "stratified_state_bootstrap_ci95": macro_ci,
            "mean_pool_auc_over_evaluable_pools": finite(
                np.mean(
                    np.concatenate(
                        [
                            evaluated[task][method]["pool_auc"][
                                np.isfinite(evaluated[task][method]["pool_auc"])
                            ]
                            for task in grids
                        ]
                    )
                )
            ),
        }

    increments = {}
    for name, (with_route, baseline) in INCREMENT_PAIRS.items():
        differences = {
            task: (
                evaluated[task][with_route]["selected"]
                - evaluated[task][baseline]["selected"]
            ).mean(axis=1)
            for task in grids
        }
        task_ci, macro_ci = bootstrap_ci(differences, args.bootstrap, rng)
        per_task = {
            task: {
                "selected_success_increment": finite(values.mean()),
                "state_bootstrap_ci95": task_ci[task],
                "pool_auc_increment": finite(
                    methods[with_route]["per_task"][task]["mean_pool_auc"]
                    - methods[baseline]["per_task"][task]["mean_pool_auc"]
                )
                if methods[with_route]["per_task"][task]["mean_pool_auc"] is not None
                else None,
            }
            for task, values in differences.items()
        }
        increments[name] = {
            "with_route": with_route,
            "baseline": baseline,
            "per_task": per_task,
            "task_macro_selected_success_increment": finite(
                np.mean([row["selected_success_increment"] for row in per_task.values()])
            ),
            "stratified_state_bootstrap_ci95": macro_ci,
            "pooled_evaluable_auc_increment": finite(
                methods[with_route]["mean_pool_auc_over_evaluable_pools"]
                - methods[baseline]["mean_pool_auc_over_evaluable_pools"]
            ),
        }

    return {
        "schema": "himoe-task-seed-heldout-increment-audit-v1",
        "status": "posthoc_exploratory",
        "protocol": {
            "source": "five right-16x32 runs; first policy query per episode",
            "tasks": 5,
            "states_per_task": 16,
            "candidate_pools": "four fixed modulo-four K8 folds per K32 state",
            "test_exclusion": (
                "complete held task and the held fold's eight seed identities absent from training"
            ),
            "features": {
                "noise": "initial flow noise, ten tokens x seven live dimensions",
                "action": "complete first 10x7 action chunk, checkpoint-std normalized",
                "hidden": "d0 HB2-5 gate-input hidden, mean over layers and action tokens",
                "route": "d0 HB2-5 normalized router probabilities, layer-resolved token mean",
            },
            "pool_preprocessing": "each feature column standardized inside its test/train K8 pool",
            "block_reduction": (
                "train-only StandardScaler + randomized PCA(12), deterministic per outer fold"
            ),
            "probe": f"linear ridge alpha={RIDGE_ALPHA}",
            "target": "eventual episode success",
            "bootstrap": "states resampled independently within task",
            "bootstrap_draws": args.bootstrap,
            "random_seed": args.seed,
        },
        "methods": methods,
        "route_increments": increments,
        "limitations": [
            "Post-hoc model family and fixed hyperparameters were not preregistered.",
            "Eventual success is a seed-stream outcome, not a first-chunk CRN value label.",
            "Only 80 state pools and five tasks are available; one task is all-success.",
            "Pool standardization is label-free but transductive over all eight test candidates.",
            "Hidden is a pre-router gate input and route is its deterministic downstream function; this is predictive, not causal, increment.",
        ],
    }


def report_markdown(summary: dict[str, Any]) -> str:
    labels = {
        "noise": "noise",
        "action": "action",
        "hidden": "hidden",
        "route": "route",
        "action_noise": "action + noise",
        "action_noise_route": "action + noise + route",
        "action_noise_hidden": "action + noise + hidden",
        "action_noise_hidden_route": "action + noise + hidden + route",
    }
    lines = [
        "# Task + seed-held-out incremental audit",
        "",
        "This is an exploratory shadow probe. Every outer prediction excludes the "
        "complete test task and the test K8 seed identities.",
        "",
        "| features | K8 AUC | selected success | random mean | delta | state-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, row in summary["methods"].items():
        ci = row["stratified_state_bootstrap_ci95"]
        lines.append(
            f"| {labels[method]} | {row['mean_pool_auc_over_evaluable_pools']:.3f} | "
            f"{row['task_macro_selected_success_rate']:.3f} | "
            f"{row['task_macro_random_pool_mean']:.3f} | "
            f"{row['task_macro_delta_vs_random']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines += ["", "## Conditional route increments", ""]
    for name, row in summary["route_increments"].items():
        ci = row["stratified_state_bootstrap_ci95"]
        lines.append(
            f"- `{name}`: selected-success {row['task_macro_selected_success_increment']:+.3f} "
            f"(95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]); pooled evaluable-pool "
            f"AUC increment {row['pooled_evaluable_auc_increment']:+.3f}."
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The label is eventual episode success from one rollout per seed. The seed also "
        "controls later replanning noise, so conditional predictive increment cannot be "
        "read as route information about the causal value of the first action chunk.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")
    hub = args.hub_root.resolve()
    cache_root = hub / "cache" / "HiMoE-VLA"
    action_stds = load_action_stds(REPO.parent)
    grids = {}
    canonical_seeds = None
    for run in discover_runs(hub):
        suite = run.relative_to(cache_root).parts[0]
        print(f"loading {run.relative_to(cache_root)}", flush=True)
        task = load_task(run, cache_root, action_stds[suite])
        grid = build_task_grid(task, action_stds[suite])
        if canonical_seeds is None:
            canonical_seeds = grid.seeds
        elif not np.array_equal(canonical_seeds, grid.seeds):
            raise RuntimeError("tasks do not share the same ordered seed grid")
        grids[grid.task] = grid
    predictions = crossfit(grids, args.seed)
    summary = summarize(grids, predictions, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = args.out.with_suffix(".md")
    report.write_text(report_markdown(summary), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
