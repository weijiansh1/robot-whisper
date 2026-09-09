#!/usr/bin/env python3
"""Frozen cross-task test of the intra-query MoE Markov selector.

Three released-left LIBERO-Goal captures each contain one fixed simulator
snapshot and the same 64 flow-noise seeds.  For every held task and K8 seed
fold, both the layer codebooks and success/failure Markov counts are fitted on
the other two tasks after removing those eight seed IDs.  The frozen split
model then scores:

* the held task in the original ``within64`` capture; and
* the matching ``objstate`` recapture, which is never used for fitting.

Thus every score holds out the complete task and candidate seed IDs.  This is a
transfer test across three snapshots, not a per-query closed-loop intervention.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zarr

from analyze_intraquery_markov import (
    HB_LAYERS,
    METHODS,
    N_ACTION_TOKENS,
    N_DENOISE,
    N_EXPERTS,
    TOP_K_VALUES,
    assign_layer_codebooks,
    finite,
    fit_layer_codebooks,
    fit_route_markov_model,
    normalize_router_probabilities,
    sequence_log_likelihoods,
    topk_tie_weights,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "analysis" / "intraquery-markov-transfer"
N_SEEDS = 64
SEED_FOLDS = 8
CANDIDATES = N_SEEDS // SEED_FOLDS
PRIMARY_K = 16
PRIMARY_ROUNDS = 3
PRIMARY_TOP_K = 2
CHECKPOINT_SHA256 = "98ee29d09d1855716e341532a3d0f76068de6fa131f8fe16ffd52f982df1b953"
LAYOUT = "released-left"


@dataclass(frozen=True)
class TaskSpec:
    key: str
    name: str
    task_id: int
    init_state_id: int
    development_server: Path
    development_client: Path
    replication_server: Path
    replication_client: Path


@dataclass(frozen=True)
class Capture:
    task: TaskSpec
    cohort: str
    seeds: np.ndarray
    success: np.ndarray
    routes: np.ndarray
    index_grid: np.ndarray
    probability_mass_max_error: float


TASKS = (
    TaskSpec(
        key="goal-t0-s24",
        name="open_the_middle_drawer_of_the_cabinet",
        task_id=0,
        init_state_id=24,
        development_server=HERE / "runs" / "within64-s24",
        development_client=HERE / "runs" / "within64-s24-client",
        replication_server=HERE / "runs" / "objstate-t0s24",
        replication_client=HERE / "runs" / "objstate-t0s24-client",
    ),
    TaskSpec(
        key="goal-t1-s19",
        name="put_the_bowl_on_the_stove",
        task_id=1,
        init_state_id=19,
        development_server=HERE / "runs" / "within64-t1s19",
        development_client=HERE / "runs" / "within64-t1s19-client",
        replication_server=HERE / "runs" / "objstate-t1s19",
        replication_client=HERE / "runs" / "objstate-t1s19-client",
    ),
    TaskSpec(
        key="goal-t3-s0",
        name="open_the_top_drawer_and_put_the_bowl_inside",
        task_id=3,
        init_state_id=0,
        development_server=HERE / "runs" / "within64-t3s0",
        development_client=HERE / "runs" / "within64-t3s0-client",
        replication_server=HERE / "runs" / "objstate-t3s0",
        replication_client=HERE / "runs" / "objstate-t3s0-client",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clusters", type=int, default=PRIMARY_K)
    parser.add_argument("--rounds", type=int, default=PRIMARY_ROUNDS)
    parser.add_argument("--top-k", type=int, default=PRIMARY_TOP_K)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--position-alpha", type=float, default=10.0)
    parser.add_argument("--permutations", type=int, default=19_999)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def candidate_grid(seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(seeds, dtype=np.int64)
    unique = np.sort(np.unique(values))
    if len(values) != N_SEEDS or len(unique) != N_SEEDS:
        raise ValueError("expected 64 unique flow-noise seeds")
    rank = {int(seed): index for index, seed in enumerate(unique)}
    ranks = np.asarray([rank[int(seed)] for seed in values], dtype=np.int64)
    folds = ranks % SEED_FOLDS
    positions = ranks // SEED_FOLDS
    grid = np.full((SEED_FOLDS, CANDIDATES), -1, dtype=np.int64)
    for fold in range(SEED_FOLDS):
        indices = np.flatnonzero(folds == fold)
        if len(indices) != CANDIDATES:
            raise ValueError("seed fold is not K8")
        grid[fold, positions[indices]] = indices
    if np.any(grid < 0) or len(np.unique(grid)) != N_SEEDS:
        raise ValueError("K8 grid does not cover all seeds")
    return grid, folds


def _first_route_rows(summaries: list[dict[str, Any]], trace_length: int) -> np.ndarray:
    calls = np.asarray([int(row["inference_calls"]) for row in summaries], dtype=np.int64)
    if np.any(calls <= 0) or int(calls.sum()) != trace_length:
        raise RuntimeError("client inference counts and route trace disagree")
    return np.r_[0, np.cumsum(calls)[:-1]]


def load_capture(task: TaskSpec, cohort: str) -> Capture:
    if cohort == "development":
        server, client = task.development_server, task.development_client
    elif cohort == "replication":
        server, client = task.replication_server, task.replication_client
    else:
        raise ValueError("unknown cohort %r" % cohort)

    summaries = sorted(
        json.loads((client / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    if len(summaries) != N_SEEDS:
        raise RuntimeError("%s does not contain 64 episodes" % client)
    if {int(row["init_state_id"]) for row in summaries} != {task.init_state_id}:
        raise RuntimeError("capture has the wrong initial state")
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries], dtype=np.int64)
    success = np.asarray([bool(row["success"]) for row in summaries], dtype=bool)
    grid, _folds = candidate_grid(seeds)

    metadata = json.loads((client / "server_metadata.json").read_text())
    if metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint differs in %s" % client)
    if metadata.get("libero_wrist_layout") != LAYOUT:
        raise RuntimeError("wrist layout differs in %s" % client)
    if tuple(metadata.get("routing_hb_layer_indices", ())) != HB_LAYERS:
        raise RuntimeError("HB layers differ in %s" % client)

    group = zarr.open_group(str(server / "routes.zarr"), mode="r")
    rows = _first_route_rows(summaries, group["hb_router_probs"].shape[0])
    raw = np.asarray(
        group["hb_router_probs"].oindex[rows, :, :, 1:, :], dtype=np.float32
    )
    expected = (N_SEEDS, len(HB_LAYERS), N_DENOISE, N_ACTION_TOKENS, N_EXPERTS)
    if raw.shape != expected:
        raise RuntimeError("unexpected route shape %s" % (raw.shape,))
    mass_error = float(np.max(np.abs(raw.sum(axis=-1) - 1.0)))
    return Capture(
        task=task,
        cohort=cohort,
        seeds=seeds,
        success=success,
        routes=normalize_router_probabilities(raw),
        index_grid=grid,
        probability_mass_max_error=mass_error,
    )


def _empty_score_table() -> dict[str, np.ndarray]:
    return {
        method: np.full((N_SEEDS, N_DENOISE), np.nan, dtype=np.float64)
        for method in METHODS
    }


def crossfit_transfer_scores(
    development: list[Capture],
    replication: list[Capture],
    clusters: int,
    alpha: float,
    position_alpha: float,
    random_seed: int,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, np.ndarray]]],
    list[dict[str, Any]],
]:
    if [row.task.key for row in development] != [row.task.key for row in replication]:
        raise ValueError("development and replication tasks are not aligned")
    scores = {
        cohort: {capture.task.key: _empty_score_table() for capture in captures}
        for cohort, captures in (
            ("development", development),
            ("replication", replication),
        )
    }
    true_logp = {
        cohort: {capture.task.key: _empty_score_table() for capture in captures}
        for cohort, captures in (
            ("development", development),
            ("replication", replication),
        )
    }
    diagnostics: list[dict[str, Any]] = []

    for held_axis, held_capture in enumerate(development):
        held_grid = held_capture.index_grid
        for fold in range(SEED_FOLDS):
            held_indices = held_grid[fold]
            held_seeds = held_capture.seeds[held_indices]
            train_routes = []
            train_success = []
            source_tasks = []
            for source_axis, source in enumerate(development):
                if source_axis == held_axis:
                    continue
                take = ~np.isin(source.seeds, held_seeds)
                if np.any(np.isin(source.seeds[take], held_seeds)):
                    raise RuntimeError("held seed leaked into source routes")
                train_routes.append(source.routes[take])
                train_success.append(source.success[take])
                source_tasks.append(source.task.key)
            route_train = np.concatenate(train_routes)
            outcome_train = np.concatenate(train_success)
            if len(np.unique(outcome_train)) != 2:
                raise RuntimeError("transfer split lacks an outcome class")

            codebooks = fit_layer_codebooks(
                route_train,
                clusters,
                random_seed + held_axis * 1000 + fold * 20,
            )
            models = {
                label: fit_route_markov_model(
                    codebooks.codes[outcome_train == label],
                    clusters,
                    alpha,
                    position_alpha,
                )
                for label in (False, True)
            }

            for cohort, captures in (
                ("development", development),
                ("replication", replication),
            ):
                test_capture = captures[held_axis]
                test_indices = test_capture.index_grid[fold]
                if not np.array_equal(
                    np.sort(test_capture.seeds[test_indices]), np.sort(held_seeds)
                ):
                    raise RuntimeError("cohorts do not share the held seed fold")
                codes = assign_layer_codebooks(
                    test_capture.routes[test_indices], codebooks.centers
                )
                logp = {
                    label: sequence_log_likelihoods(models[label], codes)
                    for label in (False, True)
                }
                labels = test_capture.success[test_indices]
                for method in METHODS:
                    scores[cohort][test_capture.task.key][method][test_indices] = (
                        logp[True][method] - logp[False][method]
                    )
                    true_logp[cohort][test_capture.task.key][method][test_indices] = (
                        np.where(labels[:, None], logp[True][method], logp[False][method])
                    )

            diagnostics.append(
                {
                    "held_task": held_capture.task.key,
                    "held_fold": fold,
                    "held_seeds": [int(seed) for seed in np.sort(held_seeds)],
                    "source_tasks": source_tasks,
                    "train_examples": len(outcome_train),
                    "train_successes": int(outcome_train.sum()),
                    "train_failures": int((~outcome_train).sum()),
                    "codebook_inertia_mean": finite(codebooks.inertia.mean()),
                }
            )

    for table in (scores, true_logp):
        for cohort in table.values():
            for task in cohort.values():
                for values in task.values():
                    if np.any(~np.isfinite(values)):
                        raise RuntimeError("cross-task scoring left non-finite entries")
    return scores, true_logp, diagnostics


def _pool_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = float((positive[:, None] > negative[None, :]).sum())
    ties = float((positive[:, None] == negative[None, :]).sum())
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def evaluate_fixed_scores(
    capture: Capture, score: np.ndarray, top_k: int
) -> tuple[dict[str, Any], np.ndarray]:
    score_grid = np.asarray(score, dtype=np.float64)[capture.index_grid]
    label_grid = capture.success[capture.index_grid]
    weights = np.stack([topk_tie_weights(row, top_k) for row in score_grid])
    selected = (weights * label_grid).sum(axis=-1) / top_k
    baseline = label_grid.mean(axis=-1)
    pool_auc = np.asarray(
        [_pool_auc(labels, values) for labels, values in zip(label_grid, score_grid)]
    )
    seed_grid = capture.seeds[capture.index_grid]
    seed_mass = np.asarray(
        [weights[seed_grid == seed].sum() for seed in np.unique(capture.seeds)]
    )
    probability = seed_mass / seed_mass.sum()
    positive = probability[probability > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return (
        {
            "top_k": top_k,
            "selected_success_rate": finite(selected.mean()),
            "exact_random_success_rate": finite(baseline.mean()),
            "delta_vs_random": finite((selected - baseline).mean()),
            "mean_pool_auc": finite(np.nanmean(pool_auc)),
            "evaluable_auc_pools": int(np.isfinite(pool_auc).sum()),
            "oracle_selected_success_rate": finite(
                (np.minimum(label_grid.sum(axis=-1), top_k) / top_k).mean()
            ),
            "seed_concentration": {
                "positive_weight_seeds": int(np.count_nonzero(seed_mass)),
                "effective_seed_count": finite(math.exp(entropy)),
                "maximum_expected_seed_share": finite(probability.max()),
            },
        },
        weights,
    )


def coherent_seed_permutation(
    label_grids: np.ndarray,
    left_weights: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    right_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Permute K8 seed columns per fold, reusing each order across tasks."""

    labels = np.asarray(label_grids, dtype=bool)
    left = np.asarray(left_weights, dtype=np.float64)
    if labels.shape != left.shape or labels.ndim != 3:
        raise ValueError("expected aligned [task,fold,candidate] grids")
    top_k = float(left.sum(axis=-1)[0, 0])
    if right_weights is None:
        observed = float(((left * labels).sum(axis=-1) / top_k - labels.mean(axis=-1)).mean())
        contrast = left
    else:
        right = np.asarray(right_weights, dtype=np.float64)
        if right.shape != left.shape or not np.allclose(right.sum(axis=-1), top_k):
            raise ValueError("selector weight grids differ")
        contrast = left - right
        observed = float((contrast * labels).sum(axis=-1).mean() / top_k)

    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        permuted = np.empty_like(labels)
        for fold in range(SEED_FOLDS):
            order = rng.permutation(CANDIDATES)
            permuted[:, fold] = labels[:, fold, order]
        if right_weights is None:
            null[draw] = float(
                ((left * permuted).sum(axis=-1) / top_k - permuted.mean(axis=-1)).mean()
            )
        else:
            null[draw] = float((contrast * permuted).sum(axis=-1).mean() / top_k)
    return {
        "observed": finite(observed),
        "one_sided_p": finite(
            (np.count_nonzero(null >= observed - 1e-15) + 1) / (draws + 1)
        ),
        "null_mean": finite(null.mean()),
        "null_95": [finite(value) for value in np.percentile(null, [2.5, 97.5])],
        "draws": draws,
        "scheme": "one K8 column permutation per seed fold, reused across held tasks",
    }


def predictive_summary(true_logp: dict[str, np.ndarray]) -> dict[str, Any]:
    events = N_DENOISE * len(HB_LAYERS)
    output = {
        method: {
            "full_path_cross_entropy_bits_per_event": finite(
                -values[:, -1].mean() / (events * math.log(2.0))
            )
        }
        for method, values in true_logp.items()
    }
    for kind in ("periodic", "position"):
        markov = true_logp["markov_" + kind][:, -1]
        occupancy = true_logp["occupancy_" + kind][:, -1]
        output["markov_" + kind]["gain_over_occupancy_bits_per_transition"] = finite(
            (markov - occupancy).mean() / ((events - 1) * math.log(2.0))
        )
    return output


def evaluate_cohort(
    captures: list[Capture],
    scores: dict[str, dict[str, np.ndarray]],
    true_logp: dict[str, dict[str, np.ndarray]],
    rounds: int,
    top_k: int,
    permutations: int,
    rng: np.random.Generator,
    predictive_function: Callable[[dict[str, np.ndarray]], dict[str, Any]] = predictive_summary,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_results: dict[str, Any] = {}
    primary_weights = []
    occupancy_weights = []
    label_grids = []
    internal: dict[str, Any] = {}
    for capture in captures:
        methods: dict[str, Any] = {}
        task_internal: dict[str, Any] = {}
        for method in METHODS:
            methods[method] = {}
            task_internal[method] = {}
            for prefix in range(1, N_DENOISE + 1):
                methods[method][str(prefix)] = {}
                task_internal[method][str(prefix)] = {}
                for selected_k in TOP_K_VALUES:
                    row, weights = evaluate_fixed_scores(
                        capture, scores[capture.task.key][method][:, prefix - 1], selected_k
                    )
                    methods[method][str(prefix)][str(selected_k)] = row
                    task_internal[method][str(prefix)][str(selected_k)] = weights

        primary = methods["markov_periodic"][str(rounds)][str(top_k)]
        markov_weights = task_internal["markov_periodic"][str(rounds)][str(top_k)]
        zero = methods["occupancy_periodic"][str(rounds)][str(top_k)]
        zero_weights = task_internal["occupancy_periodic"][str(rounds)][str(top_k)]
        labels = capture.success[capture.index_grid][None, ...]
        primary["permutation_vs_random"] = coherent_seed_permutation(
            labels, markov_weights[None, ...], permutations, rng
        )
        primary["permutation_vs_periodic_occupancy"] = coherent_seed_permutation(
            labels,
            markov_weights[None, ...],
            permutations,
            rng,
            zero_weights[None, ...],
        )
        primary["selected_success_difference_vs_periodic_occupancy"] = finite(
            primary["selected_success_rate"] - zero["selected_success_rate"]
        )
        task_results[capture.task.key] = {
            "task_name": capture.task.name,
            "task_id": capture.task.task_id,
            "init_state_id": capture.task.init_state_id,
            "successes": int(capture.success.sum()),
            "success_rate": finite(capture.success.mean()),
            "probability_mass_max_error": capture.probability_mass_max_error,
            "selection": methods,
            "predictive": predictive_function(true_logp[capture.task.key]),
        }
        primary_weights.append(markov_weights)
        occupancy_weights.append(zero_weights)
        label_grids.append(labels[0])
        internal[capture.task.key] = task_internal

    labels = np.stack(label_grids)
    markov = np.stack(primary_weights)
    occupancy = np.stack(occupancy_weights)
    task_primary = [
        task_results[capture.task.key]["selection"]["markov_periodic"][str(rounds)][
            str(top_k)
        ]
        for capture in captures
    ]
    task_zero = [
        task_results[capture.task.key]["selection"]["occupancy_periodic"][str(rounds)][
            str(top_k)
        ]
        for capture in captures
    ]
    macro = {
        "selected_success_rate": finite(
            np.mean([row["selected_success_rate"] for row in task_primary])
        ),
        "exact_random_success_rate": finite(
            np.mean([row["exact_random_success_rate"] for row in task_primary])
        ),
        "delta_vs_random": finite(np.mean([row["delta_vs_random"] for row in task_primary])),
        "mean_pool_auc": finite(np.mean([row["mean_pool_auc"] for row in task_primary])),
        "periodic_occupancy_delta_vs_random": finite(
            np.mean([row["delta_vs_random"] for row in task_zero])
        ),
        "markov_selected_success_difference_vs_occupancy": finite(
            np.mean(
                [
                    left["selected_success_rate"] - right["selected_success_rate"]
                    for left, right in zip(task_primary, task_zero)
                ]
            )
        ),
        "task_deltas": {
            capture.task.key: task_results[capture.task.key]["selection"][
                "markov_periodic"
            ][str(rounds)][str(top_k)]["delta_vs_random"]
            for capture in captures
        },
        "permutation_vs_random": coherent_seed_permutation(
            labels, markov, permutations, rng
        ),
        "permutation_vs_periodic_occupancy": coherent_seed_permutation(
            labels, markov, permutations, rng, occupancy
        ),
    }
    return {"tasks": task_results, "macro": macro}, internal


def _fmt(value: float, signed: bool = False) -> str:
    return ("%+.3f" if signed else "%.3f") % value


def _primary(task: dict[str, Any], protocol: dict[str, Any], method: str) -> dict[str, Any]:
    return task["selection"][method][str(protocol["rounds"])][str(protocol["top_k"])]


def render_report(summary: dict[str, Any]) -> str:
    protocol = summary["protocol"]
    development = summary["cohorts"]["development"]
    replication = summary["cohorts"]["replication"]
    lines = [
        "# 其他 Goal 任务：query 内 Markov 迁移测试",
        "",
        "## 协议",
        "",
        (
            "使用 3 个 `released-left` Goal 任务，每个任务固定一个 simulator "
            "snapshot、64 个共同 seed。每次留出完整测试任务，并把正在排序的 "
            "8 个 seed 从另外两个训练任务中删除；逐层 K16 codebook 也只在该"
            "训练 split 上拟合。测试池因此同时 task-disjoint 和 seed-disjoint。"
        ),
        "",
        (
            "规则保持不变：一次 query 为 `10 denoise × 8 HB layer = 80` 个"
            "微事件，看前 3 轮后按 success/failure periodic-Markov 路径似然比"
            "选 top-2。`within64` 是开发 capture；`objstate` 是相同任务、状态"
            "和 seed 的独立重采集，仅复用冻结 split 模型，不参与任何拟合。"
        ),
        "",
    ]
    for cohort_key, title in (
        ("development", "原始 within64"),
        ("replication", "objstate 重采集"),
    ):
        cohort = summary["cohorts"][cohort_key]
        lines.extend(
            [
                "## %s" % title,
                "",
                "| held-out 任务 | 随机成功率 | 零阶占用 delta | Markov 成功率 | Markov delta | AUC | p vs random | Markov - 占用 | p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for task_key, task in cohort["tasks"].items():
            markov = _primary(task, protocol, "markov_periodic")
            occupancy = _primary(task, protocol, "occupancy_periodic")
            lines.append(
                "| `%s` | %s | %s | %s | %s | %s | %.4f | %s | %.4f |"
                % (
                    task_key,
                    _fmt(markov["exact_random_success_rate"]),
                    _fmt(occupancy["delta_vs_random"], True),
                    _fmt(markov["selected_success_rate"]),
                    _fmt(markov["delta_vs_random"], True),
                    _fmt(markov["mean_pool_auc"]),
                    markov["permutation_vs_random"]["one_sided_p"],
                    _fmt(
                        markov["selected_success_difference_vs_periodic_occupancy"],
                        True,
                    ),
                    markov["permutation_vs_periodic_occupancy"]["one_sided_p"],
                )
            )
        macro = cohort["macro"]
        lines.extend(
            [
                "",
                (
                    "三任务宏平均：Markov 相对随机 `%s`（一致 seed 置换 "
                    "`p=%.4f`），零阶占用相对随机 `%s`；Markov 减占用 `%s` "
                    "（`p=%.4f`）。"
                    % (
                        _fmt(macro["delta_vs_random"], True),
                        macro["permutation_vs_random"]["one_sided_p"],
                        _fmt(macro["periodic_occupancy_delta_vs_random"], True),
                        _fmt(
                            macro[
                                "markov_selected_success_difference_vs_occupancy"
                            ],
                            True,
                        ),
                        macro["permutation_vs_periodic_occupancy"]["one_sided_p"],
                    )
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 前缀敏感性",
            "",
            "| denoise 轮数 | within64 Markov macro delta | objstate Markov macro delta |",
            "|---:|---:|---:|",
        ]
    )
    for rounds in (1, 2, 3, 5, 10):
        values = []
        for cohort in (development, replication):
            values.append(
                np.mean(
                    [
                        task["selection"]["markov_periodic"][str(rounds)][
                            str(protocol["top_k"])
                        ]["delta_vs_random"]
                        for task in cohort["tasks"].values()
                    ]
                )
            )
        lines.append(
            "| %d | %s | %s |"
            % (rounds, _fmt(values[0], True), _fmt(values[1], True))
        )

    dev_macro = development["macro"]
    rep_macro = replication["macro"]
    replicated = (
        dev_macro["delta_vs_random"] > 0.0
        and rep_macro["delta_vs_random"] > 0.0
        and dev_macro["permutation_vs_random"]["one_sided_p"] < 0.05
        and rep_macro["permutation_vs_random"]["one_sided_p"] < 0.05
        and rep_macro["markov_selected_success_difference_vs_occupancy"] > 0.0
        and rep_macro["permutation_vs_periodic_occupancy"]["one_sided_p"] < 0.05
    )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "跨任务 top-2 规则%s在独立重采集中同时满足‘优于随机且优于"
                "零阶占用’。开发/复验的 Markov 宏平均 delta 分别为 `%s` 和 "
                "`%s`；复验 Markov 减占用为 `%s`。"
                % (
                    "" if replicated else "未",
                    _fmt(dev_macro["delta_vs_random"], True),
                    _fmt(rep_macro["delta_vs_random"], True),
                    _fmt(
                        rep_macro[
                            "markov_selected_success_difference_vs_occupancy"
                        ],
                        True,
                    ),
                )
            ),
            "",
            (
                "因此，其他任务数据可以检验该思路，但当前证据仍不足以支持"
                "通用 selector。这里每个任务只有一个 snapshot；置换 p 只反映"
                "这些 K8 池内的关联，不替代更多状态上的闭环分叉实验。"
            ),
            "",
            (
                "30-task corpus 没有纳入：每个观测只有一个 seed，无法形成同观测"
                "候选池。objstate 虽使用相同配置和 seed，但流水线不是 bit-exact，"
                "其 outcome 差异正是本报告把它当重采集而非重复行的原因。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.clusters < 2:
        raise ValueError("clusters must be at least two")
    if args.rounds < 1 or args.rounds > N_DENOISE:
        raise ValueError("rounds must be between one and ten")
    if args.top_k not in TOP_K_VALUES:
        raise ValueError("top-k must be one of %s" % (TOP_K_VALUES,))
    if args.permutations < 100:
        raise ValueError("permutations must be at least 100")

    development = [load_capture(task, "development") for task in TASKS]
    replication = [load_capture(task, "replication") for task in TASKS]
    scores, true_logp, split_diagnostics = crossfit_transfer_scores(
        development,
        replication,
        args.clusters,
        args.alpha,
        args.position_alpha,
        args.seed,
    )
    rng = np.random.default_rng(args.seed)
    cohorts = {}
    for cohort_name, captures in (
        ("development", development),
        ("replication", replication),
    ):
        cohorts[cohort_name], _internal = evaluate_cohort(
            captures,
            scores[cohort_name],
            true_logp[cohort_name],
            args.rounds,
            args.top_k,
            args.permutations,
            rng,
        )

    summary = {
        "schema": "himoe-intraquery-markov-transfer-v1",
        "protocol": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "wrist_layout": LAYOUT,
            "tasks": [task.key for task in TASKS],
            "development": "within64 captures",
            "replication": "objstate recaptures; never used in fitting",
            "candidate_pool": "eight K8 folds from common seeds 1000..1063",
            "held_out": "complete target task and the eight candidate seed IDs",
            "codebook_fit": "source tasks and non-held seeds only; no labels",
            "state": "layer-specific Hellinger K-means cell of all action-token routes",
            "clusters": args.clusters,
            "rounds": args.rounds,
            "micro_events_observed": args.rounds * len(HB_LAYERS),
            "top_k": args.top_k,
            "method": "success/failure periodic-Markov path likelihood ratio",
            "alpha": args.alpha,
            "position_alpha": args.position_alpha,
            "permutations": args.permutations,
            "random_seed": args.seed,
        },
        "split_diagnostics": split_diagnostics,
        "cohorts": cohorts,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = build(args)
    for cohort_name, cohort in summary["cohorts"].items():
        macro = cohort["macro"]
        print(
            "%s macro delta=%+.4f p=%.4f markov-minus-occupancy=%+.4f"
            % (
                cohort_name,
                macro["delta_vs_random"],
                macro["permutation_vs_random"]["one_sided_p"],
                macro["markov_selected_success_difference_vs_occupancy"],
            )
        )
    print("wrote %s" % args.out_dir.resolve())


if __name__ == "__main__":
    main()
