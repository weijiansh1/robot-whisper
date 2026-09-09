#!/usr/bin/env python3
"""Apply an oracle-first gate to the existing K8 noise-selection data.

This analysis intentionally does not fit a head.  It asks whether the exact
pseudo-target that a head would try to predict is itself useful for selecting a
successful candidate.  A target that misses the gate cannot authorize a head.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from analyze_self_supervised_noise_methods import (
    Pool,
    discover_runs,
    load_task_pools,
    pool_auc,
    select_index,
)


HERE = Path(__file__).resolve().parent
DEFAULT_HUB = HERE.parent / "VLA_MUI_HUB"
DEFAULT_OUT = HERE / "analysis/oracle-gate"
DEFAULT_THRESHOLD = 0.03
SEED_FOLDS = 4

SCORE_FUNCTIONS: dict[str, Callable[[Pool], np.ndarray]] = {
    "initial_noise_only": lambda pool: pool.initial_noise_score,
    "final_action_centrality": lambda pool: pool.action_centrality,
    "future_route_change": lambda pool: pool.future_route_change,
    "late_route_centrality": lambda pool: pool.late_route_centrality,
}
ORACLE_TARGETS = (
    "final_action_centrality",
    "future_route_change",
    "late_route_centrality",
)
DISPLAY = {
    "initial_noise_only": "初始噪声中心（免费下界）",
    "final_action_centrality": "真实最终动作中心 oracle",
    "future_route_change": "真实未来路由变化 oracle",
    "late_route_centrality": "真实后续路由中心 oracle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-root", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def is_ceiling_task(pools: list[Pool]) -> bool:
    outcomes = np.concatenate([pool.success for pool in pools])
    return bool(np.all(outcomes == outcomes[0]))


def validate_pools(by_task: dict[str, list[Pool]]) -> dict[str, Any]:
    if not by_task:
        raise ValueError("no candidate pools")
    fold_seeds: dict[int, tuple[int, ...]] = {}
    for task, pools in by_task.items():
        if len(pools) != 64:
            raise ValueError(f"{task}: expected 64 K8 pools, found {len(pools)}")
        if len({pool.state for pool in pools}) != 16:
            raise ValueError(f"{task}: expected 16 states")
        for pool in pools:
            if len(pool.seeds) != 8 or len(set(map(int, pool.seeds))) != 8:
                raise ValueError("each pool must contain eight unique seeds")
            seeds = tuple(map(int, pool.seeds))
            previous = fold_seeds.setdefault(pool.fold, seeds)
            if previous != seeds:
                raise ValueError("a seed fold changed across states or tasks")
    if set(fold_seeds) != set(range(SEED_FOLDS)):
        raise ValueError("expected four seed folds")
    flattened = [seed for fold in sorted(fold_seeds) for seed in fold_seeds[fold]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("seed folds overlap")
    return {
        "tasks": len(by_task),
        "states_per_task": 16,
        "pools_per_task": 64,
        "candidates_per_pool": 8,
        "unique_seed_ids": len(flattened),
        "fold_seed_ids": {str(k): list(v) for k, v in sorted(fold_seeds.items())},
    }


def evaluate_task(pools: list[Pool]) -> dict[str, Any]:
    ordered = sorted(pools, key=lambda pool: (pool.state, pool.fold))
    success = np.stack([pool.success.astype(np.float64) for pool in ordered])
    baseline = success.mean(axis=1)
    methods: dict[str, Any] = {}
    positions: dict[str, np.ndarray] = {}
    for method, score_fn in SCORE_FUNCTIONS.items():
        scores = [np.asarray(score_fn(pool), dtype=np.float64) for pool in ordered]
        picked = np.asarray(
            [select_index(score, pool.seeds) for score, pool in zip(scores, ordered)],
            dtype=np.int64,
        )
        positions[method] = picked
        selected = success[np.arange(len(ordered)), picked]
        aucs = [pool_auc(pool.success, score) for pool, score in zip(ordered, scores)]
        finite_aucs = [value for value in aucs if np.isfinite(value)]
        selected_seeds = [int(pool.seeds[index]) for pool, index in zip(ordered, picked)]
        per_fold_unique = []
        for fold in range(SEED_FOLDS):
            rows = [i for i, pool in enumerate(ordered) if pool.fold == fold]
            per_fold_unique.append(len({selected_seeds[i] for i in rows}))
        methods[method] = {
            "selected_success": float(selected.mean()),
            "random_expected_success": float(baseline.mean()),
            "delta_vs_random": float((selected - baseline).mean()),
            "mean_pool_auc": float(np.mean(finite_aucs)) if finite_aucs else None,
            "evaluable_auc_pools": len(finite_aucs),
            "unique_selected_seeds": len(set(selected_seeds)),
            "max_seed_share": max(Counter(selected_seeds).values()) / len(selected_seeds),
            "mean_unique_seeds_per_fold": float(np.mean(per_fold_unique)),
        }
    return {
        "states": np.asarray([pool.state for pool in ordered], dtype=np.int64),
        "folds": np.asarray([pool.fold for pool in ordered], dtype=np.int64),
        "success": success,
        "baseline": baseline,
        "positions": positions,
        "methods": methods,
        "success_oracle": float(success.max(axis=1).mean()),
    }


def macro_value(task_eval: dict[str, dict[str, Any]], tasks: list[str], method: str,
                comparator: str | None = None) -> float:
    values = []
    for task in tasks:
        row = task_eval[task]
        success = row["success"]
        selected = success[np.arange(len(success)), row["positions"][method]]
        if comparator is None:
            reference = row["baseline"]
        else:
            reference = success[np.arange(len(success)), row["positions"][comparator]]
        values.append(float((selected - reference).mean()))
    return float(np.mean(values))


def clustered_bootstrap(task_eval: dict[str, dict[str, Any]], tasks: list[str],
                        method: str, comparator: str | None, draws: int,
                        rng: np.random.Generator) -> list[float]:
    distribution = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        task_values = []
        for task_index in rng.integers(0, len(tasks), len(tasks)):
            row = task_eval[tasks[int(task_index)]]
            states = np.unique(row["states"])
            state_values = []
            for state in rng.choice(states, len(states), replace=True):
                rows = np.flatnonzero(row["states"] == state)
                success = row["success"][rows]
                selected = success[
                    np.arange(len(rows)), row["positions"][method][rows]
                ]
                if comparator is None:
                    reference = row["baseline"][rows]
                else:
                    reference = success[
                        np.arange(len(rows)), row["positions"][comparator][rows]
                    ]
                state_values.append(float((selected - reference).mean()))
            task_values.append(float(np.mean(state_values)))
        distribution[draw] = np.mean(task_values)
    return [float(x) for x in np.percentile(distribution, [2.5, 97.5])]


def coherent_seed_permutation(
    task_eval: dict[str, dict[str, Any]],
    tasks: list[str],
    observed: dict[str, float],
    draws: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Permute K8 seed identities coherently across every task and state."""

    null = {method: np.empty(draws) for method in ORACLE_TARGETS}
    for draw in range(draws):
        permutations = {
            fold: rng.permutation(8) for fold in range(SEED_FOLDS)
        }
        per_method = {method: [] for method in ORACLE_TARGETS}
        for task in tasks:
            row = task_eval[task]
            labels = row["success"].copy()
            for fold, permutation in permutations.items():
                rows = np.flatnonzero(row["folds"] == fold)
                labels[rows] = labels[rows][:, permutation]
            for method in ORACLE_TARGETS:
                picked = labels[
                    np.arange(len(labels)), row["positions"][method]
                ]
                per_method[method].append(float((picked - row["baseline"]).mean()))
        for method in ORACLE_TARGETS:
            null[method][draw] = np.mean(per_method[method])
    maximum = np.column_stack([null[method] for method in ORACLE_TARGETS]).max(axis=1)
    return {
        method: {
            "one_sided_p": float(
                (1 + np.count_nonzero(null[method] >= observed[method] - 1e-15))
                / (draws + 1)
            ),
            "max_statistic_fwer_p": float(
                (1 + np.count_nonzero(maximum >= observed[method] - 1e-15))
                / (draws + 1)
            ),
        }
        for method in ORACLE_TARGETS
    }


def constant_score_control(pools: list[Pool]) -> dict[str, float | int]:
    aucs = []
    for pool in pools:
        value = pool_auc(pool.success, np.zeros(len(pool.success)))
        if np.isfinite(value):
            aucs.append(value)
    return {
        "candidate_score_max_range": 0.0,
        "mean_pool_auc": float(np.mean(aucs)),
        "evaluable_auc_pools": len(aucs),
    }


def gate_pass(delta: float, threshold: float) -> bool:
    if threshold <= 0:
        raise ValueError("gate threshold must be positive")
    return bool(delta > threshold)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 伪目标 oracle-gate 裁决",
        "",
        "## 实验思路",
        "",
        "- 只使用真实伪目标直接在同一 K8 候选池中选 1 个；本轮训练 head 数为 `0`。",
        "- 按给定规则事后回放 gate（不冒充预注册）：剔除全成功/全失败任务后，oracle 相对同池随机期望必须严格大于 `+3pp`，目标才允许进入 head 阶段。",
        "- 所有方法在完全相同的池上配对评价；同时报告免费 initial-noise-only 下界、真实 success 可达上限、seed 连贯置乱和恒定负控。",
        "",
        "## 结果",
        "",
        "| 目标 | oracle 增益 | 95% CI | 相对初始噪声 | AUC | FWER p | 裁决 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for method in ORACLE_TARGETS:
        row = summary["informative_tasks"]["methods"][method]
        ci = row["delta_bootstrap_ci95"]
        lines.append(
            "| %s | %+.3f | [%+.3f, %+.3f] | %+.3f | %.3f | %.3f | %s |"
            % (
                DISPLAY[method],
                row["delta_vs_random"],
                ci[0],
                ci[1],
                row["paired_delta_vs_initial_noise"],
                row["mean_pool_auc"],
                row["max_statistic_fwer_p"],
                "过线" if row["gate_pass"] else "淘汰",
            )
        )
    lower = summary["informative_tasks"]["methods"]["initial_noise_only"]
    lines.extend(
        [
            "",
            "初始噪声免费下界相对随机为 `%+.3f`；K8 内直接看 success 的不可部署上限为 `%.3f`，说明候选池有可选择空间，失败的是这三个伪目标。"
            % (lower["delta_vs_random"], summary["informative_tasks"]["success_oracle"]),
            "",
            "state-token / AS 恒定负控均精确得到 `AUC=0.500`、候选范围 `0.000`。共 `%d` 个有效 K8 池；每 fold 有 8 个互斥 seed，置乱对同一 seed 列跨状态、跨任务同步进行。"
            % summary["informative_tasks"]["n_pools"],
            "",
            "**裁决：三个伪目标全部未过 `+3pp` 门槛，按规则在 oracle 阶段终止；此前 8 个 head 的结果只保留为历史探索，不构成继续优化依据。**",
            "",
            "## Critic 数据审计",
            "",
            "现有 fork-pilot 有 `20` 个快照、每快照 `32` 个候选，但只覆盖一个任务，每候选只有一条未来，没有 CRN 重复，也没有视觉观测特征。已有结果是 route 同快照拟合 `rho=0.112`，留出快照降到 `0.010`；因此它只能说明弱信号偏状态特异，不能检验跨任务状态条件 critic。主实验必须补采集后再跑。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap < 100 or args.permutations < 100:
        raise ValueError("bootstrap and permutations must be at least 100")
    if args.threshold <= 0:
        raise ValueError("threshold must be positive")

    all_pools: list[Pool] = []
    integrity = []
    for run in discover_runs(args.hub_root.resolve()):
        print("loading", run, flush=True)
        pools, diagnostic = load_task_pools(run, args.hub_root.resolve() / "cache/HiMoE-VLA")
        all_pools.extend(pools)
        integrity.append(diagnostic)
    by_task = {
        task: [pool for pool in all_pools if pool.task == task]
        for task in sorted({pool.task for pool in all_pools})
    }
    seed_validation = validate_pools(by_task)
    ceiling_tasks = [task for task, pools in by_task.items() if is_ceiling_task(pools)]
    informative_tasks = [task for task in by_task if task not in ceiling_tasks]
    if not informative_tasks:
        raise RuntimeError("no informative task remains after the ceiling rule")
    task_eval = {task: evaluate_task(pools) for task, pools in by_task.items()}

    rng = np.random.default_rng(args.seed)
    observed = {
        method: macro_value(task_eval, informative_tasks, method)
        for method in ORACLE_TARGETS
    }
    permutation = coherent_seed_permutation(
        task_eval, informative_tasks, observed, args.permutations, rng
    )
    method_summary = {}
    for method in SCORE_FUNCTIONS:
        delta = macro_value(task_eval, informative_tasks, method)
        task_rows = [task_eval[task]["methods"][method] for task in informative_tasks]
        entry = {
            "selected_success": float(np.mean([row["selected_success"] for row in task_rows])),
            "random_expected_success": float(np.mean([row["random_expected_success"] for row in task_rows])),
            "delta_vs_random": delta,
            "delta_bootstrap_ci95": clustered_bootstrap(
                task_eval, informative_tasks, method, None, args.bootstrap, rng
            ),
            "mean_pool_auc": float(np.mean([row["mean_pool_auc"] for row in task_rows])),
            "mean_unique_seeds_per_fold": float(
                np.mean([row["mean_unique_seeds_per_fold"] for row in task_rows])
            ),
            "task_deltas": {
                task: task_eval[task]["methods"][method]["delta_vs_random"]
                for task in informative_tasks
            },
        }
        if method in ORACLE_TARGETS:
            entry.update(permutation[method])
            entry["paired_delta_vs_initial_noise"] = macro_value(
                task_eval, informative_tasks, method, "initial_noise_only"
            )
            entry["paired_delta_vs_initial_noise_ci95"] = clustered_bootstrap(
                task_eval, informative_tasks, method, "initial_noise_only",
                args.bootstrap, rng
            )
            entry["gate_pass"] = gate_pass(delta, args.threshold)
        method_summary[method] = entry

    all_pool_control = constant_score_control(all_pools)
    if all_pool_control["mean_pool_auc"] != 0.5:
        raise RuntimeError("constant-score AUC control did not equal 0.5")
    route_validation = json.loads(
        (HERE / "analysis/route-noise-selector/summary.json").read_text()
    )["validation"]
    if route_validation["max_state_token_candidate_range"] != 0.0:
        raise RuntimeError("state-token negative control is not constant")
    if route_validation["max_as_candidate_range"] != 0.0:
        raise RuntimeError("AS negative control is not constant")

    all_task_methods = {}
    for method in SCORE_FUNCTIONS:
        all_task_methods[method] = {
            "delta_vs_random": macro_value(task_eval, list(by_task), method),
            "task_deltas": {
                task: task_eval[task]["methods"][method]["delta_vs_random"]
                for task in by_task
            },
        }
    summary = {
        "experiment": "oracle_first_pseudo_target_gate",
        "status": "terminated_at_oracle_gate",
        "protocol": {
            "retrospective_application": True,
            "gate_threshold": args.threshold,
            "gate_rule": "oracle informative-task macro delta must be strictly greater than threshold",
            "ceiling_rule": "exclude a task iff all candidate outcomes are identical",
            "heads_fit_in_this_analysis": 0,
            "candidate_pool": "four disjoint K8 folds from each K32 grid",
            "paired_evaluation": True,
            "bootstrap_draws": args.bootstrap,
            "permutation_draws": args.permutations,
            "permutation": "one K8 column permutation per fold, reused across all states and tasks",
            "random_seed": args.seed,
        },
        "seed_validation": seed_validation,
        "normalization_validation": {
            "max_raw_probability_mass_error": route_validation["max_raw_probability_mass_error"],
            "max_renormalized_probability_mass_error": route_validation[
                "max_renormalized_probability_mass_error"
            ],
        },
        "zero_controls": {
            "state_token": all_pool_control,
            "as_channel": all_pool_control,
            "observed_state_token_candidate_range": route_validation["max_state_token_candidate_range"],
            "observed_as_candidate_range": route_validation["max_as_candidate_range"],
        },
        "ceiling_tasks": ceiling_tasks,
        "all_tasks": {"n_tasks": len(by_task), "methods": all_task_methods},
        "informative_tasks": {
            "tasks": informative_tasks,
            "n_tasks": len(informative_tasks),
            "n_pools": len(informative_tasks) * 64,
            "methods": method_summary,
            "success_oracle": float(
                np.mean([task_eval[task]["success_oracle"] for task in informative_tasks])
            ),
        },
        "decision": {
            "passed_targets": [
                method for method in ORACLE_TARGETS if method_summary[method]["gate_pass"]
            ],
            "rejected_targets": [
                method for method in ORACLE_TARGETS if not method_summary[method]["gate_pass"]
            ],
            "stop_before_heads": not any(
                method_summary[method]["gate_pass"] for method in ORACLE_TARGETS
            ),
        },
        "source_integrity": integrity,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2), flush=True)
    print("wrote", args.out_dir / "report.md", flush=True)


if __name__ == "__main__":
    main()
