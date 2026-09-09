#!/usr/bin/env python3
"""Cross-task test of a continuous-state intra-query MoE Markov selector.

This keeps the first-order Markov model but removes K-means.  At each denoise
round the state is the complete Hellinger-embedded router tensor over eight HB
layers, ten action tokens and 32 experts.  Outcome-conditional diagonal AR(1)
Gaussian transitions are fitted on source tasks only, after removing the K8
candidate seeds being tested.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_intraquery_markov import (
    HB_LAYERS,
    METHODS,
    N_ACTION_TOKENS,
    N_DENOISE,
    N_EXPERTS,
    finite,
)
from analyze_intraquery_markov_transfer import (
    SEED_FOLDS,
    TASKS,
    Capture,
    evaluate_cohort,
    load_capture,
)
from continuous_route_markov import (
    continuous_route_states,
    continuous_sequence_log_likelihoods,
    fit_continuous_route_models,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "analysis" / "intraquery-continuous-markov-transfer"
DEFAULT_DISCRETE = HERE / "analysis" / "intraquery-markov-transfer" / "summary.json"
PRIMARY_ROUNDS = 3
PRIMARY_TOP_K = 2
STATE_WIDTH = len(HB_LAYERS) * N_ACTION_TOKENS * N_EXPERTS
SENSITIVITY_GRID = (
    (5.0, 20.0),
    (20.0, 5.0),
    (20.0, 20.0),
    (20.0, 100.0),
    (100.0, 20.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--discrete-summary", type=Path, default=DEFAULT_DISCRETE)
    parser.add_argument("--rounds", type=int, default=PRIMARY_ROUNDS)
    parser.add_argument("--top-k", type=int, default=PRIMARY_TOP_K)
    parser.add_argument("--ridge", type=float, default=20.0)
    parser.add_argument("--variance-prior", type=float, default=20.0)
    parser.add_argument("--permutations", type=int, default=19_999)
    parser.add_argument("--sensitivity-permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def _empty_score_table() -> dict[str, np.ndarray]:
    return {
        method: np.full((64, N_DENOISE), np.nan, dtype=np.float64) for method in METHODS
    }


def crossfit_continuous_scores(
    development: list[Capture],
    replication: list[Capture],
    ridge_strength: float,
    variance_prior_strength: float,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, np.ndarray]]],
    list[dict[str, Any]],
]:
    if [row.task.key for row in development] != [row.task.key for row in replication]:
        raise ValueError("development and replication tasks are not aligned")
    states = {
        cohort: {
            capture.task.key: continuous_route_states(capture.routes)
            for capture in captures
        }
        for cohort, captures in (
            ("development", development),
            ("replication", replication),
        )
    }
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
        for fold in range(SEED_FOLDS):
            held_indices = held_capture.index_grid[fold]
            held_seeds = held_capture.seeds[held_indices]
            train_states = []
            train_success = []
            source_tasks = []
            for source_axis, source in enumerate(development):
                if source_axis == held_axis:
                    continue
                take = ~np.isin(source.seeds, held_seeds)
                if np.any(np.isin(source.seeds[take], held_seeds)):
                    raise RuntimeError("held seed leaked into source states")
                train_states.append(states["development"][source.task.key][take])
                train_success.append(source.success[take])
                source_tasks.append(source.task.key)
            state_train = np.concatenate(train_states)
            outcome_train = np.concatenate(train_success)
            models = fit_continuous_route_models(
                state_train,
                outcome_train,
                ridge_strength,
                variance_prior_strength,
            )

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
                test_states = states[cohort][test_capture.task.key][test_indices]
                logp = {
                    label: continuous_sequence_log_likelihoods(
                        models[label], test_states
                    )
                    for label in (False, True)
                }
                labels = test_capture.success[test_indices]
                for method in METHODS:
                    scores[cohort][test_capture.task.key][method][test_indices] = (
                        logp[True][method] - logp[False][method]
                    )
                    true_logp[cohort][test_capture.task.key][method][test_indices] = (
                        np.where(
                            labels[:, None], logp[True][method], logp[False][method]
                        )
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
                }
            )

    for table in (scores, true_logp):
        for cohort in table.values():
            for task in cohort.values():
                for values in task.values():
                    if np.any(~np.isfinite(values)):
                        raise RuntimeError(
                            "continuous cross-fitting left non-finite entries"
                        )
    return scores, true_logp, diagnostics


def continuous_predictive_summary(true_logp: dict[str, np.ndarray]) -> dict[str, Any]:
    coordinates = N_DENOISE * STATE_WIDTH
    output = {
        method: {
            "full_path_cross_entropy_bits_per_coordinate": finite(
                -values[:, -1].mean() / (coordinates * math.log(2.0))
            )
        }
        for method, values in true_logp.items()
    }
    for kind in ("periodic", "position"):
        markov = true_logp["markov_" + kind][:, -1]
        occupancy = true_logp["occupancy_" + kind][:, -1]
        output["markov_" + kind][
            "gain_over_occupancy_bits_per_coordinate_transition"
        ] = finite(
            (markov - occupancy).mean()
            / ((N_DENOISE - 1) * STATE_WIDTH * math.log(2.0))
        )
    return output


def evaluate_continuous_cohorts(
    development: list[Capture],
    replication: list[Capture],
    scores: dict[str, dict[str, dict[str, np.ndarray]]],
    true_logp: dict[str, dict[str, dict[str, np.ndarray]]],
    rounds: int,
    top_k: int,
    permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    cohorts = {}
    for cohort_name, captures in (
        ("development", development),
        ("replication", replication),
    ):
        cohorts[cohort_name], _internal = evaluate_cohort(
            captures,
            scores[cohort_name],
            true_logp[cohort_name],
            rounds,
            top_k,
            permutations,
            rng,
            continuous_predictive_summary,
        )
    return cohorts


def compact_sensitivity_result(
    ridge_strength: float,
    variance_prior_strength: float,
    cohorts: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ridge_strength": ridge_strength,
        "variance_prior_strength": variance_prior_strength,
    }
    for cohort_name in ("development", "replication"):
        macro = cohorts[cohort_name]["macro"]
        row[cohort_name] = {
            "delta_vs_random": macro["delta_vs_random"],
            "permutation_vs_random_one_sided_p": macro["permutation_vs_random"][
                "one_sided_p"
            ],
            "markov_selected_success_difference_vs_occupancy": macro[
                "markov_selected_success_difference_vs_occupancy"
            ],
            "permutation_vs_occupancy_one_sided_p": macro[
                "permutation_vs_periodic_occupancy"
            ]["one_sided_p"],
        }
    return row


def _compact_discrete_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("discrete transfer summary is missing: %s" % path)
    source = json.loads(path.read_text())
    return {
        "source": str(path.resolve()),
        "schema": source["schema"],
        "development_macro": source["cohorts"]["development"]["macro"],
        "replication_macro": source["cohorts"]["replication"]["macro"],
    }


def _fmt(value: float, signed: bool = False) -> str:
    return ("%+.3f" if signed else "%.3f") % value


def _primary(
    task: dict[str, Any], protocol: dict[str, Any], method: str
) -> dict[str, Any]:
    return task["selection"][method][str(protocol["rounds"])][str(protocol["top_k"])]


def render_report(summary: dict[str, Any]) -> str:
    protocol = summary["protocol"]
    lines = [
        "# 连续状态 query 内 Markov：其他 Goal 任务",
        "",
        "## 仍然是马尔科夫模型",
        "",
        (
            "状态不再经过 K-means。第 `tau` 个 denoise round 的状态 "
            "`X[tau]` 直接拼接 `8 layer × 10 action-token × 32 expert = 2560` "
            "个 sqrt(router probability) 分量；一条 query 有 10 个内部状态。"
        ),
        "",
        (
            "一阶假设为 `p(X[tau+1] | X[tau], outcome)`。每个分量使用 ridge "
            "AR(1) 线性高斯转移；success/failure 有不同的条件均值，但共享残差"
            "方差。它是连续状态、denoise-time、按坐标因子化的 Markov 模型，不是"
            "普通分类器，也不使用专家职责。"
        ),
        "",
        (
            "评估仍同时留出完整目标任务和 K8 的 8 个 seed。训练只来自另外两个"
            " Goal 任务；同一冻结模型再评估 objstate 重采集。固定规则为前 3 轮"
            "选 top-2。"
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
                "| held-out 任务 | 随机成功率 | 连续零阶 delta | 连续 Markov 成功率 | Markov delta | AUC | p vs random | Markov - 零阶 | p |",
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
                    "三任务宏平均：连续 Markov 相对随机 `%s`（`p=%.4f`），"
                    "连续零阶为 `%s`；Markov 减零阶 `%s`（`p=%.4f`）。"
                    % (
                        _fmt(macro["delta_vs_random"], True),
                        macro["permutation_vs_random"]["one_sided_p"],
                        _fmt(macro["periodic_occupancy_delta_vs_random"], True),
                        _fmt(
                            macro["markov_selected_success_difference_vs_occupancy"],
                            True,
                        ),
                        macro["permutation_vs_periodic_occupancy"]["one_sided_p"],
                    )
                ),
                "",
            ]
        )

    discrete = summary["discrete_reference"]
    lines.extend(
        [
            "## 与离散 Markov 对照",
            "",
            "| 模型 | within64 delta | objstate delta |",
            "|---|---:|---:|",
            "| K16 离散 Markov | %s | %s |"
            % (
                _fmt(discrete["development_macro"]["delta_vs_random"], True),
                _fmt(discrete["replication_macro"]["delta_vs_random"], True),
            ),
            "| 连续 Markov | %s | %s |"
            % (
                _fmt(
                    summary["cohorts"]["development"]["macro"]["delta_vs_random"], True
                ),
                _fmt(
                    summary["cohorts"]["replication"]["macro"]["delta_vs_random"], True
                ),
            ),
            "",
            "## 前缀敏感性",
            "",
            "| denoise 轮数 | within64 连续 Markov delta | objstate 连续 Markov delta |",
            "|---:|---:|---:|",
        ]
    )
    for rounds in (1, 2, 3, 5, 10):
        values = []
        for cohort_name in ("development", "replication"):
            cohort = summary["cohorts"][cohort_name]
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
            "| %d | %s | %s |" % (rounds, _fmt(values[0], True), _fmt(values[1], True))
        )

    lines.extend(
        [
            "",
            "## 正则敏感性",
            "",
            "| ridge | 方差先验 | within64 delta | objstate delta | objstate Markov - 零阶 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["regularization_sensitivity"]:
        lines.append(
            "| %g | %g | %s | %s | %s |"
            % (
                row["ridge_strength"],
                row["variance_prior_strength"],
                _fmt(row["development"]["delta_vs_random"], True),
                _fmt(row["replication"]["delta_vs_random"], True),
                _fmt(
                    row["replication"][
                        "markov_selected_success_difference_vs_occupancy"
                    ],
                    True,
                ),
            )
        )

    development = summary["cohorts"]["development"]["macro"]
    replication = summary["cohorts"]["replication"]["macro"]
    replication_density_gain = np.mean(
        [
            task["predictive"]["markov_periodic"][
                "gain_over_occupancy_bits_per_coordinate_transition"
            ]
            for task in summary["cohorts"]["replication"]["tasks"].values()
        ]
    )
    supported = (
        development["delta_vs_random"] > 0.0
        and replication["delta_vs_random"] > 0.0
        and development["permutation_vs_random"]["one_sided_p"] < 0.05
        and replication["permutation_vs_random"]["one_sided_p"] < 0.05
        and replication["markov_selected_success_difference_vs_occupancy"] > 0.0
        and replication["permutation_vs_periodic_occupancy"]["one_sided_p"] < 0.05
    )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "去掉 K-means 后，连续 Markov 在开发/复验上的宏平均 delta 为 "
                "`%s/%s`。固定规则%s同时通过开发、重采集以及相对连续零阶的"
                "检验。"
                % (
                    _fmt(development["delta_vs_random"], True),
                    _fmt(replication["delta_vs_random"], True),
                    "" if supported else "未",
                )
            ),
            "",
            (
                "Markov 转移使复验集的真实类别路径对数密度平均提高 "
                "`%.3f bit/坐标/转移`，说明相邻去噪轮次确有可预测动力学；但"
                "成功选择 AUC 只有 `%.3f`，所以缺失的是可迁移的 outcome 关联，"
                "不是 Markov 动力学本身。"
                % (replication_density_gain, replication["mean_pool_auc"])
            ),
            "",
            (
                "所以回答仍是：这是 Markov；连续化只消除了随机 codebook。是否"
                "能选出好候选由上述 held-task 结果决定，不能因模型形式更平滑就"
                "预设为有效。"
            ),
            "",
            (
                "限制仍然是每任务只有一个 snapshot，且这是 episode-start shadow "
                "outcome。真正部署前仍需多 snapshot 的同状态 K8 闭环分叉。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.rounds < 1 or args.rounds > N_DENOISE:
        raise ValueError("rounds must be between one and ten")
    if args.top_k not in (1, 2, 4):
        raise ValueError("top-k must be one, two or four")
    if args.ridge <= 0.0 or args.variance_prior <= 0.0:
        raise ValueError("continuous priors must be positive")
    if args.permutations < 100:
        raise ValueError("permutations must be at least 100")
    if args.sensitivity_permutations < 100:
        raise ValueError("sensitivity permutations must be at least 100")

    development = [load_capture(task, "development") for task in TASKS]
    replication = [load_capture(task, "replication") for task in TASKS]
    scores, true_logp, split_diagnostics = crossfit_continuous_scores(
        development,
        replication,
        args.ridge,
        args.variance_prior,
    )
    cohorts = evaluate_continuous_cohorts(
        development,
        replication,
        scores,
        true_logp,
        args.rounds,
        args.top_k,
        args.permutations,
        args.seed,
    )

    settings = list(SENSITIVITY_GRID)
    primary_setting = (args.ridge, args.variance_prior)
    if primary_setting not in settings:
        settings.insert(0, primary_setting)
    sensitivity = []
    for index, (ridge_strength, variance_prior_strength) in enumerate(settings):
        if (ridge_strength, variance_prior_strength) == primary_setting:
            alternate_cohorts = cohorts
        else:
            alternate_scores, alternate_logp, _diagnostics = crossfit_continuous_scores(
                development,
                replication,
                ridge_strength,
                variance_prior_strength,
            )
            alternate_cohorts = evaluate_continuous_cohorts(
                development,
                replication,
                alternate_scores,
                alternate_logp,
                args.rounds,
                args.top_k,
                args.sensitivity_permutations,
                args.seed + index + 1,
            )
        sensitivity.append(
            compact_sensitivity_result(
                ridge_strength,
                variance_prior_strength,
                alternate_cohorts,
            )
        )

    summary = {
        "schema": "himoe-intraquery-continuous-markov-transfer-v1",
        "protocol": {
            "state": "sqrt full router probabilities [8 layers,10 action tokens,32 experts]",
            "state_width": STATE_WIDTH,
            "markov_clock": "ten intra-query denoise rounds",
            "transition": "outcome-conditional coordinate-wise ridge AR(1) Gaussian",
            "shared_variance_across_outcomes": True,
            "quantization": None,
            "held_out": "complete target task and K8 candidate seed fold",
            "rounds": args.rounds,
            "top_k": args.top_k,
            "ridge_strength": args.ridge,
            "variance_prior_strength": args.variance_prior,
            "permutations": args.permutations,
            "sensitivity_permutations": args.sensitivity_permutations,
            "random_seed_only_for_permutations": args.seed,
        },
        "split_diagnostics": split_diagnostics,
        "cohorts": cohorts,
        "regularization_sensitivity": sensitivity,
        "discrete_reference": _compact_discrete_reference(args.discrete_summary),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build(args)
    for cohort_name, cohort in summary["cohorts"].items():
        macro = cohort["macro"]
        print(
            "%s continuous delta=%+.4f p=%.4f markov-minus-zero=%+.4f"
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
