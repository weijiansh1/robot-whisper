#!/usr/bin/env python3
"""Fuse causal route activity with full graph-transfer response transitions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
BASE = BUNDLE / "results"
GRAPH = BASE / "graph_stage2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "fusion")
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_metrics(target: pd.Series | np.ndarray, score: pd.Series | np.ndarray) -> tuple[float, float, int]:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if len(np.unique(target[finite])) < 2:
        return float("nan"), float("nan"), int(finite.sum())
    return (
        float(roc_auc_score(target[finite], score[finite])),
        float(average_precision_score(target[finite], score[finite])),
        int(finite.sum()),
    )


def load_graph(anchor: str) -> pd.DataFrame:
    metadata = pd.read_csv(GRAPH / f"{anchor}_metadata_and_composites.csv")
    with np.load(GRAPH / f"{anchor}_graph_ranks.npz", allow_pickle=False) as archive:
        names = archive["feature_names"].astype(str).tolist()
        ranks = archive["ranks"].astype(float)
    if len(metadata) != len(ranks):
        raise ValueError(f"{anchor}: graph metadata/rank mismatch")
    for name, output in (
        ("query|action_conditional|front_d1", "conditional_front_d1"),
        ("query|action_conditional|gap_d1", "conditional_gap_d1"),
        (
            "flow|state_shape|back_settling_log_ratio",
            "state_shape_back_settling",
        ),
    ):
        metadata[output] = ranks[:, names.index(name)]
    return metadata


def joined_anchor(anchor: str) -> pd.DataFrame:
    graph = load_graph(anchor)
    simple = pd.read_csv(BASE / f"{anchor}_prefix_features.csv")
    keys = ["cohort", "row", "task", "suite", "outcome"]
    simple_columns = keys + [
        "route_activity_w4",
        "response_history_score",
        "front_minus_back_w4",
    ]
    return graph.merge(simple[simple_columns], on=keys, validate="one_to_one")


def alarm_scores(alarm: pd.DataFrame) -> pd.DataFrame:
    alarm = alarm.copy()
    alarm["quiet_route_history"] = 1.0 - alarm["response_history_score"]
    alarm["quiet_graph_response"] = 1.0 - alarm["graph_query_responsiveness"]
    alarm["quiet_conditional_front"] = 1.0 - alarm["conditional_front_d1"]
    alarm["quiet_state_shape_settling"] = 1.0 - alarm["state_shape_back_settling"]
    alarm["quiet_route_graph_equal"] = np.nanmean(
        alarm[["quiet_route_history", "quiet_graph_response"]], axis=1
    )
    return alarm


def transition_scores(alarm: pd.DataFrame, terminal: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort", "row", "task", "suite", "episode", "outcome"]
    alarm = alarm[alarm["outcome"].isin(["late", "persistent"])]
    merged = alarm.merge(terminal, on=keys, suffixes=("_alarm", "_terminal"), validate="one_to_one")
    merged["graph_conditional_release"] = (
        merged["conditional_front_d1_terminal"]
        - merged["conditional_front_d1_alarm"]
    )
    merged["graph_gap_release"] = (
        merged["conditional_gap_d1_terminal"] - merged["conditional_gap_d1_alarm"]
    )
    merged["graph_homeostasis_release"] = (
        merged["graph_homeostasis_terminal"] - merged["graph_homeostasis_alarm"]
    )
    merged["route_activity_release"] = (
        merged["route_activity_w4_terminal"] - merged["route_activity_w4_alarm"]
    )
    merged["layer_back_handoff"] = (
        merged["front_minus_back_w4_alarm"]
        - merged["front_minus_back_w4_terminal"]
    )
    merged["restart_graph_route_equal"] = np.nanmean(
        merged[["graph_conditional_release", "route_activity_release"]], axis=1
    )
    merged["restart_graph_route_layer_equal"] = np.nanmean(
        merged[
            [
                "graph_conditional_release",
                "route_activity_release",
                "layer_back_handoff",
            ]
        ],
        axis=1,
    )
    merged["restart_homeostasis_route_layer_equal"] = np.nanmean(
        merged[
            [
                "graph_homeostasis_release",
                "route_activity_release",
                "layer_back_handoff",
            ]
        ],
        axis=1,
    )
    return merged


def metric_rows(frame: pd.DataFrame, context: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for cohort in ("development_main", "external_8b"):
        cohort_frame = frame[frame["cohort"] == cohort]
        populations = [("risk_only_late_vs_persistent", cohort_frame)]
        if context == "alarm":
            populations.append(("all_alarms_continue_vs_intervene", cohort_frame))
        for population, block in populations:
            if population.startswith("risk_only"):
                block = block[block["outcome"].isin(["late", "persistent"])]
                target = block["outcome"].eq("late").astype(int)
            else:
                target = block["outcome"].ne("persistent").astype(int)
            for feature in features:
                auc, ap, finite_n = safe_metrics(target, block[feature])
                rows.append(
                    {
                        "context": context,
                        "cohort": cohort,
                        "population": population,
                        "feature": feature,
                        "n": len(block),
                        "finite_n": finite_n,
                        "positive_n": int(target.sum()),
                        "prevalence": float(target.mean()),
                        "auc": auc,
                        "average_precision": ap,
                    }
                )
    return pd.DataFrame(rows)


def by_suite(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for cohort in ("development_main", "external_8b"):
        cohort_frame = frame[frame["cohort"] == cohort]
        for suite, block in cohort_frame.groupby("suite", sort=True):
            target = block["outcome"].eq("late").astype(int)
            if target.nunique() < 2:
                continue
            for feature in features:
                auc, ap, finite_n = safe_metrics(target, block[feature])
                rows.append(
                    {
                        "cohort": cohort,
                        "suite": suite,
                        "feature": feature,
                        "n": len(block),
                        "finite_n": finite_n,
                        "late_n": int(target.sum()),
                        "auc": auc,
                        "average_precision": ap,
                    }
                )
    return pd.DataFrame(rows)


def within_task(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    aucs = []
    weights = []
    for _, block in frame.groupby("task", sort=True):
        finite = block[feature].notna()
        block = block[finite]
        target = block["outcome"].eq("late").astype(int)
        if target.nunique() < 2:
            continue
        aucs.append(roc_auc_score(target, block[feature]))
        weights.append(int(target.sum() * (1 - target).sum()))
    return {
        "weighted_auc": float(np.average(aucs, weights=weights)),
        "macro_auc": float(np.mean(aucs)),
        "tasks": len(aucs),
    }


def bootstrap_ci(
    target: np.ndarray,
    score: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    target = target[finite]
    score = score[finite]
    negative = np.flatnonzero(target == 0)
    positive = np.flatnonzero(target == 1)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        take = np.r_[
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ]
        values[draw] = roc_auc_score(target[take], score[take])
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def task_cluster_bootstrap_ci(
    frame: pd.DataFrame,
    feature: str,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    groups = []
    for _, block in frame.groupby("task", sort=True):
        finite = block[feature].notna()
        groups.append(
            (
                block.loc[finite, "outcome"].eq("late").astype(int).to_numpy(),
                block.loc[finite, feature].to_numpy(dtype=float),
            )
        )
    values = []
    for _ in range(draws):
        take = rng.integers(0, len(groups), len(groups))
        target = np.concatenate([groups[index][0] for index in take])
        score = np.concatenate([groups[index][1] for index in take])
        if len(np.unique(target)) == 2:
            values.append(roc_auc_score(target, score))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def allocation_rows(
    development: pd.DataFrame, external: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    rows = []
    for feature in features:
        for quantile in (0.50, 0.75, 0.90):
            threshold = float(np.nanquantile(development[feature], quantile))
            selected = external[feature].to_numpy(dtype=float) >= threshold
            late = external["outcome"].eq("late").to_numpy()
            persistent = external["outcome"].eq("persistent").to_numpy()
            extra_query_column = "late_success_first_extra_query_alarm"
            extension_cost = 10 * int((selected & persistent).sum())
            extension_cost += int(
                external.loc[selected & late, extra_query_column].sum()
            )
            recovered = int((selected & late).sum())
            extend_all_cost = 10 * int(persistent.sum()) + int(
                external.loc[late, extra_query_column].sum()
            )
            extend_all_cost_per_recovered = extend_all_cost / int(late.sum())
            rows.append(
                {
                    "feature": feature,
                    "development_outcome_blind_quantile": quantile,
                    "threshold": threshold,
                    "external_n": len(external),
                    "selected_n": int(selected.sum()),
                    "late_selected": int((selected & late).sum()),
                    "late_n": int(late.sum()),
                    "late_recall": float((selected & late).sum() / late.sum()),
                    "persistent_selected": int((selected & persistent).sum()),
                    "persistent_n": int(persistent.sum()),
                    "persistent_extension_rate": float(
                        (selected & persistent).sum() / persistent.sum()
                    ),
                    "late_precision": float(
                        (selected & late).sum() / max(selected.sum(), 1)
                    ),
                    "precision_lift_over_prevalence": float(
                        ((selected & late).sum() / max(selected.sum(), 1))
                        / late.mean()
                    ),
                    "extra_query_cost": extension_cost,
                    "extra_query_cost_per_recovered": float(
                        extension_cost / max(recovered, 1)
                    ),
                    "extend_all_cost_per_recovered": float(
                        extend_all_cost_per_recovered
                    ),
                    "cost_reduction_vs_extend_all": float(
                        1.0
                        - (extension_cost / max(recovered, 1))
                        / extend_all_cost_per_recovered
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    alarm = alarm_scores(joined_anchor("alarm"))
    terminal = joined_anchor("terminal")
    transition = transition_scores(alarm, terminal)
    alarm_features = [
        "quiet_route_history",
        "quiet_graph_response",
        "quiet_conditional_front",
        "quiet_state_shape_settling",
        "quiet_route_graph_equal",
    ]
    transition_features = [
        "graph_conditional_release",
        "graph_gap_release",
        "graph_homeostasis_release",
        "route_activity_release",
        "layer_back_handoff",
        "restart_graph_route_equal",
        "restart_graph_route_layer_equal",
        "restart_homeostasis_route_layer_equal",
    ]
    metrics = pd.concat(
        [
            metric_rows(alarm, "alarm", alarm_features),
            metric_rows(transition, "terminal_minus_alarm", transition_features),
        ],
        ignore_index=True,
    )
    metrics.to_csv(args.output / "fusion_metrics.csv", index=False)
    suites = by_suite(transition, transition_features)
    suites.to_csv(args.output / "fusion_by_suite.csv", index=False)
    development_transition = transition[transition["cohort"] == "development_main"]
    external_transition = transition[transition["cohort"] == "external_8b"]
    allocation = allocation_rows(
        development_transition, external_transition, transition_features
    )
    allocation.to_csv(args.output / "allocation_metrics.csv", index=False)
    transition.to_csv(args.output / "transition_scores.csv", index=False)

    primary = "restart_graph_route_layer_equal"
    external_target = external_transition["outcome"].eq("late").astype(int).to_numpy()
    external_score = external_transition[primary].to_numpy(dtype=float)
    rng = np.random.default_rng(args.seed)
    auc, ap, finite_n = safe_metrics(external_target, external_score)
    summary = {
        "schema": "himoe.feedback_responsiveness.fusion.v1",
        "trained_model": False,
        "primary_score": {
            "name": primary,
            "status": "post-hoc equal-weight mechanism fusion",
            "components": [
                "state-conditioned front action-graph response release",
                "route-activity release",
                "front-to-back layer-response handoff",
            ],
            "development_auc": safe_metrics(
                development_transition["outcome"].eq("late"),
                development_transition[primary],
            )[0],
            "external_n": len(external_transition),
            "external_finite_n": finite_n,
            "external_late_n": int(external_target.sum()),
            "external_auc": auc,
            "external_auc_ci95_row_stratified": bootstrap_ci(
                external_target, external_score, args.bootstrap, rng
            ),
            "external_auc_ci95_task_clustered": task_cluster_bootstrap_ci(
                external_transition, primary, args.bootstrap, rng
            ),
            "external_average_precision": ap,
            "external_prevalence": float(external_target.mean()),
            "external_within_task": within_task(external_transition, primary),
        },
        "artifacts": {
            "metrics": "fusion_metrics.csv",
            "by_suite": "fusion_by_suite.csv",
            "allocation": "allocation_metrics.csv",
            "scores": "transition_scores.csv",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plain(summary), indent=2))
    print("\nExternal transition metrics")
    print(
        metrics[
            (metrics["cohort"] == "external_8b")
            & (metrics["context"] == "terminal_minus_alarm")
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nPrimary allocation")
    print(
        allocation[allocation["feature"] == primary].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )


if __name__ == "__main__":
    main()
