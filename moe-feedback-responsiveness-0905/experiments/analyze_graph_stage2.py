#!/usr/bin/env python3
"""Test full routing-transfer graphs as a causal v4 Stage-2 monitor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
V4 = WORKSPACE / "moe-v4-0904"
REFERENCE_ROOT = (
    WORKSPACE
    / "moe-routing-transfer-v11-0905/results/conditional_profiles/grid50x8"
)
EXTERNAL_ROOT = BUNDLE / "results/graph_profiles/external_8b"
ALARM_TABLE = V4 / "results/cache_new_v4/episode_alarms.csv"
LABEL_ROOT = WORKSPACE / "double-selete/trainfree/results/timeout_extension_plus10"

LABEL_PATHS = {
    "development_main": LABEL_ROOT / "development_main_clean_labels.csv",
    "external_8b": LABEL_ROOT / "external_8b_clean_labels.csv",
}

VIEWS = ("action_shape", "action_conditional", "action_partial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BUNDLE / "results/graph_stage2")
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


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


def profile_path(root: Path, task: str) -> Path:
    suite, name = task.split("/", 1)
    path = root / suite / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def profile_matrix(profile: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    names = np.r_[
        profile["within_feature_names"].astype(str),
        profile["query_feature_names"].astype(str),
    ]
    matrix = np.column_stack(
        (profile["within_features"], profile["query_features"])
    ).astype(np.float32)
    return names, matrix


def target_table(cohort: str, anchor: str) -> pd.DataFrame:
    alarms = pd.read_csv(ALARM_TABLE)
    alarms = alarms[alarms["cohort"] == cohort].sort_values("row").reset_index(drop=True)
    labels = pd.read_csv(LABEL_PATHS[cohort])
    if len(labels) != len(alarms):
        raise ValueError(f"{cohort}: label/alarm mismatch")
    if not np.array_equal(labels["task"].astype(str), alarms["task"].astype(str)):
        raise ValueError(f"{cohort}: label/alarm task mismatch")
    alarms["outcome"] = np.where(
        ~alarms["original_failure"].astype(bool),
        "timely",
        np.where(alarms["late_success_plus10_queries"], "late", "persistent"),
    )
    alarms["late_success_first_extra_query"] = labels[
        "late_success_first_extra_query"
    ]
    if anchor == "alarm":
        alarms = alarms[alarms["first_dual_regime_or_query"] >= 0].copy()
        alarms["query"] = alarms["first_dual_regime_or_query"].astype(int)
    elif anchor == "terminal":
        alarms = alarms[alarms["original_failure"].astype(bool)].copy()
        alarms["query"] = alarms["length"].astype(int) - 1
    elif anchor.startswith("after_alarm_"):
        offset = int(anchor.rsplit("_", 1)[-1])
        alarms = alarms[
            alarms["original_failure"].astype(bool)
            & (alarms["first_dual_regime_or_query"] >= 0)
            & (
                alarms["first_dual_regime_or_query"] + offset
                < alarms["length"]
            )
        ].copy()
        alarms["query"] = (
            alarms["first_dual_regime_or_query"].astype(int) + offset
        )
    else:
        raise KeyError(anchor)
    alarms["anchor"] = anchor
    return alarms.reset_index(drop=True)


def empirical_ranks(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Vectorized columnwise midranks for a small batch of current rows."""
    reference = np.asarray(reference, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    output = np.full(current.shape, np.nan, dtype=np.float32)
    reference_finite = np.isfinite(reference)
    count = reference_finite.sum(axis=0)
    for start in range(0, len(current), 16):
        stop = min(start + 16, len(current))
        values = current[start:stop]
        finite = np.isfinite(values) & (count[None, :] > 0)
        lower = (
            (reference[None, :, :] < values[:, None, :])
            & reference_finite[None, :, :]
        ).sum(axis=1)
        upper = (
            (reference[None, :, :] <= values[:, None, :])
            & reference_finite[None, :, :]
        ).sum(axis=1)
        ranked = (lower + upper) / np.maximum(2 * count[None, :], 1)
        ranked[~finite] = np.nan
        output[start:stop] = ranked
    return output


def extract_anchor(cohort: str, anchor: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    targets = target_table(cohort, anchor)
    output = None
    canonical_names = None
    evaluation_root = REFERENCE_ROOT if cohort == "development_main" else EXTERNAL_ROOT
    for task, block in targets.groupby("task", sort=True):
        reference = load_npz(profile_path(REFERENCE_ROOT, task))
        evaluation = reference if cohort == "development_main" else load_npz(
            profile_path(evaluation_root, task)
        )
        reference_names, reference_matrix = profile_matrix(reference)
        evaluation_names, evaluation_matrix = profile_matrix(evaluation)
        if not np.array_equal(reference_names, evaluation_names):
            raise ValueError(f"{task}: feature schema mismatch")
        if canonical_names is None:
            canonical_names = reference_names
            output = np.full(
                (len(targets), len(canonical_names)), np.nan, dtype=np.float32
            )
        elif not np.array_equal(canonical_names, reference_names):
            raise ValueError(f"{task}: noncanonical feature names")

        reference_query = reference["query"].astype(int)
        evaluation_query = evaluation["query"].astype(int)
        evaluation_episode = evaluation["episode_id"].astype(int)
        for query, query_block in block.groupby("query", sort=True):
            reference_rows = np.flatnonzero(reference_query == int(query))
            current_rows = []
            for episode in query_block["episode"].astype(int):
                take = np.flatnonzero(
                    (evaluation_episode == episode) & (evaluation_query == int(query))
                )
                if len(take) != 1:
                    raise ValueError(
                        f"{cohort}/{task}/episode={episode}/q={query}: found {len(take)}"
                    )
                current_rows.append(int(take[0]))
            ranked = empirical_ranks(
                reference_matrix[reference_rows], evaluation_matrix[current_rows]
            )
            output[query_block.index.to_numpy()] = ranked
    assert output is not None and canonical_names is not None
    return targets, canonical_names, output


def column_mean(matrix: np.ndarray, index: dict[str, int], names: list[str]) -> np.ndarray:
    return np.nanmean(matrix[:, [index[name] for name in names]], axis=1)


def add_fixed_composites(
    table: pd.DataFrame, names: np.ndarray, ranks: np.ndarray
) -> pd.DataFrame:
    table = table.copy()
    index = {name: position for position, name in enumerate(names.astype(str))}
    front_d1 = [f"query|{view}|front_d1" for view in VIEWS]
    back_d1 = [f"query|{view}|back_d1" for view in VIEWS]
    response_ratio = [f"query|{view}|response_log_ratio" for view in VIEWS]
    front_accel = [f"query|{view}|front_acceleration" for view in VIEWS]
    back_accel = [f"query|{view}|back_acceleration" for view in VIEWS]
    front_settling = [f"flow|{view}|front_settling_log_ratio" for view in VIEWS]
    back_settling = [f"flow|{view}|back_settling_log_ratio" for view in VIEWS]
    flow_handoff = [f"flow|{view}|back_front_late_log_ratio" for view in VIEWS]
    handoff_delay = [f"flow|{view}|handoff_delay" for view in VIEWS]
    table["graph_front_response"] = column_mean(ranks, index, front_d1)
    table["graph_back_response"] = column_mean(ranks, index, back_d1)
    table["graph_query_responsiveness"] = column_mean(
        ranks, index, front_d1 + back_d1
    )
    table["graph_response_ratio"] = column_mean(ranks, index, response_ratio)
    table["graph_front_acceleration"] = column_mean(ranks, index, front_accel)
    table["graph_back_acceleration"] = column_mean(ranks, index, back_accel)
    table["graph_front_absorption"] = 1.0 - column_mean(
        ranks, index, front_settling
    )
    table["graph_back_structure"] = column_mean(ranks, index, back_settling)
    table["graph_flow_handoff"] = column_mean(ranks, index, flow_handoff)
    table["graph_handoff_delay"] = column_mean(ranks, index, handoff_delay)
    table["graph_homeostasis"] = np.nanmean(
        np.column_stack(
            (
                table["graph_front_absorption"],
                table["graph_back_structure"],
                table["graph_flow_handoff"],
            )
        ),
        axis=1,
    )
    return table


def safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if len(np.unique(target[finite])) < 2:
        return float("nan")
    return float(roc_auc_score(target[finite], score[finite]))


def safe_ap(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if len(np.unique(target[finite])) < 2:
        return float("nan")
    return float(average_precision_score(target[finite], score[finite]))


def scan_axes(
    development_table: pd.DataFrame,
    development_ranks: np.ndarray,
    external_table: pd.DataFrame,
    external_ranks: np.ndarray,
    names: np.ndarray,
    context: str,
) -> pd.DataFrame:
    development_take = development_table["outcome"].isin(["late", "persistent"])
    external_take = external_table["outcome"].isin(["late", "persistent"])
    development_target = development_table.loc[development_take, "outcome"].eq("late").astype(int).to_numpy()
    external_target = external_table.loc[external_take, "outcome"].eq("late").astype(int).to_numpy()
    development_matrix = development_ranks[development_take.to_numpy()]
    external_matrix = external_ranks[external_take.to_numpy()]
    rows = []
    for column, name in enumerate(names.astype(str)):
        development_score = development_matrix[:, column]
        external_score = external_matrix[:, column]
        development_auc = safe_auc(development_target, development_score)
        development_coverage = float(np.isfinite(development_score).mean())
        external_coverage = float(np.isfinite(external_score).mean())
        if not np.isfinite(development_auc):
            continue
        sign = 1 if development_auc >= 0.5 else -1
        rows.append(
            {
                "context": context,
                "feature": name,
                "development_auc_raw": development_auc,
                "development_selected_sign": sign,
                "development_separation": abs(development_auc - 0.5),
                "development_coverage": development_coverage,
                "external_auc_frozen_direction": safe_auc(
                    external_target, sign * external_score
                ),
                "external_ap_frozen_direction": safe_ap(
                    external_target, sign * external_score
                ),
                "external_coverage": external_coverage,
                "external_late_prevalence": float(external_target.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["development_separation", "external_auc_frozen_direction"],
        ascending=False,
    )


def composite_metrics(
    development: pd.DataFrame, external: pd.DataFrame, context: str
) -> pd.DataFrame:
    features = [column for column in development if column.startswith("graph_")]
    rows = []
    for population, positive in (
        ("risk_only_late_vs_persistent", "late"),
        ("all_alarms_continue_vs_intervene", "nonpersistent"),
    ):
        if context != "alarm" and population.startswith("all_alarms"):
            continue
        development_block = development
        external_block = external
        if population.startswith("risk_only"):
            development_block = development[
                development["outcome"].isin(["late", "persistent"])
            ]
            external_block = external[
                external["outcome"].isin(["late", "persistent"])
            ]
        development_target = (
            development_block["outcome"].eq("late")
            if positive == "late"
            else development_block["outcome"].ne("persistent")
        ).astype(int)
        external_target = (
            external_block["outcome"].eq("late")
            if positive == "late"
            else external_block["outcome"].ne("persistent")
        ).astype(int)
        for feature in features:
            # Fixed raw direction is reported. No outcome direction is hidden.
            rows.append(
                {
                    "context": context,
                    "population": population,
                    "feature": feature,
                    "development_auc_raw": safe_auc(
                        development_target, development_block[feature]
                    ),
                    "development_ap_raw": safe_ap(
                        development_target, development_block[feature]
                    ),
                    "external_auc_raw": safe_auc(
                        external_target, external_block[feature]
                    ),
                    "external_ap_raw": safe_ap(
                        external_target, external_block[feature]
                    ),
                    "external_prevalence": float(external_target.mean()),
                }
            )
    return pd.DataFrame(rows)


def merge_deltas(
    alarm: pd.DataFrame, terminal: pd.DataFrame
) -> pd.DataFrame:
    alarm = alarm[alarm["outcome"].isin(["late", "persistent"])]
    keys = ["cohort", "row", "task", "suite", "episode", "outcome"]
    columns = keys + [column for column in alarm if column.startswith("graph_")]
    merged = alarm[columns].merge(
        terminal[columns], on=keys, suffixes=("_alarm", "_terminal")
    )
    for feature in [column for column in alarm if column.startswith("graph_")]:
        merged[f"delta_{feature}"] = (
            merged[f"{feature}_terminal"] - merged[f"{feature}_alarm"]
        )
    return merged


def delta_rank_matrices(
    alarm_table: pd.DataFrame,
    alarm_ranks: np.ndarray,
    terminal_table: pd.DataFrame,
    terminal_ranks: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    alarm_lookup = {int(row): position for position, row in enumerate(alarm_table["row"])}
    terminal_lookup = {
        int(row): position for position, row in enumerate(terminal_table["row"])
    }
    rows = sorted(set(alarm_lookup) & set(terminal_lookup))
    metadata = alarm_table.set_index("row").loc[rows].reset_index()
    delta = np.stack(
        [
            terminal_ranks[terminal_lookup[row]] - alarm_ranks[alarm_lookup[row]]
            for row in rows
        ]
    )
    return metadata, delta


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


def within_task_auc(
    table: pd.DataFrame, score: np.ndarray
) -> tuple[float, float, int]:
    work = table[["task", "outcome"]].copy()
    work["score"] = score
    aucs = []
    weights = []
    for _, block in work.groupby("task", sort=True):
        finite = block["score"].notna()
        block = block[finite]
        target = block["outcome"].eq("late").astype(int)
        if target.nunique() < 2:
            continue
        aucs.append(roc_auc_score(target, block["score"]))
        weights.append(int(target.sum() * (1 - target).sum()))
    return (
        float(np.average(aucs, weights=weights)) if aucs else float("nan"),
        float(np.mean(aucs)) if aucs else float("nan"),
        len(aucs),
    )


def post_alarm_curve(
    development_alarm: pd.DataFrame,
    development_alarm_ranks: np.ndarray,
    external_alarm: pd.DataFrame,
    external_alarm_ranks: np.ndarray,
    names: np.ndarray,
) -> pd.DataFrame:
    index = {name: position for position, name in enumerate(names.astype(str))}
    axes = {
        "conditional_front_release": "query|action_conditional|front_d1",
        "conditional_gap_release": "query|action_conditional|gap_d1",
    }
    rows = []
    for offset in (1, 2, 4, 8):
        snapshots = {}
        for cohort in ("development_main", "external_8b"):
            table, current_names, ranks = extract_anchor(
                cohort, f"after_alarm_{offset}"
            )
            if not np.array_equal(names, current_names):
                raise ValueError("post-alarm graph feature schema mismatch")
            table = add_fixed_composites(table, names, ranks)
            snapshots[cohort] = (table, ranks)
        for cohort, baseline_table, baseline_ranks in (
            ("development_main", development_alarm, development_alarm_ranks),
            ("external_8b", external_alarm, external_alarm_ranks),
        ):
            current_table, current_ranks = snapshots[cohort]
            baseline_position = {
                int(row): position for position, row in enumerate(baseline_table["row"])
            }
            current_position = {
                int(row): position for position, row in enumerate(current_table["row"])
            }
            common = sorted(set(baseline_position) & set(current_position))
            target = (
                baseline_table.set_index("row").loc[common, "outcome"].eq("late").astype(int).to_numpy()
            )
            for score, axis in axes.items():
                values = np.asarray(
                    [
                        current_ranks[current_position[row], index[axis]]
                        - baseline_ranks[baseline_position[row], index[axis]]
                        for row in common
                    ],
                    dtype=float,
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "offset_queries": offset,
                        "score": score,
                        "n": len(common),
                        "late_n": int(target.sum()),
                        "late_prevalence": float(target.mean()),
                        "auc": safe_auc(target, values),
                        "average_precision": safe_ap(target, values),
                    }
                )
            for score in (
                "graph_front_response",
                "graph_back_response",
                "graph_query_responsiveness",
                "graph_homeostasis",
            ):
                baseline_values = baseline_table.set_index("row").loc[common, score].to_numpy(dtype=float)
                current_values = current_table.set_index("row").loc[common, score].to_numpy(dtype=float)
                values = current_values - baseline_values
                rows.append(
                    {
                        "cohort": cohort,
                        "offset_queries": offset,
                        "score": score + "_release",
                        "n": len(common),
                        "late_n": int(target.sum()),
                        "late_prevalence": float(target.mean()),
                        "auc": safe_auc(target, values),
                        "average_precision": safe_ap(target, values),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    extracted = {}
    canonical_names = None
    for cohort in ("development_main", "external_8b"):
        for anchor in ("alarm", "terminal"):
            table, names, ranks = extract_anchor(cohort, anchor)
            table = add_fixed_composites(table, names, ranks)
            if canonical_names is None:
                canonical_names = names
            elif not np.array_equal(canonical_names, names):
                raise ValueError("cross-cohort graph feature schema mismatch")
            extracted[(cohort, anchor)] = (table, ranks)
    assert canonical_names is not None

    for anchor in ("alarm", "terminal"):
        frames = []
        matrices = []
        for cohort in ("development_main", "external_8b"):
            table, ranks = extracted[(cohort, anchor)]
            frames.append(table)
            matrices.append(ranks)
        metadata = pd.concat(frames, ignore_index=True)
        matrix = np.concatenate(matrices, axis=0)
        metadata.to_csv(args.output / f"{anchor}_metadata_and_composites.csv", index=False)
        np.savez_compressed(
            args.output / f"{anchor}_graph_ranks.npz",
            feature_names=canonical_names,
            ranks=matrix.astype(np.float16),
        )

    development_alarm, development_alarm_ranks = extracted[("development_main", "alarm")]
    external_alarm, external_alarm_ranks = extracted[("external_8b", "alarm")]
    development_terminal, development_terminal_ranks = extracted[("development_main", "terminal")]
    external_terminal, external_terminal_ranks = extracted[("external_8b", "terminal")]

    alarm_scan = scan_axes(
        development_alarm,
        development_alarm_ranks,
        external_alarm,
        external_alarm_ranks,
        canonical_names,
        "alarm",
    )
    terminal_scan = scan_axes(
        development_terminal,
        development_terminal_ranks,
        external_terminal,
        external_terminal_ranks,
        canonical_names,
        "terminal",
    )
    axis_scan = pd.concat([alarm_scan, terminal_scan], ignore_index=True)
    axis_scan.to_csv(args.output / "axis_metrics.csv", index=False)

    composite = pd.concat(
        [
            composite_metrics(development_alarm, external_alarm, "alarm"),
            composite_metrics(development_terminal, external_terminal, "terminal"),
        ],
        ignore_index=True,
    )
    composite.to_csv(args.output / "fixed_composite_metrics.csv", index=False)

    development_delta_table, development_delta_ranks = delta_rank_matrices(
        development_alarm,
        development_alarm_ranks,
        development_terminal,
        development_terminal_ranks,
    )
    external_delta_table, external_delta_ranks = delta_rank_matrices(
        external_alarm,
        external_alarm_ranks,
        external_terminal,
        external_terminal_ranks,
    )
    delta_scan = scan_axes(
        development_delta_table,
        development_delta_ranks,
        external_delta_table,
        external_delta_ranks,
        canonical_names,
        "terminal_minus_alarm",
    )
    delta_scan.to_csv(args.output / "delta_axis_metrics.csv", index=False)
    development_delta = merge_deltas(development_alarm, development_terminal)
    external_delta = merge_deltas(external_alarm, external_terminal)
    delta_composite = composite_metrics(
        development_delta.rename(
            columns={
                column: column.removeprefix("delta_")
                for column in development_delta
                if column.startswith("delta_graph_")
            }
        ),
        external_delta.rename(
            columns={
                column: column.removeprefix("delta_")
                for column in external_delta
                if column.startswith("delta_graph_")
            }
        ),
        "terminal_minus_alarm",
    )
    delta_composite.to_csv(args.output / "delta_composite_metrics.csv", index=False)

    curve = post_alarm_curve(
        development_alarm,
        development_alarm_ranks,
        external_alarm,
        external_alarm_ranks,
        canonical_names,
    )
    curve.to_csv(args.output / "post_alarm_curve.csv", index=False)

    # Effect transfer across all sufficiently covered axes is more informative
    # than the single best value from a 584-axis scan.
    transfer_take = alarm_scan[
        (alarm_scan["development_coverage"] >= 0.95)
        & (alarm_scan["external_coverage"] >= 0.95)
    ].copy()
    development_effect = transfer_take["development_auc_raw"] - 0.5
    external_effect = (
        transfer_take["external_auc_frozen_direction"] - 0.5
    ) * transfer_take["development_selected_sign"]
    effect_correlation = spearmanr(development_effect, external_effect)

    rng = np.random.default_rng(args.seed)
    external_delta_target = external_delta_table["outcome"].eq("late").astype(int).to_numpy()
    development_delta_target = development_delta_table["outcome"].eq("late").astype(int).to_numpy()
    graph_axis = "query|action_conditional|front_d1"
    graph_axis_position = list(canonical_names.astype(str)).index(graph_axis)
    primary_delta = external_delta_ranks[:, graph_axis_position].astype(float)
    development_primary_delta = development_delta_ranks[:, graph_axis_position].astype(float)
    primary_auc = safe_auc(external_delta_target, primary_delta)
    within_weighted, within_macro, within_tasks = within_task_auc(
        external_delta_table, primary_delta
    )
    fixed_delta = external_delta["delta_graph_homeostasis"].to_numpy(dtype=float)
    summary = {
        "schema": "himoe.feedback_responsiveness.graph_stage2.v1",
        "method": {
            "trained_model": False,
            "runtime_moe_only": True,
            "features_per_query": len(canonical_names),
            "raw_router_shape": [8, 10, 11, 32],
            "normalization": "task- and query-matched outcome-blind empirical ranks",
            "development_axis_selection_uses_outcomes": True,
            "fixed_composites_use_outcomes": False,
            "external_is_pristine_holdout": False,
        },
        "counts": {
            "development_alarm": development_alarm["outcome"].value_counts().to_dict(),
            "external_alarm": external_alarm["outcome"].value_counts().to_dict(),
            "development_terminal": development_terminal["outcome"].value_counts().to_dict(),
            "external_terminal": external_terminal["outcome"].value_counts().to_dict(),
            "development_alarmed_risk_delta": development_delta["outcome"].value_counts().to_dict(),
            "external_alarmed_risk_delta": external_delta["outcome"].value_counts().to_dict(),
        },
        "axis_effect_transfer": {
            "axes_with_95pct_coverage": len(transfer_take),
            "spearman_r": effect_correlation.statistic,
            "p_value": effect_correlation.pvalue,
            "top10_development_axes_external_auc": alarm_scan.head(10)[
                "external_auc_frozen_direction"
            ].tolist(),
        },
        "fixed_dynamic_graph_result": {
            "score": "terminal minus alarm graph_homeostasis",
            "development_auc": safe_auc(
                development_delta["outcome"].eq("late").astype(int),
                development_delta["delta_graph_homeostasis"],
            ),
            "external_auc": safe_auc(external_delta_target, fixed_delta),
            "external_auc_ci95_row_stratified": bootstrap_ci(
                external_delta_target, fixed_delta, args.bootstrap, rng
            ),
            "external_ap": safe_ap(external_delta_target, fixed_delta),
            "external_late_prevalence": float(external_delta_target.mean()),
        },
        "exploratory_dynamic_graph_result": {
            "score": f"terminal minus alarm {graph_axis}",
            "selection": (
                "mechanistically relevant axis inspected after development-axis scan; "
                "not a preregistered confirmatory score"
            ),
            "development_auc": safe_auc(
                development_delta_target, development_primary_delta
            ),
            "external_auc": primary_auc,
            "external_auc_ci95_row_stratified": bootstrap_ci(
                external_delta_target, primary_delta, args.bootstrap, rng
            ),
            "external_ap": safe_ap(external_delta_target, primary_delta),
            "external_late_prevalence": float(external_delta_target.mean()),
            "external_within_task_auc_weighted": within_weighted,
            "external_within_task_auc_macro": within_macro,
            "external_within_task_tasks": within_tasks,
        },
        "post_alarm_curve": curve.to_dict(orient="records"),
        "top_alarm_axes": alarm_scan.head(20).to_dict(orient="records"),
        "top_delta_axes": delta_scan.head(20).to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plain(summary), indent=2)[:20_000])


if __name__ == "__main__":
    main()
