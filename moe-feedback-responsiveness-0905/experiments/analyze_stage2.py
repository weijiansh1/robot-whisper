#!/usr/bin/env python3
"""Evaluate MoE-only extend-vs-intervene evidence at the frozen v4 alarm.

The analysis is intentionally train-free. Raw query features are converted to
task- and query-matched empirical ranks using outcome-blind reference routes.
No outcome, configured horizon, future query, or physical state enters a score.
"""

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
WORKSPACE = BUNDLE.parent
V4 = WORKSPACE / "moe-v4-0904"
BASE = WORKSPACE / "double-selete/trainfree/results"

FEATURE_PATHS = {
    "development_main": BASE / "online_multihead_hub/unlabeled_query_features.npz",
    "external_8b": BASE
    / "online_precision_cascade_external/unlabeled_query_features.npz",
}
MOBILITY_PATHS = {
    "development_main": V4 / "results/layerwise_mobility/main_reference.npz",
    "external_8b": V4 / "results/layerwise_mobility/external_8b.npz",
}
EXTERNAL_REFERENCE = (
    BASE / "online_precision_cascade_external/deployment_profiles.npz"
)
ALARMS = V4 / "results/cache_new_v4/episode_alarms.csv"
SEALED = V4 / "results/cache_new_v4/sealed_first_alarms.npz"
LABEL_PATHS = {
    "development_main": BASE
    / "timeout_extension_plus10/development_main_clean_labels.csv",
    "external_8b": BASE / "timeout_extension_plus10/external_8b_clean_labels.csv",
}

META_COLUMNS = {
    "cohort",
    "row",
    "task",
    "suite",
    "outcome",
    "risk",
    "q",
    "length",
    "phase",
    "lock_alarm",
    "instability_alarm",
    "late_success_first_extra_query",
}

# These directions are mechanism-defined. Positive means more reason to keep
# executing rather than intervene. They are not selected from outcome labels.
PREDECLARED_DIRECTIONS = {
    "route_activity_now": 1,
    "route_activity_w4": 1,
    "route_activity_prefix_mean": 1,
    "response_history_score": 1,
    "route_mobility_w4": 1,
    "route_mobility_prefix_mean": 1,
    "route_mobility_prefix_std": 1,
    "mobility_high_fraction": 1,
    "mobility_center_cross_rate": 1,
    "mobility_burst_count_norm": 1,
    "mobility_recent_highness": 1,
    "mobility_settlement_drop": 1,
    "lock_head_w4": -1,
    "instability_head_w4": -1,
    "front_minus_back_w4": 1,
    "front_lead_asymmetry": 1,
    "layer_dynamics_effrank": 1,
    "layer_mean_abs_corr": 1,
    "layer_profile_dispersion_w4": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BUNDLE / "results")
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


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


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if (
        finite.sum() < 4
        or np.std(left[finite]) < 1e-9
        or np.std(right[finite]) < 1e-9
    ):
        return float("nan")
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def safe_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return float("nan")
    query = np.arange(len(values), dtype=np.float64)[finite]
    return float(np.polyfit(query, values[finite], 1)[0])


def dynamics_effective_rank(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite]
    if len(matrix) < 3:
        return float("nan")
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0)
    active = scale > 1e-9
    matrix[:, active] /= scale[active]
    matrix[:, ~active] = 0.0
    singular = np.linalg.svd(matrix, compute_uv=False)
    mass = singular**2
    if mass.sum() <= 0:
        return 0.0
    probability = mass / mass.sum()
    probability = probability[probability > 0]
    rank = np.exp(-(probability * np.log(probability)).sum())
    return float(rank / min(matrix.shape))


def empirical_rank(reference: np.ndarray, value: float) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    reference = np.sort(reference[np.isfinite(reference)])
    if not len(reference) or not np.isfinite(value):
        return float("nan")
    left = np.searchsorted(reference, value, side="left")
    right = np.searchsorted(reference, value, side="right")
    return float((left + right) / (2.0 * len(reference)))


def ranked_prefix(
    episode_features: np.ndarray,
    reference_features: np.ndarray,
    reference_task_index: np.ndarray,
    task_position: int,
    final_query: int,
) -> np.ndarray:
    output = np.full(
        (final_query + 1, episode_features.shape[-1]), np.nan, dtype=np.float32
    )
    rows = np.flatnonzero(reference_task_index == task_position)
    for query in range(final_query + 1):
        current = episode_features[query]
        reference = reference_features[rows, query]
        for column in range(episode_features.shape[-1]):
            output[query, column] = empirical_rank(
                reference[:, column], current[column]
            )
    return output


def add_window_features(
    output: dict[str, Any], name: str, values: np.ndarray
) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    output[f"{name}_now"] = values[-1]
    output[f"{name}_w4"] = float(values[-4:].mean())
    output[f"{name}_w8"] = float(values[-8:].mean())
    output[f"{name}_prefix_mean"] = float(values.mean())
    output[f"{name}_prefix_std"] = float(values.std())
    output[f"{name}_last8_slope"] = safe_slope(values[-8:])
    output[f"{name}_last8_range"] = float(np.ptp(values[-8:]))


def add_response_cycle_features(
    output: dict[str, Any], mobility_rank: np.ndarray
) -> None:
    values = np.asarray(mobility_rank, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    high = values > 0.75
    center = values > 0.50
    output["mobility_high_fraction"] = float(high.mean())
    output["mobility_center_cross_rate"] = (
        float(np.mean(center[1:] != center[:-1])) if len(center) > 1 else 0.0
    )
    starts = high & np.r_[True, ~high[:-1]]
    output["mobility_burst_count_norm"] = float(starts.sum() / len(high))
    positions = np.flatnonzero(high)
    output["mobility_recent_highness"] = (
        0.0
        if not len(positions)
        else float(1.0 - (len(values) - 1 - positions[-1]) / len(values))
    )
    output["mobility_settlement_drop"] = (
        float(values[-8:-4].mean() - values[-4:].mean())
        if len(values) >= 8
        else float("nan")
    )


def add_layer_structure(
    output: dict[str, Any], mobility: np.ndarray, lock_cutoff: float
) -> None:
    mobility = np.asarray(mobility, dtype=np.float64)
    mobility = mobility[np.isfinite(mobility).all(axis=1)]
    if not len(mobility):
        return
    cutoff = max(lock_cutoff, 1e-8)
    median = np.median(mobility, axis=1) / cutoff
    front = np.mean(mobility[:, :4], axis=1) / cutoff
    back = np.mean(mobility[:, 4:], axis=1) / cutoff
    add_window_features(output, "layer_median_ratio", median)
    add_window_features(output, "layer_front_ratio", front)
    add_window_features(output, "layer_back_ratio", back)
    output["front_minus_back_w4"] = float(np.mean(front[-4:] - back[-4:]))
    output["front_back_corr0"] = safe_corr(front, back)
    output["front_leads_back_corr1"] = safe_corr(front[:-1], back[1:])
    output["back_leads_front_corr1"] = safe_corr(back[:-1], front[1:])
    output["front_lead_asymmetry"] = (
        output["front_leads_back_corr1"] - output["back_leads_front_corr1"]
    )
    output["layer_dynamics_effrank"] = dynamics_effective_rank(mobility)
    correlations = np.corrcoef(mobility, rowvar=False)
    upper = np.abs(correlations[np.triu_indices(mobility.shape[1], 1)])
    finite = upper[np.isfinite(upper)]
    output["layer_mean_abs_corr"] = (
        float(finite.mean()) if len(finite) else float("nan")
    )
    recent = mobility[-4:]
    dispersion = np.std(recent, axis=1) / (np.mean(recent, axis=1) + 1e-8)
    output["layer_profile_dispersion_w4"] = float(dispersion.mean())


def load_label_metadata(cohort: str, expected: int) -> pd.DataFrame:
    labels = pd.read_csv(LABEL_PATHS[cohort])
    if len(labels) != expected:
        raise ValueError(f"{cohort}: label row mismatch")
    return labels


def extract_features(cohort: str, anchor: str) -> pd.DataFrame:
    route = load_npz(FEATURE_PATHS[cohort])
    layer = load_npz(MOBILITY_PATHS[cohort])
    sealed = load_npz(SEALED)
    external = cohort == "external_8b"
    prefix = "external" if external else "main"
    first = sealed[f"{prefix}_dual"].astype(int)
    first_lock = sealed[f"{prefix}_lock"].astype(int)
    first_instability = sealed[f"{prefix}_instability"].astype(int)
    lock_threshold = sealed[f"{prefix}_lock_threshold"].astype(float)

    alarm_table = pd.read_csv(ALARMS)
    alarm_table = (
        alarm_table[alarm_table["cohort"] == cohort]
        .sort_values("row")
        .reset_index(drop=True)
    )
    labels = load_label_metadata(cohort, len(first))
    if len(alarm_table) != len(first):
        raise ValueError(f"{cohort}: alarm row mismatch")
    task_names = route["task_names"].astype(str)
    episode_tasks = task_names[route["task_index"].astype(int)]
    if not np.array_equal(episode_tasks, alarm_table["task"].astype(str)):
        raise ValueError(f"{cohort}: route/alarm task mismatch")
    if not np.array_equal(route["length"], layer["length"]):
        raise ValueError(f"{cohort}: route/layer length mismatch")

    if external:
        profile = load_npz(EXTERNAL_REFERENCE)
        if not np.array_equal(task_names, profile["task_names"].astype(str)):
            raise ValueError("external reference task order mismatch")
        reference_features = profile["route_reference"]
        reference_task_index = profile["reference_task_index"].astype(int)
    else:
        # This is outcome-blind in-sample empirical ranking. Self-inclusion changes
        # a rank by at most 1/400 and is disclosed in the report.
        reference_features = route["features"]
        reference_task_index = route["task_index"].astype(int)

    feature_names = route["feature_names"].astype(str).tolist()
    feature_index = {name: position for position, name in enumerate(feature_names)}
    rows: list[dict[str, Any]] = []
    for row_index, alarm_query in enumerate(first):
        risk = bool(alarm_table.loc[row_index, "original_failure"])
        if anchor == "alarm":
            if alarm_query < 0:
                continue
            final_query = int(alarm_query)
        elif anchor == "terminal":
            if not risk:
                continue
            final_query = int(route["length"][row_index]) - 1
        else:
            raise KeyError(anchor)

        late = bool(alarm_table.loc[row_index, "late_success_plus10_queries"])
        outcome = "timely" if not risk else ("late" if late else "persistent")
        ranks = ranked_prefix(
            route["features"][row_index],
            reference_features,
            reference_task_index,
            int(route["task_index"][row_index]),
            final_query,
        )
        start = min(4, final_query)
        ranks = ranks[start : final_query + 1]
        output: dict[str, Any] = {
            "cohort": cohort,
            "row": row_index,
            "task": episode_tasks[row_index],
            "suite": episode_tasks[row_index].split("/", 1)[0],
            "outcome": outcome,
            "risk": risk,
            "q": final_query,
            "length": int(route["length"][row_index]),
            # Audit only. Phase and q are never candidate score columns.
            "phase": final_query / max(int(route["length"][row_index]) - 1, 1),
            "lock_alarm": int(first_lock[row_index]),
            "instability_alarm": int(first_instability[row_index]),
            "late_success_first_extra_query": labels.loc[
                row_index, "late_success_first_extra_query"
            ],
        }
        for name in feature_names:
            add_window_features(output, name, ranks[:, feature_index[name]])

        route_activity = (
            ranks[:, feature_index["route_mobility"]]
            + ranks[:, feature_index["front_feedback_split"]]
            + (1.0 - ranks[:, feature_index["lag_recurrence"]])
        ) / 3.0
        add_window_features(output, "route_activity", route_activity)
        instability = (
            ranks[:, feature_index["late_flow_volatility"]]
            + ranks[:, feature_index["route_acceleration"]]
            + ranks[:, feature_index["route_mobility"]]
            + ranks[:, feature_index["lag_periodicity"]]
        ) / 4.0
        lock = (
            (1.0 - ranks[:, feature_index["route_mobility"]])
            + ranks[:, feature_index["lag_recurrence"]]
            + (1.0 - ranks[:, feature_index["top4_union"]])
            + (1.0 - ranks[:, feature_index["token_disagreement"]])
        ) / 4.0
        add_window_features(output, "instability_head", instability)
        add_window_features(output, "lock_head", lock)
        add_response_cycle_features(
            output, ranks[:, feature_index["route_mobility"]]
        )

        valid = layer["valid"][row_index, : final_query + 1].astype(bool)
        add_layer_structure(
            output,
            layer["mobility"][row_index, : final_query + 1][valid],
            -float(lock_threshold[row_index]),
        )
        output["response_history_score"] = float(
            np.mean(
                [
                    output["route_activity_prefix_mean"],
                    output["mobility_high_fraction"],
                    min(1.0, 2.0 * output["mobility_center_cross_rate"]),
                    output["mobility_recent_highness"],
                ]
            )
        )
        rows.append(output)
    return pd.DataFrame(rows)


def safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    if len(np.unique(target[finite])) < 2:
        return float("nan")
    return float(roc_auc_score(target[finite], score[finite]))


def safe_ap(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    if len(np.unique(target[finite])) < 2:
        return float("nan")
    return float(average_precision_score(target[finite], score[finite]))


def within_task_auc(frame: pd.DataFrame, score: str, sign: int) -> dict[str, Any]:
    aucs: list[float] = []
    weights: list[int] = []
    for _, block in frame.groupby("task", sort=True):
        target = (block["outcome"] == "late").astype(int).to_numpy()
        values = sign * block[score].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if len(np.unique(target[finite])) < 2:
            continue
        aucs.append(float(roc_auc_score(target[finite], values[finite])))
        weights.append(int(target[finite].sum() * (1 - target[finite]).sum()))
    return {
        "within_task_auc_weighted": (
            float(np.average(aucs, weights=weights)) if aucs else float("nan")
        ),
        "within_task_auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
        "within_task_tasks": len(aucs),
    }


def candidate_metrics(
    development: pd.DataFrame, external: pd.DataFrame
) -> pd.DataFrame:
    development = development[
        development["outcome"].isin(["late", "persistent"])
    ]
    external = external[external["outcome"].isin(["late", "persistent"])]
    columns = sorted(
        (set(development.columns) & set(external.columns)) - META_COLUMNS
    )
    development_target = (development["outcome"] == "late").astype(int).to_numpy()
    external_target = (external["outcome"] == "late").astype(int).to_numpy()
    rows = []
    for column in columns:
        development_auc = safe_auc(development_target, development[column])
        if not np.isfinite(development_auc):
            continue
        sign = 1 if development_auc >= 0.5 else -1
        row = {
            "feature": column,
            "development_auc_raw": development_auc,
            "development_selected_sign": sign,
            "development_separation": abs(development_auc - 0.5),
            "external_auc_frozen_direction": safe_auc(
                external_target, sign * external[column]
            ),
            "external_ap_frozen_direction": safe_ap(
                external_target, sign * external[column]
            ),
            "external_late_prevalence": float(external_target.mean()),
        }
        row.update(within_task_auc(external, column, sign))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["development_separation", "external_auc_frozen_direction"],
        ascending=False,
    )


def predeclared_metrics(
    development: pd.DataFrame, external: pd.DataFrame, context: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, positive, development_take, external_take in (
        (
            "risk_only_late_vs_persistent",
            "late",
            development["outcome"].isin(["late", "persistent"]),
            external["outcome"].isin(["late", "persistent"]),
        ),
        (
            "all_alarms_continue_vs_intervene",
            "nonpersistent",
            np.ones(len(development), dtype=bool),
            np.ones(len(external), dtype=bool),
        ),
    ):
        if context != "alarm" and population.startswith("all_alarms"):
            continue
        development_block = development[development_take]
        external_block = external[external_take]
        development_target = (
            (development_block["outcome"] == "late")
            if positive == "late"
            else (development_block["outcome"] != "persistent")
        ).astype(int)
        external_target = (
            (external_block["outcome"] == "late")
            if positive == "late"
            else (external_block["outcome"] != "persistent")
        ).astype(int)
        for score, sign in PREDECLARED_DIRECTIONS.items():
            if score not in development_block or score not in external_block:
                continue
            rows.append(
                {
                    "context": context,
                    "population": population,
                    "feature": score,
                    "predeclared_sign": sign,
                    "development_n": len(development_block),
                    "development_positive_n": int(development_target.sum()),
                    "development_auc": safe_auc(
                        development_target, sign * development_block[score]
                    ),
                    "development_ap": safe_ap(
                        development_target, sign * development_block[score]
                    ),
                    "external_n": len(external_block),
                    "external_positive_n": int(external_target.sum()),
                    "external_auc": safe_auc(
                        external_target, sign * external_block[score]
                    ),
                    "external_ap": safe_ap(
                        external_target, sign * external_block[score]
                    ),
                    "external_prevalence": float(external_target.mean()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_interval(
    target: np.ndarray,
    score: np.ndarray,
    metric: str,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    target = target[finite]
    score = score[finite]
    negative = np.flatnonzero(target == 0)
    positive = np.flatnonzero(target == 1)
    values = np.empty(draws, dtype=np.float64)
    function = roc_auc_score if metric == "auc" else average_precision_score
    for draw in range(draws):
        take = np.r_[
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ]
        values[draw] = function(target[take], score[take])
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def primary_intervals(
    external_alarm: pd.DataFrame, draws: int, rng: np.random.Generator
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for population, block, target in (
        (
            "risk_only",
            external_alarm[
                external_alarm["outcome"].isin(["late", "persistent"])
            ],
            None,
        ),
        ("all_alarms", external_alarm, None),
    ):
        if population == "risk_only":
            target = (block["outcome"] == "late").astype(int).to_numpy()
        else:
            target = (block["outcome"] != "persistent").astype(int).to_numpy()
        for score in ("route_activity_w4", "response_history_score"):
            values = block[score].to_numpy(dtype=float)
            output[f"{population}_{score}"] = {
                "n": len(block),
                "positive_n": int(target.sum()),
                "auc": safe_auc(target, values),
                "auc_ci95": bootstrap_interval(target, values, "auc", draws, rng),
                "ap": safe_ap(target, values),
                "ap_ci95": bootstrap_interval(target, values, "ap", draws, rng),
                "prevalence": float(target.mean()),
            }
    return output


def threshold_policies(
    development: pd.DataFrame,
    external: pd.DataFrame,
    context: str,
) -> pd.DataFrame:
    rows = []
    for score in ("route_activity_w4", "response_history_score"):
        for quantile in (0.50, 0.75):
            threshold = float(np.nanquantile(development[score], quantile))
            selected = external[score].to_numpy(dtype=float) >= threshold
            late = external["outcome"].eq("late").to_numpy()
            persistent = external["outcome"].eq("persistent").to_numpy()
            timely = external["outcome"].eq("timely").to_numpy()
            nonpersistent = ~persistent
            rows.append(
                {
                    "context": context,
                    "feature": score,
                    "calibration_quantile": quantile,
                    "calibration_used_outcomes": False,
                    "threshold": threshold,
                    "external_n": len(external),
                    "selected_n": int(selected.sum()),
                    "selected_fraction": float(selected.mean()),
                    "late_selected": int((selected & late).sum()),
                    "late_n": int(late.sum()),
                    "late_recall": float((selected & late).sum() / max(late.sum(), 1)),
                    "persistent_selected": int((selected & persistent).sum()),
                    "persistent_n": int(persistent.sum()),
                    "persistent_extension_rate": float(
                        (selected & persistent).sum() / max(persistent.sum(), 1)
                    ),
                    "timely_selected": int((selected & timely).sum()),
                    "timely_n": int(timely.sum()),
                    "nonpersistent_precision": float(
                        (selected & nonpersistent).sum() / max(selected.sum(), 1)
                    ),
                    "late_precision_risk_only": float(
                        (selected & late).sum()
                        / max((selected & (late | persistent)).sum(), 1)
                    ),
                }
            )
    return pd.DataFrame(rows)


def outcome_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame["outcome"].value_counts().items()}


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    development_alarm = extract_features("development_main", "alarm")
    external_alarm = extract_features("external_8b", "alarm")
    development_terminal = extract_features("development_main", "terminal")
    external_terminal = extract_features("external_8b", "terminal")

    alarm_features = pd.concat(
        [development_alarm, external_alarm], ignore_index=True
    )
    terminal_features = pd.concat(
        [development_terminal, external_terminal], ignore_index=True
    )
    alarm_features.to_csv(args.output / "alarm_prefix_features.csv", index=False)
    terminal_features.to_csv(args.output / "terminal_prefix_features.csv", index=False)

    candidates = candidate_metrics(development_alarm, external_alarm)
    candidates.to_csv(args.output / "candidate_axis_metrics.csv", index=False)
    predeclared = pd.concat(
        [
            predeclared_metrics(
                development_alarm, external_alarm, context="alarm"
            ),
            predeclared_metrics(
                development_terminal, external_terminal, context="terminal"
            ),
        ],
        ignore_index=True,
    )
    predeclared.to_csv(args.output / "predeclared_metrics.csv", index=False)

    alarm_policies = threshold_policies(
        development_alarm, external_alarm, context="alarm_all_alarms"
    )
    terminal_policies = threshold_policies(
        development_terminal,
        external_terminal,
        context="terminal_original_failures",
    )
    policies = pd.concat([alarm_policies, terminal_policies], ignore_index=True)
    policies.to_csv(args.output / "allocation_policies.csv", index=False)

    rng = np.random.default_rng(args.seed)
    summary = {
        "schema": "himoe.feedback_responsiveness.stage2.v1",
        "method": {
            "trained_model": False,
            "runtime_moe_only": True,
            "runtime_query_causal": True,
            "score_uses_query_or_horizon": False,
            "normalization_uses_outcomes": False,
            "development_rank_self_inclusion": (
                "outcome-blind; at most one of 400 task references"
            ),
            "external_reference": str(EXTERNAL_REFERENCE.relative_to(WORKSPACE)),
            "primary_score": (
                "equal mean of route-mobility rank, front-feedback-split rank, "
                "and inverse lag-recurrence rank over the latest four queries"
            ),
            "response_history_score": (
                "equal mean of prefix route activity, high-mobility fraction, "
                "twice the center-crossing rate (clipped), and recency of the "
                "last high-mobility response"
            ),
        },
        "counts": {
            "development_alarm": outcome_counts(development_alarm),
            "external_alarm": outcome_counts(external_alarm),
            "development_terminal": outcome_counts(development_terminal),
            "external_terminal": outcome_counts(external_terminal),
        },
        "external_primary_intervals": primary_intervals(
            external_alarm, args.bootstrap, rng
        ),
        "top_development_selected_candidates": candidates.head(10).to_dict(
            orient="records"
        ),
        "artifacts": {
            "candidate_axis_metrics": "candidate_axis_metrics.csv",
            "predeclared_metrics": "predeclared_metrics.csv",
            "allocation_policies": "allocation_policies.csv",
            "alarm_prefix_features": "alarm_prefix_features.csv",
            "terminal_prefix_features": "terminal_prefix_features.csv",
        },
        "validity": {
            "external_is_pristine_holdout": False,
            "late_success_window_queries": 10,
            "persistent_is_right_censored": True,
            "external_alarm_late_n": int(
                external_alarm["outcome"].eq("late").sum()
            ),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Alarm counts")
    print(pd.DataFrame(summary["counts"]).fillna(0).to_string())
    print("\nPrimary external intervals")
    print(json.dumps(plain(summary["external_primary_intervals"]), indent=2))
    print("\nPredeclared alarm metrics")
    take = predeclared[
        (predeclared["context"] == "alarm")
        & predeclared["feature"].isin(
            [
                "route_activity_w4",
                "response_history_score",
                "front_minus_back_w4",
                "front_lead_asymmetry",
                "layer_dynamics_effrank",
                "layer_mean_abs_corr",
            ]
        )
    ]
    print(take.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nFrozen outcome-blind allocation policies")
    print(policies.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
