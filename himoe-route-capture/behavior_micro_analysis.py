"""Pure statistics for the formal route--outcome micro-pilot.

Pair rows describe geometry, but they are never inference units.  This module
labels and matches pairs inside a snapshot, reduces snapshots inside episodes,
and only then performs task/episode/snapshot resampling.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


Q_MARGIN = 0.1
SCREEN_Q_SAME = 0.125
SCREEN_Q_DIFFERENT = 0.25
PHYSICAL_RESOLUTION_FLOORS = {
    "arm_qpos_native": 0.01,
    "arm_qvel_native": 0.01,
    "eef_orientation_rad": 0.01,
    "eef_position_m": 0.001,
    "gripper_qpos_native": 0.001,
    "object_joint_native": 0.001,
    "object_orientation_rad": 0.01,
    "object_qvel_native": 0.001,
    "object_translation_m": 0.001,
    "robot_qpos_native": 0.01,
    "robot_qvel_native": 0.01,
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"cannot interpret {value!r} as boolean")


def snapshot_uid(row: Mapping[str, Any]) -> str:
    state_hash = str(row.get("snapshot_state_sha256", ""))
    if len(state_hash) == 64:
        return f"task{int(row['task_id'])}/{state_hash}"
    bundle = str(row.get("bundle_id", "bundle"))
    return f"task{int(row['task_id'])}/{bundle}/{row['snapshot']}"


def formal_outcome_relation(
    row: Mapping[str, Any], q_margin: float = Q_MARGIN
) -> str:
    """Label an exact event-tape and paired-CRN outcome comparison."""

    if not np.isfinite(q_margin) or q_margin <= 0.0:
        raise ValueError("q_margin must be finite and positive")
    event_equal = _as_bool(row["events_equal"])
    lower = float(row["q_ci_lower"])
    upper = float(row["q_ci_upper"])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("paired Q interval must be finite and ordered")
    q_same = lower >= -q_margin and upper <= q_margin
    q_different = lower > q_margin or upper < -q_margin
    if event_equal and q_same:
        return "same"
    if not event_equal or q_different:
        return "different"
    return "ambiguous"


def physical_schema(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError("physical schema requires pair rows")
    schema = sorted(
        {
            str(key)[3:]
            for row in rows
            for key in row
            if str(key).startswith("dX_")
        }
    )
    if not schema:
        raise ValueError("pair rows contain no dense physical trajectory components")
    return schema


def physical_presence_masks(
    rows: Sequence[Mapping[str, Any]], schema: Sequence[str]
) -> dict[int, dict[str, bool]]:
    by_task: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(row)
    masks = {}
    for task, material in sorted(by_task.items()):
        mask = {}
        for name in schema:
            present = [f"dX_{name}" in row for row in material]
            if any(present) and not all(present):
                raise ValueError(
                    f"task {task} inconsistently stores physical component {name}"
                )
            mask[name] = bool(all(present))
        if not any(mask.values()):
            raise ValueError(f"task {task} has no component from the frozen physical schema")
        masks[task] = mask
    return masks


def fit_physical_scales(
    rows: Sequence[Mapping[str, Any]], schema: Sequence[str]
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for name in schema:
        values = np.asarray(
            [float(row[f"dX_{name}"]) for row in rows if f"dX_{name}" in row],
            dtype=np.float64,
        )
        if not len(values):
            raise ValueError(f"calibration never observes physical component {name}")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"invalid calibration physical distances for {name}")
        positive = values[values > 0.0]
        empirical = float(np.median(positive)) if len(positive) else 0.0
        floor = PHYSICAL_RESOLUTION_FLOORS.get(name)
        if floor is None:
            raise ValueError(f"no preregistered resolution floor for physical component {name}")
        scales[name] = max(empirical, floor)
    return scales


def apply_physical_scales(
    rows: Iterable[dict[str, Any]], schema: Sequence[str], scales: Mapping[str, float]
) -> None:
    if set(schema) != set(scales):
        raise ValueError("physical scale keys do not match the frozen schema")
    for row in rows:
        mask = [f"dX_{name}" in row for name in schema]
        values = [
            float(row[f"dX_{name}"]) / float(scales[name])
            for name, present in zip(schema, mask)
            if present
        ]
        if not values:
            raise ValueError("evaluation row has no component in the frozen physical schema")
        row["d_physics"] = float(np.sqrt(np.mean(np.square(values))))
        row["d_physics_presence_mask"] = "".join("1" if value else "0" for value in mask)
        row["d_physics_component_count"] = len(values)


def _snapshot_equal_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[snapshot_uid(row)] += 1
    return np.asarray([1.0 / counts[snapshot_uid(row)] for row in rows], dtype=np.float64)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    centers = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    centers /= sorted_weights.sum()
    return float(np.interp(quantile, centers, sorted_values))


def _weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.average(values, weights=weights))
    return float(np.sqrt(np.average(np.square(values - mean), weights=weights)))


def _quadratic_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> list[float]:
    if len(x) < 3 or len(np.unique(x)) < 3:
        raise ValueError("quadratic residual fit needs at least three distinct action distances")
    design = np.column_stack([np.ones(len(x)), x, np.square(x)])
    root_weight = np.sqrt(weights / weights.mean())
    coefficients, *_ = np.linalg.lstsq(
        design * root_weight[:, None], y * root_weight, rcond=None
    )
    return [float(value) for value in coefficients]


def _quadratic_predict(x: float, coefficients: Sequence[float]) -> float:
    if len(coefficients) != 3:
        raise ValueError("quadratic model must contain three coefficients")
    return float(coefficients[0] + coefficients[1] * x + coefficients[2] * x * x)


@dataclass(frozen=True)
class FormalCalibration:
    action_near_q20: float
    action_far_q80: float
    physics_near_q20: float
    physics_far_q80: float
    caliper_near: float
    caliper_far: float
    physical_schema: tuple[str, ...]
    physical_scales: dict[str, float]
    quadratic_coefficients: dict[str, dict[str, list[float]]]
    physical_presence_masks: dict[int, dict[str, bool]]
    calibration_task_ids: tuple[int, ...] = (0,)
    near_quantile: float = 0.2
    far_quantile: float = 0.8
    caliper_multiplier: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["physical_schema"] = list(self.physical_schema)
        value["calibration_task_ids"] = list(self.calibration_task_ids)
        value["physical_presence_masks"] = {
            str(key): mask for key, mask in sorted(self.physical_presence_masks.items())
        }
        value["weighting"] = (
            "snapshot-equal: every calibration snapshot has unit total pair weight"
        )
        return value


def _qualifies(
    row: Mapping[str, Any], stratum: str, calibration: FormalCalibration
) -> bool:
    action = float(row["d_action"])
    if stratum == "near":
        return action <= calibration.action_near_q20
    if stratum == "far":
        return (
            action >= calibration.action_far_q80
            and float(row["d_physics"]) >= calibration.physics_far_q80
        )
    raise ValueError(f"unknown action stratum {stratum!r}")


def fit_formal_calibration(
    rows: Sequence[dict[str, Any]],
    *,
    calibration_task_ids: Sequence[int] = (0,),
    near_quantile: float = 0.2,
    far_quantile: float = 0.8,
    caliper_multiplier: float = 0.2,
) -> FormalCalibration:
    if not 0.0 < near_quantile < far_quantile < 1.0:
        raise ValueError("calibration quantiles must satisfy 0 < near < far < 1")
    task_ids = tuple(sorted({int(value) for value in calibration_task_ids}))
    selected = [row for row in rows if int(row["task_id"]) in task_ids]
    if not selected:
        raise ValueError("no rows belong to the calibration tasks")
    if {int(row["task_id"]) for row in selected} != set(task_ids):
        raise ValueError("one or more calibration tasks have no pair rows")
    schema = physical_schema(selected)
    scales = fit_physical_scales(selected, schema)
    apply_physical_scales(rows, schema, scales)
    action = np.asarray([float(row["d_action"]) for row in selected], dtype=np.float64)
    physics = np.asarray([float(row["d_physics"]) for row in selected], dtype=np.float64)
    all_presence_masks = physical_presence_masks(rows, schema)
    weights = _snapshot_equal_weights(selected)
    provisional = FormalCalibration(
        action_near_q20=_weighted_quantile(action, weights, near_quantile),
        action_far_q80=_weighted_quantile(action, weights, far_quantile),
        physics_near_q20=_weighted_quantile(physics, weights, near_quantile),
        physics_far_q80=_weighted_quantile(physics, weights, far_quantile),
        caliper_near=0.0,
        caliper_far=0.0,
        physical_schema=tuple(schema),
        physical_scales=scales,
        quadratic_coefficients={},
        physical_presence_masks=all_presence_masks,
        calibration_task_ids=task_ids,
        near_quantile=near_quantile,
        far_quantile=far_quantile,
        caliper_multiplier=caliper_multiplier,
    )
    calipers: dict[str, float] = {}
    coefficients: dict[str, dict[str, list[float]]] = {}
    for stratum in ("near", "far"):
        subset = [row for row in selected if _qualifies(row, stratum, provisional)]
        x = np.asarray([float(row["d_action"]) for row in subset], dtype=np.float64)
        subset_weights = _snapshot_equal_weights(subset)
        stratum_sd = _weighted_sd(x, subset_weights)
        if len(x) < 2 or stratum_sd <= 0.0:
            raise ValueError(f"calibration {stratum} stratum has zero action-distance spread")
        calipers[stratum] = caliper_multiplier * stratum_sd
        y = np.asarray([float(row["d_route"]) for row in subset], dtype=np.float64)
        coefficients[stratum] = {
            "d_route": _quadratic_fit(x, y, subset_weights)
        }
    return FormalCalibration(
        **{
            **asdict(provisional),
            "caliper_near": calipers["near"],
            "caliper_far": calipers["far"],
            "quadratic_coefficients": coefficients,
        }
    )


def apply_formal_calibration(
    rows: Iterable[dict[str, Any]], calibration: FormalCalibration
) -> None:
    for row in rows:
        relation = formal_outcome_relation(row)
        row["outcome_relation"] = relation
        if _qualifies(row, "near", calibration):
            stratum = "near"
        elif _qualifies(row, "far", calibration):
            stratum = "far"
        else:
            stratum = "middle"
        row["action_stratum"] = stratum
        if stratum not in {"near", "far"}:
            continue
        for metric, coefficients in calibration.quadratic_coefficients[stratum].items():
            row[f"{metric}_residual"] = float(row[metric]) - _quadratic_predict(
                float(row["d_action"]), coefficients
            )


def minimum_cost_action_matching(
    same_action: Sequence[float],
    different_action: Sequence[float],
    caliper: float,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, then minimum-cost, action-only matching."""

    left = np.asarray(same_action, dtype=np.float64)
    right = np.asarray(different_action, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("matching inputs must be one-dimensional")
    if not len(left) or not len(right):
        return []
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("matching inputs must be finite")
    if not np.isfinite(caliper) or caliper <= 0.0:
        raise ValueError("matching caliper must be finite and positive")
    n_left, n_right = len(left), len(right)
    size = n_left + n_right
    penalty = (size + 1.0) * caliper
    invalid = (size + 1.0) * penalty
    cost = np.full((size, size), invalid, dtype=np.float64)
    pair_cost = np.abs(left[:, None] - right[None, :])
    cost[:n_left, :n_right] = np.where(pair_cost <= caliper, pair_cost, invalid)
    cost[:n_left, n_right:] = penalty
    cost[n_left:, :n_right] = penalty
    cost[n_left:, n_right:] = 0.0
    row_index, column_index = linear_sum_assignment(cost)
    matches = []
    for left_index, right_index in zip(row_index, column_index):
        if left_index < n_left and right_index < n_right:
            value = float(pair_cost[left_index, right_index])
            if value <= caliper:
                matches.append((int(left_index), int(right_index), value))
    return sorted(matches)


def matched_snapshot_effects(
    rows: Sequence[Mapping[str, Any]],
    calibration: FormalCalibration,
    *,
    min_matches: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if min_matches < 1:
        raise ValueError("min_matches must be positive")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        stratum = str(row.get("action_stratum", "middle"))
        if stratum in {"near", "far"}:
            grouped[(snapshot_uid(row), stratum)].append(row)
    effects: list[dict[str, Any]] = []
    censoring: list[dict[str, Any]] = []
    for (uid, stratum), material in sorted(grouped.items()):
        same = [row for row in material if row.get("outcome_relation") == "same"]
        different = [row for row in material if row.get("outcome_relation") == "different"]
        ambiguous = sum(row.get("outcome_relation") == "ambiguous" for row in material)
        caliper = calibration.caliper_near if stratum == "near" else calibration.caliper_far
        matches = minimum_cost_action_matching(
            [float(row["d_action"]) for row in same],
            [float(row["d_action"]) for row in different],
            caliper,
        )
        first = material[0]
        audit = {
            "snapshot_uid": uid,
            "snapshot": str(first["snapshot"]),
            "task_id": int(first["task_id"]),
            "episode": int(first["episode"]),
            "stratum": stratum,
            "same_pair_count": len(same),
            "different_pair_count": len(different),
            "same_unique_candidate_count": len(
                {
                    int(row[key])
                    for row in same
                    for key in ("candidate_i", "candidate_j")
                }
            ),
            "different_unique_candidate_count": len(
                {
                    int(row[key])
                    for row in different
                    for key in ("candidate_i", "candidate_j")
                }
            ),
            "ambiguous_pair_count": int(ambiguous),
            "matched_count": len(matches),
            "caliper": caliper,
            "max_matched_action_gap": max((item[2] for item in matches), default=None),
        }
        if (
            len(matches) < min_matches
            or audit["same_unique_candidate_count"] < 4
            or audit["different_unique_candidate_count"] < 4
        ):
            audit["reason"] = (
                "insufficient action-only matches or fewer than four unique candidates "
                "in an outcome group"
            )
            censoring.append(audit)
            continue
        metric_keys = {
            "route_effect": "d_route",
            "route_residual_effect": "d_route_residual",
            "hidden_effect": "d_hidden",
            "flow_effect": "d_flow",
            "physics_effect": "d_physics",
        }
        result = dict(audit)
        for output, metric in metric_keys.items():
            differences = [
                float(different[right][metric]) - float(same[left][metric])
                for left, right, _cost in matches
            ]
            result[output] = float(np.mean(differences))
        result["matched_pairs"] = [
            {
                "same": [int(same[left]["candidate_i"]), int(same[left]["candidate_j"])],
                "different": [
                    int(different[right]["candidate_i"]),
                    int(different[right]["candidate_j"]),
                ],
                "action_gap": cost,
            }
            for left, right, cost in matches
        ]
        effects.append(result)
    return effects, censoring


def screen_snapshot_flags(
    rows: Sequence[Mapping[str, Any]],
    calibration: FormalCalibration,
    *,
    min_pairs: int = 2,
    min_candidates: int = 3,
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for stratum in ("near", "far"):
            if _qualifies(row, stratum, calibration):
                grouped[(snapshot_uid(row), stratum)].append(row)
                break
    result: dict[str, dict[str, Any]] = {}
    for (uid, stratum), material in sorted(grouped.items()):
        same = []
        different = []
        for row in material:
            event_equal = _as_bool(row["events_equal"])
            gap = abs(float(row["q_i"]) - float(row["q_j"]))
            if event_equal and gap <= SCREEN_Q_SAME:
                same.append(row)
            if not event_equal or gap >= SCREEN_Q_DIFFERENT:
                different.append(row)
        def candidate_support(values: Sequence[Mapping[str, Any]]) -> set[int]:
            return {
                int(row[key])
                for row in values
                for key in ("candidate_i", "candidate_j")
            }
        flagged = (
            len(same) >= min_pairs
            and len(different) >= min_pairs
            and len(candidate_support(same)) >= min_candidates
            and len(candidate_support(different)) >= min_candidates
        )
        result.setdefault(uid, {})[stratum] = {
            "screen_positive": bool(flagged),
            "same_pair_count": len(same),
            "different_pair_count": len(different),
            "same_candidate_count": len(candidate_support(same)),
            "different_candidate_count": len(candidate_support(different)),
        }
    return result


def screening_metrics(
    formal_snapshots: Sequence[Mapping[str, Any]],
    eligible_uids: Mapping[str, set[str]],
    screen_flags: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for stratum in ("near", "far"):
        counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        raw = {key: 0 for key in counts}
        for snapshot in formal_snapshots:
            uid = str(snapshot["snapshot_uid"])
            probability = float(snapshot.get("inclusion_probability", 1.0))
            if not 0.0 < probability <= 1.0:
                raise ValueError("formal inclusion probability must lie in (0,1]")
            weight = 1.0 / probability
            predicted = bool(
                screen_flags.get(uid, {}).get(stratum, {}).get("screen_positive", False)
            )
            actual = uid in eligible_uids.get(stratum, set())
            cell = ("t" if actual else "f") + ("p" if predicted else "n")
            counts[cell] += weight
            raw[cell] += 1
        recall_denominator = counts["tp"] + counts["fn"]
        precision_denominator = counts["tp"] + counts["fp"]
        output[stratum] = {
            "weighted_counts": counts,
            "raw_counts": raw,
            "recall": (
                counts["tp"] / recall_denominator if recall_denominator else None
            ),
            "ppv": counts["tp"] / precision_denominator if precision_denominator else None,
        }
    return output


def q_label_coverage_and_topup(
    rows: Sequence[Mapping[str, Any]],
    repeats: Sequence[int],
    *,
    eligible_counts: Mapping[str, int] | None = None,
    evaluation_task_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Report pair-label coverage and the preregistered global top-up decision.

    Ambiguous paired-Q intervals are unlabeled pairs. They are deliberately not
    called horizon-censored continuations; that is audited from the execution
    tapes separately.
    """

    repeat_set = sorted({int(value) for value in repeats})
    if not repeat_set:
        raise ValueError("continuation repeats are unavailable")
    uniform = len(repeat_set) == 1
    current = repeat_set[0] if uniform else None
    cohorts: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        stratum = str(row.get("action_stratum", "middle"))
        if stratum in {"near", "far"}:
            cohorts[(int(row["task_id"]), stratum)].append(row)
    task_ids = tuple(
        sorted(
            {int(value) for value in evaluation_task_ids}
            if evaluation_task_ids is not None
            else {int(row["task_id"]) for row in rows}
        )
    )
    coverage = {}
    any_low_coverage = False
    ambiguous_total = 0
    relevant_total = 0
    for task in task_ids:
        for stratum in ("near", "far"):
            material = cohorts.get((task, stratum), [])
            labeled = sum(
                row.get("outcome_relation") in {"same", "different"}
                for row in material
            )
            ambiguous = len(material) - labeled
            by_snapshot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in material:
                by_snapshot[snapshot_uid(row)].append(row)
            snapshot_coverage = [
                sum(
                    row.get("outcome_relation") in {"same", "different"}
                    for row in snapshot_rows
                )
                / len(snapshot_rows)
                for snapshot_rows in by_snapshot.values()
            ]
            value = float(np.mean(snapshot_coverage)) if snapshot_coverage else 0.0
            any_low_coverage |= value < 0.70
            ambiguous_total += ambiguous
            relevant_total += len(material)
            coverage[f"task{task}/{stratum}"] = {
                "pairs": len(material),
                "labeled": int(labeled),
                "ambiguous": int(ambiguous),
                "coverage": value,
                "snapshot_count": len(by_snapshot),
                "snapshot_equal_weighting": True,
            }
    counts = {key: int(value) for key, value in (eligible_counts or {}).items()}
    too_few_eligible = any(counts.get(stratum, 0) < 4 for stratum in ("near", "far"))
    topup = bool(uniform and current == 48 and (any_low_coverage or too_few_eligible))
    return {
        "uniform_repeats": uniform,
        "observed_repeats": repeat_set,
        "label_coverage_by_eval_task_and_stratum": coverage,
        "q_ambiguous_pairs": ambiguous_total,
        "relevant_pairs": relevant_total,
        "eligible_snapshot_counts": counts,
        "topup_trigger_any_cohort_coverage_below_0_70": any_low_coverage,
        "topup_trigger_main_stratum_eligible_below_4": too_few_eligible,
        "global_topup_required": topup,
        "global_topup_target_repeats": 96 if topup else None,
        "partial_snapshot_topup_forbidden": True,
    }
def _macro_effect(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> tuple[float, dict[int, float]]:
    by_task: dict[int, dict[int, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_task[int(row["task_id"])][int(row["episode"])].append(row)
    task_effects: dict[int, float] = {}
    for task, episodes in by_task.items():
        episode_effects = []
        for snapshots in episodes.values():
            values = np.asarray([float(row[metric]) for row in snapshots], dtype=np.float64)
            weights = np.asarray(
                [1.0 / float(row.get("inclusion_probability", 1.0)) for row in snapshots],
                dtype=np.float64,
            )
            episode_effects.append(float(np.average(values, weights=weights)))
        task_effects[task] = float(np.mean(episode_effects))
    return float(np.mean(list(task_effects.values()))), task_effects


def hierarchical_effect_summary(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    draws: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {"available": False, "reason": "no eligible matched snapshots"}
    if draws < 100:
        raise ValueError("hierarchical bootstrap needs at least 100 draws")
    mean, task_effects = _macro_effect(rows, metric)
    task_ids = sorted(task_effects)
    result: dict[str, Any] = {
        "available": len(task_ids) >= 2,
        "mean": mean,
        "task_effects": {str(key): value for key, value in sorted(task_effects.items())},
        "task_count": len(task_ids),
        "episode_count": len({(int(row["task_id"]), int(row["episode"])) for row in rows}),
        "snapshot_count": len(rows),
        "inference_hierarchy": "task -> episode -> snapshot; pairs reduced before inference",
    }
    if len(task_ids) < 2:
        result.update({"lower": None, "upper": None, "draws": 0})
        return result
    nested: dict[int, dict[int, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        nested[int(row["task_id"])][int(row["episode"])].append(row)
    rng = np.random.default_rng(seed)
    distribution = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
        task_values = []
        for task in sampled_tasks:
            episode_ids = sorted(nested[int(task)])
            sampled_episodes = rng.choice(episode_ids, size=len(episode_ids), replace=True)
            episode_values = []
            for episode in sampled_episodes:
                snapshots = nested[int(task)][int(episode)]
                selected = rng.integers(0, len(snapshots), size=len(snapshots))
                material = [snapshots[int(index)] for index in selected]
                values = np.asarray([float(row[metric]) for row in material])
                weights = np.asarray(
                    [1.0 / float(row.get("inclusion_probability", 1.0)) for row in material]
                )
                episode_values.append(float(np.average(values, weights=weights)))
            task_values.append(float(np.mean(episode_values)))
        distribution[draw] = float(np.mean(task_values))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(distribution, [tail, 1.0 - tail])
    result.update({"lower": float(lower), "upper": float(upper), "draws": draws})
    return result


def deterministic_hidden_projection(
    hidden: np.ndarray, *, output_width: int, seed: int
) -> np.ndarray:
    value = np.asarray(hidden, dtype=np.float32)
    if value.ndim < 2 or value.shape[-1] < 1:
        raise ValueError("hidden tensor must have a non-empty feature axis")
    if output_width < 1:
        raise ValueError("projection output width must be positive")
    rng = np.random.default_rng(seed)
    projection = rng.choice((-1.0, 1.0), size=(value.shape[-1], output_width)).astype(
        np.float32
    )
    projection /= np.sqrt(float(output_width))
    return np.tensordot(value, projection, axes=([-1], [0]))


def pairwise_rms_distance(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or len(array) < 2:
        raise ValueError("pairwise distance needs at least two candidate tensors")
    flat = array.reshape(len(array), -1)
    difference = flat[:, None, :] - flat[None, :, :]
    return np.sqrt(np.mean(np.square(difference), axis=-1))
